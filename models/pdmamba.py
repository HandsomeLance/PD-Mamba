# ============================================================================
# PD-Mamba: PPG-Dedicated Koopman-Enhanced Mamba Encoder
# ============================================================================
# This module contains the PPG encoder used for downstream classification.
# It is composed of two stages:
#
#   1) KoopmanDecomposer (see koopman_decomposer.py): a finite-dimensional
#      Koopman operator approximation that lifts sliding-window PPG states
#      into a Hilbert observation space and projects the linearly evolved
#      coordinates back to complementary physiological components.
#
#   2) MambaEncoderMAE: a multi-scale patch embedding + bidirectional Mamba
#      (Bi-Mamba) sequence encoder that produces patch-level token
#      representations from the decomposed components.
#
# The PDMBambaEncoder wrapper chains the two stages and emits the patch-level
# latent sequence consumed by the downstream SDF module.
#
# Dependencies:
#   - torch >= 1.13
#   - mamba-ssm (provides `from mamba_ssm import Mamba`)
#
# Only the model architecture and forward inference logic are included here.
# Training-specific code (losses, data loaders, pretraining routines, etc.)
# is intentionally omitted.
# ============================================================================

import torch
import torch.nn as nn
from mamba_ssm import Mamba

from koopman_decomposer import KoopmanDecomposer


# ----------------------------------------------------------------------------
# MultiScalePatchEmbedding
# ----------------------------------------------------------------------------
class MultiScalePatchEmbedding(nn.Module):
    """Multi-scale 1D patch embedding.

    A shared initial projection is followed by three parallel strided
    convolution branches (kernel sizes 3/7/15) whose features are fused by a
    1x1 convolution and a squeeze-and-excitation block. The output is
    organized as patch-level token sequences for the Mamba backbone.
    """

    class _SEBlock(nn.Module):
        """Squeeze-and-excitation channel recalibration block."""

        def __init__(self, channels, reduction=4):
            super().__init__()
            self.squeeze = nn.AdaptiveAvgPool1d(1)
            self.dims = nn.Sequential(
                nn.Linear(channels, channels // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channels // reduction, channels),
                nn.Sigmoid(),
            )

        def forward(self, x):
            b, c, _ = x.size()
            y = self.squeeze(x).view(b, c)
            y = self.dims(y).view(b, c, 1)
            return x * y

    def __init__(self, in_chans=4, embed_dim=64, patch_len=160, dropout_rate=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = 4  # stride-based downsampling instead of global pooling

        self.initial_proj = nn.Sequential(
            nn.Conv1d(in_chans, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim // 2),
            nn.ReLU(inplace=True),
        )

        self.k_sizes = [3, 7, 15]
        self.multiscale_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim // 2, embed_dim // 2, k,
                          padding=k // 2, stride=self.stride),
                nn.BatchNorm1d(embed_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            ) for k in self.k_sizes
        ])

        self.fusion = nn.Sequential(
            nn.Conv1d((embed_dim // 2) * len(self.k_sizes), embed_dim, kernel_size=1),
            self._SEBlock(embed_dim),
        )

    def forward(self, x):
        # x: (B, N, C, L)
        B, N, C, L = x.shape
        x = x.reshape(B * N, C, L)

        x = self.initial_proj(x)
        feats = [block(x) for block in self.multiscale_blocks]
        x = torch.cat(feats, dim=1)
        x = self.fusion(x)  # (B*N, embed_dim, L/stride)

        _, D, L_new = x.shape
        # Reorganize for Mamba: (B, N, L_new, D)
        x = x.view(B, N, D, L_new).permute(0, 1, 3, 2).contiguous()
        return x, L_new


# ----------------------------------------------------------------------------
# BiMambaBlock
# ----------------------------------------------------------------------------
class BiMambaBlock(nn.Module):
    """Physiological-signal enhanced bidirectional Mamba block.

    A forward scan and a backward scan (via sequence flip) are fused by a
    linear projection. The bidirectional scanning mitigates phase offset
    artifacts in PPG pulse-wave feature extraction.
    """

    def __init__(self, embed_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)

        # Forward and backward Mamba (d_state/d_conv/expand match the source
        # encoder definition).
        self.fwd_mamba = Mamba(d_model=embed_dim, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        self.bwd_mamba = Mamba(d_model=embed_dim, d_state=d_state,
                               d_conv=d_conv, expand=expand)

        # Fuse 2*embed_dim back to embed_dim.
        self.final_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x):
        res = x
        x = self.norm(x)

        # Forward path.
        x_fwd = self.fwd_mamba(x)

        # Backward path (reverse-order processing via flip).
        x_bwd = self.bwd_mamba(x.flip(dims=[1])).flip(dims=[1])

        # Feature fusion.
        out = self.final_proj(torch.cat([x_fwd, x_bwd], dim=-1))
        return out + res


# ----------------------------------------------------------------------------
# MambaEncoderMAE
# ----------------------------------------------------------------------------
class MambaEncoderMAE(nn.Module):
    """Multi-scale patch embedding + Bi-Mamba encoder.

    The encoder consumes patch tokens produced from the Koopman-decomposed
    components, optionally keeps a subset of patches (MAE-style masking), and
    returns patch-level token representations.

    Note: the final scatter is functional (`torch.scatter`), not in-place
    (`scatter_`), so gradients flow back through the encoder output.
    """

    def __init__(self, patch_len=64, in_chans=4, embed_dim=64, depth=3,
                 num_patches=30):
        super().__init__()
        self.num_patches = num_patches
        self.patch_embed = MultiScalePatchEmbedding(in_chans, embed_dim, patch_len)

        # Block-level positional embedding.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.layers = nn.ModuleList([
            BiMambaBlock(embed_dim=embed_dim) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x, ids_keep):
        # 1. Patch embedding -> (B, N, L_new, D).
        x_reshaped, L_new = self.patch_embed(x)
        B, N, _, D = x_reshaped.shape

        # 2. Gather the visible patches.
        keep_idx = ids_keep.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, L_new, D)
        x_visible = torch.gather(x_reshaped, dim=1, index=keep_idx)  # (B, num_keep, L_new, D)

        # 3. Inject positional embeddings (broadcast inside each patch).
        vis_pos = torch.gather(self.pos_embed.expand(B, -1, -1), dim=1,
                               index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        x_visible = x_visible + vis_pos.unsqueeze(2)

        # 4. Sort by temporal order and flatten into a long sequence.
        sort_idx = torch.argsort(ids_keep, dim=1)
        x_ordered = torch.gather(
            x_visible, dim=1,
            index=sort_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, L_new, D))

        # Mamba scan length: num_keep * L_new (e.g. 30 * 16 = 480).
        x_mamba_in = x_ordered.view(B, -1, D)

        # 5. Deep temporal modeling with Bi-Mamba layers.
        for layer in self.layers:
            x_mamba_in = layer(x_mamba_in)
        x_mamba_out = self.norm(x_mamba_in)

        # 6. Post-pooling: recover patch-level tokens (B, num_keep, D).
        x_rebuilt = x_mamba_out.view(B, -1, L_new, D)
        out_patches = x_rebuilt.mean(dim=2)

        # 7. Restore the original patch order (functional scatter keeps graph).
        out = torch.scatter(
            torch.zeros_like(out_patches),
            dim=1,
            index=sort_idx.unsqueeze(-1).expand(-1, -1, D),
            src=out_patches,
        )
        return out


# ----------------------------------------------------------------------------
# PDMBambaEncoder (wrapper for downstream use)
# ----------------------------------------------------------------------------
class PDMBambaEncoder(nn.Module):
    """PD-Mamba encoder: Koopman decomposition -> patch embedding -> Bi-Mamba.

    This is the full PPG encoder used in downstream classification. It wraps
    the KoopmanDecomposer and MambaEncoderMAE stages and mirrors the inference
    path of the original downstream network (see DownstreamWESADNet.forward):

        x_comp   = decomposer(x_ppg)[0]                        # (B, C, L)
        patches  = x_comp.view(B, C, num_patches, -1).permute(0, 2, 1, 3)
        z_ppg    = mae_encoder(patches, ids_all)               # (B, N, D)

    Args:
        latent_dim: embedding dimension D (Koopman dim == encoder embed dim).
        num_patches: number of patch tokens N.
        patch_len: length of each patch after windowing (L / num_patches).
        num_components: number of Koopman components C (default 4).
        koop_window / koop_stride: sliding-window params of the decomposer.
        depth: number of Bi-Mamba layers.
        mamba_d_state / mamba_d_conv / mamba_expand: Mamba SSM parameters.

    Input:
        x_ppg: (B, 1, L) raw PPG segment.

    Output:
        z_ppg: (B, N, D) patch-level latent sequence for the SDF module.
    """

    def __init__(self, latent_dim=64, num_patches=30, patch_len=64,
                 num_components=4, koop_window=16, koop_stride=16,
                 depth=3, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_patches = num_patches
        self.patch_len = patch_len
        self.num_components = num_components

        self.decomposer = KoopmanDecomposer(
            in_channels=1,
            num_components=num_components,
            koop_dim=latent_dim,
            window=koop_window,
            stride=koop_stride,
        )
        self.mae_encoder = MambaEncoderMAE(
            patch_len=patch_len,
            in_chans=num_components,
            embed_dim=latent_dim,
            depth=depth,
            num_patches=num_patches,
        )

    def forward(self, x_ppg):
        B = x_ppg.shape[0]

        # Koopman decomposition -> complementary components (B, C, L).
        x_comp = self.decomposer(x_ppg)[0]

        # Reshape components into patches: (B, N, C, patch_len).
        patches = x_comp.view(B, self.num_components,
                              self.num_patches, -1).permute(0, 2, 1, 3)

        # Inference uses all patches (no masking).
        ids_all = torch.arange(self.num_patches, device=x_ppg.device).repeat(B, 1)

        # Bi-Mamba encoding -> (B, N, D).
        z_ppg = self.mae_encoder(patches, ids_all)
        return z_ppg
