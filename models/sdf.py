# ============================================================================
# SDF: Symplectic Dynamic Fusion
# ============================================================================
# This module implements the downstream fusion network that
# combines PPG latent representations produced by PD-Mamba with an auxiliary
# stress (SKT) signal:
#
#   1) STEncoder           : spatio-temporal feature extractor that encodes
#                            the SKT signal into a global condition z_st.
#   2) HamiltonianDynamics : learnable Hamiltonian dynamics H_theta(z, z_st)
#                            with a hand-written symplectic (leapfrog)
#                            integrator evolving z_ppg from t=0 to t=1.
#   4) PhysioFusion_v3_Symplectic (SDF): final fusion of the evolved state
#                            with sequence context and skip connections.
#
# Only the model architecture and forward inference logic are included here.
# Training-specific code (losses, data loaders, cross-validation, etc.) is
# intentionally omitted.
#
# Dependencies:
#   - torch >= 1.13
# ============================================================================

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# TemporalAttentionBlock
# ----------------------------------------------------------------------------
class TemporalAttentionBlock(nn.Module):
    """Temporal channel-attention block (squeeze-excitation over time)."""

    def __init__(self, channels):
        super().__init__()
        self.atn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv1d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.atn(x)


# ----------------------------------------------------------------------------
# FeatureExtractor
# ----------------------------------------------------------------------------
class FeatureExtractor(nn.Module):
    """Dilated residual feature extractor with temporal attention."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.SiLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.SiLU(),
        )
        self.atn = TemporalAttentionBlock(out_ch)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        x = self.conv(x)
        x = self.atn(x)
        return x + res


# ----------------------------------------------------------------------------
# STEncoder
# ----------------------------------------------------------------------------
class STEncoder(nn.Module):
    """Spatio-temporal (ST) encoder for the auxiliary SKT signal.

    A micro branch (small kernel, small dilation) and a macro branch (large
    kernel, large dilation) extract short- and long-range dynamics from the
    first/second-order differences of the input signal. The fused bottleneck
    is projected to a global condition vector z_st.
    """

    def __init__(self, input_channels=1, output_dim=32):
        super().__init__()
        self.micro_branch = nn.Sequential(
            FeatureExtractor(input_channels, 24, kernel_size=7, dilation=1),
            FeatureExtractor(24, 24, kernel_size=7, dilation=2),
        )
        self.macro_branch = nn.Sequential(
            FeatureExtractor(input_channels, 24, kernel_size=11, dilation=4),
            FeatureExtractor(24, 24, kernel_size=11, dilation=8),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(48, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # First and second temporal differences.
        d1 = torch.diff(x, dim=-1, prepend=x[:, :, :1])
        d2 = torch.diff(d1, dim=-1, prepend=d1[:, :, :1])

        f_micro = self.micro_branch(d2 + x)
        f_macro = self.macro_branch(d1)
        combined = torch.cat([f_micro, f_macro], dim=1)

        bottleneck = self.fusion(combined).view(x.size(0), -1)
        z_st = self.head(bottleneck)
        return z_st


# ----------------------------------------------------------------------------
# STGuidedAttention (interface only)
# ----------------------------------------------------------------------------
class STGuidedAttention(nn.Module):
    def __init__(self, latent_dim=64, num_patches=10):
        super().__init__()
        pass


# ----------------------------------------------------------------------------
# HamiltonianDynamics
# ----------------------------------------------------------------------------
class HamiltonianDynamics(nn.Module):
    """Learnable Hamiltonian dynamics.

        H_theta(z, z_st) = MLP_theta(concat(z, expand(z_st))),  z = [q, p]
        dz/dt = J grad_z H,  J = [[0, I], [-I, 0]]

    A hand-written symplectic integrator (leapfrog / Störmer-Verlet
    kick-drift-kick form) evolves the state from t=0 to t=1. The computation
    graph is fully preserved (no detach), so gradients flow back to z_ppg and
    H_net parameters.
    """

    def __init__(self, feature_dim, st_dim):
        super().__init__()
        self.dim = feature_dim // 2
        # Hamiltonian energy function H(q, p; z_st). No bias in the output
        # layer: a constant shift does not affect dz/dt = J grad H.
        self.H_net = nn.Sequential(
            nn.Linear(feature_dim + st_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False),
        )

    def _grad_H(self, z, st_cond):
        # During training z inherits the z_ppg graph (requires_grad=True) and
        # gradients flow back fully. Under inference (outer torch.no_grad())
        # z is locally re-enabled to compute the evolution only.
        with torch.enable_grad():
            if not z.requires_grad:
                z = z.detach().requires_grad_(True)
            H = self.H_net(torch.cat([z, st_cond], dim=-1)).sum()
            return torch.autograd.grad(H, z, create_graph=True)[0]

    def _leapfrog_step(self, q, p, st_cond, h):
        # Kick half-step: p -= (h/2) dH/dq at (q, p).
        g = self._grad_H(torch.cat([q, p], dim=-1), st_cond)
        gq, _ = torch.split(g, self.dim, dim=-1)
        p_half = p - (h / 2.0) * gq

        # Drift full-step: q += h dH/dp at (q, p_half).
        g_mid = self._grad_H(torch.cat([q, p_half], dim=-1), st_cond)
        _, gp_mid = torch.split(g_mid, self.dim, dim=-1)
        q_new = q + h * gp_mid

        # Kick half-step: p -= (h/2) dH/dq at (q_new, p_half).
        g_new = self._grad_H(torch.cat([q_new, p_half], dim=-1), st_cond)
        gq_new, _ = torch.split(g_new, self.dim, dim=-1)
        p_new = p_half - (h / 2.0) * gq_new
        return q_new, p_new

    def forward(self, z0, st_cond, n_steps=8):
        # z0: (B, N, D) initial state (z_ppg); st_cond: (B, N, d_st).
        q, p = torch.split(z0, self.dim, dim=-1)
        h = 1.0 / n_steps
        for _ in range(n_steps):
            q, p = self._leapfrog_step(q, p, st_cond, h)
        return torch.cat([q, p], dim=-1)


# ----------------------------------------------------------------------------
# PhysioFusion_v3_Symplectic (SDF core)
# ----------------------------------------------------------------------------
class PhysioFusion_v3_Symplectic(nn.Module):
    """SDF core: Symplectic Dynamic Fusion.

        z_evolved = SymplecticIntegrator(J grad_z H_theta, z_ppg, [0, 1])
        z_final   = Refine(z_evolved + z_seq + z_skip)

    The interface matches PhysioFusion_v3_MoE: returns (z_final_mean, None).
    """

    def __init__(self, feature_dim=128, st_dim=32, n_steps=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_steps = n_steps
        self.dynamics = HamiltonianDynamics(feature_dim, st_dim)

        # Final normalization + SiLU.
        self.refine = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
        )

    def forward(self, z_ppg, z_refined_seq, skip_max, z_st):
        # Inputs: z_ppg (B, N, D), z_refined_seq (B, N, D),
        #         skip_max (B, D), z_st (B, d_st).
        B, N, D = z_ppg.shape

        # 1. Broadcast the global SKT statistic to every patch as a
        #    conditional potential field: (B, N, d_st).
        st_cond = z_st.unsqueeze(1).expand(-1, N, -1)

        # 2. Hand-written symplectic integration of z_ppg (t=0 -> t=1).
        z_evolved = self.dynamics(z_ppg, st_cond, n_steps=self.n_steps)

        # 3. Fuse evolved state with sequence context and skip connection.
        z_fused = z_evolved + z_refined_seq + skip_max.unsqueeze(1)

        # 4. Refine and mean-pool to align with the downstream head.
        z_final = self.refine(z_fused)
        return z_final.mean(dim=1), None


# ----------------------------------------------------------------------------
# SDFNet (wrapper for standalone use)
# ----------------------------------------------------------------------------
class SDFNet(nn.Module):
    """Standalone SDF module: ST encoder -> ST-guided attention -> fusion.

    This wrapper mirrors the downstream path of the original integrated model
    (STEncoder -> ST-guided attention interface -> skip projection -> fusion).

    Args:
        latent_dim: embedding dimension D (must match PD-Mamba output).
        num_patches: number of patch tokens N (must match PD-Mamba).
        st_dim: output dimension of the ST encoder (default 32).
        n_steps: number of symplectic integration steps (default 8).

    Inputs:
        z_ppg: (B, N, D) PPG latent sequence from PD-Mamba.
        x_st:  (B, 1, L) auxiliary SKT signal (same length as the PPG input).

    Outputs:
        z_final: (B, D) fused classification feature.
        (None,): auxiliary output kept for interface compatibility.
    """

    def __init__(self, latent_dim=64, num_patches=30, st_dim=32, n_steps=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_patches = num_patches

        self.st_encoder = STEncoder(input_channels=1, output_dim=st_dim)
        self.st_guided_attn = STGuidedAttention(latent_dim=latent_dim,
                                                num_patches=num_patches)
        self.skip_proj = nn.Linear(latent_dim, latent_dim)
        self.fusion = PhysioFusion_v3_Symplectic(feature_dim=latent_dim,
                                                 st_dim=st_dim,
                                                 n_steps=n_steps)

    def forward(self, z_ppg, x_st):
        z_st = self.st_encoder(x_st)
        _, z_hgr_seq, _ = self.st_guided_attn(z_ppg, x_st)
        skip_max, _ = torch.max(self.skip_proj(z_ppg), dim=1)
        z_final, aux = self.fusion(z_ppg, z_hgr_seq, skip_max, z_st)
        return z_final, aux
