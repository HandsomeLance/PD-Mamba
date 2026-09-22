# ============================================================================
# KoopmanDecomposer: Finite-Dimensional Koopman Operator Approximation
# ============================================================================
# This module contains the Koopman decomposition stage of the PD-Mamba
# encoder (paper II. PRELIMINARY, Koopman Operator Theory). The decomposer
# lifts sliding-window PPG states into a Hilbert observation space, evolves
# them with a trainable finite-dimensional transition matrix K, and projects
# the evolved coordinates to C complementary physiological components.
#
# It also produces the Koopman consistency supervision quantities
# (y_true_next / y_pred_next / K) used during pretraining.
#
# Dependencies:
#   - torch >= 1.13
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class KoopmanDecomposer(nn.Module):
    """Finite-dimensional Koopman operator approximation.

    Theoretical correspondence (paper II. PRELIMINARY, Koopman operator theory):

      1) Observation function phi: lifts each sliding-window state
         l_t in R^w to the Hilbert observation space y_t = phi(l_t) in R^K,
         where K >> w (nonlinear expansion).
      2) Trainable finite-dimensional transition matrix K in R^{K x K}:
             y_evolved_t = K y_t
         approximates the true adjacent-state evolution y_{t+1}, which is
         directly obtained by applying phi to the next window.
      3) Linear projection D (Koopman mode readout): projects the evolved
         coordinates in the lifted space to C complementary physiological
         components, z_t^c = D_c . y_evolved_t.

    Args:
        in_channels: number of input signal channels (always 1 for PPG).
        num_components: number of complementary components C (default 4).
        koop_dim: dimension of the Hilbert observation space K (default 64).
        window: sliding-window length used for state sampling (default 16).
        stride: sliding-window stride (default 16, non-overlapping windows).

    Input:
        x: (B, 1, L) raw PPG segment.

    Output:
        components: (B, C, L) complementary components upsampled back to length L.
        y_pred_next / y_true_next / K: Koopman consistency supervision quantities
            (used during pretraining; downstream inference only needs `components`).
    """

    def __init__(self, in_channels=1, num_components=4,
                 koop_dim=64, window=16, stride=16):
        super().__init__()
        self.koop_dim = koop_dim
        self.window = window
        self.stride = stride
        self.num_components = num_components

        # Observation function phi: window state -> Hilbert observation space R^K.
        self.phi = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, koop_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(koop_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # -> (B*T, koop_dim, 1)
        )

        # Finite-dimensional Koopman transition matrix.
        self.K = nn.Parameter(torch.randn(koop_dim, koop_dim) * 0.1)

        # Koopman mode readout: lifted-space coordinates -> C components.
        self.projector = nn.Linear(koop_dim, num_components)

    def forward(self, x):
        B, C, L = x.shape

        # 1. Sample non-overlapping sliding windows.
        #    unfold -> (B, 1, T, w), T = (L - window) // stride + 1
        windows = x.unfold(-1, self.window, self.stride)  # (B, 1, T, w)
        windows = windows.squeeze(1)                      # (B, T, w)

        # 2. Observation lifting: each window -> one Hilbert observation point.
        BT, T, w = windows.shape
        y = self.phi(windows.reshape(BT * T, 1, w))       # (BT*T, koop_dim, 1)
        y = y.reshape(B, T, self.koop_dim)                # (B, T, koop_dim)

        # 3. Linear time evolution: y_evolved_t = K y_t.
        y_evolved = torch.matmul(y, self.K)               # (B, T, koop_dim)

        # 4. Project to C complementary components and upsample to length L.
        proj = self.projector(y_evolved)                  # (B, T, C)
        components = proj.permute(0, 2, 1)                # (B, C, T)
        components = F.interpolate(components, size=L, mode='linear',
                                   align_corners=False)

        # 5. Koopman consistency supervision quantities:
        #    true next state y_{t+1} (from phi on the next window)
        #    predicted next state K y_t
        y_true_next = y[:, 1:]                            # (B, T-1, koop_dim)
        y_pred_next = torch.matmul(y[:, :-1], self.K)     # (B, T-1, koop_dim)

        return components, y_pred_next, y_true_next, self.K
