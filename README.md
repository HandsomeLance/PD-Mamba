# PD-Mamba & SDF — Model Code

Model code for the paper's PPG-based physiological signal recognition pipeline.
Only the **model architecture and forward inference logic** are released here;
training scripts, data loaders, losses, and pretraining routines are **not**
included.

```
models/
├── koopman_decomposer.py   # Koopman decomposition stage of PD-Mamba
├── pdmamba.py              # PD-Mamba: Koopman-enhanced PPG encoder (Bi-Mamba backbone)
├── sdf.py                  # SDF: Symplectic Dynamic Fusion
└── README.md               # this file
```

---

## 1. Dependencies

- Python >= 3.8, PyTorch >= 1.13
- [mamba-ssm](https://github.com/state-spaces/mamba) (`pip install mamba-ssm`)
- **GPU is required**: the `mamba-ssm` causal convolution kernel is CUDA-only.
  Move the model and inputs to `cuda` before inference.

---

## 2. PD-Mamba (PPG Encoder)

Files: `koopman_decomposer.py`, `pdmamba.py`

### Architecture

1. **KoopmanDecomposer** (`koopman_decomposer.py`) — finite-dimensional
   Koopman operator approximation: sliding-window states are lifted by `phi`
   into a Hilbert observation space, evolved by a trainable transition matrix
   `K`, and projected to 4 complementary physiological components (paper
   II. PRELIMINARY).
2. **MultiScalePatchEmbedding** — shared initial projection followed by three
   parallel strided convolution branches (kernel sizes 3/7/15), fused by a
   1x1 convolution and a squeeze-and-excitation block, producing patch-level
   token sequences.
3. **BiMambaBlock** — bidirectional Mamba block: a forward scan and a
   backward scan (via sequence flip) are fused by a linear projection,
   mitigating phase offset artifacts in PPG pulse-wave extraction.
4. **MambaEncoderMAE** — multi-scale patch embedding + bidirectional Mamba
   (Bi-Mamba) sequence encoder that turns the decomposed components into
   patch-level latent tokens.

`PDMBambaEncoder` chains KoopmanDecomposer + MambaEncoderMAE and is the
recommended entry point.

### Initialization (real training hyper-parameters)

```python
from pdmamba import PDMBambaEncoder

model = PDMBambaEncoder(
    latent_dim=64,        # Koopman dim == encoder embed dim
    num_patches=30,       # number of patch tokens N
    patch_len=64,         # patch length (L / N)
    num_components=4,     # Koopman complementary components
    koop_window=16,       # Koopman sliding window
    koop_stride=16,       # Koopman sliding stride
    depth=3,              # number of Bi-Mamba layers
    mamba_d_state=16,     # Mamba SSM state size
    mamba_d_conv=4,       # Mamba local convolution width
    mamba_expand=2,       # Mamba expansion factor
)
```

### Input / Output

| Item  | Shape     | Description                                  |
| ----- | --------- | -------------------------------------------- |
| Input | `(B, 1, L)` | Raw PPG segment, `L = num_patches * patch_len` (e.g. 30 × 64 = 1920) |
| Output| `(B, N, D)` | Patch-level latent sequence `z_ppg` consumed by SDF, `D = latent_dim` |

```python
import torch

x_ppg = torch.randn(2, 1, 30 * 64)      # (B, 1, L)
z_ppg = model(x_ppg)                    # (B, 30, 64)
```

---

## 3. SDF (Symplectic Dynamic Fusion)

File: `sdf.py`

### Architecture

1. **STEncoder** — spatio-temporal encoder for the auxiliary SKT signal
   (micro/macro dilated branches + global projection `z_st`).
2. **HamiltonianDynamics** — learnable Hamiltonian `H_theta(z, z_st)` with a
   hand-written leapfrog (symplectic) integrator evolving `z_ppg` from t=0 to
   t=1 (paper Eq.5-Eq.6).
3. **PhysioFusion_v3_Symplectic** — final fusion of the evolved state, the
   sequence context, and a max-pooled skip connection, followed by a refine
   layer (paper Eq.7).

`SDFNet` wraps STEncoder + a skip projection + the fusion core and is the
recommended entry point.

### Initialization (real training hyper-parameters)

```python
from sdf import SDFNet

model = SDFNet(
    latent_dim=64,     # must match PD-Mamba latent_dim
    num_patches=30,    # must match PD-Mamba num_patches
    st_dim=32,         # ST encoder output dim
    n_steps=8,         # symplectic integration steps
)
```

### Input / Output

| Item  | Shape        | Description                                          |
| ----- | ------------ | ---------------------------------------------------- |
| Input | `z_ppg: (B, N, D)` | PPG latent sequence from PD-Mamba            |
| Input | `x_st: (B, 1, L)`   | Auxiliary SKT signal, `L = N * patch_len` |
| Output| `z_final: (B, D)`   | Fused classification feature vector      |
| Output| `aux: None`         | Placeholder for interface compatibility  |

```python
import torch

z_ppg = torch.randn(2, 30, 64)          # (B, N, D)
x_st  = torch.randn(2, 1, 30 * 64)      # (B, 1, L)
z_final, _ = model(z_ppg, x_st)         # (B, 64)
```

---

## 4. End-to-End Pipeline

```python
import torch
from pdmamba import PDMBambaEncoder
from sdf import SDFNet

ppg_encoder = PDMBambaEncoder(latent_dim=64, num_patches=30, patch_len=64, depth=3)
sdf_net     = SDFNet(latent_dim=64, num_patches=30, st_dim=32, n_steps=8)

ppg_encoder.eval()
sdf_net.eval()

x_ppg = torch.randn(2, 1, 1920)         # raw PPG
x_st  = torch.randn(2, 1, 1920)         # SKT signal

with torch.no_grad():
    z_ppg = ppg_encoder(x_ppg)          # (B, 30, 64)
    z_final, _ = sdf_net(z_ppg, x_st)   # (B, 64)

# z_final can be fed into any downstream classifier head.
```

## 5. Notes

- `MambaEncoderMAE.forward` accepts an `ids_keep` mask for MAE-style masking;
  standalone inference should pass all patch indices (the `PDMBambaEncoder`
  wrapper does this automatically).
- The Koopman consistency quantities returned by `KoopmanDecomposer`
  (`y_pred_next`, `y_true_next`, `K`) are used only during pretraining;
  inference uses the first returned value `components`.
- The SDF core preserves the computation graph through the hand-written
  symplectic integrator, so gradients flow back to the PPG encoder when the
  two modules are stacked and fine-tuned end-to-end.
- Due to intellectual property protection requirements arising from a university-enterprise 
  collaborative project, and to avoid potential impacts on subsequent research that has not yet been published, 
  the detailed implementation code of the STGA model is temporarily not released. 
  Readers may implement it independently based on the methods, figures, and parameter tables provided in the paper. 
  We appreciate your understanding
