# Improved CNN Autoencoder for Jet Image Reconstruction

> **Context:** LHC jet images · Shape `(N, 3, 125, 125)` (ECAL / HCAL / Tracks) · Pre-processed with `log1p` + standardisation · Anomaly score = MSE reconstruction error

---

## 1. Weaknesses in the Current Architecture

| Issue | Root Cause | Effect |
|---|---|---|
| **Loss plateau** | No skip connections; gradient vanishes through 3 encoder stages | Early saturation, mediocre reconstruction |
| **Upsample + Conv in decoder** | Bilinear upsample introduces checkerboard artefacts | Blurry outputs |
| **ReLU throughout** | Dying-ReLU problem in deeper layers | Dead neurons, training stalls |
| **No bottleneck** | Latent space is the full spatial feature map `(128, 16, 16)` | Too many degrees of freedom — autoencoder is not selective |
| **No dropout** | Network memorises training set | Poor generalisation, overfitting on background jets |
| **BatchNorm in decoder** | BN statistics shift during inference, especially at small batch sizes | Unstable reconstructions |
| **No residual paths** | Network cannot easily learn identity mapping | Harder to train deep encoder |

---

## 2. Improved Architecture — PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    Pre-activation residual block with GroupNorm.
    Keeps the same spatial size and channel count.
    """
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),          # GroupNorm: stable at any batch size
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),              # spatial dropout: regularise feature maps
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)               # residual addition

class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res = ResBlock(out_ch, dropout)

    def forward(self, x):
        return self.res(self.conv(x))

class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.05):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res = ResBlock(out_ch, dropout)

    def forward(self, x):
        return self.res(self.up(x))

class JetAutoencoder(nn.Module):
    """
    Input:  (B, 3, 125, 125)
    Latent: (B, 256, 8, 8)  →  flat dim ≈ 16 384  (down from ~327 680)
    Output: (B, 3, 125, 125)  (nearest-neighbour pad to match input)
    """
    def __init__(self, latent_channels: int = 256, dropout: float = 0.1):
        super().__init__()

        self.enc1 = DownBlock(3,   32,  dropout)   # → (B,  32, 62, 62)
        self.enc2 = DownBlock(32,  64,  dropout)   # → (B,  64, 31, 31)
        self.enc3 = DownBlock(64,  128, dropout)   # → (B, 128, 15, 15)
        self.enc4 = DownBlock(128, latent_channels, dropout)  # → (B, 256, 7, 7)

        self.bottleneck = nn.Sequential(
            ResBlock(latent_channels, dropout=0.0),
            ResBlock(latent_channels, dropout=0.0),
        )

        self.dec4 = UpBlock(latent_channels, 128, dropout)  # → (B, 128, 14, 14)
        self.dec3 = UpBlock(128, 64,  dropout)              # → (B,  64, 28, 28)
        self.dec2 = UpBlock(64,  32,  dropout)              # → (B,  32, 56, 56)
        self.dec1 = UpBlock(32,  16,  dropout)              # → (B,  16,112,112)

        self.head = nn.Sequential(
            nn.Conv2d(16, 3, 3, padding=1),
        )

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        return self.bottleneck(x)

    def decode(self, z):
        x = self.dec4(z)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        x = self.head(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        x_hat = F.interpolate(x_hat, size=x.shape[-2:], mode='bilinear',
                              align_corners=False)
        return x_hat

    def anomaly_score(self, x):
        """Per-sample MSE reconstruction error (use as anomaly metric)."""
        with torch.no_grad():
            x_hat = self(x)
            score = ((x - x_hat) ** 2).mean(dim=[1, 2, 3])
        return score
```

---

## 3. Architecture Improvements — Rationale

### 3·1 Residual Blocks
- Gradient flows directly through the skip connection, eliminating vanishing gradients across 4 encoder stages.
- Network learns *corrections* to identity, making it easier to train.

### 3·2 `ConvTranspose2d` instead of `Upsample + Conv`
- Learned transposed convolution produces sharper, artefact-free reconstructions.
- Avoids the checkerboard / blur trade-off inherent to bilinear upsampling.

### 3·3 `LeakyReLU(0.2)` instead of `ReLU`
- Keeps a small gradient for negative pre-activations (slope = 0.2).
- Prevents dead neurons in early training when features are near zero.
- Standard choice for image autoencoders (used in all modern GAN-style architectures).

### 3·4 `GroupNorm` instead of `BatchNorm`
- `GroupNorm` is **batch-size independent** — critical for small or variable batches.
- No running statistics to go stale → stable behaviour at inference without `model.eval()` concerns.
- Use `num_groups=8`; tune down to 4 if channels ≤ 32.

### 3·5 `Dropout2d` in Residual Blocks (encoder only)
- Spatial dropout zeros entire feature-map channels, forcing distributed representations.
- Placed *inside* residual blocks so it regularises the learned corrections, not the skip path.
- Use `p=0.1` in encoder, `p=0.05` in decoder (reconstruction needs finer control).
- **Remove** during `model.eval()` — PyTorch does this automatically.

### 3·6 Compact Bottleneck `(256, 7, 7)`
- Latent flat dim ≈ 12 544 vs. original ≈ 327 680 — **26× compression**.
- Sufficient for capturing dominant jet substructure (core shape, energy distribution).
- Anomalous jets deviate from the learned manifold → higher reconstruction error.

### 3·7 4 Encoder/Decoder Stages
- Extra stage captures finer-grained jet substructure compared to 3 stages.
- Gradual channel expansion `3 → 32 → 64 → 128 → 256` avoids abrupt information bottleneck.

### 3·8 No Activation in Output Head
- Pre-processed data lives in ℝ (standardised, can be negative).
- `Tanh` or `Sigmoid` would artificially clamp outputs → unfaithful reconstruction.

---

## 4. Training Hyperparameters

```python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

model = JetAutoencoder(latent_channels=256, dropout=0.1).cuda()

optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-4,           # sweet spot for AdamW on image models
    betas=(0.9, 0.999),
    weight_decay=1e-4, # mild L2 regularisation on weights
)


scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    steps_per_epoch=len(train_loader),
    epochs=100,
    pct_start=0.1,        # 10% warmup
    anneal_strategy='cos',
)

def combined_loss(x_hat, x, alpha=0.8):
    mse = F.mse_loss(x_hat, x)
    return mse  # start with pure MSE; add SSIM if reconstruction is blurry

EPOCHS = 100
BATCH  = 64
GRAD_CLIP = 1.0   # gradient clipping — key stability tool

for epoch in range(EPOCHS):
    model.train()
    for x, in train_loader:
        x = x.cuda()
        x_hat = model(x)
        loss = combined_loss(x_hat, x)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # ← clip!
        optimizer.step()
        scheduler.step()
```

| Hyperparameter | Recommended Value | Rationale |
|---|---|---|
| Optimizer | `AdamW` | Decoupled weight decay → better regularisation |
| Learning rate | `3e-4` (peak `1e-3`) | One-Cycle schedule handles warm-up and cooldown |
| Batch size | `64` | Already good; try `128` if GPU allows |
| Weight decay | `1e-4` | Mild regularisation |
| Gradient clip | `1.0` | Prevents exploding gradients at start of training |
| Epochs | `80–120` | Plateau usually occurs before 100 with LR scheduling |
| Dropout | `0.1` (enc), `0.05` (dec) | Spatial dropout for feature-map regularisation |

---

## 5. Expected Loss Behaviour

```
Epoch   1–10 :  MSE drops sharply  (~0.8 → ~0.15)     [learning global structure]
Epoch  10–40 :  Gradual decline    (~0.15 → ~0.05)     [learning fine substructure]
Epoch  40–80 :  Slow tail          (~0.05 → ~0.025)    [optimal reconstruction]
Epoch  80+   :  Flatten / marginal gains               [diminishing returns]
```

> [!TIP]
> If loss is still flat by epoch 20, lower `weight_decay` to `1e-5` and re-run. If NaN appears at any epoch, lower `max_lr` to `5e-4`.

**Anomaly separation check** — plot reconstruction MSE distributions:
```python
import matplotlib.pyplot as plt
scores_bg = model.anomaly_score(bg_loader_sample).cpu().numpy()
scores_sig = model.anomaly_score(sig_loader_sample).cpu().numpy()
plt.hist(scores_bg, bins=60, alpha=0.7, label='Background (QCD)')
plt.hist(scores_sig, bins=60, alpha=0.7, label='Signal (anomalous)')
plt.xlabel('Reconstruction MSE'); plt.legend()
```
Good models show a clean separation between distributions.

---

## 6. Training Stability Debugging Checklist

```python
assert not torch.isnan(loss), "NaN loss — check data normalisation"
assert not torch.isinf(loss), "Inf loss — lower learning rate or clip gradients"

total_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
total_norm = total_norm ** 0.5

import matplotlib.pyplot as plt
model.eval()
with torch.no_grad():
    x_sample = next(iter(val_loader))[0][:4].cuda()
    x_hat = model(x_sample)
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i in range(4):
    axes[0, i].imshow(x_sample[i, 0].cpu(), cmap='hot')   # ECAL original
    axes[1, i].imshow(x_hat[i, 0].cpu(),    cmap='hot')   # ECAL reconstruction
plt.suptitle('Top: original | Bottom: reconstruction')

print(f"Input mean: {x_sample.mean():.4f}  std: {x_sample.std():.4f}")

print(f"Recon pixel std: {x_hat.std():.4f}")

from torch_lr_finder import LRFinder
finder = LRFinder(model, optimizer, combined_loss, device='cuda')
finder.range_test(train_loader, end_lr=0.1, num_iter=100)
finder.plot()  # pick lr just before the loss rises steeply
```

---

## 7. Optional Enhancements (Research Direction)

| Technique | When to Try | Expected Gain |
|---|---|---|
| **Perceptual loss** (VGG features) | If MSE reconstructions are blurry | Sharper jet core features |
| **β-VAE bottleneck** | If anomaly separation is poor | Better-structured latent manifold |
| **Channel attention (SE blocks)** | After baseline works well | Differential weighting of ECAL/HCAL/Tracks |
| **Adversarial training (AE-GAN)** | Final model refinement | Photorealistic jet detail |
| **Normalising Flow posterior** | Full probabilistic anomaly score | Calibrated p-values |

> [!IMPORTANT]
> For the ML4Sci / GSoC prototyping phase, stick to the architecture above without optional enhancements. Get a stable MSE-only baseline first, then iterate.
