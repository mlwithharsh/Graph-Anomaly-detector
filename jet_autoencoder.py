"""
jet_autoencoder.py
==================
Improved CNN Autoencoder for LHC jet image reconstruction.

Architecture highlights
-----------------------
* Residual blocks  → stable gradient flow through 4 encoder stages
* ConvTranspose2d  → learned, artefact-free upsampling (vs bilinear Upsample)
* LeakyReLU(0.2)  → no dead neurons (vs ReLU)
* GroupNorm        → batch-size-independent normalisation (vs BatchNorm)
* Dropout2d        → spatial regularisation
* Compact bottleneck (256, 7, 7) ≈ 12 544 dims (vs ~230 400)
* No output activation → data lives in ℝ (standardised / log-transformed)

Training utilities
------------------
* AdamW + OneCycleLR  → fast convergence, avoids loss plateaus
* Gradient clipping    → training stability
* anomaly_score()      → per-sample MSE for anomaly detection

Usage (notebook)
----------------
    import sys, importlib
    sys.path.insert(0, "..") # or the directory containing this file
    import jet_autoencoder as jae
    importlib.reload(jae)    # reload after edits

    model = jae.JetAutoencoder().to(device)
    optimizer, scheduler = jae.make_optimizer(model, train_loader, epochs=100)
    jae.train(model, train_loader, optimizer, scheduler, epochs=100, device=device)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

class ResBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)   # residual skip


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
    def __init__(self, latent_channels: int = 256, dropout: float = 0.1):
        super().__init__()

        self.enc1 = DownBlock(3,   32,  dropout)      # (B,  32,  60,  60)
        self.enc2 = DownBlock(32,  64,  dropout)      # (B,  64,  30,  30)
        self.enc3 = DownBlock(64,  128, dropout)      # (B, 128,  15,  15)
        self.enc4 = DownBlock(128, latent_channels, dropout)   # (B, 256,  7,   7)

        self.bottleneck = nn.Sequential(
            ResBlock(latent_channels, dropout=0.0),
            ResBlock(latent_channels, dropout=0.0),
        )

        self.dec4 = UpBlock(latent_channels, 128, dropout)  # (B, 128, 14, 14)
        self.dec3 = UpBlock(128, 64,  dropout)              # (B,  64, 28, 28)
        self.dec2 = UpBlock(64,  32,  dropout)              # (B,  32, 56, 56)
        self.dec1 = UpBlock(32,  16,  dropout)              # (B,  16,112,112)

        self.head = nn.Conv2d(16, 3, 3, padding=1)

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
        return self.head(x)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        # Resize to exactly match input spatial dims (handles any input size)
        x_hat = F.interpolate(x_hat, size=x.shape[-2:],
                              mode="bilinear", align_corners=False)
        return x_hat

    @torch.no_grad()
    def anomaly_score(self, x):
        self.eval()
        x_hat = self(x)
        return ((x - x_hat) ** 2).mean(dim=[1, 2, 3])

def make_optimizer(model, train_loader, epochs: int = 100,
                   base_lr: float = 3e-4, max_lr: float = 1e-3,
                   weight_decay: float = 1e-4):
    """
    Returns (optimizer, scheduler) configured with AdamW + OneCycleLR.

    Parameters
    ----------
    model        : JetAutoencoder (or any nn.Module)
    train_loader : DataLoader used during training
    epochs       : total training epochs
    base_lr      : initial/final learning rate
    max_lr       : peak learning rate
    weight_decay : L2 regularisation strength
    """
    optimizer = optim.AdamW(model.parameters(),
                            lr=base_lr,
                            betas=(0.9, 0.999),
                            weight_decay=weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,          # 10 % warmup
        anneal_strategy="cos",
    )
    return optimizer, scheduler


def train(model, train_loader, optimizer, scheduler,
          epochs: int = 100, device: str = "cpu",
          grad_clip: float = 1.0, print_every: int = 1):
    model.to(device)
    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            images = batch[0].to(device)

            optimizer.zero_grad()
            x_hat = model(images)
            loss = F.mse_loss(x_hat, images)

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(
                    f"Epoch {epoch}: loss is {loss.item():.6f}. "
                    "Check data normalisation or lower max_lr."
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        loss_history.append(epoch_loss)

        if epoch % print_every == 0:
            print(f"Epoch {epoch:>4d} | Loss: {epoch_loss:.6f}")

    return loss_history


def visualise_reconstructions(model, val_loader, device: str = "cpu",
                              n: int = 4):
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        x = batch[0][:n].to(device)
        x_hat = model(x)

    x     = x.cpu()
    x_hat = x_hat.cpu()

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for i in range(n):
        axes[0, i].imshow(x[i, 0], cmap="inferno")
        axes[0, i].set_title(f"Original {i}")
        axes[0, i].axis("off")

        axes[1, i].imshow(x_hat[i, 0], cmap="inferno")
        axes[1, i].set_title(f"Recon {i}")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Reconstruction", fontsize=12)
    plt.suptitle("ECAL channel — Original vs Reconstruction")
    plt.tight_layout()
    plt.show()
