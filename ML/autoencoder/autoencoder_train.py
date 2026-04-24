#!/usr/bin/env python3
"""
Grass Classifier — Autoencoder Training (grass-only dataset)

Trains a 1D convolutional autoencoder to reconstruct grass range profiles.
At inference, high reconstruction error = not grass.

Usage:
    python autoencoder_train.py
    python autoencoder_train.py --csv path/to/data.csv
    python autoencoder_train.py --epochs 300 --lr 0.001

CSV format (produced by grass_logger.py):
    label, bin0, bin1, ..., binN
    Only rows with label=1 (grass) are used for training.
    Label=0 rows are used for threshold calibration if present.
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ── Model ────────────────────────────────────────────────────────
class GrassAutoencoder(nn.Module):
    """
    1D convolutional autoencoder for radar range profiles.
    Learns to compress and reconstruct grass profiles.
    High reconstruction error on non-grass = anomaly.
    """
    def __init__(self, n_bins):
        super().__init__()
        self.n_bins = n_bins

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(n_bins // 4),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(size=n_bins),
            nn.Conv1d(8, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error(self, x):
        """Per-sample mean squared error."""
        with torch.no_grad():
            x_hat = self.forward(x)
            mse = ((x - x_hat) ** 2).mean(dim=(1, 2))
        return mse


# ── Normalization ────────────────────────────────────────────────
def normalize_per_sample(X):
    mu = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mu) / std


# ── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train grass autoencoder")
    parser.add_argument("--csv", default=os.path.join("..", "range_profiles", "dataset.csv"),
                        help="Path to dataset CSV")
    parser.add_argument("--out", default="autoencoder_model.pth", help="Output model path")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────
    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found. Run grass_logger.py first.")
        sys.exit(1)

    print(f"Loading {args.csv} ...")
    data = np.loadtxt(args.csv, delimiter=",")
    labels = data[:, 0].astype(int)
    features = data[:, 1:].astype(np.float32)
    n_bins = features.shape[1]

    grass_mask = labels == 1
    other_mask = labels == 0
    grass_data = features[grass_mask]
    other_data = features[other_mask] if other_mask.any() else None

    print(f"  Grass samples:     {len(grass_data)}")
    print(f"  Not-grass samples: {len(other_data) if other_data is not None else 0}  (threshold calibration only)")
    print(f"  ROI bins:          {n_bins}")

    if len(grass_data) < 30:
        print("WARNING: Very small dataset. Collect more grass data.")

    # ── Normalize ────────────────────────────────────────────────
    grass_norm = normalize_per_sample(grass_data)

    np.random.seed(42)
    idx = np.random.permutation(len(grass_norm))
    split = int(0.8 * len(idx))
    train_data = grass_norm[idx[:split]]
    val_data = grass_norm[idx[split:]]

    X_train = torch.tensor(train_data).unsqueeze(1)
    X_val = torch.tensor(val_data).unsqueeze(1)

    train_ds = TensorDataset(X_train)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)

    # ── Train ────────────────────────────────────────────────────
    model = GrassAutoencoder(n_bins)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    print(f"\nTraining for {args.epochs} epochs (lr={args.lr}, batch={args.batch})")
    print("-" * 60)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for (xb,) in train_dl:
            x_hat = model(xb)
            loss = criterion(x_hat, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        scheduler.step()
        epoch_loss /= len(train_ds)

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_hat = model(X_val)
                val_loss = criterion(val_hat, X_val).item()
                val_errors = model.reconstruction_error(X_val)
                val_mean = val_errors.mean().item()
                val_max = val_errors.max().item()

            tag = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                tag = " *best*"

            print(f"  Epoch {epoch:4d}  train={epoch_loss:.6f}  "
                  f"val={val_loss:.6f}  err=[mean={val_mean:.4f} max={val_max:.4f}]{tag}")

    # ── Compute threshold ────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        all_grass = torch.tensor(grass_norm).unsqueeze(1)
        grass_errors = model.reconstruction_error(all_grass).numpy()

    grass_mean = grass_errors.mean()
    grass_std = grass_errors.std()
    threshold = grass_mean + 3 * grass_std

    print(f"\n  Grass recon error:  mean={grass_mean:.6f}  std={grass_std:.6f}")
    print(f"  Auto threshold:     {threshold:.6f}  (mean + 3*std)")

    if other_data is not None and len(other_data) > 0:
        other_norm = normalize_per_sample(other_data)
        other_t = torch.tensor(other_norm).unsqueeze(1)
        other_errors = model.reconstruction_error(other_t).numpy()
        other_mean = other_errors.mean()
        detected = (other_errors > threshold).mean() * 100
        print(f"  Not-grass recon error: mean={other_mean:.6f}")
        print(f"  Not-grass detection:   {detected:.1f}% flagged at auto threshold")

    # ── Save ─────────────────────────────────────────────────────
    save_dict = {
        "model_state_dict": best_state if best_state else model.state_dict(),
        "n_bins": n_bins,
        "threshold": float(threshold),
        "grass_error_mean": float(grass_mean),
        "grass_error_std": float(grass_std),
    }
    torch.save(save_dict, args.out)

    print("-" * 60)
    print(f"Model saved to: {args.out}")
    print(f"  threshold={threshold:.6f}  (tune with --threshold in classifier)")


if __name__ == "__main__":
    main()
