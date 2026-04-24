#!/usr/bin/env python3
"""
Grass Classifier — Train & Export

Trains a small 1D CNN on range-profile ROI data collected by grass_logger.py.
Exports a .pth model file for use in live inference.

Usage:
    python grass_train.py                         # train on range_profiles/dataset.csv
    python grass_train.py --csv my_data.csv       # custom CSV path
    python grass_train.py --epochs 200 --lr 0.0005

CSV format (produced by grass_logger.py):
    label, bin0, bin1, ..., binN
    1, 23.45, 22.10, ...
    0, 18.30, 17.55, ...
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


# ── Model ────────────────────────────────────────────────────────
class GrassNet(nn.Module):
    """
    Small 1D CNN for binary classification of radar range profiles.
    Input:  (batch, 1, n_bins)
    Output: (batch, 1) logits
    """
    def __init__(self, n_bins):
        super().__init__()
        self.n_bins = n_bins
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # → (batch, 32, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Normalization ────────────────────────────────────────────────
def normalize_per_sample(X):
    """Zero-mean, unit-variance per sample (makes model robust to overall power shifts)."""
    mu = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mu) / std


# ── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train grass classifier")
    parser.add_argument("--csv", default=os.path.join("..", "range_profiles", "dataset.csv"), help="Path to dataset CSV")
    parser.add_argument("--out", default="cnn_model.pth", help="Output model path")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────
    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found. Run grass_logger.py first to collect data.")
        sys.exit(1)

    print(f"Loading {args.csv} ...")
    data = np.loadtxt(args.csv, delimiter=",")
    labels = data[:, 0].astype(int)
    features = data[:, 1:].astype(np.float32)

    n_samples, n_bins = features.shape
    n_grass = (labels == 1).sum()
    n_other = (labels == 0).sum()

    print(f"  Samples:  {n_samples}  ({n_grass} grass, {n_other} not-grass)")
    print(f"  ROI bins: {n_bins}")

    if n_samples < 50:
        print("WARNING: Very small dataset. Collect more data for reliable results.")
    if n_grass == 0 or n_other == 0:
        print("ERROR: Need both grass and not-grass samples.")
        sys.exit(1)

    # ── Normalize & split ────────────────────────────────────────
    features = normalize_per_sample(features)

    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=args.test_size, random_state=42, stratify=labels
    )

    X_train_t = torch.tensor(X_train).unsqueeze(1)   # (N, 1, bins)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val).unsqueeze(1)
    y_val_t   = torch.tensor(y_val, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)

    # ── Class weighting (handles imbalanced data) ────────────────
    pos_weight = torch.tensor([n_other / max(n_grass, 1)], dtype=torch.float32)

    # ── Model, optimizer, loss ───────────────────────────────────
    model = GrassNet(n_bins)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"\nTraining for {args.epochs} epochs (lr={args.lr}, batch={args.batch})")
    print("-" * 55)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        # ── Train ────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_dl:
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        scheduler.step()
        epoch_loss /= len(train_ds)

        # ── Validate ─────────────────────────────────────────────
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t).squeeze(-1)
                val_preds = (torch.sigmoid(val_logits) > 0.5).long().numpy()
                val_acc = (val_preds == y_val).mean()

                # Per-class accuracy
                grass_mask = y_val == 1
                other_mask = y_val == 0
                grass_acc = (val_preds[grass_mask] == y_val[grass_mask]).mean() if grass_mask.any() else 0
                other_acc = (val_preds[other_mask] == y_val[other_mask]).mean() if other_mask.any() else 0

            tag = ""
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                tag = " *best*"

            print(f"  Epoch {epoch:4d}  loss={epoch_loss:.4f}  "
                  f"val_acc={val_acc:.3f}  (grass={grass_acc:.3f} other={other_acc:.3f}){tag}")

    # ── Save best model ──────────────────────────────────────────
    if best_state is None:
        best_state = model.state_dict()

    save_dict = {
        "model_state_dict": best_state,
        "n_bins": n_bins,
        "best_val_acc": best_val_acc,
    }
    torch.save(save_dict, args.out)

    print("-" * 55)
    print(f"Best val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to:    {args.out}")
    print(f"\nTo use in your live script:")
    print(f"  checkpoint = torch.load('{args.out}')")
    print(f"  model = GrassNet(checkpoint['n_bins'])")
    print(f"  model.load_state_dict(checkpoint['model_state_dict'])")
    print(f"  model.eval()")


if __name__ == "__main__":
    main()
