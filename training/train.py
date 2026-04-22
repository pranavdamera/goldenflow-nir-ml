"""Train HoneyNIRNet on spectral CSV data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from app.ml.honey_nir_net import HoneyNIRNet
from training.focal_loss import FocalLoss

LABEL_TO_INDEX = {
    "pure": 0,
    "rice_syrup": 1,
    "hfcs": 2,
    "jaggery_syrup": 3,
    "invert_sugar": 4,
    "sugar_fed": 5,
    "unknown": 6,
}


@dataclass
class DatasetBundle:
    """Container for train and validation tensors."""

    x_train: Tensor
    x_val: Tensor
    y_train_cls: Tensor
    y_val_cls: Tensor
    y_train_reg: Tensor
    y_val_reg: Tensor


def load_csv_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load spectral features and targets from CSV."""
    df = pd.read_csv(path)
    feature_cols = [f"wavelength_{i}" for i in range(1, 257)]
    missing = [col for col in feature_cols + ["label", "adulteration_pct"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x = df[feature_cols].to_numpy(dtype=np.float32)
    y_cls = df["label"].map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)
    y_reg = df["adulteration_pct"].to_numpy(dtype=np.float32)
    return x, y_cls, y_reg


def build_splits(x: np.ndarray, y_cls: np.ndarray, y_reg: np.ndarray, val_size: float, seed: int) -> DatasetBundle:
    """Create stratified train/validation splits and convert to tensors."""
    x_train, x_val, y_train_cls, y_val_cls, y_train_reg, y_val_reg = train_test_split(
        x, y_cls, y_reg, test_size=val_size, random_state=seed, stratify=y_cls
    )

    return DatasetBundle(
        x_train=torch.from_numpy(x_train).unsqueeze(1),
        x_val=torch.from_numpy(x_val).unsqueeze(1),
        y_train_cls=torch.from_numpy(y_train_cls),
        y_val_cls=torch.from_numpy(y_val_cls),
        y_train_reg=torch.from_numpy(y_train_reg).unsqueeze(1),
        y_val_reg=torch.from_numpy(y_val_reg).unsqueeze(1),
    )


def train(args: argparse.Namespace) -> None:
    """Execute end-to-end training loop and save best checkpoint."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x, y_cls, y_reg = load_csv_dataset(args.data_csv)
    bundle = build_splits(x, y_cls, y_reg, args.val_size, args.seed)

    train_ds = TensorDataset(bundle.x_train, bundle.y_train_cls, bundle.y_train_reg)
    val_ds = TensorDataset(bundle.x_val, bundle.y_val_cls, bundle.y_val_reg)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = HoneyNIRNet()
    cls_criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    reg_criterion = nn.L1Loss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = -1.0
    output_path = Path(args.output_checkpoint)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb_cls, yb_reg in train_loader:
            optimizer.zero_grad()
            logits, reg = model(xb)
            cls_loss = cls_criterion(logits, yb_cls)
            reg_loss = reg_criterion(reg, yb_reg)
            loss = cls_loss + args.reg_loss_weight * reg_loss
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * xb.size(0)

        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        mae_sum = 0.0
        with torch.no_grad():
            for xb, yb_cls, yb_reg in val_loader:
                logits, reg = model(xb)
                preds = torch.argmax(logits, dim=1)
                correct += int((preds == yb_cls).sum().item())
                total += int(yb_cls.size(0))
                mae_sum += float(torch.abs(reg - yb_reg).sum().item())

        val_acc = correct / total if total else 0.0
        val_mae = mae_sum / total if total else 0.0
        epoch_loss = running_loss / len(train_ds)
        print(f"epoch={epoch} loss={epoch_loss:.6f} val_acc={val_acc:.4f} val_mae={val_mae:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)

    print(f"Best checkpoint saved to {output_path} with val_acc={best_val_acc:.4f}")


def parse_args() -> argparse.Namespace:
    """Parse command-line hyperparameters for model training."""
    parser = argparse.ArgumentParser(description="Train HoneyNIRNet on spectral CSV data.")
    parser.add_argument("--data-csv", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--reg-loss-weight", type=float, default=0.5)
    parser.add_argument("--output-checkpoint", type=str, default="honey_nir_net.pt")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
