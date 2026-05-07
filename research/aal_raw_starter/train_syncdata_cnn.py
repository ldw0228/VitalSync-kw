from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from train_syncdata_snn import SyncDataWindowDataset


class RespirationCNN(nn.Module):
    """Continuous-input 1D CNN baseline for SyncData-like matrices."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def corrcoef(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.detach().flatten().cpu()
    target = target.detach().flatten().cpu()
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = torch.sqrt((pred * pred).sum() * (target * target).sum())
    if float(denom) == 0.0:
        return 0.0
    return float((pred * target).sum() / denom)


def split_indices(n: int, train_ratio: float = 0.7, val_ratio: float = 0.15) -> tuple[list[int], list[int], list[int]]:
    train_end = max(1, int(n * train_ratio))
    val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))
    val_end = min(val_end, n - 1)
    return list(range(0, train_end)), list(range(train_end, val_end)), list(range(val_end, n))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_fn = nn.MSELoss()
    losses: list[float] = []
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            losses.append(float(loss_fn(pred, y).item()))
            preds.append(pred.cpu())
            targets.append(y.cpu())
    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    return {
        "loss": float(np.mean(losses)),
        "mae": float(torch.mean(torch.abs(pred_all - target_all)).item()),
        "rmse": float(torch.sqrt(torch.mean((pred_all - target_all) ** 2)).item()),
        "corr": corrcoef(pred_all, target_all),
    }


def save_prediction_plot(
    model: nn.Module,
    dataset: SyncDataWindowDataset,
    index: int,
    out_path: Path,
    device: torch.device,
) -> None:
    model.eval()
    x, y = dataset[index]
    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()[0]
    target = y.numpy()[0]
    signal = x.numpy()[0]
    t = np.arange(target.shape[0]) / dataset.fs

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, label="continuous UWB input")
    plt.legend(loc="upper right")
    plt.title("Input window")
    plt.subplot(2, 1, 2)
    plt.plot(t, target, label="target respiration")
    plt.plot(t, pred, label="CNN prediction", alpha=0.85)
    plt.legend(loc="upper right")
    plt.xlabel("time (s)")
    plt.title("SyncData CNN respiration reconstruction")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", required=True, help="Path to UWB_Biopac_SyncData-like .mat")
    parser.add_argument("--radar-field", default="com_row", choices=["com_row", "com_col", "tv_row", "tv_col"])
    parser.add_argument("--target-field", default="biopac_resp", choices=["biopac_resp", "radar_resp"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=2.0)
    parser.add_argument("--bin-index", type=int, default=None)
    parser.add_argument("--bin-radius", type=int, default=0)
    parser.add_argument("--preprocess", choices=["none", "moving_average", "fft_bandpass"], default="none")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.mat).resolve().parent / "syncdata_cnn_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(7)
    np.random.seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SyncDataWindowDataset(
        args.mat,
        radar_field=args.radar_field,
        target_field=args.target_field,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        bin_index=args.bin_index,
        bin_radius=args.bin_radius,
        preprocess=args.preprocess,
        encode="none",
    )
    if len(dataset) < 4:
        raise RuntimeError("Not enough windows. Use a shorter window or more data.")

    train_idx, val_idx, test_idx = split_indices(len(dataset))
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=8, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=8)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=8)

    in_channels = int(dataset[0][0].shape[0])
    model = RespirationCNN(in_channels=in_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_path = out_dir / "syncdata_cnn.pt"
    best_val = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(model.state_dict(), best_path)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:03d} | train {row['train_loss']:.4f} | "
                f"val rmse {row['val_rmse']:.4f} | val corr {row['val_corr']:.4f}"
            )

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, device)
    metrics = {
        **dataset.metadata,
        "device": str(device),
        "model": "cnn",
        "input_channels": in_channels,
        "dataset_windows": len(dataset),
        "train_windows": len(train_idx),
        "val_windows": len(val_idx),
        "test_windows": len(test_idx),
        "test": test_metrics,
        "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_prediction_plot(model, dataset, test_idx[0], out_dir / "prediction.png", device)

    print("test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"saved model: {best_path}")
    print(f"saved metrics: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
