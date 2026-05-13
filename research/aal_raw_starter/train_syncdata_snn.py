from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

MOBIVITAL_DIR = Path(__file__).resolve().parents[1] / "mobivital_snn_starter"
sys.path.insert(0, str(MOBIVITAL_DIR))

from preprocessing import apply_preprocess  # noqa: E402
from spike_encoding import (  # noqa: E402
    delta_rate_hybrid_encode,
    delta_spike_encode,
    level_crossing_encode,
    rate_spike_encode,
    zscore,
)
from train_delta_snn import DeltaSNN, SpikingTCN, corrcoef, split_indices  # noqa: E402


def get_mat_field(struct: object, name: str) -> np.ndarray:
    value = getattr(struct, name)
    return np.asarray(value, dtype=np.float32)


def best_bin_by_corr(feature_range_time: np.ndarray, target: np.ndarray) -> tuple[int, np.ndarray]:
    feature_z = zscore(feature_range_time, axis=1)
    target_z = zscore(target, axis=0).reshape(1, -1)
    corr = np.mean(feature_z * target_z, axis=1)
    return int(np.argmax(np.abs(corr))), corr


class SyncDataWindowDataset(Dataset):
    """Window dataset for graduate-style UWB_Biopac_SyncData.mat files."""

    def __init__(
        self,
        mat_path: str | Path,
        radar_field: str = "com_row",
        target_field: str = "biopac_resp",
        window_sec: float = 10.0,
        stride_sec: float = 2.0,
        bin_index: int | None = None,
        bin_radius: int = 0,
        preprocess: str = "none",
        encode: str = "delta_rate_hybrid",
        threshold_scale: float = 0.75,
        threshold_mode: str = "std",
        threshold_percentile: float = 75.0,
        target_spike_rate: float = 0.2,
        levels: int = 5,
    ) -> None:
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        sync = mat["UWB_Biopac_SyncData"]
        self.fs = float(np.asarray(getattr(sync, "Fs_uwb")).squeeze())
        if target_field == "radar_resp":
            fs_ref = float(np.asarray(getattr(sync, "Fs_radar_resp", getattr(sync, "Fs_uwb"))).squeeze())
        else:
            fs_ref = float(np.asarray(getattr(sync, "Fs_biopac")).squeeze())
        radar = get_mat_field(sync, radar_field)
        reference = get_mat_field(sync, target_field).squeeze()

        if radar.ndim != 2:
            raise ValueError(f"{radar_field} must be a range x time matrix, got shape {radar.shape}")
        if reference.ndim != 1:
            raise ValueError(f"biopac_resp must be 1-D, got shape {reference.shape}")

        radar = apply_preprocess(radar.T, mode=preprocess, fs=int(round(self.fs))).T
        t_radar = np.arange(radar.shape[1], dtype=np.float32) / self.fs
        t_ref = np.arange(reference.shape[0], dtype=np.float32) / fs_ref
        target = np.interp(t_radar, t_ref, reference).astype(np.float32)
        target = zscore(target).astype(np.float32)

        use_bin = bin_index
        if use_bin is None:
            use_bin, _ = best_bin_by_corr(radar, target)
        lo = max(0, use_bin - bin_radius)
        hi = min(radar.shape[0], use_bin + bin_radius + 1)
        selected = zscore(radar[lo:hi, :], axis=1).astype(np.float32)

        self.window = int(round(window_sec * self.fs))
        self.stride = int(round(stride_sec * self.fs))
        self.items: list[tuple[np.ndarray, np.ndarray]] = []
        self.metadata = {
            "mat_path": str(mat_path),
            "radar_field": radar_field,
            "target_field": target_field,
            "fs_uwb": self.fs,
            "fs_reference": fs_ref,
            "radar_shape_range_time": list(radar.shape),
            "reference_samples": int(reference.shape[0]),
            "selected_bin": int(use_bin),
            "bin_radius": int(bin_radius),
            "encode": encode,
            "preprocess": preprocess,
        }

        for start in range(0, selected.shape[1] - self.window + 1, self.stride):
            end = start + self.window
            x_win = selected[:, start:end]
            y_win = target[start:end]
            if encode == "delta":
                x_encoded = delta_spike_encode(
                    x_win,
                    threshold_scale=threshold_scale,
                    threshold_mode=threshold_mode,
                    threshold_percentile=threshold_percentile,
                    target_spike_rate=target_spike_rate,
                    bipolar=True,
                )
                x_encoded = x_encoded.reshape(-1, x_encoded.shape[-1])
            elif encode == "delta_rate_hybrid":
                x_encoded = delta_rate_hybrid_encode(
                    x_win,
                    threshold_scale=threshold_scale,
                    threshold_mode=threshold_mode,
                    threshold_percentile=threshold_percentile,
                    target_spike_rate=target_spike_rate,
                    seed=start,
                )
            elif encode == "rate":
                x_encoded = rate_spike_encode(x_win, seed=start)
            elif encode == "level_crossing":
                x_encoded = level_crossing_encode(x_win, levels=levels)
            elif encode == "none":
                x_encoded = x_win
            else:
                raise ValueError(f"Unknown encode mode: {encode}")
            self.items.append((x_encoded.astype(np.float32), y_win[None, :]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.items[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_fn = nn.MSELoss()
    losses: list[float] = []
    rates: list[float] = []
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred, rate = model(x)
            losses.append(float(loss_fn(pred, y).item()))
            rates.append(float(rate.item()))
            preds.append(pred.cpu())
            targets.append(y.cpu())
    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    return {
        "loss": float(np.mean(losses)),
        "mae": float(torch.mean(torch.abs(pred_all - target_all)).item()),
        "rmse": float(torch.sqrt(torch.mean((pred_all - target_all) ** 2)).item()),
        "corr": corrcoef(pred_all, target_all),
        "hidden_spike_rate": float(np.mean(rates)),
    }


def save_prediction_plot(model: nn.Module, dataset: SyncDataWindowDataset, index: int, out_path: Path, device: torch.device) -> None:
    model.eval()
    x, y = dataset[index]
    with torch.no_grad():
        pred, _ = model(x.unsqueeze(0).to(device))
    pred_np = pred.squeeze(0).cpu().numpy()[0]
    target = y.numpy()[0]
    t = np.arange(target.shape[0]) / dataset.fs

    plt.figure(figsize=(12, 6))
    plt.plot(t, target, label="reference respiration")
    plt.plot(t, pred_np, label="SNN prediction", alpha=0.85)
    plt.xlabel("time (s)")
    plt.title("SyncData SNN respiration reconstruction")
    plt.legend(loc="upper right")
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
    parser.add_argument("--preprocess", choices=["none", "moving_average", "fft_bandpass"], default="moving_average")
    parser.add_argument("--encode", choices=["rate", "delta", "delta_rate_hybrid", "level_crossing"], default="delta_rate_hybrid")
    parser.add_argument("--threshold-scale", type=float, default=0.75)
    parser.add_argument("--threshold-mode", choices=["std", "mad", "percentile", "target_rate"], default="std")
    parser.add_argument("--threshold-percentile", type=float, default=75.0)
    parser.add_argument("--target-spike-rate", type=float, default=0.2)
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--model", choices=["lif_cnn", "spiking_tcn"], default="lif_cnn")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rate-reg", type=float, default=1e-3)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.mat).resolve().parent / "syncdata_snn_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(11)
    np.random.seed(11)
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
        encode=args.encode,
        threshold_scale=args.threshold_scale,
        threshold_mode=args.threshold_mode,
        threshold_percentile=args.threshold_percentile,
        target_spike_rate=args.target_spike_rate,
        levels=args.levels,
    )
    if len(dataset) < 4:
        raise RuntimeError("Not enough windows. Use a shorter window or more data.")

    train_idx, val_idx, test_idx = split_indices(len(dataset))
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=8, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=8)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=8)

    in_channels = int(dataset[0][0].shape[0])
    model = DeltaSNN(in_channels=in_channels) if args.model == "lif_cnn" else SpikingTCN(in_channels=in_channels)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_path = out_dir / "syncdata_snn.pt"
    best_val = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_rates: list[float] = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred, rate = model(x)
            loss = loss_fn(pred, y) + args.rate_reg * rate
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_rates.append(float(rate.item()))

        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_hidden_spike_rate": float(np.mean(train_rates)),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(model.state_dict(), best_path)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:03d} | train {row['train_loss']:.4f} | "
                f"val rmse {row['val_rmse']:.4f} | val corr {row['val_corr']:.4f} | "
                f"rate {row['val_hidden_spike_rate']:.4f}"
            )

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, device)
    input_spike_density = float(torch.stack([dataset[i][0].mean() for i in range(len(dataset))]).mean())
    input_spikes_per_window = float(torch.stack([dataset[i][0].sum() for i in range(len(dataset))]).mean())
    input_spikes_per_second = input_spikes_per_window / args.window_sec
    metrics = {
        **dataset.metadata,
        "device": str(device),
        "model": args.model,
        "input_channels": in_channels,
        "dataset_windows": len(dataset),
        "train_windows": len(train_idx),
        "val_windows": len(val_idx),
        "test_windows": len(test_idx),
        "threshold_scale": args.threshold_scale,
        "threshold_mode": args.threshold_mode,
        "threshold_percentile": args.threshold_percentile,
        "target_spike_rate": args.target_spike_rate,
        "levels": args.levels,
        "input_spike_density": input_spike_density,
        "input_spikes_per_window": input_spikes_per_window,
        "input_spikes_per_second": input_spikes_per_second,
        "test": test_metrics,
        "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_prediction_plot(model, dataset, test_idx[0], out_dir / "prediction.png", device)

    print("test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"input spike density: {input_spike_density:.4f}")
    print(f"input spikes/window: {input_spikes_per_window:.1f}")
    print(f"input spikes/second: {input_spikes_per_second:.1f}")
    print(f"saved model: {best_path}")
    print(f"saved metrics: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
