from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from mobivital_dataset import MobiVitalWindowDataset


class SurrogateSpike(torch.autograd.Function):
    """Step spike with a sigmoid-shaped surrogate gradient."""

    @staticmethod
    def forward(ctx, membrane: torch.Tensor, threshold: float, beta: float) -> torch.Tensor:
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        ctx.beta = beta
        return (membrane >= threshold).to(membrane.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (membrane,) = ctx.saved_tensors
        beta = ctx.beta
        threshold = ctx.threshold
        sig = torch.sigmoid(beta * (membrane - threshold))
        grad = beta * sig * (1.0 - sig)
        return grad_output * grad, None, None


def spike_fn(membrane: torch.Tensor, threshold: float = 1.0, beta: float = 10.0) -> torch.Tensor:
    return SurrogateSpike.apply(membrane, threshold, beta)


class LIF1d(nn.Module):
    """Leaky integrate-and-fire layer over the time axis."""

    def __init__(self, decay: float = 0.85, threshold: float = 1.0, beta: float = 10.0) -> None:
        super().__init__()
        self.decay = decay
        self.threshold = threshold
        self.beta = beta

    def forward(self, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # current: [batch, channels, time]
        mem = torch.zeros_like(current[:, :, 0])
        spikes = []
        for t in range(current.shape[-1]):
            mem = self.decay * mem + current[:, :, t]
            spk = spike_fn(mem, self.threshold, self.beta)
            mem = mem * (1.0 - spk)
            spikes.append(spk)
        spike_train = torch.stack(spikes, dim=-1)
        return spike_train, spike_train.mean()


class DeltaSNN(nn.Module):
    """Small spiking CNN for respiration waveform reconstruction."""

    def __init__(self, in_channels: int = 2, hidden: int = 32) -> None:
        super().__init__()
        self.current1 = nn.Conv1d(in_channels, hidden, kernel_size=9, padding=4)
        self.lif1 = LIF1d(decay=0.88, threshold=0.6, beta=8.0)
        self.current2 = nn.Conv1d(hidden, hidden, kernel_size=7, padding=3)
        self.lif2 = LIF1d(decay=0.90, threshold=0.7, beta=8.0)
        self.readout = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cur1 = self.current1(x)
        spk1, rate1 = self.lif1(cur1)
        cur2 = self.current2(spk1)
        spk2, rate2 = self.lif2(cur2)
        out = self.readout(spk2)
        return out, (rate1 + rate2) / 2.0


class SpikingTCNBlock(nn.Module):
    """Dilated temporal convolution followed by LIF spiking dynamics."""

    def __init__(self, channels: int, dilation: int, kernel_size: int = 5) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.lif = LIF1d(decay=0.90, threshold=0.7, beta=8.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current = self.norm(self.conv(x))
        spikes, rate = self.lif(current)
        return spikes, rate


class SpikingTCN(nn.Module):
    """Spiking temporal convolution network for longer respiration patterns."""

    def __init__(self, in_channels: int = 2, hidden: int = 32) -> None:
        super().__init__()
        self.input_current = nn.Conv1d(in_channels, hidden, kernel_size=1)
        self.input_lif = LIF1d(decay=0.88, threshold=0.6, beta=8.0)
        self.blocks = nn.ModuleList(
            [
                SpikingTCNBlock(hidden, dilation=1),
                SpikingTCNBlock(hidden, dilation=2),
                SpikingTCNBlock(hidden, dilation=4),
                SpikingTCNBlock(hidden, dilation=8),
            ]
        )
        self.readout = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spikes, rate = self.input_lif(self.input_current(x))
        rates = [rate]
        for block in self.blocks:
            next_spikes, block_rate = block(spikes)
            spikes = torch.clamp(spikes + next_spikes, 0.0, 1.0)
            rates.append(block_rate)
        out = self.readout(spikes)
        return out, torch.stack(rates).mean()


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
    losses: list[float] = []
    rates: list[float] = []
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    loss_fn = nn.MSELoss()
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
    mae = torch.mean(torch.abs(pred_all - target_all)).item()
    rmse = torch.sqrt(torch.mean((pred_all - target_all) ** 2)).item()
    return {
        "loss": float(np.mean(losses)),
        "mae": float(mae),
        "rmse": float(rmse),
        "corr": corrcoef(pred_all, target_all),
        "hidden_spike_rate": float(np.mean(rates)),
    }


def save_prediction_plot(
    model: nn.Module,
    dataset: MobiVitalWindowDataset,
    index: int,
    out_path: Path,
    device: torch.device,
) -> None:
    model.eval()
    x, y = dataset[index]
    with torch.no_grad():
        pred, _ = model(x.unsqueeze(0).to(device))
    pred_np = pred.squeeze(0).cpu().numpy()[0]
    target = y.numpy()[0]
    spikes = x.numpy()
    t = np.arange(target.shape[0]) / dataset.fs

    plt.figure(figsize=(12, 7))
    plt.subplot(3, 1, 1)
    spike_events = [np.where(channel > 0)[0] / dataset.fs for channel in spikes]
    lineoffsets = np.arange(spikes.shape[0] - 1, -1, -1)
    plt.eventplot(spike_events, lineoffsets=lineoffsets)
    if spikes.shape[0] == 1:
        labels = ["rate"]
    elif spikes.shape[0] == 2:
        labels = ["positive", "negative"]
    elif spikes.shape[0] == 3:
        labels = ["delta+", "delta-", "rate"]
    else:
        labels = [f"ch{i}" for i in range(spikes.shape[0])]
    plt.yticks(lineoffsets, labels)
    plt.title("Spike input")

    plt.subplot(3, 1, 2)
    plt.plot(t, target, label="respiration target")
    plt.plot(t, pred_np, label="SNN prediction", alpha=0.85)
    plt.legend(loc="upper right")
    plt.title("SNN respiration reconstruction")

    plt.subplot(3, 1, 3)
    plt.plot(t, target - pred_np)
    plt.title("Residual")
    plt.xlabel("time (s)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, nargs="+", help="One or more MobiVital CSV files")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=2.0)
    parser.add_argument("--bin-radius", type=int, default=0)
    parser.add_argument("--preprocess", choices=["none", "moving_average", "fft_bandpass"], default="none")
    parser.add_argument("--threshold-scale", type=float, default=0.75)
    parser.add_argument("--threshold-mode", choices=["std", "mad", "percentile", "target_rate"], default="std")
    parser.add_argument("--threshold-percentile", type=float, default=75.0)
    parser.add_argument("--target-spike-rate", type=float, default=0.2)
    parser.add_argument("--levels", type=int, default=5, help="Number of level-crossing thresholds")
    parser.add_argument("--encode", choices=["rate", "delta", "delta_rate_hybrid", "level_crossing"], default="delta")
    parser.add_argument("--model", choices=["lif_cnn", "spiking_tcn"], default="lif_cnn")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rate-reg", type=float, default=1e-3, help="Penalty for hidden spike activity")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.csv[0]).resolve().parent / "delta_snn_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(11)
    np.random.seed(11)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MobiVitalWindowDataset(
        args.csv,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        bin_radius=args.bin_radius,
        encode=args.encode,
        threshold_scale=args.threshold_scale,
        threshold_mode=args.threshold_mode,
        threshold_percentile=args.threshold_percentile,
        target_spike_rate=args.target_spike_rate,
        levels=args.levels,
        preprocess=args.preprocess,
    )
    if len(dataset) < 4:
        raise RuntimeError("Not enough windows. Use a shorter window or more CSV files.")

    train_idx, val_idx, test_idx = split_indices(len(dataset))
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=8, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=8)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=8)

    in_channels = int(dataset[0][0].shape[0])
    if args.model == "lif_cnn":
        model = DeltaSNN(in_channels=in_channels).to(device)
    elif args.model == "spiking_tcn":
        model = SpikingTCN(in_channels=in_channels).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_path = out_dir / "snn.pt"
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
        "dataset_windows": len(dataset),
        "train_windows": len(train_idx),
        "val_windows": len(val_idx),
        "test_windows": len(test_idx),
        "device": str(device),
        "bin_radius": args.bin_radius,
        "preprocess": args.preprocess,
        "encode": args.encode,
        "model": args.model,
        "input_channels": in_channels,
        "threshold_scale": args.threshold_scale,
        "threshold_mode": args.threshold_mode,
        "threshold_percentile": args.threshold_percentile,
        "target_spike_rate": args.target_spike_rate,
        "levels": args.levels,
        "input_spike_rate": input_spike_density,
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
    print(f"saved plot: {out_dir / 'prediction.png'}")
    print(f"saved metrics: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
