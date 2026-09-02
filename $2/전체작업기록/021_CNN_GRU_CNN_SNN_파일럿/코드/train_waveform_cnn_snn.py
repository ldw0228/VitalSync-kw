from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "waveform_training_dataset.npz"
SPLIT_PATH = HERE / "data" / "split_manifest.csv"
RUN_ROOT = HERE / "outputs"
MODEL_FS = 10.0
LOW_HZ, HIGH_HZ = 0.07, 0.70
MODEL_SEEDS = [42, 314, 2718]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class WaveDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray, rr_mean: float, rr_std: float):
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=int)
        self.rr_mean = rr_mean
        self.rr_std = rr_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        i = self.indices[position]
        rr = float(self.arrays["rr"][i])
        return {
            "index": i,
            "x": torch.from_numpy(self.arrays["x"][i]),
            "wave": torch.from_numpy(self.arrays["wave"][i]),
            "wave_mask": torch.tensor(float(self.arrays["wave_mask"][i])),
            "rr": torch.tensor(0.0 if not math.isfinite(rr) else (rr - self.rr_mean) / self.rr_std),
            "rr_mask": torch.tensor(float(math.isfinite(rr))),
            "motion": torch.tensor(float(max(self.arrays["motion"][i], 0))),
            "motion_mask": torch.tensor(float(self.arrays["motion"][i] >= 0)),
            "apnea": torch.tensor(float(self.arrays["apnea"][i])),
        }


class RadarEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(24, 32, 7, padding=3), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class MultiRadarFrontEnd(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = RadarEncoder()
        self.gate = nn.Linear(64, 1)

    def forward(self, x):
        batch, _channels, steps = x.shape
        encoded = self.encoder(x.reshape(batch * 3, 24, steps))
        encoded = encoded.reshape(batch, 3, 64, encoded.shape[-1])
        weights = torch.softmax(self.gate(encoded.mean(dim=-1)).squeeze(-1), dim=1)
        fused = torch.sum(encoded * weights[:, :, None, None], dim=1)
        return fused, weights


class WaveDecoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 48, 5, padding=2), nn.GELU(),
            nn.Conv1d(48, 24, 5, padding=2), nn.GELU(),
            nn.Conv1d(24, 1, 7, padding=3),
        )

    def forward(self, sequence, output_steps=200):
        sequence = F.interpolate(sequence, size=output_steps, mode="linear", align_corners=False)
        return self.net(sequence).squeeze(1)


class CNNGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.front = MultiRadarFrontEnd()
        self.gru = nn.GRU(64, 96, batch_first=True)
        self.decoder = WaveDecoder(96)
        self.head = nn.Sequential(nn.Linear(96, 48), nn.GELU(), nn.Dropout(0.15))
        self.rr = nn.Linear(48, 1)
        self.motion = nn.Linear(48, 1)
        self.apnea = nn.Linear(48, 1)

    def forward(self, x):
        encoded, gates = self.front(x)
        sequence, _ = self.gru(encoded.transpose(1, 2))
        temporal = sequence.transpose(1, 2)
        feature = self.head(sequence.mean(dim=1))
        return {
            "wave": self.decoder(temporal),
            "rr": self.rr(feature).squeeze(-1),
            "motion": self.motion(feature).squeeze(-1),
            "apnea": self.apnea(feature).squeeze(-1),
            "spike_rate": x.new_tensor(0.0),
            "gates": gates,
        }


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage):
        ctx.save_for_backward(voltage)
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (voltage,) = ctx.saved_tensors
        surrogate = 0.35 / (1.0 + voltage.abs()).pow(2)
        return grad_output * surrogate


class RecurrentLIF(nn.Module):
    def __init__(self, input_dim, hidden_dim, beta=0.92):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.recurrent = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.full((hidden_dim,), 0.02))
        self.beta = beta

    def step(self, x, voltage, previous_spike):
        voltage = self.beta * voltage + self.input(x) + self.recurrent(previous_spike) + self.bias
        voltage = voltage - previous_spike
        spike = SurrogateSpike.apply(voltage - 1.0)
        return voltage, spike


class CNNSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.front = MultiRadarFrontEnd()
        self.lif1 = RecurrentLIF(64, 128)
        self.lif2 = RecurrentLIF(128, 64)
        self.decoder = WaveDecoder(64)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.15))
        self.rr = nn.Linear(64, 1)
        self.motion = nn.Linear(64, 1)
        self.apnea = nn.Linear(64, 1)

    def forward(self, x):
        encoded, gates = self.front(x)
        batch, _channels, steps = encoded.shape
        v1 = x.new_zeros((batch, 128)); s1 = x.new_zeros((batch, 128))
        v2 = x.new_zeros((batch, 64)); s2 = x.new_zeros((batch, 64))
        spikes, voltages = [], []
        for t in range(steps):
            v1, s1 = self.lif1.step(encoded[:, :, t], v1, s1)
            v2, s2 = self.lif2.step(s1, v2, s2)
            spikes.append(s2)
            voltages.append(v2)
        spike_sequence = torch.stack(spikes, dim=2)
        voltage_sequence = torch.stack(voltages, dim=2)
        feature = self.head(torch.cat((spike_sequence.mean(dim=2), voltage_sequence.mean(dim=2)), dim=1))
        return {
            "wave": self.decoder(spike_sequence),
            "rr": self.rr(feature).squeeze(-1),
            "motion": self.motion(feature).squeeze(-1),
            "apnea": self.apnea(feature).squeeze(-1),
            "spike_rate": spike_sequence.mean(),
            "gates": gates,
        }


def normalized_wave(x):
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-5)


def waveform_loss(pred, target, mask):
    keep = mask > 0.5
    if not torch.any(keep):
        return pred.sum() * 0.0
    p = normalized_wave(pred[keep])
    y = normalized_wave(target[keep])
    corr = torch.mean(p * y, dim=1)
    sign = torch.where(corr.detach() >= 0, 1.0, -1.0)[:, None]
    aligned = p * sign
    correlation_loss = torch.mean(1.0 - torch.abs(corr))
    point_loss = F.smooth_l1_loss(aligned, y)
    derivative_loss = F.smooth_l1_loss(aligned[:, 1:] - aligned[:, :-1], y[:, 1:] - y[:, :-1])
    return 0.60 * correlation_loss + 0.30 * point_loss + 0.10 * derivative_loss


def masked_smooth_l1(pred, target, mask):
    keep = mask > 0.5
    return F.smooth_l1_loss(pred[keep], target[keep]) if torch.any(keep) else pred.sum() * 0.0


def masked_bce(logit, target, mask, pos_weight):
    keep = mask > 0.5
    if not torch.any(keep):
        return logit.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logit[keep], target[keep], pos_weight=pos_weight)


def total_loss(output, batch, motion_weight, apnea_weight):
    wave = waveform_loss(output["wave"], batch["wave"], batch["wave_mask"])
    rr = masked_smooth_l1(output["rr"], batch["rr"], batch["rr_mask"])
    motion = masked_bce(output["motion"], batch["motion"], batch["motion_mask"], motion_weight)
    apnea = F.binary_cross_entropy_with_logits(output["apnea"], batch["apnea"], pos_weight=apnea_weight)
    return wave + 0.50 * rr + 0.20 * motion + 0.20 * apnea, wave, rr, motion, apnea


def load_arrays():
    z = np.load(DATA_PATH, allow_pickle=True)
    return {
        "x": z["inputs"].astype(np.float32),
        "wave": z["waveform_targets"].astype(np.float32),
        "rr": z["rr_targets"].astype(np.float32),
        "wave_mask": z["waveform_masks"].astype(np.int8),
        "motion": z["motion_targets"].astype(np.int8),
        "apnea": z["apnea_targets"].astype(np.int8),
        "subjects": z["subjects"].astype(str),
        "states": z["states"].astype(str),
        "starts": z["starts"].astype(np.float32),
    }


def load_splits():
    rows = list(csv.DictReader(SPLIT_PATH.open(encoding="utf-8-sig", newline="")))
    result = []
    for fold in range(1, 6):
        current = [r for r in rows if int(r["fold"]) == fold]
        result.append({role: [r["subject"] for r in current if r["role"] == role] for role in ("train", "validation", "test")})
    return result


def class_weights(arrays, train_idx, device):
    motion = arrays["motion"][train_idx]
    motion = motion[motion >= 0]
    apnea = arrays["apnea"][train_idx]
    motion_pos = max(int(np.sum(motion == 1)), 1)
    apnea_pos = max(int(np.sum(apnea == 1)), 1)
    return (
        torch.tensor(float(np.sum(motion == 0)) / motion_pos, device=device),
        torch.tensor(float(np.sum(apnea == 0)) / apnea_pos, device=device),
    )


@torch.no_grad()
def validation_loss(model, loader, device, weights):
    model.eval(); total, count = 0.0, 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        loss, *_ = total_loss(model(batch["x"]), batch, *weights)
        total += float(loss) * len(batch["x"]); count += len(batch["x"])
    return total / max(count, 1)


def rr_from_wave(wave):
    wave = np.asarray(wave, dtype=float)
    wave = wave - np.mean(wave)
    power = np.abs(np.fft.rfft(wave * np.hanning(len(wave)), n=4096)) ** 2
    freq = np.fft.rfftfreq(4096, 1.0 / MODEL_FS)
    keep = (freq >= LOW_HZ) & (freq <= HIGH_HZ)
    return float(freq[keep][int(np.argmax(power[keep]))] * 60.0)


def balanced_accuracy(target, prediction):
    target = np.asarray(target); prediction = np.asarray(prediction)
    sens = np.mean(prediction[target == 1] == 1) if np.any(target == 1) else math.nan
    spec = np.mean(prediction[target == 0] == 0) if np.any(target == 0) else math.nan
    return float(np.nanmean([sens, spec]))


@torch.no_grad()
def evaluate(model, loader, device, rr_mean, rr_std, arrays, candidate, fold, seed):
    model.eval(); rows = []
    for batch in loader:
        index = batch["index"].numpy()
        batch_device = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        output = model(batch_device["x"])
        wave_pred = output["wave"].cpu().numpy()
        rr_pred = (output["rr"].cpu().numpy() * rr_std + rr_mean)
        motion_prob = torch.sigmoid(output["motion"]).cpu().numpy()
        apnea_prob = torch.sigmoid(output["apnea"]).cpu().numpy()
        gates = output["gates"].cpu().numpy()
        for j, original in enumerate(index):
            true_rr = float(arrays["rr"][original])
            wave_ok = bool(arrays["wave_mask"][original])
            corr = ""
            derived_rr = ""
            if wave_ok:
                p = (wave_pred[j] - np.mean(wave_pred[j])) / (np.std(wave_pred[j]) + 1e-8)
                y = arrays["wave"][original]
                corr = float(abs(np.mean(p * y)))
                derived_rr = rr_from_wave(p)
            rows.append({
                "candidate": candidate, "fold": fold, "seed": seed, "index": int(original),
                "subject": arrays["subjects"][original], "state": arrays["states"][original],
                "start_s": float(arrays["starts"][original]),
                "rr_true_bpm": "" if not math.isfinite(true_rr) else true_rr,
                "rr_head_bpm": "" if not math.isfinite(true_rr) else float(rr_pred[j]),
                "rr_wave_bpm": derived_rr,
                "wave_abs_correlation": corr,
                "motion_true": "" if arrays["motion"][original] < 0 else int(arrays["motion"][original]),
                "motion_probability": float(motion_prob[j]),
                "apnea_true": int(arrays["apnea"][original]),
                "apnea_probability": float(apnea_prob[j]),
                "radar1_gate": float(gates[j, 0]), "radar2_gate": float(gates[j, 1]), "radar3_gate": float(gates[j, 2]),
            })
    return rows


def summarize(rows):
    rr_rows = [r for r in rows if r["rr_true_bpm"] != ""]
    head_error = np.asarray([abs(float(r["rr_head_bpm"]) - float(r["rr_true_bpm"])) for r in rr_rows])
    wave_error = np.asarray([abs(float(r["rr_wave_bpm"]) - float(r["rr_true_bpm"])) for r in rr_rows])
    corr = np.asarray([float(r["wave_abs_correlation"]) for r in rr_rows])
    motion_rows = [r for r in rows if r["motion_true"] != ""]
    motion_true = [int(r["motion_true"]) for r in motion_rows]
    motion_pred = [float(r["motion_probability"]) >= 0.5 for r in motion_rows]
    apnea_true = [int(r["apnea_true"]) for r in rows]
    apnea_pred = [float(r["apnea_probability"]) >= 0.5 for r in rows]
    subject_mae, subject_corr = [], []
    for subject in sorted(set(r["subject"] for r in rr_rows)):
        current = [r for r in rr_rows if r["subject"] == subject]
        subject_mae.append(np.mean([abs(float(r["rr_head_bpm"]) - float(r["rr_true_bpm"])) for r in current]))
        subject_corr.append(np.mean([float(r["wave_abs_correlation"]) for r in current]))
    return {
        "rr_n": len(rr_rows),
        "rr_head_mae_bpm": float(np.mean(head_error)),
        "rr_head_within2": float(np.mean(head_error <= 2.0)),
        "rr_wave_mae_bpm": float(np.mean(wave_error)),
        "wave_abs_correlation": float(np.mean(corr)),
        "subject_macro_rr_head_mae_bpm": float(np.mean(subject_mae)),
        "subject_macro_wave_abs_correlation": float(np.mean(subject_corr)),
        "motion_balanced_accuracy": balanced_accuracy(motion_true, motion_pred),
        "apnea_balanced_accuracy": balanced_accuracy(apnea_true, apnea_pred),
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def build_model(candidate):
    return CNNGRU() if candidate == "CNN_GRU" else CNNSNN()


def train_run(candidate, fold, seed, split, arrays, max_epochs, min_epochs, patience, run_dir, device):
    seed_everything(seed)
    train_idx = np.flatnonzero(np.isin(arrays["subjects"], split["train"]))
    val_idx = np.flatnonzero(np.isin(arrays["subjects"], split["validation"]))
    test_idx = np.flatnonzero(np.isin(arrays["subjects"], split["test"]))
    valid_rr = arrays["rr"][train_idx]
    rr_mean = float(np.nanmean(valid_rr)); rr_std = float(np.nanstd(valid_rr) + 1e-6)
    train_loader = DataLoader(WaveDataset(arrays, train_idx, rr_mean, rr_std), batch_size=32, shuffle=True)
    val_loader = DataLoader(WaveDataset(arrays, val_idx, rr_mean, rr_std), batch_size=64, shuffle=False)
    test_loader = DataLoader(WaveDataset(arrays, test_idx, rr_mean, rr_std), batch_size=64, shuffle=False)
    model = build_model(candidate).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4 if candidate == "CNN_GRU" else 5e-4, weight_decay=1e-4)
    weights = class_weights(arrays, train_idx, device)
    best_loss = math.inf; best_epoch = 0; wait = 0; best_state = None; history = []
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train(); sums = np.zeros(6, dtype=float); count = 0
        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["x"])
            loss, wave_loss, rr_loss, motion_loss, apnea_loss = total_loss(output, batch, *weights)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            n = len(batch["x"]); count += n
            sums += np.asarray([
                float(loss.detach()), float(wave_loss.detach()), float(rr_loss.detach()),
                float(motion_loss.detach()), float(apnea_loss.detach()), float(output["spike_rate"].detach()),
            ]) * n
        val = validation_loss(model, val_loader, device, weights)
        row = {"candidate": candidate, "fold": fold, "seed": seed, "epoch": epoch,
               "train_loss": sums[0]/count, "wave_loss": sums[1]/count, "rr_loss": sums[2]/count,
               "motion_loss": sums[3]/count, "apnea_loss": sums[4]/count,
               "spike_rate": sums[5]/count, "validation_loss": val}
        history.append(row)
        if val < best_loss - 1e-4:
            best_loss = val; best_epoch = epoch; wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"[{candidate}] fold={fold} seed={seed} epoch={epoch} train={row['train_loss']:.4f} val={val:.4f} spike={row['spike_rate']:.3f}", flush=True)
        if patience is not None and epoch >= min_epochs and wait >= patience:
            break
    model.load_state_dict(best_state)
    predictions = evaluate(model, test_loader, device, rr_mean, rr_std, arrays, candidate, fold, seed)
    summary = summarize(predictions)
    checkpoint = {
        "model_state": model.state_dict(), "candidate": candidate, "fold": fold, "seed": seed,
        "rr_mean": rr_mean, "rr_std": rr_std, "best_epoch": best_epoch,
        "train_subjects": split["train"], "validation_subjects": split["validation"], "test_subjects": split["test"],
        "input_shape": [72, 200], "outputs": ["respiration_waveform", "respiratory_rate", "motion", "apnea"],
    }
    torch.save(checkpoint, run_dir / f"{candidate}_fold{fold}_seed{seed}.pt")
    metric = {"candidate": candidate, "fold": fold, "seed": seed, "best_epoch": best_epoch,
              "epochs_run": len(history), "parameters": sum(p.numel() for p in model.parameters()),
              "training_seconds": time.perf_counter()-started, **summary}
    print(f"[result] {candidate} fold={fold} seed={seed} epoch={best_epoch}/{len(history)} RR={summary['subject_macro_rr_head_mae_bpm']:.3f} corr={summary['subject_macro_wave_abs_correlation']:.3f}", flush=True)
    return metric, predictions, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "pilot", "formal"], default="smoke")
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device("cpu")
    arrays = load_arrays(); splits = load_splits()
    candidates = ["CNN_GRU", "CNN_SNN"]
    if args.mode == "smoke":
        max_epochs, min_epochs, patience = 3, 1, None
        run_dir = RUN_ROOT / "smoke"; seeds = [42]; fold_ids = [1]
    elif args.mode == "pilot":
        max_epochs, min_epochs, patience = 100, 25, 15
        run_dir = RUN_ROOT / "pilot"; seeds = [42]; fold_ids = [1]
    else:
        max_epochs, min_epochs, patience = 150, 35, 20
        run_dir = RUN_ROOT / "formal"; seeds = MODEL_SEEDS; fold_ids = list(range(1, 6))
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics, predictions, histories = [], [], []
    for candidate in candidates:
        for fold in fold_ids:
            for seed in seeds:
                metric, current_predictions, current_history = train_run(
                    candidate, fold, seed, splits[fold-1], arrays,
                    max_epochs, min_epochs, patience, run_dir, device,
                )
                metrics.append(metric); predictions.extend(current_predictions); histories.extend(current_history)
                write_csv(run_dir / "fold_metrics.csv", metrics)
                write_csv(run_dir / "oof_predictions.csv", predictions)
                write_csv(run_dir / "training_history.csv", histories)
    result = {
        "status": f"{args.mode}_complete",
        "dataset": {"subjects": len(set(arrays["subjects"])), "windows": len(arrays["subjects"]), "input_shape": list(arrays["x"].shape[1:])},
        "architecture": {
            "front_end": "shared 3-block 1D CNN for each radar + learned 3-radar softmax gate",
            "CNN_GRU": "one 96-unit GRU + waveform decoder + RR/motion/apnea heads",
            "CNN_SNN": "128-unit recurrent LIF + 64-unit recurrent LIF + waveform decoder + RR/motion/apnea heads",
        },
        "protocol": {"mode": args.mode, "max_epochs": max_epochs, "min_epochs": min_epochs, "patience": patience, "seeds": seeds, "folds": fold_ids},
        "metrics": metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
