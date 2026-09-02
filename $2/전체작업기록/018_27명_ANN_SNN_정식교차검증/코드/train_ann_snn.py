from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DATA_PATH = HERE / "data" / "development_training_windows.npz"


def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


class Adam:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for key, value in self.params.items():
            grad = grads[key]
            self.m[key] = 0.9 * self.m[key] + 0.1 * grad
            self.v[key] = 0.999 * self.v[key] + 0.001 * grad * grad
            mhat = self.m[key] / (1.0 - 0.9 ** self.t)
            vhat = self.v[key] / (1.0 - 0.999 ** self.t)
            value -= self.lr * mhat / (np.sqrt(vhat) + 1e-8)


def multitask_loss(pred, rr, motion):
    rr_mask = np.isfinite(rr)
    motion_mask = motion >= 0
    d = np.zeros_like(pred, dtype=np.float32)
    rr_loss = 0.0
    motion_loss = 0.0
    if np.any(rr_mask):
        error = pred[rr_mask, 0] - rr[rr_mask]
        rr_loss = float(0.5 * np.mean(error * error))
        d[rr_mask, 0] = error / max(int(rr_mask.sum()), 1)
    if np.any(motion_mask):
        logits = pred[motion_mask, 1]
        target = motion[motion_mask].astype(np.float32)
        prob = sigmoid(logits)
        positives = max(float(np.sum(target == 1)), 1.0)
        negatives = max(float(np.sum(target == 0)), 1.0)
        pos_weight = negatives / positives
        weights = np.where(target == 1, pos_weight, 1.0)
        norm = max(float(weights.sum()), 1.0)
        motion_loss = float(np.sum(weights * (np.logaddexp(0.0, logits) - target * logits)) / norm)
        d[motion_mask, 1] = 0.2 * weights * (prob - target) / norm
    return rr_loss + 0.2 * motion_loss, rr_loss, motion_loss, d


def clip_grads(grads, max_norm=5.0):
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if norm > max_norm:
        return {k: g * (max_norm / norm) for k, g in grads.items()}
    return grads


class ANN:
    def __init__(self, steps, input_dim, hidden=64, seed=42):
        rng = np.random.default_rng(seed)
        flat = steps * input_dim
        self.params = {
            "W1": (rng.standard_normal((flat, hidden)) * math.sqrt(2.0 / flat)).astype(np.float32),
            "b1": np.zeros(hidden, dtype=np.float32),
            "W2": (rng.standard_normal((hidden, 2)) * math.sqrt(2.0 / hidden)).astype(np.float32),
            "b2": np.zeros(2, dtype=np.float32),
        }

    def forward(self, x, store=False):
        flat = x.reshape(len(x), -1)
        pre = flat @ self.params["W1"] + self.params["b1"]
        hidden = np.maximum(pre, 0.0)
        pred = hidden @ self.params["W2"] + self.params["b2"]
        return (pred, flat, pre, hidden) if store else pred

    def loss_grads(self, x, rr, motion):
        pred, flat, pre, hidden = self.forward(x, True)
        loss, rr_loss, motion_loss, dpred = multitask_loss(pred, rr, motion)
        grads = {
            "W2": hidden.T @ dpred + 1e-4 * self.params["W2"],
            "b2": dpred.sum(axis=0),
        }
        dh = dpred @ self.params["W2"].T
        dpre = dh * (pre > 0)
        grads["W1"] = flat.T @ dpre + 1e-4 * self.params["W1"]
        grads["b1"] = dpre.sum(axis=0)
        return loss, rr_loss, motion_loss, clip_grads(grads), 0.0


class SNN:
    def __init__(self, input_dim, hidden=64, seed=42):
        rng = np.random.default_rng(seed)
        self.beta = np.float32(0.92)
        self.threshold = np.float32(1.0)
        self.params = {
            "W1": (rng.standard_normal((input_dim, hidden)) * math.sqrt(1.5 / input_dim)).astype(np.float32),
            "b1": np.full(hidden, 0.02, dtype=np.float32),
            "W2": (rng.standard_normal((hidden, 2)) * math.sqrt(1.5 / hidden)).astype(np.float32),
            "b2": np.zeros(2, dtype=np.float32),
        }

    def forward(self, x, store=False):
        batch, steps, _ = x.shape
        hidden = self.params["W1"].shape[1]
        voltage = np.zeros((batch, hidden), dtype=np.float32)
        previous_spike = np.zeros_like(voltage)
        voltages = np.empty((batch, steps, hidden), dtype=np.float32)
        spikes = np.empty_like(voltages)
        currents = np.empty((batch, steps, 2), dtype=np.float32)
        for t in range(steps):
            voltage = self.beta * voltage + x[:, t] @ self.params["W1"] + self.params["b1"]
            voltage -= previous_spike * self.threshold
            spike = (voltage >= self.threshold).astype(np.float32)
            current = spike @ self.params["W2"] + self.params["b2"]
            voltages[:, t] = voltage
            spikes[:, t] = spike
            currents[:, t] = current
            previous_spike = spike
        pred = currents.mean(axis=1)
        return (pred, voltages, spikes, currents) if store else pred

    def loss_grads(self, x, rr, motion):
        pred, voltages, spikes, currents = self.forward(x, True)
        loss, rr_loss, motion_loss, dpred = multitask_loss(pred, rr, motion)
        batch, steps, _ = currents.shape
        dcurrent = np.repeat((dpred / steps)[:, None, :], steps, axis=1)
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        for t in range(steps):
            grads["W2"] += spikes[:, t].T @ dcurrent[:, t]
        grads["b2"] = dcurrent.sum(axis=(0, 1))
        dspike = dcurrent @ self.params["W2"].T
        carry = np.zeros((batch, self.params["W1"].shape[1]), dtype=np.float32)
        for t in range(steps - 1, -1, -1):
            surrogate = 0.35 / (1.0 + np.abs(voltages[:, t] - self.threshold)) ** 2
            dvoltage = dspike[:, t] * surrogate + self.beta * carry
            grads["W1"] += x[:, t].T @ dvoltage
            grads["b1"] += dvoltage.sum(axis=0)
            carry = dvoltage
        grads["W1"] += 1e-4 * self.params["W1"]
        grads["W2"] += 1e-4 * self.params["W2"]
        return loss, rr_loss, motion_loss, clip_grads(grads), float(spikes.mean())


def load_data():
    data = np.load(DATA_PATH, allow_pickle=True)
    wave = data["waveforms"].astype(np.float32)
    scalar = data["scalars"].astype(np.float32)
    rr = data["rr_targets"].astype(np.float32)
    motion = data["motion_targets"].astype(np.int64)
    subjects = data["subjects"].astype(str)
    return wave, scalar, rr, motion, subjects


def fit_transform(wave, scalar, rr, train_mask):
    wave_mean = wave[train_mask].mean(axis=(0, 1), keepdims=True)
    wave_std = wave[train_mask].std(axis=(0, 1), keepdims=True) + 1e-6
    scalar_mean = scalar[train_mask].mean(axis=0, keepdims=True)
    scalar_std = scalar[train_mask].std(axis=0, keepdims=True) + 1e-6
    rr_mean = float(np.nanmean(rr[train_mask]))
    rr_std = float(np.nanstd(rr[train_mask]) + 1e-6)
    wave_n = np.clip((wave - wave_mean) / wave_std, -5, 5)
    scalar_n = np.clip((scalar - scalar_mean) / scalar_std, -5, 5)
    scalar_time = np.repeat(scalar_n[:, None, :], wave.shape[1], axis=1)
    x = np.concatenate([wave_n, scalar_time], axis=2).astype(np.float32)
    rr_n = ((rr - rr_mean) / rr_std).astype(np.float32)
    scaler = {
        "wave_mean": wave_mean.reshape(-1), "wave_std": wave_std.reshape(-1),
        "scalar_mean": scalar_mean.reshape(-1), "scalar_std": scalar_std.reshape(-1),
        "rr_mean": np.asarray([rr_mean], dtype=np.float32),
        "rr_std": np.asarray([rr_std], dtype=np.float32),
    }
    return x, rr_n, scaler


def encode_input(x, method):
    """Encode six temporal radar channels; keep 27 radar-quality scalars as direct current."""
    dynamic = x[:, :, :6]
    static = x[:, :, 6:]
    if method == "direct":
        return x.astype(np.float32)
    if method == "signed_rate":
        magnitude = np.clip(np.abs(dynamic) / 3.0, 0.0, 1.0)
        accumulator = np.zeros((len(x), dynamic.shape[2]), dtype=np.float32)
        positive = np.zeros_like(dynamic, dtype=np.float32)
        negative = np.zeros_like(dynamic, dtype=np.float32)
        for t in range(dynamic.shape[1]):
            accumulator += magnitude[:, t]
            event = (accumulator >= 1.0).astype(np.float32)
            accumulator -= event
            positive[:, t] = event * (dynamic[:, t] >= 0)
            negative[:, t] = event * (dynamic[:, t] < 0)
        return np.concatenate([positive, negative, static], axis=2).astype(np.float32)
    if method == "delta_event":
        change = np.diff(dynamic, axis=1, prepend=dynamic[:, :1])
        threshold = 0.25
        positive = (change >= threshold).astype(np.float32)
        negative = (change <= -threshold).astype(np.float32)
        return np.concatenate([positive, negative, static], axis=2).astype(np.float32)
    raise ValueError(f"unknown encoding: {method}")


def metrics(model, x, rr_n, motion, mask, scaler):
    pred = model.forward(x[mask])
    rr_true = rr_n[mask]
    motion_true = motion[mask]
    rr_mask = np.isfinite(rr_true)
    motion_mask = motion_true >= 0
    result = {}
    if np.any(rr_mask):
        rr_pred_bpm = pred[rr_mask, 0] * float(scaler["rr_std"][0]) + float(scaler["rr_mean"][0])
        rr_true_bpm = rr_true[rr_mask] * float(scaler["rr_std"][0]) + float(scaler["rr_mean"][0])
        error = np.abs(rr_pred_bpm - rr_true_bpm)
        result.update({"rr_n": int(rr_mask.sum()), "rr_mae_bpm": float(error.mean()), "rr_within2": float(np.mean(error <= 2.0))})
    if np.any(motion_mask):
        motion_pred = sigmoid(pred[motion_mask, 1]) >= 0.5
        target = motion_true[motion_mask]
        positive = target == 1
        negative = target == 0
        sensitivity = float(np.mean(motion_pred[positive])) if np.any(positive) else math.nan
        specificity = float(np.mean(~motion_pred[negative])) if np.any(negative) else math.nan
        result.update({
            "motion_n": int(motion_mask.sum()),
            "motion_accuracy": float(np.mean(motion_pred == target)),
            "motion_sensitivity": sensitivity,
            "motion_specificity": specificity,
            "motion_balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
        })
    return result


def save_model(path, model, scaler, metadata):
    payload = {f"param_{k}": v for k, v in model.params.items()}
    payload.update({f"scaler_{k}": v for k, v in scaler.items()})
    payload["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    np.savez_compressed(path, **payload)


def train_one(name, model, lr, x, rr, motion, train_mask, val_mask, epochs, seed, patience=None, min_epochs=1, verbose=True):
    optimizer = Adam(model.params, lr=lr)
    rng = np.random.default_rng(seed)
    train_idx = np.flatnonzero(train_mask & (np.isfinite(rr) | (motion >= 0)))
    history = []
    best_params = {k: v.copy() for k, v in model.params.items()}
    best_val = math.inf
    best_epoch = 0
    wait = 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        order = rng.permutation(train_idx)
        batch_losses, spikes = [], []
        for start in range(0, len(order), 16):
            idx = order[start:start + 16]
            loss, rr_loss, motion_loss, grads, spike_rate = model.loss_grads(x[idx], rr[idx], motion[idx])
            optimizer.step(grads)
            batch_losses.append((loss, rr_loss, motion_loss))
            spikes.append(spike_rate)
        val_pred = model.forward(x[val_mask])
        val_loss, val_rr_loss, val_motion_loss, _ = multitask_loss(val_pred, rr[val_mask], motion[val_mask])
        row = {
            "model": name, "epoch": epoch,
            "train_loss": float(np.mean([v[0] for v in batch_losses])),
            "train_rr_loss": float(np.mean([v[1] for v in batch_losses])),
            "train_motion_loss": float(np.mean([v[2] for v in batch_losses])),
            "val_loss": val_loss, "val_rr_loss": val_rr_loss, "val_motion_loss": val_motion_loss,
            "spike_rate": float(np.mean(spikes)), "epoch_seconds": time.perf_counter() - started,
        }
        history.append(row)
        if verbose:
            print(f"[{name}] epoch {epoch}/{epochs} train={row['train_loss']:.4f} val={row['val_loss']:.4f} spike={row['spike_rate']:.4f}", flush=True)
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_params = {k: v.copy() for k, v in model.params.items()}
            wait = 0
        else:
            wait += 1
        if patience is not None and epoch >= min_epochs and wait >= patience:
            break
    model.params = best_params
    return history, best_epoch, best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "screen"], default="smoke")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    seed = 42
    wave, scalar, rr, motion, subjects = load_data()
    unique_subjects = sorted(set(subjects.tolist()))
    val_subjects = unique_subjects[-2:]
    val_mask = np.isin(subjects, val_subjects)
    train_mask = ~val_mask
    x_direct, rr_n, scaler = fit_transform(wave, scalar, rr, train_mask)

    epochs = args.epochs if args.epochs is not None else (3 if args.mode == "smoke" else 40)
    methods = ["direct"] if args.mode == "smoke" else ["direct", "signed_rate", "delta_event"]

    run_dir = OUT / ("models_smoke" if args.mode == "smoke" else "encoding_screen")
    run_dir.mkdir(parents=True, exist_ok=True)
    all_history = []
    results = {}
    ann = ANN(x_direct.shape[1], x_direct.shape[2], hidden=64, seed=seed)
    ann_history, ann_best_epoch, ann_best_val = train_one(
        "ANN_direct", ann, 1e-3, x_direct, rr_n, motion, train_mask, val_mask,
        epochs, seed, patience=10 if args.mode == "screen" else None, min_epochs=10,
    )
    all_history.extend(ann_history)
    results["ANN_direct"] = {
        "best_epoch": ann_best_epoch, "best_val_loss": ann_best_val,
        "validation": metrics(ann, x_direct, rr_n, motion, val_mask, scaler),
        "parameters": int(sum(v.size for v in ann.params.values())),
    }
    save_model(run_dir / "ann_direct.npz", ann, scaler, {"status": f"{args.mode}_not_final", "model": "ANN", "encoding": "direct"})

    snn_models = {}
    for method in methods:
        encoded = encode_input(x_direct, method)
        snn = SNN(encoded.shape[2], hidden=64, seed=seed)
        name = f"SNN_{method}"
        history, best_epoch, best_val = train_one(
            name, snn, 5e-4, encoded, rr_n, motion, train_mask, val_mask,
            epochs, seed, patience=10 if args.mode == "screen" else None, min_epochs=10,
        )
        all_history.extend(history)
        results[name] = {
            "best_epoch": best_epoch, "best_val_loss": best_val,
            "validation": metrics(snn, encoded, rr_n, motion, val_mask, scaler),
            "parameters": int(sum(v.size for v in snn.params.values())),
            "mean_input_event_rate": float(np.mean(encoded[:, :, :12])) if method != "direct" else None,
            "mean_hidden_spike_rate_best_training": float(history[best_epoch - 1]["spike_rate"]),
        }
        save_model(run_dir / f"snn_{method}.npz", snn, scaler, {
            "status": f"{args.mode}_not_final", "model": "SNN", "encoding": method,
            "beta": float(snn.beta), "threshold": float(snn.threshold),
        })
        snn_models[method] = snn

    metadata = {
        "status": f"{args.mode}_not_final",
        "epochs_max": epochs,
        "seed": seed,
        "train_subjects": [s for s in unique_subjects if s not in val_subjects],
        "validation_subjects": val_subjects,
        "input_shape_direct": list(x_direct.shape[1:]),
        "hidden_units": 64,
        "encoding": "direct temporal current; 3 radar phase + delta, scalar quality broadcast over time",
        "outputs": ["normalized_rr_regression", "motion_logit"],
        "rr_decoder": "mean output current, inverse target normalization",
        "not_used": "independent-validation subjects",
    }
    with (run_dir / f"{args.mode}_history.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_history[0].keys()))
        writer.writeheader(); writer.writerows(all_history)
    result = {
        **metadata,
        "results": results,
        "history": all_history,
        "interpretation": "Screening uses one fixed development split and one seed; it selects candidates but is not research performance.",
    }
    (run_dir / f"{args.mode}_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
