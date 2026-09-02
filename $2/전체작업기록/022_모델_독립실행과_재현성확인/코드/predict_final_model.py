from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def encode_input(x: np.ndarray, method: str) -> np.ndarray:
    dynamic, static = x[:, :, :6], x[:, :, 6:]
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
        positive = (change >= 0.25).astype(np.float32)
        negative = (change <= -0.25).astype(np.float32)
        return np.concatenate([positive, negative, static], axis=2).astype(np.float32)
    raise ValueError(method)


def load_and_normalize(model_data, input_data, method):
    wave = input_data["waveforms"].astype(np.float32)
    scalar = input_data["scalars"].astype(np.float32)
    wave_mean = model_data["scaler_wave_mean"].reshape(1, 1, -1)
    wave_std = model_data["scaler_wave_std"].reshape(1, 1, -1)
    scalar_mean = model_data["scaler_scalar_mean"].reshape(1, -1)
    scalar_std = model_data["scaler_scalar_std"].reshape(1, -1)
    wave_n = np.clip((wave - wave_mean) / wave_std, -5, 5)
    scalar_n = np.clip((scalar - scalar_mean) / scalar_std, -5, 5)
    scalar_time = np.repeat(scalar_n[:, None, :], wave.shape[1], axis=1)
    direct = np.concatenate([wave_n, scalar_time], axis=2).astype(np.float32)
    return encode_input(direct, method)


def forward(model_data, metadata, x):
    if metadata["candidate"].startswith("ANN"):
        flat = x.reshape(len(x), -1)
        hidden = np.maximum(flat @ model_data["param_W1"] + model_data["param_b1"], 0.0)
        return hidden @ model_data["param_W2"] + model_data["param_b2"]
    beta, threshold = 0.92, 1.0
    voltage = np.zeros((len(x), model_data["param_W1"].shape[1]), dtype=np.float32)
    previous_spike = np.zeros_like(voltage)
    current_sum = np.zeros((len(x), 2), dtype=np.float32)
    for t in range(x.shape[1]):
        voltage = beta * voltage + x[:, t] @ model_data["param_W1"] + model_data["param_b1"]
        voltage -= previous_spike * threshold
        spike = (voltage >= threshold).astype(np.float32)
        current_sum += spike @ model_data["param_W2"] + model_data["param_b2"]
        previous_spike = spike
    return current_sum / x.shape[1]


def main():
    parser = argparse.ArgumentParser(description="Run a saved all-27 breathing model on preprocessed 30-second radar windows.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="NPZ containing waveforms[N,150,6] and scalars[N,27]")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    model_data = np.load(args.model, allow_pickle=True)
    input_data = np.load(args.input, allow_pickle=True)
    metadata = json.loads(str(model_data["metadata_json"]))
    x = load_and_normalize(model_data, input_data, metadata["encoding"])
    if list(x.shape[1:]) != list(metadata["input_shape"]):
        raise ValueError(f"input shape {list(x.shape[1:])} != model shape {metadata['input_shape']}")
    pred = forward(model_data, metadata, x)
    rr = pred[:, 0] * float(model_data["scaler_rr_std"][0]) + float(model_data["scaler_rr_mean"][0])
    motion_probability = sigmoid(pred[:, 1])

    subjects = input_data["subjects"].astype(str) if "subjects" in input_data else np.asarray([""] * len(rr))
    starts = input_data["starts"] if "starts" in input_data else np.arange(len(rr), dtype=float)
    rows = [{
        "index": i,
        "subject": subjects[i],
        "start_s": float(starts[i]),
        "respiratory_rate_bpm": float(rr[i]),
        "motion_probability": float(motion_probability[i]),
        "reject_due_to_motion": int(motion_probability[i] >= 0.5),
    } for i in range(len(rr))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "model": str(args.model),
        "candidate": metadata["candidate"],
        "windows_processed": len(rows),
        "output": str(args.output),
        "finite_predictions": int(np.isfinite(rr).sum()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
