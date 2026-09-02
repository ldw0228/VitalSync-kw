import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
AUDIT_JSON = ROOT / "$2" / "audit_build" / "marker_audit_data.json"
DATA_ROOT = ROOT / "HAI_EXPERIMENT"
OUT = ROOT / "$2" / "outputs" / "snn_training"
CACHE = OUT / "feature_cache.npz"

FPS = 40
FRAME_LENGTH = 185
RANGE_BINS = 92
N_ZONES = 6
TIME_STEPS = 32
HIDDEN = 32
N_CLASSES = 7
CLASS_NAMES = ["1번 호흡", "2번 호흡", "3번 픽업", "4번 낙상", "5번 코스", "6번 왕복", "7번 자유"]
PAIR_TO_CLASS = [0, 1, 2, 2, 3, 3, 3, 3, 4, 5, 6]

MAX_EPOCHS = 90
PATIENCE = 16
BATCH_SIZE = 16
LEARNING_RATE = 0.004
WEIGHT_DECAY = 1e-4
BETA = 0.90
THRESHOLD = 1.0
SURROGATE_GAMMA = 0.35
SEED = 20260830


def load_font(size=20, bold=False):
    names = [
        Path(r"C:\Windows\Fonts\malgunbd.ttf" if bold else r"C:\Windows\Fonts\malgun.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for name in names:
        if name.exists():
            return ImageFont.truetype(str(name), size=size)
    return ImageFont.load_default()


def selected_records():
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    records = [r for r in audit["records"] if r["overall_status"] == "자동 점검 통과" and r["selected_count"] == 22]
    records.sort(key=lambda r: r["number"])
    return records


def radar_files(subject, radar_index):
    folder = DATA_ROOT / subject / str(radar_index)
    return sorted(folder.rglob("xethru_datafloat_*.dat"))


def read_motion_file(path):
    raw = np.memmap(path, dtype="<f4", mode="r")
    n_frames = raw.size // FRAME_LENGTH
    matrix = raw[: n_frames * FRAME_LENGTH].reshape(n_frames, FRAME_LENGTH)
    zones = np.array_split(np.arange(RANGE_BINS), N_ZONES)
    output = np.empty((max(n_frames - 1, 0), N_ZONES), dtype=np.float32)
    chunk_size = 4096
    prev = None
    pos = 0
    for start in range(0, n_frames, chunk_size):
        stop = min(n_frames, start + chunk_size)
        iq = np.asarray(matrix[start:stop, 1:], dtype=np.float32)
        complex_iq = iq[:, 0::2] + 1j * iq[:, 1::2]
        if prev is not None:
            complex_iq = np.concatenate([prev, complex_iq], axis=0)
        if complex_iq.shape[0] < 2:
            prev = complex_iq[-1:]
            continue
        diff = np.abs(np.diff(complex_iq, axis=0))
        block = np.stack([diff[:, zone].mean(axis=1) for zone in zones], axis=1).astype(np.float32)
        output[pos : pos + len(block)] = block
        pos += len(block)
        prev = complex_iq[-1:]
    return output[:pos]


def read_subject_motion(subject):
    radar_motion = []
    for radar_index in range(1, 4):
        files = radar_files(subject, radar_index)
        if not files:
            raise FileNotFoundError(f"{subject} radar{radar_index} 원본 없음")
        parts = [read_motion_file(path) for path in files]
        radar_motion.append(np.concatenate(parts, axis=0))
    length = min(len(motion) for motion in radar_motion)
    return np.concatenate([motion[:length] for motion in radar_motion], axis=1)


def temporal_pool(segment, steps=TIME_STEPS):
    edges = np.linspace(0, len(segment), steps + 1).astype(int)
    pooled = np.empty((steps, segment.shape[1]), dtype=np.float32)
    for i in range(steps):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            hi = min(len(segment), lo + 1)
        pooled[i] = segment[lo:hi].mean(axis=0)
    return pooled


def extract_dataset(force=False):
    if CACHE.exists() and not force:
        cached = np.load(CACHE, allow_pickle=True)
        return {name: cached[name] for name in cached.files}

    records = selected_records()
    features, durations, labels, subjects, pairs, starts, ends = [], [], [], [], [], [], []
    manifest_rows = []
    for record in records:
        subject = record["subject"]
        print(f"[extract] {subject}", flush=True)
        motion = read_subject_motion(subject)
        max_time = len(motion) / FPS
        markers = record.get("selected_slots") or record["selected_markers"]
        if len(markers) != 22 or any(value is None for value in markers):
            raise ValueError(f"{subject}: 22개 선택 마커가 완전하지 않음")
        for pair_idx in range(11):
            start_s = float(markers[2 * pair_idx])
            end_s = float(markers[2 * pair_idx + 1])
            start_s = max(0.0, min(start_s, max_time))
            end_s = max(start_s + 0.2, min(end_s, max_time))
            lo = max(0, int(math.floor(start_s * FPS)))
            hi = min(len(motion), int(math.ceil(end_s * FPS)))
            if hi - lo < 8:
                raise ValueError(f"{subject} pair {pair_idx + 1}: 구간이 너무 짧음")
            pooled = temporal_pool(motion[lo:hi])
            class_id = PAIR_TO_CLASS[pair_idx]
            features.append(pooled)
            durations.append(end_s - start_s)
            labels.append(class_id)
            subjects.append(subject)
            pairs.append(pair_idx + 1)
            starts.append(start_s)
            ends.append(end_s)
            manifest_rows.append({
                "subject": subject,
                "pair_index": pair_idx + 1,
                "scenario_class": class_id + 1,
                "scenario": CLASS_NAMES[class_id],
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "duration_s": round(end_s - start_s, 3),
                "selection_status": record["overall_status"],
                "selection_source": record["selected_source"],
            })

    result = {
        "features": np.asarray(features, dtype=np.float32),
        "durations": np.asarray(durations, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "subjects": np.asarray(subjects, dtype=object),
        "pairs": np.asarray(pairs, dtype=np.int64),
        "starts": np.asarray(starts, dtype=np.float32),
        "ends": np.asarray(ends, dtype=np.float32),
    }
    np.savez_compressed(CACHE, **result)
    with (OUT / "dataset_manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return result


def fit_scaler(features, durations, indices):
    logged = np.log1p(features[indices])
    flat = logged.reshape(-1, logged.shape[-1])
    q05 = np.percentile(flat, 5, axis=0).astype(np.float32)
    q95 = np.percentile(flat, 95, axis=0).astype(np.float32)
    scale = np.maximum(q95 - q05, 1e-5).astype(np.float32)
    duration_scale = float(max(np.percentile(durations[indices], 95), 1.0))
    return q05, scale, duration_scale


def encode(features, durations, scaler, include_duration=True):
    q05, scale, duration_scale = scaler
    base = np.clip((np.log1p(features) - q05) / scale, 0.0, 1.0)
    duration = np.clip(durations / duration_scale, 0.0, 1.5).astype(np.float32)
    duration_channel = np.repeat(duration[:, None, None], base.shape[1], axis=1)
    if not include_duration:
        duration_channel[:] = 0.0
    delta = np.diff(base, axis=1, prepend=base[:, :1])
    on = np.clip(delta * 3.0, 0.0, 1.0)
    off = np.clip(-delta * 3.0, 0.0, 1.0)
    analog = np.concatenate([base, duration_channel, on, off], axis=2).astype(np.float32)

    accumulator = np.zeros((analog.shape[0], analog.shape[2]), dtype=np.float32)
    spikes = np.zeros_like(analog, dtype=np.float32)
    gain = 1.35
    for t in range(analog.shape[1]):
        accumulator += analog[:, t] * gain
        fired = accumulator >= 1.0
        spikes[:, t] = fired
        accumulator -= fired.astype(np.float32)
    return spikes


class Adam:
    def __init__(self, params, lr=LEARNING_RATE):
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
            self.v[key] = 0.999 * self.v[key] + 0.001 * (grad * grad)
            m_hat = self.m[key] / (1 - 0.9 ** self.t)
            v_hat = self.v[key] / (1 - 0.999 ** self.t)
            value -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


class LIFSNN:
    def __init__(self, input_dim, hidden=HIDDEN, seed=SEED):
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": (rng.standard_normal((input_dim, hidden)) * math.sqrt(1.5 / input_dim)).astype(np.float32),
            "b1": np.full(hidden, 0.03, dtype=np.float32),
            "W2": (rng.standard_normal((hidden, N_CLASSES)) * math.sqrt(1.5 / hidden)).astype(np.float32),
            "b2": np.zeros(N_CLASSES, dtype=np.float32),
        }

    def forward(self, x, store=False):
        w1, b1, w2, b2 = (self.params[k] for k in ("W1", "b1", "W2", "b2"))
        batch, steps, _ = x.shape
        voltage = np.zeros((batch, w1.shape[1]), dtype=np.float32)
        previous_spike = np.zeros_like(voltage)
        voltages = np.empty((batch, steps, w1.shape[1]), dtype=np.float32)
        spikes = np.empty_like(voltages)
        for t in range(steps):
            voltage = BETA * voltage + x[:, t] @ w1 + b1 - THRESHOLD * previous_spike
            spike = (voltage >= THRESHOLD).astype(np.float32)
            voltages[:, t] = voltage
            spikes[:, t] = spike
            previous_spike = spike
        logits = (spikes @ w2 + b2).mean(axis=1)
        if store:
            return logits, voltages, spikes
        return logits

    def loss_and_grads(self, x, y, class_weights):
        logits, voltages, spikes = self.forward(x, store=True)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(shifted)
        probs /= probs.sum(axis=1, keepdims=True)
        sample_weights = class_weights[y]
        normalizer = max(float(sample_weights.sum()), 1e-8)
        loss = float(np.sum(-np.log(np.maximum(probs[np.arange(len(y)), y], 1e-8)) * sample_weights) / normalizer)
        loss += 0.5 * WEIGHT_DECAY * (float(np.sum(self.params["W1"] ** 2)) + float(np.sum(self.params["W2"] ** 2)))

        dlogits = probs
        dlogits[np.arange(len(y)), y] -= 1.0
        dlogits *= (sample_weights / normalizer)[:, None]
        steps = x.shape[1]
        grads = {key: np.zeros_like(value) for key, value in self.params.items()}
        for t in range(steps):
            grads["W2"] += spikes[:, t].T @ (dlogits / steps)
        grads["b2"] = dlogits.sum(axis=0)

        d_spikes = (dlogits @ self.params["W2"].T) / steps
        d_voltage_next = np.zeros_like(d_spikes)
        for t in range(steps - 1, -1, -1):
            surrogate = SURROGATE_GAMMA / (1.0 + np.abs(voltages[:, t] - THRESHOLD)) ** 2
            d_voltage = d_spikes * surrogate + BETA * d_voltage_next
            grads["W1"] += x[:, t].T @ d_voltage
            grads["b1"] += d_voltage.sum(axis=0)
            d_voltage_next = d_voltage
        grads["W1"] += WEIGHT_DECAY * self.params["W1"]
        grads["W2"] += WEIGHT_DECAY * self.params["W2"]
        norm = math.sqrt(sum(float(np.sum(value * value)) for value in grads.values()))
        if norm > 5.0:
            factor = 5.0 / norm
            grads = {key: value * factor for key, value in grads.items()}
        return loss, grads, float(spikes.mean())


def metrics(y_true, y_pred):
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        cm[int(true), int(pred)] += 1
    per_class = []
    recalls, f1s = [], []
    for c in range(N_CLASSES):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"class_id": c + 1, "class_name": CLASS_NAMES[c], "precision": precision, "recall": recall, "f1": f1, "support": int(cm[c].sum())})
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def evaluate(model, spikes, labels, indices):
    logits = model.forward(spikes[indices])
    pred = logits.argmax(axis=1)
    result = metrics(labels[indices], pred)
    result["predictions"] = pred.tolist()
    return result


def train_one_fold(spikes, labels, subjects, test_subject, val_subject, fold_seed):
    train_idx = np.flatnonzero((subjects != test_subject) & (subjects != val_subject))
    val_idx = np.flatnonzero(subjects == val_subject)
    test_idx = np.flatnonzero(subjects == test_subject)
    counts = np.bincount(labels[train_idx], minlength=N_CLASSES)
    class_weights = len(train_idx) / (N_CLASSES * np.maximum(counts, 1))
    model = LIFSNN(spikes.shape[2], seed=fold_seed)
    optimizer = Adam(model.params)
    rng = np.random.default_rng(fold_seed)
    best_state = None
    best_score = -1.0
    wait = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        order = rng.permutation(train_idx)
        losses, spike_rates = [], []
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            loss, grads, spike_rate = model.loss_and_grads(spikes[idx], labels[idx], class_weights)
            optimizer.step(grads)
            losses.append(loss)
            spike_rates.append(spike_rate)
        train_result = evaluate(model, spikes, labels, train_idx)
        val_result = evaluate(model, spikes, labels, val_idx)
        history.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "train_accuracy": train_result["accuracy"],
            "val_accuracy": val_result["accuracy"],
            "val_balanced_accuracy": val_result["balanced_accuracy"],
            "hidden_spike_rate": float(np.mean(spike_rates)),
        })
        score = val_result["balanced_accuracy"] + 0.05 * val_result["accuracy"]
        if score > best_score + 1e-7:
            best_score = score
            best_state = {key: value.copy() for key, value in model.params.items()}
            wait = 0
        else:
            wait += 1
        if wait >= PATIENCE:
            break
    model.params = best_state
    test_result = evaluate(model, spikes, labels, test_idx)
    return model, history, test_result, train_idx, val_idx, test_idx


def nearest_centroid_baseline(features, durations, labels, train_idx, test_idx, mode="all"):
    motion = np.concatenate([np.log1p(features).mean(axis=1), np.log1p(features).std(axis=1)], axis=1)
    duration = durations[:, None] / 450.0
    if mode == "motion":
        flat = motion
    elif mode == "duration":
        flat = duration
    else:
        flat = np.concatenate([motion, duration], axis=1)
    mean = flat[train_idx].mean(axis=0)
    std = np.maximum(flat[train_idx].std(axis=0), 1e-6)
    z = (flat - mean) / std
    centroids = np.stack([z[train_idx][labels[train_idx] == c].mean(axis=0) for c in range(N_CLASSES)])
    distance = ((z[test_idx, None, :] - centroids[None, :, :]) ** 2).mean(axis=2)
    return distance.argmin(axis=1)


def draw_confusion(cm, path):
    cm = np.asarray(cm)
    cell = 74
    left, top = 220, 100
    width, height = left + cell * N_CLASSES + 80, top + cell * N_CLASSES + 140
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, value_font = load_font(25, True), load_font(17), load_font(20, True)
    draw.text((30, 24), "LOSO 교차검증 혼동행렬", fill="#1F4E78", font=title_font)
    max_value = max(int(cm.max()), 1)
    for i in range(N_CLASSES):
        draw.text((8, top + i * cell + 22), CLASS_NAMES[i], fill="#333333", font=label_font)
        draw.text((left + i * cell + 8, top - 42), str(i + 1), fill="#333333", font=label_font)
        for j in range(N_CLASSES):
            value = int(cm[i, j])
            intensity = value / max_value
            base = (236, 243, 250) if i != j else (207, 226, 243)
            fill = tuple(int(channel * (1 - 0.45 * intensity)) for channel in base)
            x0, y0 = left + j * cell, top + i * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=fill, outline="#AAB7C4")
            text = str(value)
            bbox = draw.textbbox((0, 0), text, font=value_font)
            draw.text((x0 + (cell - (bbox[2] - bbox[0])) / 2, y0 + 22), text, fill="#1F1F1F", font=value_font)
    draw.text((left + 120, height - 42), "예측 시나리오 번호", fill="#333333", font=label_font)
    draw.text((70, top - 40), "실제 시나리오", fill="#333333", font=label_font)
    image.save(path)


def draw_history(histories, path):
    width, height = 1200, 650
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, font = load_font(25, True), load_font(17)
    draw.text((35, 22), "교차검증 평균 학습 경과", fill="#1F4E78", font=title_font)
    left, top, right, bottom = 90, 90, 1140, 560
    draw.rectangle((left, top, right, bottom), outline="#6B7280", width=2)
    max_epoch = max(len(history) for history in histories)
    for y in np.linspace(0, 1, 6):
        py = bottom - y * (bottom - top)
        draw.line((left, py, right, py), fill="#E5E7EB")
        draw.text((35, py - 10), f"{y:.1f}", fill="#555555", font=font)
    train = np.full((len(histories), max_epoch), np.nan)
    val = np.full_like(train, np.nan)
    for i, history in enumerate(histories):
        train[i, : len(history)] = [row["train_accuracy"] for row in history]
        val[i, : len(history)] = [row["val_accuracy"] for row in history]
    mean_train = np.nanmean(train, axis=0)
    mean_val = np.nanmean(val, axis=0)
    for values, color in [(mean_train, "#4472C4"), (mean_val, "#C55A11")]:
        points = []
        for idx, value in enumerate(values):
            x = left + idx / max(max_epoch - 1, 1) * (right - left)
            y = bottom - float(value) * (bottom - top)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=4)
    draw.text((left, bottom + 24), "Epoch", fill="#555555", font=font)
    draw.rectangle((800, 35, 828, 48), fill="#4472C4")
    draw.text((838, 28), "학습 정확도", fill="#333333", font=font)
    draw.rectangle((980, 35, 1008, 48), fill="#C55A11")
    draw.text((1018, 28), "검증 정확도", fill="#333333", font=font)
    image.save(path)


def train_final_model(features, durations, labels):
    indices = np.arange(len(labels))
    scaler = fit_scaler(features, durations, indices)
    spikes = encode(features, durations, scaler)
    counts = np.bincount(labels, minlength=N_CLASSES)
    weights = len(labels) / (N_CLASSES * np.maximum(counts, 1))
    model = LIFSNN(spikes.shape[2], seed=SEED + 999)
    optimizer = Adam(model.params)
    rng = np.random.default_rng(SEED + 999)
    for _ in range(60):
        order = rng.permutation(indices)
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            _, grads, _ = model.loss_and_grads(spikes[idx], labels[idx], weights)
            optimizer.step(grads)
    logits, _, hidden_spikes = model.forward(spikes, store=True)
    np.savez_compressed(
        OUT / "snn_final_model.npz",
        **model.params,
        scaler_q05=scaler[0],
        scaler_scale=scaler[1],
        duration_scale=np.asarray(scaler[2]),
        class_names=np.asarray(CLASS_NAMES, dtype=object),
        time_steps=np.asarray(TIME_STEPS),
        beta=np.asarray(BETA),
        threshold=np.asarray(THRESHOLD),
        feature_description=np.asarray("3 radars x 6 range zones; log motion + duration + ON/OFF delta; deterministic integrate-and-fire rate coding", dtype=object),
    )
    return float(hidden_spikes.mean()), float((logits.argmax(axis=1) == labels).mean()), spikes.shape[2]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    dataset = extract_dataset()
    features = dataset["features"]
    durations = dataset["durations"]
    labels = dataset["labels"]
    subjects = dataset["subjects"].astype(str)
    unique_subjects = sorted(np.unique(subjects).tolist())
    all_predictions = np.full(len(labels), -1, dtype=np.int64)
    baseline_predictions = np.full(len(labels), -1, dtype=np.int64)
    motion_baseline_predictions = np.full(len(labels), -1, dtype=np.int64)
    duration_baseline_predictions = np.full(len(labels), -1, dtype=np.int64)
    motion_only_snn_predictions = np.full(len(labels), -1, dtype=np.int64)
    fold_rows, histories = [], []

    for fold, test_subject in enumerate(unique_subjects):
        val_subject = unique_subjects[(fold + 1) % len(unique_subjects)]
        train_pre_idx = np.flatnonzero((subjects != test_subject) & (subjects != val_subject))
        scaler = fit_scaler(features, durations, train_pre_idx)
        spikes = encode(features, durations, scaler)
        model, history, test_result, train_idx, val_idx, test_idx = train_one_fold(
            spikes, labels, subjects, test_subject, val_subject, SEED + fold
        )
        pred = np.asarray(test_result.pop("predictions"), dtype=np.int64)
        all_predictions[test_idx] = pred
        baseline_pred = nearest_centroid_baseline(features, durations, labels, train_idx, test_idx)
        baseline_predictions[test_idx] = baseline_pred
        motion_baseline_predictions[test_idx] = nearest_centroid_baseline(features, durations, labels, train_idx, test_idx, mode="motion")
        duration_baseline_predictions[test_idx] = nearest_centroid_baseline(features, durations, labels, train_idx, test_idx, mode="duration")
        histories.append(history)
        fold_rows.append({
            "fold": fold + 1,
            "test_subject": test_subject,
            "validation_subject": val_subject,
            "epochs": len(history),
            "accuracy": test_result["accuracy"],
            "balanced_accuracy": test_result["balanced_accuracy"],
            "macro_f1": test_result["macro_f1"],
            "baseline_accuracy": float((baseline_pred == labels[test_idx]).mean()),
            "last_train_accuracy": history[-1]["train_accuracy"],
            "last_val_accuracy": history[-1]["val_accuracy"],
        })
        print(f"[fold {fold + 1:02d}] test={test_subject} acc={test_result['accuracy']:.3f} epochs={len(history)}", flush=True)

    for fold, test_subject in enumerate(unique_subjects):
        val_subject = unique_subjects[(fold + 1) % len(unique_subjects)]
        train_pre_idx = np.flatnonzero((subjects != test_subject) & (subjects != val_subject))
        scaler = fit_scaler(features, durations, train_pre_idx)
        spikes = encode(features, durations, scaler, include_duration=False)
        _, _, test_result, _, _, test_idx = train_one_fold(
            spikes, labels, subjects, test_subject, val_subject, SEED + 100 + fold
        )
        motion_only_snn_predictions[test_idx] = np.asarray(test_result["predictions"], dtype=np.int64)
        print(f"[ablation {fold + 1:02d}] test={test_subject} motion-only acc={test_result['accuracy']:.3f}", flush=True)

    aggregate = metrics(labels, all_predictions)
    baseline = metrics(labels, baseline_predictions)
    motion_only_snn = metrics(labels, motion_only_snn_predictions)
    motion_baseline = metrics(labels, motion_baseline_predictions)
    duration_baseline = metrics(labels, duration_baseline_predictions)
    final_spike_rate, final_train_accuracy, input_dim = train_final_model(features, durations, labels)
    params = input_dim * HIDDEN + HIDDEN + HIDDEN * N_CLASSES + N_CLASSES

    prediction_rows = []
    for i in range(len(labels)):
        prediction_rows.append({
            "subject": subjects[i],
            "pair_index": int(dataset["pairs"][i]),
            "true_class": int(labels[i]) + 1,
            "true_scenario": CLASS_NAMES[int(labels[i])],
            "predicted_class": int(all_predictions[i]) + 1,
            "predicted_scenario": CLASS_NAMES[int(all_predictions[i])],
            "correct": int(labels[i] == all_predictions[i]),
            "start_s": float(dataset["starts"][i]),
            "end_s": float(dataset["ends"][i]),
            "duration_s": float(durations[i]),
        })
    with (OUT / "loso_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    with (OUT / "fold_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows[0]))
        writer.writeheader()
        writer.writerows(fold_rows)

    draw_confusion(aggregate["confusion_matrix"], OUT / "confusion_matrix.png")
    draw_history(histories, OUT / "training_history.png")
    result = {
        "run_date": "2026-08-30",
        "task": "7-class scenario classification from three UWB radars",
        "selection_policy": "overall_status == 자동 점검 통과 AND selected_count == 22",
        "subjects": unique_subjects,
        "n_subjects": len(unique_subjects),
        "n_segments": int(len(labels)),
        "class_names": CLASS_NAMES,
        "class_counts": np.bincount(labels, minlength=N_CLASSES).tolist(),
        "input": {
            "radars": 3,
            "range_zones_per_radar": N_ZONES,
            "pooled_time_steps": TIME_STEPS,
            "encoded_channels": input_dim,
            "encoding": "결정적 integrate-and-fire rate encoding: log motion, duration, ON/OFF temporal delta",
        },
        "model": {
            "type": "single-hidden-layer LIF SNN with surrogate-gradient BPTT",
            "hidden_neurons": HIDDEN,
            "output_classes": N_CLASSES,
            "beta": BETA,
            "threshold": THRESHOLD,
            "surrogate_gamma": SURROGATE_GAMMA,
            "parameter_count": int(params),
        },
        "training": {
            "validation": "10-fold leave-one-subject-out; one additional subject used for early stopping in each fold",
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "class_weighting": "inverse-frequency",
            "seed": SEED,
        },
        "cross_validation": aggregate,
        "nearest_centroid_baseline": baseline,
        "ablations": {
            "motion_only_snn": motion_only_snn,
            "motion_only_nearest_centroid": motion_baseline,
            "duration_only_nearest_centroid": duration_baseline,
        },
        "folds": fold_rows,
        "final_model": {
            "trained_on_all_accepted_segments": True,
            "epochs": 60,
            "training_accuracy": final_train_accuracy,
            "hidden_spike_rate": final_spike_rate,
            "model_file": "snn_final_model.npz",
        },
        "runtime_seconds": time.time() - start_time,
        "limitations": [
            "accepted subjects are only 10, so the result is a pilot rather than a deployment-grade estimate",
            "multiple sub-scenarios from one participant share the same upper-level label",
            "duration is included as an input and may contribute strongly because protocol durations differ",
            "no external video ground truth was available for an independent label audit",
        ],
    }
    (OUT / "training_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"accuracy": aggregate["accuracy"], "balanced_accuracy": aggregate["balanced_accuracy"], "macro_f1": aggregate["macro_f1"], "subjects": len(unique_subjects), "segments": len(labels), "runtime_s": result["runtime_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
