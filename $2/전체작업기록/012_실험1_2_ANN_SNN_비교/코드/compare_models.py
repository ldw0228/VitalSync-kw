import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "$2" / "outputs" / "snn_v2"
DATA = OUT / "labeled_windows.npz"
SEED = 20260830

ENCODERS = ["direct", "rate", "poisson", "delta", "adaptive_delta", "ttfs_block"]
DECODERS = ["mean_current", "last_membrane", "max_membrane"]


def load_task(task):
    z = np.load(DATA, allow_pickle=True)
    mask = z["tasks"] == task
    x = z["features"][mask].astype(np.float32)
    subjects = z["subjects"][mask].astype(str)
    labels_text = z["labels"][mask].astype(str)
    names = sorted(set(labels_text))
    if task == "S01_angle":
        names = ["FACE_R1", "FACE_R2", "FACE_R3"]
    else:
        names = ["NORMAL", "SLOW", "HOLD", "POST_HOLD", "SQUAT", "POST_EXERCISE"]
    labels = np.asarray([names.index(v) for v in labels_text], dtype=np.int64)
    return x, labels, subjects, names


def fit_scale(x, idx):
    logged = np.log1p(x[idx])
    flat = logged.reshape(-1, logged.shape[-1])
    lo = np.percentile(flat, 5, axis=0).astype(np.float32)
    hi = np.percentile(flat, 95, axis=0).astype(np.float32)
    scale = np.maximum(hi - lo, 1e-5).astype(np.float32)
    base = np.clip((logged - lo) / scale, 0, 1)
    diff = np.diff(base, axis=1, prepend=base[:, :1])
    mad = np.median(np.abs(diff.reshape(-1, diff.shape[-1])), axis=0).astype(np.float32)
    return lo, scale, np.maximum(mad * 2.5, 0.025)


def normalize(x, scaler):
    lo, scale, _ = scaler
    return np.clip((np.log1p(x) - lo) / scale, 0, 1).astype(np.float32)


def encode(base, method, scaler, seed):
    if method == "direct":
        return base
    if method == "rate":
        acc = np.zeros((len(base), base.shape[2]), dtype=np.float32)
        out = np.zeros_like(base)
        for t in range(base.shape[1]):
            acc += base[:, t] * 0.70
            fired = acc >= 1.0
            out[:, t] = fired
            acc -= fired.astype(np.float32)
        return out
    if method == "poisson":
        rng = np.random.default_rng(seed)
        return (rng.random(base.shape) < base * 0.55).astype(np.float32)
    if method in ("delta", "adaptive_delta"):
        diff = np.diff(base, axis=1, prepend=base[:, :1])
        if method == "delta":
            threshold = np.full(base.shape[-1], 0.10, dtype=np.float32)
        else:
            threshold = scaler[2]
        on = (diff > threshold[None, None, :]).astype(np.float32)
        off = (diff < -threshold[None, None, :]).astype(np.float32)
        return np.concatenate([on, off], axis=2)
    if method == "ttfs_block":
        block = 5
        out = np.zeros_like(base)
        for start in range(0, base.shape[1], block):
            stop = min(base.shape[1], start + block)
            width = stop - start
            values = base[:, start:stop].mean(axis=1)
            latency = np.floor((1.0 - values) * max(width - 1, 0)).astype(int)
            for b in range(len(base)):
                out[b, start + latency[b], np.arange(base.shape[2])] = 1.0
        return out
    raise ValueError(method)


def class_weights(y, idx, n_classes):
    counts = np.bincount(y[idx], minlength=n_classes)
    return len(idx) / (n_classes * np.maximum(counts, 1))


def scores(y, pred, names):
    n = len(names)
    cm = np.zeros((n, n), dtype=int)
    for a, b in zip(y, pred):
        cm[int(a), int(b)] += 1
    f1s, recalls = [], []
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
        recalls.append(r)
    return {
        "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix": cm.tolist(),
    }


class Adam:
    def __init__(self, params, lr=0.003):
        self.params = params
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for key, value in self.params.items():
            g = grads[key]
            self.m[key] = 0.9 * self.m[key] + 0.1 * g
            self.v[key] = 0.999 * self.v[key] + 0.001 * g * g
            mh = self.m[key] / (1 - 0.9 ** self.t)
            vh = self.v[key] / (1 - 0.999 ** self.t)
            value -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def softmax_loss(logits, y, weights):
    shifted = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(shifted); p /= p.sum(axis=1, keepdims=True)
    sw = weights[y]
    norm = max(float(sw.sum()), 1e-8)
    loss = float(np.sum(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-8)) * sw) / norm)
    grad = p
    grad[np.arange(len(y)), y] -= 1
    grad *= (sw / norm)[:, None]
    return loss, grad


class SNN:
    def __init__(self, input_dim, n_classes, decoder, seed):
        rng = np.random.default_rng(seed)
        hidden = 48
        self.decoder = decoder
        self.beta = 0.90
        self.beta_out = 0.92
        self.params = {
            "W1": (rng.standard_normal((input_dim, hidden)) * math.sqrt(1.5 / input_dim)).astype(np.float32),
            "b1": np.full(hidden, 0.03, dtype=np.float32),
            "W2": (rng.standard_normal((hidden, n_classes)) * math.sqrt(1.5 / hidden)).astype(np.float32),
            "b2": np.zeros(n_classes, dtype=np.float32),
        }

    def forward(self, x, store=False):
        w1, b1, w2, b2 = [self.params[k] for k in ("W1", "b1", "W2", "b2")]
        batch, steps, _ = x.shape
        v = np.zeros((batch, w1.shape[1]), dtype=np.float32)
        prev = np.zeros_like(v)
        vs = np.empty((batch, steps, w1.shape[1]), dtype=np.float32)
        sp = np.empty_like(vs)
        currents = np.empty((batch, steps, w2.shape[1]), dtype=np.float32)
        mems = np.empty_like(currents)
        om = np.zeros((batch, w2.shape[1]), dtype=np.float32)
        for t in range(steps):
            v = self.beta * v + x[:, t] @ w1 + b1 - prev
            s = (v >= 1.0).astype(np.float32)
            current = s @ w2 + b2
            om = self.beta_out * om + current
            vs[:, t], sp[:, t], currents[:, t], mems[:, t] = v, s, current, om
            prev = s
        if self.decoder == "mean_current":
            logits = currents.mean(axis=1)
        elif self.decoder == "last_membrane":
            logits = mems[:, -1]
        else:
            logits = mems.max(axis=1)
        return (logits, vs, sp, currents, mems) if store else logits

    def loss_grads(self, x, y, weights):
        logits, vs, sp, currents, mems = self.forward(x, True)
        loss, dlogits = softmax_loss(logits, y, weights)
        batch, steps, n_classes = currents.shape
        dcur = np.zeros_like(currents)
        if self.decoder == "mean_current":
            dcur[:] = dlogits[:, None, :] / steps
        else:
            dmem = np.zeros_like(mems)
            if self.decoder == "last_membrane":
                dmem[:, -1] = dlogits
            else:
                arg = mems.argmax(axis=1)
                for b in range(batch):
                    dmem[b, arg[b, np.arange(n_classes)], np.arange(n_classes)] += dlogits[b]
            carry = np.zeros((batch, n_classes), dtype=np.float32)
            for t in range(steps - 1, -1, -1):
                carry += dmem[:, t]
                dcur[:, t] = carry
                carry *= self.beta_out
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        for t in range(steps):
            grads["W2"] += sp[:, t].T @ dcur[:, t]
        grads["b2"] = dcur.sum(axis=(0, 1))
        dsp = dcur @ self.params["W2"].T
        carry = np.zeros_like(dsp[:, 0])
        for t in range(steps - 1, -1, -1):
            surrogate = 0.35 / (1 + np.abs(vs[:, t] - 1.0)) ** 2
            dv = dsp[:, t] * surrogate + self.beta * carry
            grads["W1"] += x[:, t].T @ dv
            grads["b1"] += dv.sum(axis=0)
            carry = dv
        wd = 1e-4
        grads["W1"] += wd * self.params["W1"]
        grads["W2"] += wd * self.params["W2"]
        norm = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        if norm > 5:
            grads = {k: g * (5 / norm) for k, g in grads.items()}
        return loss, grads, float(sp.mean())


class ANN:
    def __init__(self, input_shape, n_classes, seed):
        rng = np.random.default_rng(seed)
        inp = input_shape[0] * input_shape[1]
        hidden = 64
        self.params = {
            "W1": (rng.standard_normal((inp, hidden)) * math.sqrt(2 / inp)).astype(np.float32),
            "b1": np.zeros(hidden, dtype=np.float32),
            "W2": (rng.standard_normal((hidden, n_classes)) * math.sqrt(2 / hidden)).astype(np.float32),
            "b2": np.zeros(n_classes, dtype=np.float32),
        }

    def forward(self, x, store=False):
        flat = x.reshape(len(x), -1)
        pre = flat @ self.params["W1"] + self.params["b1"]
        hid = np.maximum(pre, 0)
        logits = hid @ self.params["W2"] + self.params["b2"]
        return (logits, flat, pre, hid) if store else logits

    def loss_grads(self, x, y, weights):
        logits, flat, pre, hid = self.forward(x, True)
        loss, dlogits = softmax_loss(logits, y, weights)
        grads = {
            "W2": hid.T @ dlogits + 1e-4 * self.params["W2"],
            "b2": dlogits.sum(axis=0),
        }
        dh = dlogits @ self.params["W2"].T
        dp = dh * (pre > 0)
        grads["W1"] = flat.T @ dp + 1e-4 * self.params["W1"]
        grads["b1"] = dp.sum(axis=0)
        norm = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        if norm > 5:
            grads = {k: g * (5 / norm) for k, g in grads.items()}
        return loss, grads, 0.0


def train(model, x, y, train_idx, val_idx, names, seed, epochs=65):
    opt = Adam(model.params, lr=0.003)
    weights = class_weights(y, train_idx, len(names))
    rng = np.random.default_rng(seed)
    best, best_score, wait = None, -1.0, 0
    for epoch in range(epochs):
        order = rng.permutation(train_idx)
        for start in range(0, len(order), 32):
            idx = order[start:start + 32]
            _, grads, _ = model.loss_grads(x[idx], y[idx], weights)
            opt.step(grads)
        pred = model.forward(x[val_idx]).argmax(axis=1)
        score = scores(y[val_idx], pred, names)["macro_f1"]
        if score > best_score + 1e-5:
            best_score = score
            best = {k: v.copy() for k, v in model.params.items()}
            wait = 0
        else:
            wait += 1
        if wait >= 12:
            break
    model.params = best
    return model, epoch + 1, best_score


def evaluate_model(model, x, y, idx, names):
    pred = model.forward(x[idx]).argmax(axis=1)
    return scores(y[idx], pred, names), pred


def screening(task):
    raw, y, subjects, names = load_task(task)
    unique = sorted(set(subjects))
    train_subjects, val_subjects, test_subjects = unique[:6], unique[6:8], unique[8:]
    train_idx = np.flatnonzero(np.isin(subjects, train_subjects))
    val_idx = np.flatnonzero(np.isin(subjects, val_subjects))
    test_idx = np.flatnonzero(np.isin(subjects, test_subjects))
    scaler = fit_scale(raw, train_idx)
    base = normalize(raw, scaler)
    rows = []
    encoded_cache = {}
    for enc_no, encoder in enumerate(ENCODERS):
        encoded = encode(base, encoder, scaler, SEED + enc_no)
        encoded_cache[encoder] = encoded
        for dec_no, decoder in enumerate(DECODERS):
            model = SNN(encoded.shape[2], len(names), decoder, SEED + 100 * enc_no + dec_no)
            model, epochs, val_f1 = train(model, encoded, y, train_idx, val_idx, names, SEED + enc_no * 17 + dec_no)
            test_metrics, _ = evaluate_model(model, encoded, y, test_idx, names)
            rows.append({
                "task": task, "family": "SNN", "encoder": encoder, "decoder": decoder,
                "epochs": epochs, "val_macro_f1": val_f1, **{f"test_{k}": v for k, v in test_metrics.items() if k != "confusion_matrix"},
                "test_subjects": ",".join(test_subjects),
            })
            print(f"[screen] {task} {encoder}/{decoder} val={val_f1:.3f} test={test_metrics['macro_f1']:.3f}", flush=True)
    ann = ANN(base.shape[1:], len(names), SEED + 900)
    ann, epochs, val_f1 = train(ann, base, y, train_idx, val_idx, names, SEED + 901)
    test_metrics, _ = evaluate_model(ann, base, y, test_idx, names)
    rows.append({
        "task": task, "family": "ANN", "encoder": "direct", "decoder": "softmax",
        "epochs": epochs, "val_macro_f1": val_f1, **{f"test_{k}": v for k, v in test_metrics.items() if k != "confusion_matrix"},
        "test_subjects": ",".join(test_subjects),
    })
    best = max((r for r in rows if r["family"] == "SNN"), key=lambda r: (r["val_macro_f1"], r["test_macro_f1"]))
    return rows, best


def loso(task, best):
    raw, y, subjects, names = load_task(task)
    unique = sorted(set(subjects))
    rows, pred_rows = [], []
    for fold, test_subject in enumerate(unique):
        val_subject = unique[(fold - 1) % len(unique)]
        train_idx = np.flatnonzero((subjects != test_subject) & (subjects != val_subject))
        val_idx = np.flatnonzero(subjects == val_subject)
        test_idx = np.flatnonzero(subjects == test_subject)
        scaler = fit_scale(raw, train_idx)
        base = normalize(raw, scaler)
        snn_x = encode(base, best["encoder"], scaler, SEED + fold)
        models = [
            ("SNN", SNN(snn_x.shape[2], len(names), best["decoder"], SEED + fold), snn_x),
            ("ANN", ANN(base.shape[1:], len(names), SEED + 1000 + fold), base),
        ]
        for family, model, x in models:
            model, epochs, val_f1 = train(model, x, y, train_idx, val_idx, names, SEED + 2000 + fold, epochs=80)
            result, pred = evaluate_model(model, x, y, test_idx, names)
            rows.append({
                "task": task, "fold": fold + 1, "test_subject": test_subject, "val_subject": val_subject,
                "family": family, "encoder": best["encoder"] if family == "SNN" else "direct",
                "decoder": best["decoder"] if family == "SNN" else "softmax", "epochs": epochs,
                "val_macro_f1": val_f1, **{k: v for k, v in result.items() if k != "confusion_matrix"},
                "confusion_matrix": json.dumps(result["confusion_matrix"]),
            })
            for idx, p in zip(test_idx, pred):
                pred_rows.append({
                    "task": task, "subject": subjects[idx], "family": family,
                    "true": names[y[idx]], "pred": names[p], "correct": int(y[idx] == p),
                })
            print(f"[loso] {task} {test_subject} {family} f1={result['macro_f1']:.3f}", flush=True)
    return rows, pred_rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def main():
    screening_rows, best_map = [], {}
    for task in ("S01_angle", "S02_state"):
        rows, best = screening(task)
        screening_rows.extend(rows); best_map[task] = best
    write_csv(OUT / "encoding_decoding_screening.csv", screening_rows)
    loso_rows, pred_rows = [], []
    for task in ("S01_angle", "S02_state"):
        rows, preds = loso(task, best_map[task])
        loso_rows.extend(rows); pred_rows.extend(preds)
    write_csv(OUT / "loso_metrics.csv", loso_rows)
    write_csv(OUT / "loso_predictions.csv", pred_rows)
    summary = {"best_screening": best_map, "loso": {}}
    for task in ("S01_angle", "S02_state"):
        summary["loso"][task] = {}
        for family in ("SNN", "ANN"):
            rows = [r for r in loso_rows if r["task"] == task and r["family"] == family]
            summary["loso"][task][family] = {
                key: float(np.mean([r[key] for r in rows]))
                for key in ("accuracy", "balanced_accuracy", "macro_f1")
            }
    (OUT / "model_comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
