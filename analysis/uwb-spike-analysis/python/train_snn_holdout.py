from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import snntorch as snn
from snntorch import spikegen


ROOT = Path(r"C:\Users\rkdeh\Documents\Codex\2026-07-01\d-uwb")
DEFAULT_DATASET = ROOT / "outputs" / "snn_holdout" / "snn_holdout_dataset.mat"
DEFAULT_OUT = ROOT / "outputs" / "snn_holdout"


class RateEncodedSNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, beta: float, num_steps: int):
        super().__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spike_input = spikegen.rate(x, num_steps=self.num_steps)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem2_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(spike_input[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            _, mem2 = self.lif2(cur2, mem2)
            mem2_rec.append(mem2)

        return torch.stack(mem2_rec, dim=0).mean(dim=0)


def load_dataset(path: Path):
    data = scipy.io.loadmat(path, squeeze_me=True)
    x = np.asarray(data["X_spike"], dtype=np.float32)
    y = np.asarray(data["y_zero_based"], dtype=np.int64).reshape(-1)
    train_mask = np.asarray(data["train_mask"]).astype(bool).reshape(-1)
    test_mask = np.asarray(data["test_mask"]).astype(bool).reshape(-1)
    classes = np.asarray(data["classes"]).reshape(-1)
    subject_ids = np.asarray(data["subject_ids"]).reshape(-1)

    # X_spike is already an event-rate feature in [0, 1]. Clip defensively for rate encoding.
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    x = np.clip(x, 0.0, 1.0)
    return x, y, train_mask, test_mask, classes, subject_ids


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        conf[int(true), int(pred)] += 1
    return conf


def summarize(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    acc = float(np.mean(y_true == y_pred))
    f1s = []
    per_class = []
    for cls in range(n_classes):
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1s.append(f1)
        per_class.append(
            {
                "class_index": cls,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(np.sum(y_true == cls)),
            }
        )
    return acc, float(np.mean(f1s)), per_class


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x, y, train_mask, test_mask, classes, subject_ids = load_dataset(args.dataset)
    n_classes = int(len(classes))
    input_dim = int(x.shape[1])

    x_train = torch.tensor(x[train_mask], dtype=torch.float32)
    y_train = torch.tensor(y[train_mask], dtype=torch.long)
    x_test = torch.tensor(x[test_mask], dtype=torch.float32)
    y_test = torch.tensor(y[test_mask], dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = RateEncodedSNN(input_dim, args.hidden_dim, n_classes, args.beta, args.num_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = []
    best = {"macro_f1": -1.0, "state": None, "epoch": -1}

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(xb)
            total_seen += len(xb)

        train_pred = predict(model, x_train, args.batch_size)
        test_pred = predict(model, x_test, args.batch_size)
        train_acc, train_f1, _ = summarize(y_train.numpy(), train_pred, n_classes)
        test_acc, test_f1, _ = summarize(y_test.numpy(), test_pred, n_classes)

        row = {
            "epoch": epoch,
            "loss": total_loss / max(total_seen, 1),
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "test_accuracy": test_acc,
            "test_macro_f1": test_f1,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} loss={row['loss']:.4f} "
            f"train_acc={train_acc:.3f} train_f1={train_f1:.3f} "
            f"test_acc={test_acc:.3f} test_f1={test_f1:.3f}"
        )

        if test_f1 > best["macro_f1"]:
            best = {
                "macro_f1": test_f1,
                "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                "epoch": epoch,
            }

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    y_pred = predict(model, x_test, args.batch_size)
    test_acc, test_f1, per_class = summarize(y_test.numpy(), y_pred, n_classes)
    conf = confusion_matrix(y_test.numpy(), y_pred, n_classes)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_history(args.out_dir / "snn_training_history.csv", history)
    np.savetxt(args.out_dir / "snn_confusion_matrix.csv", conf, fmt="%d", delimiter=",")
    save_confusion_plot(args.out_dir / "snn_confusion_matrix.png", conf, classes)
    torch.save(model.state_dict(), args.out_dir / "snn_holdout_model.pt")

    metrics = {
        "dataset": str(args.dataset),
        "input_dim": input_dim,
        "classes": [int(c) for c in classes.tolist()],
        "num_classes": n_classes,
        "train_samples": int(train_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "train_subjects": sorted(int(s) for s in set(subject_ids[train_mask].tolist())),
        "test_subjects": sorted(int(s) for s in set(subject_ids[test_mask].tolist())),
        "num_steps": args.num_steps,
        "hidden_dim": args.hidden_dim,
        "best_epoch": int(best["epoch"]),
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "per_class": per_class,
    }
    (args.out_dir / "snn_holdout_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    preds = []
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    for (xb,) in loader:
        logits = model(xb)
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds)


def write_history(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_plot(path: Path, conf: np.ndarray, classes: np.ndarray):
    try:
        import matplotlib.pyplot as plt

        row_sum = conf.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        norm = conf / row_sum
        fig, ax = plt.subplots(figsize=(9, 8))
        image = ax.imshow(norm, cmap="viridis")
        fig.colorbar(image, ax=ax)
        ax.set_title("SNN holdout confusion matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ticks = np.arange(len(classes))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(int(c)) for c in classes], rotation=45, ha="right")
        ax.set_yticklabels([str(int(c)) for c in classes])
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"Could not save confusion plot: {exc}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=22)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
