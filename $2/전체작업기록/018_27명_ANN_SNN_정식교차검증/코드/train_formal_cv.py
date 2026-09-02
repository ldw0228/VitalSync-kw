from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from train_ann_snn import ANN, SNN, encode_input, fit_transform, metrics, save_model, sigmoid, train_one


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DATA = HERE / "data" / "all27_training_windows.npz"
RUN = OUT / "formal_cv"
CHECKPOINTS = RUN / "checkpoints"
FINAL_MODELS = RUN / "final_models"

FOLD_SEED = 20260901
MODEL_SEEDS = [42, 314, 2718]
N_FOLDS = 5
MAX_EPOCHS = 120
MIN_EPOCHS = 30
PATIENCE = 15
HIDDEN = 64
CANDIDATES = ["ANN_direct", "SNN_direct", "SNN_signed_rate", "SNN_delta_event"]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_for(candidate: str, x: np.ndarray, seed: int):
    if candidate == "ANN_direct":
        return ANN(x.shape[1], x.shape[2], hidden=HIDDEN, seed=seed), 1e-3, "direct"
    method = candidate.removeprefix("SNN_")
    return SNN(x.shape[2], hidden=HIDDEN, seed=seed), 5e-4, method


def subject_macro(pred_rows: list[dict]) -> dict:
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for row in pred_rows:
        by_subject[row["subject"]].append(row)
    rr_mae, within2, bal_acc = [], [], []
    for rows in by_subject.values():
        rr_errors = [abs(float(r["rr_pred_bpm"]) - float(r["rr_true_bpm"])) for r in rows if r["rr_true_bpm"] != ""]
        if rr_errors:
            rr_mae.append(float(np.mean(rr_errors)))
            within2.append(float(np.mean(np.asarray(rr_errors) <= 2.0)))
        motion_rows = [r for r in rows if r["motion_true"] != ""]
        if motion_rows:
            target = np.asarray([int(r["motion_true"]) for r in motion_rows])
            prediction = np.asarray([int(r["motion_pred"]) for r in motion_rows])
            sens = np.mean(prediction[target == 1] == 1) if np.any(target == 1) else math.nan
            spec = np.mean(prediction[target == 0] == 0) if np.any(target == 0) else math.nan
            if math.isfinite(sens) and math.isfinite(spec):
                bal_acc.append(float((sens + spec) / 2.0))
    return {
        "subject_count": len(by_subject),
        "rr_subject_macro_mae_bpm": float(np.mean(rr_mae)),
        "rr_subject_macro_mae_sd": float(np.std(rr_mae, ddof=1)),
        "rr_subject_macro_within2": float(np.mean(within2)),
        "motion_subject_macro_balanced_accuracy": float(np.mean(bal_acc)) if bal_acc else math.nan,
    }


def make_predictions(model, x, rr_n, motion, mask, scaler, subjects, states, starts, candidate, fold, seed):
    idx = np.flatnonzero(mask)
    pred = model.forward(x[idx])
    rr_mean = float(scaler["rr_mean"][0])
    rr_std = float(scaler["rr_std"][0])
    rows = []
    for local, original in enumerate(idx):
        rr_true = ""
        rr_pred = ""
        if math.isfinite(float(rr_n[original])):
            rr_true = float(rr_n[original] * rr_std + rr_mean)
            rr_pred = float(pred[local, 0] * rr_std + rr_mean)
        motion_true = "" if motion[original] < 0 else int(motion[original])
        rows.append({
            "candidate": candidate,
            "fold": fold,
            "seed": seed,
            "index": int(original),
            "subject": str(subjects[original]),
            "state": str(states[original]),
            "start_s": float(starts[original]),
            "rr_true_bpm": rr_true,
            "rr_pred_bpm": rr_pred,
            "motion_true": motion_true,
            "motion_probability": float(sigmoid(pred[local, 1])),
            "motion_pred": "" if motion_true == "" else int(sigmoid(pred[local, 1]) >= 0.5),
        })
    return rows


def train_fixed(candidate, model, lr, x, rr, motion, epochs, seed):
    # Epoch count was chosen only from cross-validation. No full-data early stopping is used.
    from train_ann_snn import Adam

    optimizer = Adam(model.params, lr=lr)
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(np.isfinite(rr) | (motion >= 0))
    spike_rates = []
    for _epoch in range(epochs):
        order = rng.permutation(idx)
        epoch_spikes = []
        for start in range(0, len(order), 16):
            batch = order[start:start + 16]
            _loss, _rr_loss, _motion_loss, grads, spike_rate = model.loss_grads(x[batch], rr[batch], motion[batch])
            optimizer.step(grads)
            epoch_spikes.append(spike_rate)
        spike_rates.append(float(np.mean(epoch_spikes)))
    return spike_rates


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    FINAL_MODELS.mkdir(parents=True, exist_ok=True)
    data = np.load(DATA, allow_pickle=True)
    wave = data["waveforms"].astype(np.float32)
    scalar = data["scalars"].astype(np.float32)
    rr = data["rr_targets"].astype(np.float32)
    motion = data["motion_targets"].astype(np.int64)
    subjects = data["subjects"].astype(str)
    states = data["states"].astype(str)
    starts = data["starts"].astype(np.float32)
    unique_subjects = np.asarray(sorted(set(subjects.tolist())))

    rng = np.random.default_rng(FOLD_SEED)
    shuffled = rng.permutation(unique_subjects)
    test_folds = [list(x) for x in np.array_split(shuffled, N_FOLDS)]
    split_rows = []
    split_specs = []
    for fold, test_subjects in enumerate(test_folds, start=1):
        remaining = [s for s in unique_subjects if s not in test_subjects]
        val_rng = np.random.default_rng(FOLD_SEED + fold)
        val_subjects = list(val_rng.choice(remaining, size=3, replace=False))
        train_subjects = [s for s in remaining if s not in val_subjects]
        split_specs.append((fold, train_subjects, val_subjects, test_subjects))
        for role, values in (("train", train_subjects), ("validation", val_subjects), ("test", test_subjects)):
            split_rows.extend({"fold": fold, "subject": s, "role": role} for s in values)
    write_csv(RUN / "split_manifest.csv", split_rows)

    fold_metrics, predictions, histories = [], [], []
    started_all = time.perf_counter()
    for candidate in CANDIDATES:
        for fold, train_subjects, val_subjects, test_subjects in split_specs:
            train_mask = np.isin(subjects, train_subjects)
            val_mask = np.isin(subjects, val_subjects)
            test_mask = np.isin(subjects, test_subjects)
            x_direct, rr_n, scaler = fit_transform(wave, scalar, rr, train_mask)
            method = "direct" if candidate == "ANN_direct" else candidate.removeprefix("SNN_")
            x = encode_input(x_direct, method)
            for seed in MODEL_SEEDS:
                model, lr, _ = model_for(candidate, x, seed)
                run_started = time.perf_counter()
                history, best_epoch, best_val = train_one(
                    f"{candidate}_f{fold}_s{seed}", model, lr, x, rr_n, motion,
                    train_mask, val_mask, MAX_EPOCHS, seed,
                    patience=PATIENCE, min_epochs=MIN_EPOCHS, verbose=False,
                )
                elapsed = time.perf_counter() - run_started
                for row in history:
                    histories.append({"candidate": candidate, "fold": fold, "seed": seed, **row})
                test_metric = metrics(model, x, rr_n, motion, test_mask, scaler)
                run_predictions = make_predictions(
                    model, x, rr_n, motion, test_mask, scaler,
                    subjects, states, starts, candidate, fold, seed,
                )
                predictions.extend(run_predictions)
                macro = subject_macro(run_predictions)
                params = int(sum(v.size for v in model.params.values()))
                spike = float(history[best_epoch - 1]["spike_rate"]) if candidate.startswith("SNN") else 0.0
                fold_metrics.append({
                    "candidate": candidate,
                    "fold": fold,
                    "seed": seed,
                    "train_subject_n": len(train_subjects),
                    "validation_subject_n": len(val_subjects),
                    "test_subject_n": len(test_subjects),
                    "epochs_run": len(history),
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_val,
                    "test_rr_window_mae_bpm": test_metric.get("rr_mae_bpm", math.nan),
                    "test_rr_window_within2": test_metric.get("rr_within2", math.nan),
                    "test_motion_balanced_accuracy": test_metric.get("motion_balanced_accuracy", math.nan),
                    **macro,
                    "hidden_spike_rate": spike,
                    "parameters": params,
                    "training_seconds": elapsed,
                })
                save_model(CHECKPOINTS / f"{candidate}_fold{fold}_seed{seed}.npz", model, scaler, {
                    "status": "formal_subject_cv_checkpoint",
                    "candidate": candidate,
                    "fold": fold,
                    "seed": seed,
                    "train_subjects": train_subjects,
                    "validation_subjects": val_subjects,
                    "test_subjects": test_subjects,
                    "best_epoch": best_epoch,
                    "input_shape": list(x.shape[1:]),
                })
                write_csv(RUN / "fold_metrics.csv", fold_metrics)
                write_csv(RUN / "oof_predictions.csv", predictions)
                print(
                    f"[formal] {candidate} fold={fold}/5 seed={seed} "
                    f"epoch={best_epoch}/{len(history)} macro_MAE={macro['rr_subject_macro_mae_bpm']:.3f} "
                    f"motion_BA={macro['motion_subject_macro_balanced_accuracy']:.3f} time={elapsed:.1f}s",
                    flush=True,
                )

    write_csv(RUN / "training_history.csv", histories)
    by_candidate_seed: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in predictions:
        by_candidate_seed[(row["candidate"], int(row["seed"]))].append(row)
    seed_summaries = []
    for (candidate, seed), rows in sorted(by_candidate_seed.items()):
        seed_summaries.append({"candidate": candidate, "seed": seed, **subject_macro(rows)})
    write_csv(RUN / "seed_summary.csv", seed_summaries)

    comparison = {}
    for candidate in CANDIDATES:
        rows = [r for r in seed_summaries if r["candidate"] == candidate]
        maes = np.asarray([r["rr_subject_macro_mae_bpm"] for r in rows], dtype=float)
        withins = np.asarray([r["rr_subject_macro_within2"] for r in rows], dtype=float)
        motions = np.asarray([r["motion_subject_macro_balanced_accuracy"] for r in rows], dtype=float)
        fold_rows = [r for r in fold_metrics if r["candidate"] == candidate]
        comparison[candidate] = {
            "rr_subject_macro_mae_bpm_mean_over_seeds": float(np.mean(maes)),
            "rr_subject_macro_mae_bpm_sd_over_seeds": float(np.std(maes, ddof=1)),
            "rr_subject_macro_within2_mean_over_seeds": float(np.mean(withins)),
            "motion_subject_macro_balanced_accuracy_mean_over_seeds": float(np.mean(motions)),
            "median_best_epoch": int(round(float(np.median([r["best_epoch"] for r in fold_rows])))),
            "mean_training_seconds_per_fold_seed": float(np.mean([r["training_seconds"] for r in fold_rows])),
            "parameters": int(fold_rows[0]["parameters"]),
            "mean_hidden_spike_rate": float(np.mean([r["hidden_spike_rate"] for r in fold_rows])),
        }

    selected_snn = min(
        [name for name in CANDIDATES if name.startswith("SNN")],
        key=lambda name: comparison[name]["rr_subject_macro_mae_bpm_mean_over_seeds"],
    )
    selected_overall = min(
        CANDIDATES,
        key=lambda name: comparison[name]["rr_subject_macro_mae_bpm_mean_over_seeds"],
    )

    # Train deployable reference files on all 27 after the comparison is complete.
    all_mask = np.ones(len(subjects), dtype=bool)
    x_direct, rr_n, scaler = fit_transform(wave, scalar, rr, all_mask)
    final_records = {}
    for candidate in CANDIDATES:
        method = "direct" if candidate == "ANN_direct" else candidate.removeprefix("SNN_")
        x = encode_input(x_direct, method)
        model, lr, _ = model_for(candidate, x, MODEL_SEEDS[0])
        epochs = comparison[candidate]["median_best_epoch"]
        final_started = time.perf_counter()
        spike_rates = train_fixed(candidate, model, lr, x, rr_n, motion, epochs, MODEL_SEEDS[0])
        elapsed = time.perf_counter() - final_started
        metadata = {
            "status": "trained_on_all_27_deployable_model",
            "candidate": candidate,
            "encoding": method,
            "subject_count": len(unique_subjects),
            "subjects": unique_subjects.tolist(),
            "epochs": epochs,
            "epoch_selection": "median best epoch from formal 5-fold x 3-seed subject-level CV",
            "seed": MODEL_SEEDS[0],
            "input_shape": list(x.shape[1:]),
            "outputs": ["normalized_respiratory_rate", "motion_logit"],
            "biopac_input": False,
        }
        path = FINAL_MODELS / f"{candidate}_all27.npz"
        save_model(path, model, scaler, metadata)
        final_records[candidate] = {
            "path": str(path),
            "epochs": epochs,
            "training_seconds": elapsed,
            "last_epoch_hidden_spike_rate": spike_rates[-1] if candidate.startswith("SNN") else 0.0,
        }
        print(f"[final] {candidate}: all27, epochs={epochs}, time={elapsed:.1f}s", flush=True)

    result = {
        "status": "formal_comparison_complete_and_all27_models_trained",
        "dataset": {
            "subjects": len(unique_subjects),
            "windows": len(subjects),
            "rr_windows": int(np.isfinite(rr).sum()),
            "motion_positive_windows": int(np.sum(motion == 1)),
        },
        "protocol": {
            "split": "5-fold subject-level outer test; 3 validation subjects inside each training fold",
            "model_seeds": MODEL_SEEDS,
            "max_epochs": MAX_EPOCHS,
            "min_epochs": MIN_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "primary_selection_metric": "mean subject-macro RR MAE across 3 seeds",
            "data_leakage_rule": "one subject never appears in train/validation/test simultaneously within a fold",
        },
        "comparison": comparison,
        "selected_snn": selected_snn,
        "selected_overall": selected_overall,
        "final_all27_models": final_records,
        "elapsed_seconds": time.perf_counter() - started_all,
        "interpretation": (
            "Cross-validation scores estimate performance on unseen people. Final all-27 files use every usable subject; "
            "their expected performance is the out-of-fold cross-validation score, not a re-test on their training data."
        ),
    }
    (RUN / "formal_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
