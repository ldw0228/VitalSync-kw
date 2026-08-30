#!/usr/bin/env python3
"""Lock one i3 capacity and one policy jointly across all discovery units."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "artifacts/runs/harmonic_candidate_set_snn_v2"
CAMPAIGN_ROOT = PROJECT_ROOT / "artifacts/campaigns/harmonic_candidate_set_snn_v2"
TRAINER_PATH = PROJECT_ROOT / "scripts/train_harmonic_set_snn.py"
PRESETS = ("default", "large")
FOLDS = (3, 4)
SEEDS = (20260828, 20260829, 20260830)
PARAMETER_COUNTS = {"default": 195_603, "large": 410_131}


def load_trainer():
    spec = importlib.util.spec_from_file_location("i3_common_policy_trainer", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen HCS trainer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def finite_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def assert_frozen_sources() -> dict[str, Any]:
    freeze_root = CAMPAIGN_ROOT / "source_snapshots/i3_final"
    freeze = json.loads((freeze_root / "MANIFEST.json").read_text(encoding="utf-8"))
    paths = {
        "train_harmonic_set_snn.py": TRAINER_PATH,
        "harmonic_set_models.py": PROJECT_ROOT / "src/snn_rr/harmonic_set_models.py",
        "ADAPTIVE_CAMPAIGN_CONTRACT.json": CAMPAIGN_ROOT / "ADAPTIVE_CAMPAIGN_CONTRACT.json",
        "harmonic_set_v2.yaml": PROJECT_ROOT / "configs/harmonic_set_v2.yaml",
        "run_gpu_admitted.py": PROJECT_ROOT / "scripts/run_gpu_admitted.py",
    }
    for name, path in paths.items():
        if sha256_file(path) != str(freeze["files"][name]):
            raise RuntimeError(f"frozen i3 source/config changed: {name}")
    discovery = json.loads(
        (RUN_ROOT / "nested_proposer/discovery_index.json").read_text(encoding="utf-8")
    )
    if discovery.get("outer_test_opened") is not False or int(discovery.get("completed_units", -1)) != 30:
        raise RuntimeError("strict proposer discovery is incomplete or test-opened")
    return freeze


def unit_root(preset: str, fold: int, seed: int) -> Path:
    return RUN_ROOT / f"hcs_discovery/i3_{preset}/outer_{fold}_seed_{seed}"


def load_prediction(trainer, path: Path):
    with np.load(path, allow_pickle=False) as archive:
        values = {
            "position": archive["position"].copy(),
            "cache_index": archive["cache_index"].copy(),
            "target": archive["target_rr_bpm"].copy(),
            "identity": archive["identity"].astype(str),
            "base_prediction": archive["fallback_rr_bpm"].copy(),
            "base_std": archive["fallback_std_bpm"].copy(),
            "base_available": archive["fallback_available"].copy(),
            "source_prediction": archive["source_rr_bpm"].copy(),
            "source_scale": archive["source_scale_bpm"].copy(),
            "source_available": archive["source_available"].copy(),
            "selected_index": archive["selected_index"].copy(),
            "selected_probability": archive["selected_probability"].copy(),
            "margin": archive["margin"].copy(),
            "entropy": archive["entropy"].copy(),
            "quality": archive["quality"].copy(),
            "spike_rate": archive["spike_rate"].copy(),
            "final_prediction": archive["final_rr_bpm"].copy(),
            "applied_pull": archive["applied_pull"].copy(),
            "normalized_entropy": archive["normalized_entropy"].copy(),
            "valid_candidate_count": archive["valid_candidate_count"].copy(),
            "raw_anchor_prediction": archive["raw_anchor_rr_bpm"].copy(),
            "raw_anchor_std": archive["raw_anchor_std_bpm"].copy(),
            "corrected_anchor_prediction": archive["corrected_anchor_rr_bpm"].copy(),
            "anchor_residual": archive["anchor_residual_bpm"].copy(),
            "anchor_snap_gate": archive["anchor_snap_gate"].copy(),
            "candidate_source_prediction": archive["candidate_source_rr_bpm"].copy(),
        }
    return trainer.Predictions(**values)


def validate_unit(trainer, preset: str, fold: int, seed: int) -> tuple[Any, dict[str, Any]]:
    root = unit_root(preset, fold, seed)
    files = {
        name: root / name
        for name in (
            "best_checkpoint.pt", "history.json", "run_manifest.json", "scaler.json",
            "fallback_policy.json", "selection_lock.json", "validation_metrics.json",
            "validation_predictions.npz",
        )
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing i3 unit files {preset}/{fold}/{seed}: {missing}")
    if (root / "test_predictions.npz").exists() or (root / "test_metrics.json").exists():
        raise RuntimeError(f"discovery unit contains outer-test output: {root}")
    lock = json.loads(files["selection_lock.json"].read_text(encoding="utf-8"))
    manifest = json.loads(files["run_manifest.json"].read_text(encoding="utf-8"))
    if (
        int(lock.get("outer_fold", -1)) != fold
        or int(lock.get("seed", -1)) != seed
        or int(lock.get("adaptive_iteration", -1)) != 3
        or lock.get("outer_test_not_opened_before_this_lock") is not True
        or int(manifest.get("model_config", {}).get("hidden_channels", -1))
        != (64 if preset == "default" else 96)
    ):
        raise RuntimeError(f"i3 unit identity/config mismatch: {root}")
    for key, name in (
        ("checkpoint_sha256", "best_checkpoint.pt"),
        ("history_sha256", "history.json"),
        ("run_manifest_sha256", "run_manifest.json"),
        ("scaler_sha256", "scaler.json"),
        ("policy_sha256", "fallback_policy.json"),
    ):
        if str(lock.get(key, "")) != sha256_file(files[name]):
            raise RuntimeError(f"i3 unit hash mismatch: {files[name]}")
    prediction = load_prediction(trainer, files["validation_predictions.npz"])
    source_metrics = trainer.evaluation_metrics(
        prediction.target, prediction.source_prediction, prediction.identity
    )
    recorded = json.loads(files["validation_metrics.json"].read_text(encoding="utf-8"))["source"]
    for metric in trainer.COMMERCIAL_SOURCE_GATES:
        if recorded.get(metric) is None or abs(float(recorded[metric]) - float(source_metrics[metric])) > 1e-10:
            raise RuntimeError(f"recorded source metric mismatch {root}: {metric}")
    key = trainer.commercial_gate_selection_key(source_metrics)
    record = {
        "preset": preset,
        "outer_fold": fold,
        "seed": seed,
        "rows": int(len(prediction.target)),
        "source_metrics": finite_json(source_metrics),
        "gate_key": list(key),
        "selection_lock_sha256": sha256_file(files["selection_lock.json"]),
        "validation_predictions_sha256": sha256_file(files["validation_predictions.npz"]),
        "validation_metrics_sha256": sha256_file(files["validation_metrics.json"]),
    }
    return prediction, record


def capacity_key(records: Sequence[Mapping[str, Any]], preset: str) -> tuple:
    gate_keys = [record["gate_key"] for record in records]
    return (
        sum(int(key[0]) for key in gate_keys),
        max(float(key[1]) for key in gate_keys),
        sum(float(key[2]) for key in gate_keys),
        float(np.mean([record["source_metrics"]["identity_macro_mae"] for record in records])),
        float(np.mean([record["source_metrics"]["mae"] for record in records])),
        PARAMETER_COUNTS[preset],
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=RUN_ROOT / "hcs_discovery/i3_common_lock",
    )
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"common lock output is non-empty: {output}")
    freeze = assert_frozen_sources()
    trainer = load_trainer()
    by_preset: dict[str, list[tuple[Any, dict[str, Any]]]] = {}
    for preset in PRESETS:
        by_preset[preset] = [
            validate_unit(trainer, preset, fold, seed)
            for seed in SEEDS for fold in FOLDS
        ]
    rankings = {
        preset: capacity_key([record for _, record in units], preset)
        for preset, units in by_preset.items()
    }
    selected = min(PRESETS, key=lambda preset: rankings[preset])
    selected_predictions = [prediction for prediction, _ in by_preset[selected]]
    policy, combined = trainer.select_fallback_policy_multi(
        selected_predictions,
        maximum_coverage=0.20,
        maximum_fpr=0.01,
        minimum_precision=0.80,
        minimum_correction_recall=0.20,
        gate_aware=True,
    )
    capacity_document = {
        "schema_version": 1,
        "classification": "retrospective_joint_discovery_capacity_selection",
        "outer_test_opened": False,
        "selection_order": [
            "total_failed_gates", "maximum_normalized_violation",
            "summed_normalized_violation", "mean_identity_macro_mae",
            "mean_mae", "parameter_count",
        ],
        "selected_preset": selected,
        "selected_parameter_count": PARAMETER_COUNTS[selected],
        "maximum_parameters": 750_000,
        "ranking": {preset: list(key) for preset, key in rankings.items()},
        "units": {
            preset: [record for _, record in units]
            for preset, units in by_preset.items()
        },
        "source_freeze_manifest_sha256": sha256_file(
            CAMPAIGN_ROOT / "source_snapshots/i3_final/MANIFEST.json"
        ),
    }
    per_unit: list[dict[str, Any]] = []
    for prediction, (_, record) in zip(
        selected_predictions, by_preset[selected], strict=True
    ):
        applied = trainer.apply_fallback_policy(prediction, policy)
        base = np.where(
            prediction.base_available,
            prediction.base_prediction,
            prediction.source_prediction,
        )
        locked_metrics = trainer.evaluation_metrics(
            applied.target, applied.final_prediction, applied.identity
        )
        per_unit.append(
            {
                "outer_fold": record["outer_fold"],
                "seed": record["seed"],
                "source": record["source_metrics"],
                "base": finite_json(trainer.evaluation_metrics(
                    prediction.target, base, prediction.identity
                )),
                "locked_final": finite_json(locked_metrics),
                "locked_gate_key": list(
                    trainer.commercial_gate_selection_key(locked_metrics)
                ),
            }
        )
    combined_metrics = trainer.evaluation_metrics(
        combined.target, combined.final_prediction, combined.identity
    )
    policy_document = finite_json(
        {
            "schema_version": 1,
            "classification": "retrospective_joint_discovery_policy_selection",
            "outer_test_opened": False,
            "selected_preset": selected,
            "policy": asdict(policy),
            "aggregate_locked_metrics": combined_metrics,
            "aggregate_locked_gate_key": list(
                trainer.commercial_gate_selection_key(combined_metrics)
            ),
            "per_unit": per_unit,
            "commercial_claim_authorized": False,
            "prospective_confirmation_required": True,
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    capacity_path = output / "capacity_selection.json"
    policy_path = output / "common_fallback_policy.json"
    atomic_write_json(capacity_path, capacity_document)
    atomic_write_json(policy_path, policy_document)
    lock = {
        "schema_version": 1,
        "classification": "retrospective_i3_common_discovery_lock",
        "outer_test_opened_before_lock": False,
        "selected_preset": selected,
        "selected_parameter_count": PARAMETER_COUNTS[selected],
        "capacity_selection_sha256": sha256_file(capacity_path),
        "common_fallback_policy_sha256": sha256_file(policy_path),
        "promotion_eligible": bool(policy.promotion_eligible),
        "policy_selection_status": policy.selection_status,
        "selection_objective": policy.selection_objective,
        "source_freeze": freeze["files"],
        "unit_locks": [
            {
                "outer_fold": record["outer_fold"],
                "seed": record["seed"],
                "selection_lock_sha256": record["selection_lock_sha256"],
                "validation_predictions_sha256": record["validation_predictions_sha256"],
            }
            for _, record in by_preset[selected]
        ],
        "test_access_policy": (
            "common capacity and policy locked; no outer-test input has been "
            "constructed or iterated by this selector"
        ),
        "commercial_claim_authorized": False,
    }
    lock_path = output / "selection_lock.json"
    atomic_write_json(lock_path, lock)
    for path in (capacity_path, policy_path, lock_path):
        path.chmod(0o444)
    print(json.dumps(finite_json(lock), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
