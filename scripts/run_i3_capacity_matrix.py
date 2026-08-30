#!/usr/bin/env python3
"""Resume the serialized, no-test i3 default/large discovery matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "artifacts/runs/harmonic_candidate_set_snn_v2"
CACHE_ROOT = PROJECT_ROOT / "artifacts/cache"
CAMPAIGN_ROOT = PROJECT_ROOT / "artifacts/campaigns/harmonic_candidate_set_snn_v2"
SEEDS = (20260828, 20260829, 20260830)
FOLDS = (3, 4)
PRESETS = ("default", "large")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_prelaunch_integrity() -> None:
    index_path = RUN_ROOT / "nested_proposer/discovery_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("outer_test_opened") is not False or int(index.get("completed_units", -1)) != 30:
        raise RuntimeError("strict nested discovery index is incomplete or test-opened")
    freeze_root = CAMPAIGN_ROOT / "source_snapshots/i3_final"
    freeze = json.loads((freeze_root / "MANIFEST.json").read_text(encoding="utf-8"))
    current = {
        "train_harmonic_set_snn.py": PROJECT_ROOT / "scripts/train_harmonic_set_snn.py",
        "harmonic_set_models.py": PROJECT_ROOT / "src/snn_rr/harmonic_set_models.py",
        "ADAPTIVE_CAMPAIGN_CONTRACT.json": CAMPAIGN_ROOT / "ADAPTIVE_CAMPAIGN_CONTRACT.json",
        "harmonic_set_v2.yaml": PROJECT_ROOT / "configs/harmonic_set_v2.yaml",
        "run_gpu_admitted.py": PROJECT_ROOT / "scripts/run_gpu_admitted.py",
    }
    for name, path in current.items():
        if sha256_file(path) != str(freeze["files"][name]):
            raise RuntimeError(f"i3 frozen source/config changed: {name}")


def cache_path(fold: int, seed: int) -> Path:
    return CACHE_ROOT / (
        f"harmonic_set_v2_i2r_nested_o{fold}_s{seed}_"
        "nms125_base_emap_svd12_m050"
    )


def fallback_path(fold: int, seed: int) -> Path:
    return RUN_ROOT / f"nested_fallbacks/outer_{fold}_seed_{seed}.csv"


def output_path(preset: str, fold: int, seed: int) -> Path:
    return RUN_ROOT / f"hcs_discovery/i3_{preset}/outer_{fold}_seed_{seed}"


def validate_complete(path: Path, fold: int, seed: int, preset: str) -> dict[str, object]:
    required = (
        "best_checkpoint.pt", "history.json", "run_manifest.json", "scaler.json",
        "fallback_policy.json", "selection_lock.json", "validation_metrics.json",
        "validation_predictions.npz",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"incomplete i3 output {path}: {missing}")
    if (path / "test_predictions.npz").exists() or (path / "test_metrics.json").exists():
        raise RuntimeError(f"discovery output unexpectedly contains outer-test artifacts: {path}")
    lock = json.loads((path / "selection_lock.json").read_text(encoding="utf-8"))
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    if (
        int(lock.get("outer_fold", -1)) != fold
        or int(lock.get("seed", -1)) != seed
        or int(lock.get("adaptive_iteration", -1)) != 3
        or lock.get("outer_test_not_opened_before_this_lock") is not True
        or manifest.get("model_config", {}).get("hidden_channels")
        != (64 if preset == "default" else 96)
    ):
        raise RuntimeError(f"i3 lock identity/config mismatch: {path}")
    for key, name in (
        ("checkpoint_sha256", "best_checkpoint.pt"),
        ("history_sha256", "history.json"),
        ("run_manifest_sha256", "run_manifest.json"),
        ("scaler_sha256", "scaler.json"),
        ("policy_sha256", "fallback_policy.json"),
    ):
        if str(lock.get(key, "")) != sha256_file(path / name):
            raise RuntimeError(f"i3 lock hash mismatch: {path / name}")
    metrics = json.loads((path / "validation_metrics.json").read_text(encoding="utf-8"))
    return {
        "preset": preset,
        "outer_fold": fold,
        "seed": seed,
        "best_epoch": int(lock["best_epoch"]),
        "policy_selection_status": lock.get("policy_selection_status"),
        "promotion_eligible": bool(lock.get("promotion_eligible", False)),
        "source": metrics["source"],
        "locked_final": metrics["locked_final"],
        "selection_lock_sha256": sha256_file(path / "selection_lock.json"),
    }


def trainer_command(preset: str, fold: int, seed: int, output: Path) -> list[str]:
    return [
        sys.executable, str(PROJECT_ROOT / "scripts/train_harmonic_set_snn.py"),
        "--cache", str(cache_path(fold, seed)),
        "--fallback-oof", str(fallback_path(fold, seed)),
        "--output-dir", str(output), "--fold", str(fold), "--seed", str(seed),
        "--device", "cuda", "--amp", "--deterministic", "--preset", preset,
        "--maximum-parameters", "750000", "--epochs", "120",
        "--minimum-epochs", "20", "--patience", "18",
        "--learning-rate", "0.0003", "--adaptive-iteration", "3",
        "--anchor-residual-mode", "causal_posterior",
        "--anchor-max-residual-bpm", "12",
        "--anchor-minimum-scale-bpm", "0.25",
        "--anchor-maximum-scale-bpm", "12",
        "--anchor-initial-scale-bpm", "1.5",
        "--anchor-distance-weight", "1.0",
        "--anchor-source-mode", "learned_blend",
        "--anchor-residual-weight", "0.75", "--anchor-nll-weight", "0.20",
        "--anchor-gate-weight", "0.08", "--tail-weight", "2.0",
        "--cvar-weight", "0.15", "--warmup-windows", "2",
        "--gradient-accumulation-sessions", "4", "--chunk-windows", "32",
        "--maximum-coverage", "0.20", "--maximum-fpr", "0.01",
        "--minimum-precision", "0.80", "--minimum-correction-recall", "0.20",
        "--discovery-only",
    ]


def run_matrix(presets: Sequence[str], folds: Sequence[int], seeds: Sequence[int]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    wrapper = PROJECT_ROOT / "scripts/run_gpu_admitted.py"
    lock_file = RUN_ROOT / "gpu_admission.lock"
    ledger = RUN_ROOT / "gpu_execution_ledger.jsonl"
    for preset in presets:
        for seed in seeds:
            for fold in folds:
                assert_prelaunch_integrity()
                output = output_path(preset, fold, seed)
                if (output / "selection_lock.json").is_file():
                    results.append(validate_complete(output, fold, seed, preset))
                    continue
                if output.exists() and any(output.iterdir()):
                    raise RuntimeError(f"refusing incomplete non-empty i3 output: {output}")
                command = trainer_command(preset, fold, seed, output)
                admitted = [
                    sys.executable, str(wrapper), "--lock-file", str(lock_file),
                    "--ledger", str(ledger), "--", *command,
                ]
                completed = subprocess.run(admitted, cwd=PROJECT_ROOT, check=False)
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"serialized i3 job failed ({completed.returncode}): "
                        f"preset={preset} fold={fold} seed={seed}"
                    )
                results.append(validate_complete(output, fold, seed, preset))
    return results


def parse_csv(value: str, allowed: Sequence[object], cast) -> tuple:
    parsed = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or len(set(parsed)) != len(parsed) or any(item not in allowed for item in parsed):
        raise argparse.ArgumentTypeError(f"invalid unique subset: {value}")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", default="default,large")
    parser.add_argument("--folds", default="3,4")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = parser.parse_args(argv)
    presets = parse_csv(args.presets, PRESETS, str)
    folds = parse_csv(args.folds, FOLDS, int)
    seeds = parse_csv(args.seeds, SEEDS, int)
    results = run_matrix(presets, folds, seeds)
    print(json.dumps({"status": "complete", "outer_test_opened": False, "results": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
