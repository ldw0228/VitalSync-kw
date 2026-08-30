#!/usr/bin/env python3
"""Resumable trainer/inference driver for nested proposer discovery units."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFESTS = (
    PROJECT_ROOT
    / "artifacts/campaigns/harmonic_candidate_set_snn_v2/nested_proposer/manifests"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts/runs/harmonic_candidate_set_snn_v2/nested_proposer"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def discovery_manifests(root: Path, outer_folds: Sequence[int]) -> list[Path]:
    result: list[Path] = []
    for outer in outer_folds:
        directory = root / f"outer_{int(outer)}"
        inner = sorted(directory.glob("inner_pred_*.json"))
        validation = sorted(directory.glob("validation_pred_*.json"))
        if len(inner) != 4 or len(validation) != 1:
            raise RuntimeError(
                f"outer {outer} must contain four inner and one validation manifest"
            )
        if list(directory.glob("test_pred_*.json")) and any(
            path.name.startswith("test_pred_") for path in (*inner, *validation)
        ):
            raise RuntimeError("outer-test manifest entered discovery manifest set")
        result.extend((*inner, *validation))
    return result


def _manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("fold_id"), int):
        raise RuntimeError(f"invalid custom split manifest: {path}")
    return value


def _verify_checkpoint(
    path: Path, *, manifest: Mapping[str, Any], manifest_path: Path
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "snn":
        raise RuntimeError(f"nested proposer checkpoint is not SNN: {path}")
    if int(checkpoint.get("fold", -1)) != int(manifest["fold_id"]):
        raise RuntimeError(f"nested proposer checkpoint fold mismatch: {path}")
    provenance = checkpoint.get("split_authority_provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError(f"checkpoint has no custom split provenance: {path}")
    if provenance.get("split_manifest_content_sha256") != manifest.get(
        "content_sha256"
    ):
        raise RuntimeError(f"checkpoint custom split content mismatch: {path}")
    if provenance.get("split_manifest_file_sha256") != sha256_file(manifest_path):
        raise RuntimeError(f"checkpoint custom split file mismatch: {path}")
    return checkpoint


def _unit_record(
    *,
    seed: int,
    outer: int,
    manifest_path: Path,
    output_dir: Path,
    checkpoint_path: Path,
    prediction_path: Path,
) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "outer_fold": int(outer),
        "role": (
            "hcs_validation" if manifest_path.name.startswith("validation_") else "hcs_train_oof"
        ),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "output_dir": str(output_dir.resolve()),
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        },
        "all_window_prediction": {
            "path": str(prediction_path.resolve()),
            "sha256": sha256_file(prediction_path),
            "bytes": prediction_path.stat().st_size,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all non-test nested proposer units with resumable verification"
    )
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFESTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "artifacts/cache/rf32s")
    parser.add_argument("--outer-folds", default="3,4")
    parser.add_argument("--seeds", default="20260828,20260829,20260830")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--train-device", default="cuda")
    parser.add_argument("--prediction-device", default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-units", type=int)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_root = args.manifest_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    outer_folds = [int(value) for value in args.outer_folds.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    manifests = discovery_manifests(manifest_root, outer_folds)
    units = [(seed, path) for seed in seeds for path in manifests]
    if args.max_units is not None:
        if args.max_units < 1:
            raise ValueError("--max-units must be positive")
        units = units[: args.max_units]

    index_path = output_root / "discovery_index.json"
    records: list[dict[str, Any]] = []
    for seed, manifest_path in units:
        manifest = _manifest(manifest_path)
        outer = int(manifest_path.parent.name.removeprefix("outer_"))
        stem = manifest_path.stem
        unit_dir = output_root / f"seed_{seed}" / f"outer_{outer}" / stem
        fold_dir = unit_dir / f"fold_{manifest['fold_id']}"
        checkpoint_path = fold_dir / "snn_best.pt"
        metrics_path = unit_dir / "metrics.json"
        prediction_path = fold_dir / "snn_prediction_all_windows.npz"

        if not (metrics_path.is_file() and checkpoint_path.is_file()):
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts/train.py"),
                "--identity-split-manifest",
                str(manifest_path),
                "--output-dir",
                str(unit_dir),
                "--model",
                "both",
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--batch-size",
                "48",
                "--workers",
                str(args.workers),
                "--device",
                str(args.train_device),
                "--amp",
                "--deterministic",
                "--seed",
                str(seed),
                "--causal-history",
                "--harmonic-head",
                "--alias-gate",
                "--aux-fusion",
                "structured",
                "--exact-aux-alignment",
                "--simulation-steps",
                "12",
                "--hidden-dim",
                "192",
                "--radar-dropout",
                "0.20",
                "--distill-weight",
                "0.35",
                "--distill-temperature",
                "2.0",
                "--alias-loss-weight",
                "0.05",
                "--quality-loss-weight",
                "0.15",
                "--spike-rate-weight",
                "0.0005",
                "--bootstrap-samples",
                "500",
            ]
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        _verify_checkpoint(
            checkpoint_path, manifest=manifest, manifest_path=manifest_path
        )

        if not prediction_path.is_file():
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts/predict_custom_split_all_windows.py"),
                "--cache-dir",
                str(cache_dir),
                "--checkpoint",
                str(checkpoint_path),
                "--identity-split-manifest",
                str(manifest_path),
                "--output",
                str(prediction_path),
                "--device",
                str(args.prediction_device),
                "--batch-size",
                "128",
                "--workers",
                "0",
            ]
            if str(args.prediction_device).startswith("cuda"):
                command.append("--amp")
            else:
                command.append("--no-amp")
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        records.append(
            _unit_record(
                seed=seed,
                outer=outer,
                manifest_path=manifest_path,
                output_dir=unit_dir,
                checkpoint_path=checkpoint_path,
                prediction_path=prediction_path,
            )
        )
        partial = {
            "schema_version": 1,
            "classification": "retrospective_fully_nested_discovery",
            "outer_test_opened": False,
            "requested_units": len(units),
            "completed_units": len(records),
            "records": records,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(index_path, partial)
    return partial


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(
        json.dumps(
            {
                "completed_units": result["completed_units"],
                "requested_units": result["requested_units"],
                "outer_test_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
