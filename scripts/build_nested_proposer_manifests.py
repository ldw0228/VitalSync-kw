#!/usr/bin/env python3
"""Build immutable custom identity splits for a fully nested proposer stack.

For one HCS outer fold, every HCS-training fold is predicted by a proposer
that has seen neither that prediction fold nor the HCS validation/test folds.
The HCS-validation proposer excludes the HCS test fold, and the final HCS-test
proposer is trained only after the validation policy has been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSIGNMENTS = (
    PROJECT_ROOT
    / "artifacts/runs/final_alias_gate_s12_deterministic/fold_assignments.json"
)
DEFAULT_CACHE_MANIFEST = PROJECT_ROOT / "artifacts/cache/rf32s/manifest.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts/campaigns/harmonic_candidate_set_snn_v2/nested_proposer/manifests"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_content_sha256(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "content_sha256"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _identities_for_folds(
    identity_to_fold: Mapping[str, int], folds: Sequence[int]
) -> list[str]:
    selected = {int(value) for value in folds}
    return sorted(
        identity for identity, fold in identity_to_fold.items() if int(fold) in selected
    )


def _manifest(
    *,
    fold_id: int,
    assignments_path: Path,
    cache_manifest: Path,
    train: Sequence[str],
    validation: Sequence[str],
    prediction: Sequence[str],
    excluded: Sequence[str],
) -> dict[str, Any]:
    identities = {
        "train": sorted(set(train)),
        "validation": sorted(set(validation)),
        "prediction": sorted(set(prediction)),
        "excluded": sorted(set(excluded)),
        "scaler": sorted(set(train)),
    }
    disjoint_keys = ("train", "validation", "prediction", "excluded")
    groups = [set(identities[key]) for key in disjoint_keys]
    for left in range(len(groups)):
        for right in range(left + 1, len(groups)):
            if groups[left] & groups[right]:
                raise RuntimeError(
                    f"identity leakage in custom fold {fold_id}: "
                    f"{disjoint_keys[left]} vs {disjoint_keys[right]}"
                )
    if not all(identities[key] for key in ("train", "validation", "prediction")):
        raise RuntimeError(f"empty fitted/prediction identity set in custom fold {fold_id}")
    result: dict[str, Any] = {
        "schema_version": 1,
        "fold_id": fold_id,
        "fold_assignments": {
            "path": str(assignments_path.resolve()),
            "sha256": sha256_file(assignments_path),
        },
        "cache": {
            "manifest_path": str(cache_manifest.resolve()),
            "manifest_sha256": sha256_file(cache_manifest),
        },
        "identities": identities,
    }
    result["content_sha256"] = canonical_content_sha256(result)
    return result


def build_plan(
    *,
    assignments_path: Path,
    cache_manifest: Path,
    outer_folds: Sequence[int],
    include_outer_test: bool = True,
) -> tuple[list[tuple[Path, dict[str, Any]]], dict[str, Any]]:
    assignment_payload = json.loads(assignments_path.read_text(encoding="utf-8"))
    raw_mapping = assignment_payload.get("identity_to_fold")
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise RuntimeError("fold assignment file has no identity_to_fold mapping")
    identity_to_fold = {str(key): int(value) for key, value in raw_mapping.items()}
    fold_numbers = sorted(set(identity_to_fold.values()))
    if fold_numbers != list(range(len(fold_numbers))) or len(fold_numbers) < 4:
        raise RuntimeError("fold assignments must be contiguous and contain >=4 folds")
    all_identities = set(identity_to_fold)
    records: list[tuple[Path, dict[str, Any]]] = []
    summary: dict[str, Any] = {
        "schema_version": 1,
        "classification": "retrospective_fully_nested_proposer_plan",
        "outer_folds": {},
        "fold_assignments_sha256": sha256_file(assignments_path),
        "cache_manifest_sha256": sha256_file(cache_manifest),
    }
    n_folds = len(fold_numbers)
    for outer in outer_folds:
        outer = int(outer)
        if outer not in fold_numbers:
            raise ValueError(f"outer fold is outside assignment domain: {outer}")
        outer_validation = (outer + 1) % n_folds
        training_pool = [
            fold for fold in fold_numbers if fold not in {outer, outer_validation}
        ]
        outer_dir = Path(f"outer_{outer}")
        unit_summaries: list[dict[str, Any]] = []

        for index, prediction_fold in enumerate(training_pool):
            proposer_validation = training_pool[(index + 1) % len(training_pool)]
            proposer_train = [
                fold
                for fold in training_pool
                if fold not in {prediction_fold, proposer_validation}
            ]
            fold_id = 100 * outer + prediction_fold
            manifest = _manifest(
                fold_id=fold_id,
                assignments_path=assignments_path,
                cache_manifest=cache_manifest,
                train=_identities_for_folds(identity_to_fold, proposer_train),
                validation=_identities_for_folds(
                    identity_to_fold, [proposer_validation]
                ),
                prediction=_identities_for_folds(identity_to_fold, [prediction_fold]),
                excluded=_identities_for_folds(
                    identity_to_fold, [outer, outer_validation]
                ),
            )
            relative = outer_dir / f"inner_pred_{prediction_fold}.json"
            records.append((relative, manifest))
            unit_summaries.append(
                {
                    "role": "hcs_train_oof",
                    "prediction_fold": prediction_fold,
                    "proposer_validation_fold": proposer_validation,
                    "proposer_train_folds": proposer_train,
                    "manifest": str(relative),
                    "manifest_content_sha256": manifest["content_sha256"],
                }
            )

        proposer_validation = training_pool[0]
        proposer_train = training_pool[1:]
        fold_id = 100 * outer + 50
        validation_manifest = _manifest(
            fold_id=fold_id,
            assignments_path=assignments_path,
            cache_manifest=cache_manifest,
            train=_identities_for_folds(identity_to_fold, proposer_train),
            validation=_identities_for_folds(
                identity_to_fold, [proposer_validation]
            ),
            prediction=_identities_for_folds(identity_to_fold, [outer_validation]),
            excluded=_identities_for_folds(identity_to_fold, [outer]),
        )
        relative = outer_dir / f"validation_pred_{outer_validation}.json"
        records.append((relative, validation_manifest))
        unit_summaries.append(
            {
                "role": "hcs_validation",
                "prediction_fold": outer_validation,
                "proposer_validation_fold": proposer_validation,
                "proposer_train_folds": proposer_train,
                "manifest": str(relative),
                "manifest_content_sha256": validation_manifest["content_sha256"],
            }
        )

        if include_outer_test:
            fold_id = 100 * outer + 60
            test_manifest = _manifest(
                fold_id=fold_id,
                assignments_path=assignments_path,
                cache_manifest=cache_manifest,
                train=_identities_for_folds(identity_to_fold, training_pool),
                validation=_identities_for_folds(identity_to_fold, [outer_validation]),
                prediction=_identities_for_folds(identity_to_fold, [outer]),
                excluded=[],
            )
            relative = outer_dir / f"test_pred_{outer}.json"
            records.append((relative, test_manifest))
            unit_summaries.append(
                {
                    "role": "hcs_test_open_only_after_policy_lock",
                    "prediction_fold": outer,
                    "proposer_validation_fold": outer_validation,
                    "proposer_train_folds": training_pool,
                    "manifest": str(relative),
                    "manifest_content_sha256": test_manifest["content_sha256"],
                }
            )

        covered = set()
        for _, manifest in records:
            if int(manifest["fold_id"]) // 100 == outer:
                for key in ("train", "validation", "prediction", "excluded"):
                    covered.update(manifest["identities"][key])
        if covered != all_identities:
            raise RuntimeError(f"outer {outer} manifests do not cover canonical identities")
        summary["outer_folds"][str(outer)] = {
            "outer_test_fold": outer,
            "outer_validation_fold": outer_validation,
            "hcs_training_folds": training_pool,
            "units": unit_summaries,
        }
    if not include_outer_test:
        summary["classification"] = (
            "retrospective_fully_nested_non_test_proposer_plan"
        )
        summary["outer_test_opened"] = False
        summary["outer_test_unit_count"] = 0
    summary["content_sha256"] = canonical_content_sha256(summary)
    return records, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create immutable fully nested proposer identity manifests"
    )
    parser.add_argument("--fold-assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--cache-manifest", type=Path, default=DEFAULT_CACHE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--outer-folds", default="3,4")
    parser.add_argument(
        "--exclude-outer-test",
        action="store_true",
        help="materialize only the four inner-OOF and one validation unit per outer fold",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    assignments = args.fold_assignments.expanduser().resolve()
    cache_manifest = args.cache_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    outer_folds = [int(value) for value in args.outer_folds.split(",") if value.strip()]
    records, summary = build_plan(
        assignments_path=assignments,
        cache_manifest=cache_manifest,
        outer_folds=outer_folds,
        include_outer_test=not args.exclude_outer_test,
    )
    for relative, manifest in records:
        atomic_json(output_dir / relative, manifest)
    atomic_json(output_dir / "plan.json", summary)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "manifests": len(records),
                "outer_folds": outer_folds,
                "plan_content_sha256": summary["content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
