#!/usr/bin/env python3
"""Determine which locked HCS units must be rebuilt after proposer retraining.

The discovery HCS stack validator binds the SHA-256 of each of its five
all-window proposer prediction archives.  Retraining a proposer checkpoint is
therefore harmless to an already locked HCS unit only when all five prediction
archives remain byte-identical.  This audit validates both completed proposer
campaigns with the same strict routines used by the 60+30 merge, compares the
folds-3/4 prediction bindings, and emits the exact ``--force-retrain-units``
value required by ``run_fixed_i3_pretest_campaign.py``.

No target, outer-test manifest, or score is accepted by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import merge_nested_proposer_indexes as merge  # noqa: E402


DEFAULT_OUTPUT = (
    merge.CAMPAIGN_ROOT
    / "current_source_merged/retrain_impact_audit.json"
)

InspectUnit = Callable[
    [Mapping[str, Any]], tuple[str, dict[str, Any] | None, str | None]
]


def canonical_content_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prediction_hash(record: Mapping[str, Any], *, label: str) -> str:
    binding = record.get("all_window_prediction")
    if not isinstance(binding, Mapping):
        raise RuntimeError(f"{label} lacks all_window_prediction binding")
    value = str(binding.get("sha256", ""))
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"{label} prediction SHA-256 is invalid")
    return value


def _checkpoint_hash(record: Mapping[str, Any], *, label: str) -> str:
    binding = record.get("checkpoint")
    if not isinstance(binding, Mapping):
        raise RuntimeError(f"{label} lacks checkpoint binding")
    value = str(binding.get("sha256", ""))
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"{label} checkpoint SHA-256 is invalid")
    return value


def compare_record_maps(
    main_records: Mapping[tuple[int, int, str], Mapping[str, Any]],
    retrain_records: Mapping[tuple[int, int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare the exact folds-3/4 source stacks without reading labels."""

    expected = {
        (outer, seed, name)
        for outer in sorted(merge.RETRAIN_FOLDS)
        for seed in merge.SEEDS
        for name in merge._manifest_name_for_outer(outer)
    }
    if not expected.issubset(main_records):
        raise RuntimeError("main proposer index lacks a folds-3/4 comparison unit")
    if set(retrain_records) != expected:
        raise RuntimeError("retrain proposer index is not the exact folds-3/4 30-unit cover")

    units: list[dict[str, Any]] = []
    affected_pairs: set[tuple[int, int]] = set()
    changed_prediction_units = 0
    changed_checkpoint_units = 0
    for key in sorted(expected):
        outer, seed, manifest_name = key
        main = main_records[key]
        retrain = retrain_records[key]
        main_prediction = _prediction_hash(main, label=f"main {key}")
        retrain_prediction = _prediction_hash(retrain, label=f"retrain {key}")
        main_checkpoint = _checkpoint_hash(main, label=f"main {key}")
        retrain_checkpoint = _checkpoint_hash(retrain, label=f"retrain {key}")
        prediction_identical = main_prediction == retrain_prediction
        checkpoint_identical = main_checkpoint == retrain_checkpoint
        if not prediction_identical:
            affected_pairs.add((outer, seed))
            changed_prediction_units += 1
        if not checkpoint_identical:
            changed_checkpoint_units += 1
        units.append(
            {
                "outer_fold": outer,
                "seed": seed,
                "manifest": manifest_name,
                "prediction_byte_identical": prediction_identical,
                "checkpoint_byte_identical": checkpoint_identical,
                "main_prediction_sha256": main_prediction,
                "retrain_prediction_sha256": retrain_prediction,
                "main_checkpoint_sha256": main_checkpoint,
                "retrain_checkpoint_sha256": retrain_checkpoint,
            }
        )

    ordered_pairs = sorted(affected_pairs)
    force_value = ",".join(f"{outer}:{seed}" for outer, seed in ordered_pairs)
    return {
        "comparison_units": len(units),
        "changed_prediction_units": changed_prediction_units,
        "changed_checkpoint_units": changed_checkpoint_units,
        "affected_hcs_units": [
            {"outer_fold": outer, "seed": seed} for outer, seed in ordered_pairs
        ],
        "force_retrain_unit_count": len(ordered_pairs),
        "force_retrain_units_cli_value": force_value,
        "force_retrain_argument": (
            ["--force-retrain-units", force_value] if force_value else []
        ),
        "decision_rule": (
            "force an HCS fold/seed unit iff any of its five all-window proposer "
            "prediction file SHA-256 bindings changed"
        ),
        "checkpoint_change_alone_forces_hcs_retrain": False,
        "prediction_paths_ignored_after_bound_file_validation": True,
        "units": units,
    }


def _validated_campaigns(
    *,
    full_plan_path: Path,
    main_index_path: Path,
    retrain_plan_path: Path,
    retrain_index_path: Path,
    inspect_main: InspectUnit | None = None,
    inspect_retrain: InspectUnit | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[tuple[int, int, str], dict[str, Any]],
    dict[tuple[int, int, str], dict[str, Any]],
]:
    full, full_binding, full_units = merge._validate_plan(
        full_plan_path,
        expected_folds=merge.FOLDS,
        label="full six-fold proposer plan",
    )
    retrain, retrain_binding, retrain_units = merge._validate_plan(
        retrain_plan_path,
        expected_folds=sorted(merge.RETRAIN_FOLDS),
        label="current-source folds-3/4 retrain plan",
    )
    merge._assert_compatible_plans(full, retrain, full_units, retrain_units)

    if inspect_main is None:
        main_root = merge._resolve(full["reusable_run_root"])

        def inspect_main(unit: Mapping[str, Any]):
            return merge.campaign.inspect_unit(unit, run_root=main_root)

    if inspect_retrain is None:
        retrain_root = merge._resolve(retrain["reusable_run_root"])

        def inspect_retrain(unit: Mapping[str, Any]):
            return merge.campaign.inspect_unit(unit, run_root=retrain_root)

    main_index, main_binding, main_records = merge._validate_index(
        main_index_path,
        plan_path=full_plan_path,
        plan=full,
        plan_binding=full_binding,
        planned=full_units,
        label="completed main proposer index",
        inspect_unit=inspect_main,
    )
    retrain_index, retrain_index_binding, retrain_records = merge._validate_index(
        retrain_index_path,
        plan_path=retrain_plan_path,
        plan=retrain,
        plan_binding=retrain_binding,
        planned=retrain_units,
        label="completed current-source folds-3/4 proposer index",
        inspect_unit=inspect_retrain,
    )
    inputs = {
        "full_plan": {**full_binding, "content_sha256": full["content_sha256"]},
        "main_index": {
            **main_binding,
            "content_sha256": main_index["content_sha256"],
        },
        "retrain_plan": {
            **retrain_binding,
            "content_sha256": retrain["content_sha256"],
        },
        "retrain_index": {
            **retrain_index_binding,
            "content_sha256": retrain_index["content_sha256"],
        },
    }
    return inputs, {"full": full, "retrain": retrain}, main_records, retrain_records


def audit_retrain_impact(
    *,
    full_plan_path: Path,
    main_index_path: Path,
    retrain_plan_path: Path,
    retrain_index_path: Path,
    output_path: Path,
    inspect_main: InspectUnit | None = None,
    inspect_retrain: InspectUnit | None = None,
) -> dict[str, Any]:
    destination = output_path.expanduser().resolve()
    if destination.exists():
        raise RuntimeError(f"immutable retrain impact audit already exists: {destination}")
    inputs, plans, main_records, retrain_records = _validated_campaigns(
        full_plan_path=full_plan_path.expanduser().resolve(),
        main_index_path=main_index_path.expanduser().resolve(),
        retrain_plan_path=retrain_plan_path.expanduser().resolve(),
        retrain_index_path=retrain_index_path.expanduser().resolve(),
        inspect_main=inspect_main,
        inspect_retrain=inspect_retrain,
    )
    comparison = compare_record_maps(main_records, retrain_records)
    result: dict[str, Any] = {
        "schema_version": 1,
        "classification": "retrospective_nested_proposer_retrain_impact_audit",
        "commercial_claim_authorized": False,
        "outer_test_opened": False,
        "outer_test_record_count": 0,
        "target_or_reference_accessed": False,
        "source_campaigns_hash_complete": True,
        "source_plans_compatible": True,
        "inputs": inputs,
        "plan_content_sha256": {
            "full": plans["full"]["content_sha256"],
            "retrain": plans["retrain"]["content_sha256"],
        },
        "comparison": comparison,
    }
    result["content_sha256"] = canonical_content_sha256(result)
    payload = (
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise RuntimeError(
                f"immutable retrain impact audit already exists: {destination}"
            ) from exc
        destination.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-plan", type=Path, default=merge.DEFAULT_FULL_PLAN)
    parser.add_argument("--main-index", type=Path, default=merge.DEFAULT_MAIN_INDEX)
    parser.add_argument("--retrain-plan", type=Path, default=merge.DEFAULT_RETRAIN_PLAN)
    parser.add_argument("--retrain-index", type=Path, default=merge.DEFAULT_RETRAIN_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = audit_retrain_impact(
            full_plan_path=args.full_plan,
            main_index_path=args.main_index,
            retrain_plan_path=args.retrain_plan,
            retrain_index_path=args.retrain_index,
            output_path=args.output,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
