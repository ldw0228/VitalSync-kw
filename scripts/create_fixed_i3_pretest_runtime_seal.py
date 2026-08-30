#!/usr/bin/env python3
"""Create the prelaunch byte inventory for the fixed-i3 pre-test campaign.

The fixed-i3 driver re-verifies this inventory before and after every logical
unit.  This builder validates the complete 90-unit non-test proposer index,
then binds every effective Python/config source, every referenced proposer
checkpoint/prediction, the immutable discovery locks, and the RF/SVD payload
trees.  Mutable fixed-i3 output/cache roots are deliberately excluded.

No outer-test manifest, target artifact, or evaluation score is accepted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import run_fixed_i3_pretest_campaign as fixed  # noqa: E402
import seal_runtime_inputs as runtime_seal  # noqa: E402


DEFAULT_SOURCES = (
    PROJECT_ROOT / "scripts/__init__.py",
    PROJECT_ROOT / "scripts/create_fixed_i3_pretest_runtime_seal.py",
    PROJECT_ROOT / "scripts/run_fixed_i3_pretest_campaign.py",
    PROJECT_ROOT / "scripts/run_hcs_discovery_campaign.py",
    PROJECT_ROOT / "scripts/seal_runtime_inputs.py",
    PROJECT_ROOT / "scripts/build_nested_proposer_stack.py",
    PROJECT_ROOT / "scripts/build_nested_fallback_oof.py",
    PROJECT_ROOT / "scripts/build_harmonic_set_cache.py",
    PROJECT_ROOT / "scripts/train_harmonic_set_snn.py",
    PROJECT_ROOT / "scripts/run_gpu_admitted.py",
    PROJECT_ROOT / "src/snn_rr/__init__.py",
    PROJECT_ROOT / "src/snn_rr/split_authority.py",
    PROJECT_ROOT / "src/snn_rr/harmonic_set_data.py",
    PROJECT_ROOT / "src/snn_rr/harmonic_set_models.py",
    PROJECT_ROOT / "src/snn_rr/svd_episode_models.py",
    PROJECT_ROOT / "configs/harmonic_set_v2.yaml",
)


def _resolve(raw: Any, *, relative_to: Path = PROJECT_ROOT) -> Path:
    path = Path(str(raw)).expanduser()
    return path.resolve() if path.is_absolute() else (relative_to / path).resolve()


def _forbid_outer_test(path: Path, label: str) -> None:
    lowered = [part.lower() for part in path.parts]
    if any(part.startswith("test_pred_") for part in lowered):
        raise RuntimeError(f"outer-test path is forbidden in {label}: {path}")
    if path.name.lower() in {
        "evaluation_lock.json",
        "locked_hcs_oof_joined.npz",
        "canonical_locked_hcs_targets.npz",
        "canonical_locked_hcs_targets_receipt.json",
    }:
        raise RuntimeError(f"target/evaluation path is forbidden in {label}: {path}")


def collect_inventory_paths(
    *,
    plan_path: Path,
    index_path: Path,
    groups: Mapping[tuple[int, int], Mapping[str, Any]],
    common_root: Path,
    freeze_root: Path,
    reuse_root: Path,
    rf_cache: Path,
    svd_cache: Path,
    fold_assignments: Path,
    retrain_impact_audit: Path = fixed.DEFAULT_RETRAIN_IMPACT_AUDIT,
    python_executable: Path = Path(sys.executable),
    extra_sources: Sequence[Path] = (),
) -> tuple[list[Path], list[Path], list[Path]]:
    """Return deterministic source/tree/file inventories after validation."""

    plan_resolved = plan_path.expanduser().resolve()
    index_resolved = index_path.expanduser().resolve()
    try:
        plan = json.loads(plan_resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid fixed-i3 non-test plan: {plan_resolved} ({exc})") from exc
    if not isinstance(plan, Mapping) or plan.get("outer_test_opened") is not False:
        raise RuntimeError("fixed-i3 plan is not a sealed non-test plan")
    manifest_root = _resolve(plan.get("manifest_root"), relative_to=plan_resolved.parent)

    expected_groups = {(fold, seed) for fold in fixed.FOLDS for seed in fixed.SEEDS}
    if set(groups) != expected_groups:
        raise RuntimeError("fixed-i3 proposer groups are not the exact 18-unit matrix")

    sources = [path.expanduser().resolve() for path in (*DEFAULT_SOURCES, *extra_sources)]
    if len(sources) != len(set(sources)):
        raise RuntimeError("fixed-i3 runtime source list contains duplicates")
    trees = [
        common_root.expanduser().resolve(),
        freeze_root.expanduser().resolve(),
        reuse_root.expanduser().resolve(),
        rf_cache.expanduser().resolve(),
        svd_cache.expanduser().resolve(),
        manifest_root,
    ]
    if len(trees) != len(set(trees)):
        raise RuntimeError("fixed-i3 runtime input tree list contains duplicates")

    bindings: list[Path] = [
        plan_resolved,
        index_resolved,
        fold_assignments.expanduser().resolve(),
        retrain_impact_audit.expanduser().resolve(),
        python_executable.expanduser().resolve(),
    ]
    for key in sorted(groups):
        group = groups[key]
        if group.get("status") != "ready":
            raise RuntimeError(f"fixed-i3 proposer group is not ready: {key}")
        units = group.get("units")
        if not isinstance(units, list) or len(units) != 5:
            raise RuntimeError(f"fixed-i3 proposer group lacks its five-unit cover: {key}")
        for unit in units:
            if not isinstance(unit, Mapping):
                raise RuntimeError(f"fixed-i3 proposer group contains invalid unit: {key}")
            for name in ("checkpoint", "all_window_prediction"):
                binding = unit.get(name)
                if not isinstance(binding, Mapping):
                    raise RuntimeError(f"fixed-i3 proposer unit lacks {name}: {key}")
                bindings.append(_resolve(binding.get("path")))
    # A file referenced by several semantic records must be byte-checked only
    # once; duplicate paths would otherwise make the seal needlessly unstable.
    bindings = sorted(set(bindings), key=str)

    for label, paths in (("source", sources), ("tree", trees), ("binding", bindings)):
        for path in paths:
            _forbid_outer_test(path, f"fixed-i3 runtime {label}")
    return sources, trees, bindings


def create_seal(
    *,
    plan_path: Path,
    index_path: Path,
    common_root: Path,
    freeze_root: Path,
    reuse_root: Path,
    rf_cache: Path,
    svd_cache: Path,
    fold_assignments: Path,
    output_path: Path,
    retrain_impact_audit: Path = fixed.DEFAULT_RETRAIN_IMPACT_AUDIT,
    python_executable: Path = Path(sys.executable),
    extra_sources: Sequence[Path] = (),
) -> dict[str, Any]:
    destination = output_path.expanduser().resolve()
    if destination.exists():
        raise RuntimeError(f"immutable fixed-i3 runtime seal already exists: {destination}")
    groups, validated = fixed.validate_non_test_plan_index(
        plan_path.expanduser().resolve(), index_path.expanduser().resolve()
    )
    sources, trees, bindings = collect_inventory_paths(
        plan_path=plan_path,
        index_path=index_path,
        groups=groups,
        common_root=common_root,
        freeze_root=freeze_root,
        reuse_root=reuse_root,
        rf_cache=rf_cache,
        svd_cache=svd_cache,
        fold_assignments=fold_assignments,
        retrain_impact_audit=retrain_impact_audit,
        python_executable=python_executable,
        extra_sources=extra_sources,
    )
    document = runtime_seal.inventory(
        sources=sources,
        trees=trees,
        bindings=bindings,
        post_launch_attestation=False,
    )
    document["fixed_i3_context"] = {
        "classification": "retrospective_fixed_i3_pretest_runtime_input_context",
        "outer_test_opened": False,
        "outer_test_record_count": 0,
        "target_or_evaluation_artifact_accessed": False,
        "proposer_matrix_groups": len(groups),
        "proposer_matrix_units": sum(len(group["units"]) for group in groups.values()),
        "validated_plan": validated["plan"],
        "validated_index": validated["index"],
        "mutable_output_and_cache_roots_excluded": True,
    }
    # ``inventory`` hashes before this fixed-i3 context is appended.
    document["content_sha256"] = runtime_seal.canonical_sha256(document)
    runtime_seal.atomic_json(destination, document)
    verified = runtime_seal.verify(destination)
    return {
        "status": "sealed_and_verified",
        "output": str(destination),
        "content_sha256": document["content_sha256"],
        "sources": len(sources),
        "trees": len(trees),
        "bindings": len(bindings),
        "verified_files": verified["verified_files"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=fixed.DEFAULT_PLAN)
    parser.add_argument("--index", type=Path, default=fixed.DEFAULT_INDEX)
    parser.add_argument("--common-root", type=Path, default=fixed.DEFAULT_COMMON_ROOT)
    parser.add_argument("--freeze-root", type=Path, default=fixed.DEFAULT_FREEZE_ROOT)
    parser.add_argument("--reuse-root", type=Path, default=fixed.DEFAULT_REUSE_ROOT)
    parser.add_argument("--rf-cache", type=Path, default=fixed.DEFAULT_RF_CACHE)
    parser.add_argument("--svd-cache", type=Path, default=fixed.DEFAULT_SVD_CACHE)
    parser.add_argument("--fold-assignments", type=Path, default=fixed.DEFAULT_FOLDS)
    parser.add_argument(
        "--retrain-impact-audit",
        type=Path,
        default=fixed.DEFAULT_RETRAIN_IMPACT_AUDIT,
    )
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, default=fixed.DEFAULT_RUNTIME_SEAL)
    parser.add_argument("--extra-source", action="append", type=Path, default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = create_seal(
            plan_path=args.plan,
            index_path=args.index,
            common_root=args.common_root,
            freeze_root=args.freeze_root,
            reuse_root=args.reuse_root,
            rf_cache=args.rf_cache,
            svd_cache=args.svd_cache,
            fold_assignments=args.fold_assignments,
            output_path=args.output,
            retrain_impact_audit=args.retrain_impact_audit,
            python_executable=args.python_executable,
            extra_sources=args.extra_source,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
