from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/create_fixed_i3_pretest_runtime_seal.py"
SPEC = importlib.util.spec_from_file_location("create_fixed_i3_pretest_runtime_seal", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SEAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEAL)


def _groups(tmp_path: Path) -> dict[tuple[int, int], dict[str, object]]:
    result: dict[tuple[int, int], dict[str, object]] = {}
    for fold in SEAL.fixed.FOLDS:
        for seed in SEAL.fixed.SEEDS:
            units = []
            for position in range(5):
                checkpoint = tmp_path / f"unit_{fold}_{seed}_{position}.pt"
                prediction = tmp_path / f"unit_{fold}_{seed}_{position}.npz"
                checkpoint.write_bytes(b"checkpoint")
                prediction.write_bytes(b"prediction")
                units.append(
                    {
                        "checkpoint": {"path": str(checkpoint)},
                        "all_window_prediction": {"path": str(prediction)},
                    }
                )
            result[(fold, seed)] = {"status": "ready", "units": units}
    return result


def _arguments(tmp_path: Path) -> dict[str, object]:
    manifest_root = tmp_path / "manifests"
    manifest_root.mkdir()
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "outer_test_opened": False,
                "manifest_root": str(manifest_root),
            }
        ),
        encoding="utf-8",
    )
    index = tmp_path / "index.json"
    index.write_text("{}", encoding="utf-8")
    fold_assignments = tmp_path / "fold_assignments.json"
    fold_assignments.write_text("{}", encoding="utf-8")
    retrain_impact_audit = tmp_path / "retrain_impact_audit.json"
    retrain_impact_audit.write_text("{}", encoding="utf-8")
    trees = []
    for name in ("common", "freeze", "reuse", "rf", "svd"):
        path = tmp_path / name
        path.mkdir()
        (path / "payload.bin").write_bytes(name.encode())
        trees.append(path)
    return {
        "plan_path": plan,
        "index_path": index,
        "groups": _groups(tmp_path),
        "common_root": trees[0],
        "freeze_root": trees[1],
        "reuse_root": trees[2],
        "rf_cache": trees[3],
        "svd_cache": trees[4],
        "fold_assignments": fold_assignments,
        "retrain_impact_audit": retrain_impact_audit,
    }


def test_collects_exact_18_by_5_proposer_cover(tmp_path: Path) -> None:
    args = _arguments(tmp_path)
    sources, trees, bindings = SEAL.collect_inventory_paths(**args)
    assert len(sources) == len(SEAL.DEFAULT_SOURCES)
    assert len(trees) == 6
    assert len(bindings) == 5 + 18 * 5 * 2
    assert all(path.is_absolute() for path in (*sources, *trees, *bindings))


def test_deduplicates_repeated_bound_artifacts(tmp_path: Path) -> None:
    args = _arguments(tmp_path)
    groups = args["groups"]
    assert isinstance(groups, dict)
    first = groups[(0, SEAL.fixed.SEEDS[0])]["units"][0]
    second = groups[(0, SEAL.fixed.SEEDS[0])]["units"][1]
    second["checkpoint"] = first["checkpoint"]
    _, _, bindings = SEAL.collect_inventory_paths(**args)
    assert len(bindings) == 5 + 18 * 5 * 2 - 1


def test_rejects_incomplete_group_matrix(tmp_path: Path) -> None:
    args = _arguments(tmp_path)
    groups = args["groups"]
    assert isinstance(groups, dict)
    groups.pop(next(iter(groups)))
    with pytest.raises(RuntimeError, match="exact 18-unit matrix"):
        SEAL.collect_inventory_paths(**args)


def test_rejects_outer_test_binding(tmp_path: Path) -> None:
    args = _arguments(tmp_path)
    groups = args["groups"]
    assert isinstance(groups, dict)
    unit = groups[(0, SEAL.fixed.SEEDS[0])]["units"][0]
    forbidden = tmp_path / "test_pred_0" / "prediction.npz"
    forbidden.parent.mkdir()
    forbidden.write_bytes(b"x")
    unit["all_window_prediction"] = {"path": str(forbidden)}
    with pytest.raises(RuntimeError, match="outer-test path is forbidden"):
        SEAL.collect_inventory_paths(**args)
