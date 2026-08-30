from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVALUATOR = _load(
    "evaluate_locked_hcs_radar_masks_under_test",
    ROOT / "scripts/evaluate_locked_hcs_radar_masks.py",
)
PRIMARY_FIXTURE = _load(
    "primary_evaluator_fixture_for_radar_masks",
    ROOT / "tests/test_evaluate_locked_hcs_oof.py",
)


def _write_json(
    path: Path, value: dict[str, Any], *, content_hash: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = dict(value)
    if content_hash:
        document["content_sha256"] = EVALUATOR.PRIMARY.canonical_json_sha256(document)
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")


def _binding(path: Path) -> dict[str, Any]:
    return EVALUATOR.PRIMARY.bind_file(path)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def _mask_fixture(tmp_path: Path) -> dict[str, Path]:
    primary = PRIMARY_FIXTURE._fixture(tmp_path / "primary")
    mask_root = tmp_path / "masks"
    plan_path = mask_root / "control/plan.json"
    preexecution_path = mask_root / "control/preexecution_lock.json"
    complete_seal_path = mask_root / "complete_seal.json"
    seeds = list(EVALUATOR.PRIMARY.FIXED_SEEDS)

    plan_units: list[dict[str, Any]] = []
    output_arrays: dict[tuple[int, int, str], dict[str, np.ndarray]] = {}
    output_paths: dict[tuple[int, int, str], Path] = {}
    for seed in seeds:
        for fold in range(6):
            primary_prediction = (
                primary["root"]
                / "units"
                / f"outer_{fold}_seed_{seed}"
                / "sealed_label_free_predictions.npz"
            )
            with np.load(primary_prediction, allow_pickle=False) as archive:
                primary_arrays = {name: np.asarray(archive[name]) for name in archive.files}
            for mask_position, (mask, pattern) in enumerate(EVALUATOR.MASKS.items()):
                key = (fold, seed, mask)
                unit_root = mask_root / "units" / f"outer_{fold}_seed_{seed}" / mask
                prediction_path = unit_root / "sealed_label_free_predictions.npz"
                arrays = {name: value.copy() for name, value in primary_arrays.items()}
                if mask != EVALUATOR.BASELINE_MASK:
                    delta = np.float32(mask_position * 0.15)
                    for field in (
                        "fallback_rr_bpm",
                        "source_rr_bpm",
                        "final_rr_bpm",
                    ):
                        arrays[field] = (arrays[field].astype(np.float32) + delta).astype(
                            np.float32
                        )
                output_arrays[key] = arrays
                output_paths[key] = prediction_path
                plan_units.append(
                    {
                        "unit_id": EVALUATOR.MASK_CAMPAIGN._mask_unit_id(
                            fold, seed, mask
                        ),
                        "outer_fold": fold,
                        "seed": seed,
                        "radar_mask": mask,
                        "radar_mask_pattern": list(pattern),
                        "inputs": {},
                        "primary": {
                            "sealed_prediction": _binding(primary_prediction),
                        },
                        "commands": [],
                        "outputs": {
                            "sealed_prediction": str(prediction_path.resolve()),
                            "receipt": str((unit_root / "receipt.json").resolve()),
                        },
                    }
                )

    mask_contract = {
        "mask_order_fixed_before_inference": list(EVALUATOR.MASK_NAMES),
        "best_mask_selection_allowed": False,
        "target_or_metric_dependent_mask_selection_allowed": False,
        "all_masks_are_required_conditions": True,
        "radars_123_primary_parity": {
            "required": True,
            "comparison": "dtype_shape_and_array_bytes",
            "artifacts": ["sealed_prediction"],
        },
    }
    plan = {
        "schema_version": 1,
        "classification": "locked_hcs_seven_radar_mask_label_free_plan",
        "folds": list(range(6)),
        "seeds": seeds,
        "radar_masks": [
            {"name": name, "pattern": list(pattern)}
            for name, pattern in EVALUATOR.MASKS.items()
        ],
        "primary_unit_count": 18,
        "unit_count": 126,
        "target_or_label_artifact_bound": False,
        "evaluation_permitted_before_complete_seal": False,
        "mask_selection_contract": mask_contract,
        "primary": {
            "predictions_seal": _binding(primary["root"] / "predictions_seal.json")
        },
        "effective_sources": {},
        "execution": {
            "device": "cpu",
            "amp": False,
            "batch_size": 128,
            "shell": False,
            "publication": "atomic_unit_directory_rename",
        },
        "units": plan_units,
    }
    _write_json(plan_path, plan)
    plan_binding = _binding(plan_path)
    preexecution = {
        "schema_version": 1,
        "classification": "locked_hcs_radar_mask_preexecution_input_seal",
        "plan": plan_binding,
        "unit_count": 126,
        "target_or_label_artifact_opened": False,
        "evaluation_authorized": False,
    }
    _write_json(preexecution_path, preexecution)

    seal_units: list[dict[str, Any]] = []
    for unit in plan_units:
        key = (unit["outer_fold"], unit["seed"], unit["radar_mask"])
        prediction_path = output_paths[key]
        _write_npz(prediction_path, output_arrays[key])
        receipt_path = prediction_path.parent / "receipt.json"
        parity = {
            "required": unit["radar_mask"] == EVALUATOR.BASELINE_MASK,
            "performed": unit["radar_mask"] == EVALUATOR.BASELINE_MASK,
        }
        receipt = {
            "schema_version": 1,
            "classification": "locked_hcs_radar_mask_label_free_unit_receipt",
            "unit_id": unit["unit_id"],
            "outer_fold": unit["outer_fold"],
            "seed": unit["seed"],
            "radar_mask": unit["radar_mask"],
            "radar_mask_pattern": unit["radar_mask_pattern"],
            "plan": plan_binding,
            "inputs": unit["inputs"],
            "primary": unit["primary"],
            "commands": unit["commands"],
            "outputs": {"sealed_prediction": _binding(prediction_path)},
            "radars_123_bit_exact_primary_comparison": parity,
            "runtime_guards": {
                "device": "cpu",
                "cuda_visible_devices": "",
                "amp": False,
                "shell": False,
            },
            "source_semantics": "frozen_no_action_placeholder",
            "target_or_label_fields_read": False,
            "target_or_label_fields_present": False,
            "evaluation_performed": False,
        }
        _write_json(receipt_path, receipt, content_hash=True)
        receipt_document = json.loads(receipt_path.read_text(encoding="utf-8"))
        seal_units.append(
            {
                "unit_id": unit["unit_id"],
                "outer_fold": unit["outer_fold"],
                "seed": unit["seed"],
                "radar_mask": unit["radar_mask"],
                "radar_mask_pattern": unit["radar_mask_pattern"],
                "receipt": _binding(receipt_path),
                "sealed_prediction": _binding(prediction_path),
                "receipt_content_sha256": receipt_document["content_sha256"],
            }
        )
    complete_seal = {
        "schema_version": 1,
        "classification": "locked_hcs_all_seven_radar_mask_predictions_sealed",
        "plan": plan_binding,
        "preexecution_lock": _binding(preexecution_path),
        "primary_predictions_seal": _binding(primary["root"] / "predictions_seal.json"),
        "folds": list(range(6)),
        "seeds": seeds,
        "radar_masks": plan["radar_masks"],
        "unit_count": 126,
        "complete_matrix": True,
        "target_or_label_artifact_opened_before_seal": False,
        "best_mask_selection_performed": False,
        "all_masks_retained_as_fixed_conditions": True,
        "evaluation_authorized": True,
        "mask_selection_contract": mask_contract,
        "units": seal_units,
    }
    _write_json(complete_seal_path, complete_seal)

    output = tmp_path / "evaluation"
    return {
        **primary,
        "mask_root": mask_root,
        "mask_plan": plan_path,
        "mask_seal": complete_seal_path,
        "output": output,
        "mask_report": output / "report.json",
        "mask_csv": output / "metrics.csv",
        "mask_receipt": output / "receipt.json",
    }


def _evaluate(paths: dict[str, Path]) -> dict[str, Any]:
    return EVALUATOR.evaluate_radar_masks(
        radar_mask_root=paths["mask_root"],
        primary_root=paths["root"],
        primary_evaluation_lock=paths["evaluation_lock"],
        target_receipt=paths["target_receipt"],
        evaluation_spec=paths["evaluation_spec"],
        output_dir=paths["output"],
        report_output=paths["mask_report"],
        csv_output=paths["mask_csv"],
        receipt_output=paths["mask_receipt"],
        orchestrator_command=["synthetic-radar-mask-evaluation"],
    )


def test_all_seven_masks_are_evaluated_per_seed_with_immutable_evidence(
    tmp_path: Path,
) -> None:
    paths = _mask_fixture(tmp_path)
    receipt = _evaluate(paths)
    report = json.loads(paths["mask_report"].read_text(encoding="utf-8"))

    assert report["commercial_claim_authorized"] is False
    assert report["commercial_performance_proven"] is False
    assert report["prospective_confirmation_required"] is True
    assert report["mask_selection_or_ranking_performed"] is False
    assert report["seed_pooling_ranking_or_suppression_performed"] is False
    assert report["radars_123_primary_parity_gate"]["all_fixed_seeds_passed"] is True
    assert set(report["per_seed"]) == {"20260828", "20260829", "20260830"}
    for seed_report in report["per_seed"].values():
        assert set(seed_report["radar_masks"]) == set(EVALUATOR.MASKS)
        for metrics in seed_report["radar_masks"].values():
            assert set(metrics["eight_fixed_window_phases"]) == {
                str(value) for value in range(8)
            }
            assert {"identity", "fold", "session", "protocol", "rr_band"} <= set(
                metrics["strata"]
            )
            assert "qc:reference_quality" in metrics["strata"]
        bootstrap = seed_report["paired_physical_identity_cluster_bootstrap"]
        assert bootstrap["fixed_spec"]["unit"] == "physical_identity"
        assert bootstrap["fixed_spec"]["samples"] == 128
        assert set(bootstrap["degradation_vs_radars_123"]) == set(EVALUATOR.MASKS) - {
            EVALUATOR.BASELINE_MASK
        }
        for degradation in bootstrap["degradation_vs_radars_123"].values():
            assert degradation["paired_on_exact_cache_index_and_physical_identity"] is True
            assert set(degradation["direction_adjusted_degradation_intervals"]) == set(
                EVALUATOR.PRIMARY.METRIC_FIELDS
            )

    assert receipt["commercial_claim_authorized"] is False
    assert receipt["all_seven_masks_per_seed_without_selection"] is True
    assert receipt["radars_123_primary_parity_gate_passed"] is True
    assert receipt["outputs"]["report"] == _binding(paths["mask_report"])
    assert receipt["outputs"]["metrics_csv"] == _binding(paths["mask_csv"])
    assert (paths["mask_report"].stat().st_mode & 0o777) == 0o444
    assert (paths["mask_csv"].stat().st_mode & 0o777) == 0o444
    assert (paths["mask_receipt"].stat().st_mode & 0o777) == 0o444
    with paths["mask_csv"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["record_type"] for row in rows} == {"metric", "paired_degradation"}
    assert {row["radar_mask"] for row in rows} == set(EVALUATOR.MASKS)


def test_incomplete_mask_seal_fails_before_primary_target_context_is_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _mask_fixture(tmp_path)
    seal = json.loads(paths["mask_seal"].read_text(encoding="utf-8"))
    seal["units"].pop()
    paths["mask_seal"].write_text(json.dumps(seal), encoding="utf-8")
    called = False

    def forbidden(**_: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("target context must not be opened")

    monkeypatch.setattr(EVALUATOR, "_load_primary_context", forbidden)
    with pytest.raises(EVALUATOR.LockedRadarMaskEvaluationError, match="matrix is incomplete"):
        _evaluate(paths)
    assert called is False
    assert not paths["output"].exists()


def test_evaluation_spec_and_target_receipt_tamper_fail_before_target_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_paths = _mask_fixture(tmp_path / "spec")
    spec_paths["evaluation_spec"].chmod(0o644)
    spec = json.loads(spec_paths["evaluation_spec"].read_text(encoding="utf-8"))
    spec["bootstrap"]["samples"] = 7
    spec_paths["evaluation_spec"].write_text(json.dumps(spec), encoding="utf-8")
    called = False
    original_context = EVALUATOR._load_primary_context

    def forbidden(**_: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("target context must not be opened")

    monkeypatch.setattr(EVALUATOR, "_load_primary_context", forbidden)
    with pytest.raises(EVALUATOR.LockedRadarMaskEvaluationError, match="content_sha256"):
        _evaluate(spec_paths)
    assert called is False
    monkeypatch.setattr(EVALUATOR, "_load_primary_context", original_context)

    receipt_paths = _mask_fixture(tmp_path / "receipt")
    receipt_paths["target_receipt"].chmod(0o644)
    target_receipt = json.loads(
        receipt_paths["target_receipt"].read_text(encoding="utf-8")
    )
    target_receipt["valid_reference_rows"] = 47
    receipt_paths["target_receipt"].write_text(json.dumps(target_receipt), encoding="utf-8")
    target_paths = {
        receipt_paths["target"].resolve(),
        receipt_paths["joined"].resolve(),
    }
    opened_target: list[Path] = []
    original = EVALUATOR.PRIMARY._load_npz

    def guarded(path: Path, *, label: str) -> dict[str, np.ndarray]:
        if path.resolve() in target_paths:
            opened_target.append(path.resolve())
        return original(path, label=label)

    monkeypatch.setattr(EVALUATOR.PRIMARY, "_load_npz", guarded)
    with pytest.raises(EVALUATOR.LockedRadarMaskEvaluationError, match="content_sha256"):
        _evaluate(receipt_paths)
    assert opened_target == []


def test_radars_123_primary_parity_tamper_fails_before_target_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _mask_fixture(tmp_path)
    seal = json.loads(paths["mask_seal"].read_text(encoding="utf-8"))
    unit = next(
        item
        for item in seal["units"]
        if item["seed"] == 20260828
        and item["outer_fold"] == 0
        and item["radar_mask"] == EVALUATOR.BASELINE_MASK
    )
    prediction_path = Path(unit["sealed_prediction"]["path"])
    with np.load(prediction_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["final_rr_bpm"] = arrays["final_rr_bpm"].copy()
    arrays["final_rr_bpm"][0] += np.float32(0.5)
    prediction_path.chmod(0o644)
    _write_npz(prediction_path, arrays)
    receipt_path = Path(unit["receipt"]["path"])
    receipt_path.chmod(0o644)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["outputs"]["sealed_prediction"] = _binding(prediction_path)
    receipt.pop("content_sha256")
    receipt["content_sha256"] = EVALUATOR.PRIMARY.canonical_json_sha256(receipt)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    unit["receipt"] = _binding(receipt_path)
    unit["sealed_prediction"] = _binding(prediction_path)
    unit["receipt_content_sha256"] = receipt["content_sha256"]
    paths["mask_seal"].write_text(json.dumps(seal, sort_keys=True), encoding="utf-8")

    called = False

    def forbidden(**_: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("target context must not be opened")

    monkeypatch.setattr(EVALUATOR, "_load_primary_context", forbidden)
    with pytest.raises(EVALUATOR.LockedRadarMaskEvaluationError, match="primary parity failed"):
        _evaluate(paths)
    assert called is False


def test_mask_evaluation_is_create_once(tmp_path: Path) -> None:
    paths = _mask_fixture(tmp_path)
    _evaluate(paths)
    before = {
        path: path.read_bytes()
        for path in (paths["mask_report"], paths["mask_csv"], paths["mask_receipt"])
    }
    with pytest.raises(EVALUATOR.LockedRadarMaskEvaluationError, match="already exists"):
        _evaluate(paths)
    assert all(path.read_bytes() == payload for path, payload in before.items())
