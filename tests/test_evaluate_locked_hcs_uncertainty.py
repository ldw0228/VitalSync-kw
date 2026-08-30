from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_locked_hcs_uncertainty.py"
SPEC = importlib.util.spec_from_file_location("evaluate_locked_hcs_uncertainty", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


def _json(path: Path, value: dict[str, Any], *, content_hash: bool = False) -> dict[str, Any]:
    document = dict(value)
    if content_hash:
        document["content_sha256"] = EVAL.PRIMARY.canonical_json_sha256(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return document


def _binding(path: Path) -> dict[str, Any]:
    return EVAL.PRIMARY.bind_file(path)


def _fixed_gates() -> dict[str, Any]:
    return {
        "all_seeds_required": True,
        "conformal_max_absolute_calibration_error_all_levels": 0.07,
        "conformal_90_marginal_coverage_min": 0.88,
        "conformal_90_identity_macro_coverage_min": 0.85,
        "conformal_90_fixed_phase_0_coverage_min": 0.85,
        "conformal_90_mean_full_width_bpm_max": 6.0,
        "conformal_90_p95_full_width_bpm_max": 10.0,
        "selective_80_mae_bpm_max": 1.0,
        "selective_80_catastrophic_over_5_max": 0.03,
    }


class PreTargetFixture:
    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path / "locked"
        self.seeds = EVAL.PRIMARY.FIXED_SEEDS
        spec_document = EVAL.PRIMARY.evaluation_spec_document(
            expected_rows=12,
            expected_identities=2,
            bootstrap_samples=20,
            bootstrap_seed=31,
            bootstrap_confidence=0.90,
        )
        self.spec = tmp_path / "evaluation_spec.json"
        _json(self.spec, spec_document, content_hash=True)

        pretest = self.root / "pretest_lock.json"
        _json(pretest, {"classification": "test_pretest_lock"})
        pretest_hash = EVAL.PRIMARY.sha256_file(pretest)

        dummy_root = tmp_path / "calibration_sources"
        source_units = []
        for source in range(5):
            artifacts = {}
            for kind in ("manifest", "checkpoint", "all_window_prediction"):
                path = dummy_root / f"source_{source}_{kind}.bin"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{source}/{kind}".encode())
                artifacts[kind] = _binding(path)
            source_units.append(
                {"name": f"source_{source}", "role": "nested_oof", **artifacts}
            )

        calibration_units = []
        prediction_units = []
        for seed in self.seeds:
            for fold in EVAL.FOLDS:
                calibration_units.append(
                    {
                        "outer_fold": fold,
                        "seed": seed,
                        "source_unit_count": 5,
                        "source_units": source_units,
                        "source_rows_all": 120,
                        "source_rows_valid_phase_0": 15,
                        "source_identity_count": 15,
                        "source_identities": [f"I{value:02d}" for value in range(15)],
                        "std_floor_bpm": 0.25,
                        "interval_calibration": {
                            f"{coverage:.2f}": {
                                "nominal_coverage": coverage,
                                "normalized_absolute_error_quantile": 1.5,
                            }
                            for coverage in EVAL.INTERVAL_COVERAGES
                        },
                        "selective_thresholds": {
                            f"{coverage:.2f}": {
                                "intended_acceptance_coverage": coverage,
                                "rr_std_threshold_bpm": 2.0,
                            }
                            for coverage in EVAL.SELECTIVE_COVERAGES
                        },
                    }
                )
                unit_root = self.root / "units" / f"outer_{fold}_seed_{seed}"
                index = np.asarray([fold * 2, fold * 2 + 1], dtype=np.int64)
                fallback = np.asarray([12.0 + fold, 13.0 + fold], dtype=np.float32)
                raw_path = unit_root / "raw_hcs_prediction.npz"
                point_path = unit_root / "sealed_label_free_predictions.npz"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    raw_path,
                    cache_index=index,
                    fallback_rr_bpm=fallback,
                    fallback_std_bpm=np.asarray([0.5, 0.75], dtype=np.float32),
                    fallback_available=np.ones(2, dtype=bool),
                    source_rr_bpm=fallback + np.float32(0.25),
                    source_scale_bpm=np.full(2, 0.8, dtype=np.float32),
                    source_available=np.ones(2, dtype=bool),
                    selected_probability=np.full(2, 0.7, dtype=np.float32),
                    margin=np.full(2, 0.2, dtype=np.float32),
                    entropy=np.full(2, 0.4, dtype=np.float32),
                    normalized_entropy=np.full(2, 0.3, dtype=np.float32),
                    quality=np.full(2, 0.9, dtype=np.float32),
                    valid_candidate_count=np.full(2, 3, dtype=np.int16),
                )
                np.savez_compressed(
                    point_path,
                    cache_index=index,
                    outer_fold=np.asarray(fold, dtype=np.int16),
                    seed=np.asarray(seed, dtype=np.int64),
                    fallback_rr_bpm=fallback,
                    source_rr_bpm=fallback + np.float32(0.25),
                    final_rr_bpm=fallback.copy(),
                    applied_pull=np.zeros(2, dtype=np.float32),
                    target_joined=np.asarray(False),
                )
                derived_path = unit_root / "derived_inference_lock.json"
                _json(
                    derived_path,
                    {
                        "schema_version": 1,
                        "classification": "locked_hcs_oof_derived_test_inference",
                        "outer_fold": fold,
                        "seed": seed,
                        "pretest_lock_sha256": pretest_hash,
                        "target_artifact_opened": False,
                        "frozen_policy_status": "fail_closed_no_action",
                        "no_action_bit_exact_float32_fallback": True,
                        "derived_artifacts": {"raw_hcs_prediction": _binding(raw_path)},
                        "sealed_prediction": _binding(point_path),
                    },
                )
                prediction_units.append(
                    {
                        "outer_fold": fold,
                        "seed": seed,
                        "derived_lock": _binding(derived_path),
                        "prediction": _binding(point_path),
                    }
                )
        self.calibration = tmp_path / "calibration.json"
        _json(
            self.calibration,
            {
                "schema_version": 1,
                "classification": "locked_pretest_cross_fitted_proposer_uncertainty_calibration",
                "commercial_claim_authorized": False,
                "prospective_confirmation_required": True,
                "outer_test_opened": False,
                "outer_test_record_count": 0,
                "target_artifact_opened": False,
                "point_prediction_modified": False,
                "folds": list(EVAL.FOLDS),
                "seeds": list(self.seeds),
                "unit_count": 18,
                "fixed_method": {
                    "phase_modulus": 8,
                    "phase_value": 0,
                    "std_floor_bpm": 0.25,
                    "interval_coverages": list(EVAL.INTERVAL_COVERAGES),
                    "selective_coverages": list(EVAL.SELECTIVE_COVERAGES),
                    "no_test_time_fit_or_threshold_selection": True,
                    "formal_exchangeability_claim": False,
                },
                "fixed_evaluation_gates": _fixed_gates(),
                "inputs": {},
                "units": calibration_units,
            },
            content_hash=True,
        )
        self.uncertainty_evaluation_spec = tmp_path / "uncertainty_evaluation_spec.json"
        EVAL.FREEZER.freeze_uncertainty_evaluation_spec(
            output_path=self.uncertainty_evaluation_spec,
            locked_oof_root=self.root,
            primary_spec_path=self.spec,
            calibration_path=self.calibration,
        )
        self.predictions = self.root / "predictions_seal.json"
        _json(
            self.predictions,
            {
                "schema_version": 1,
                "classification": "locked_hcs_oof_all_label_free_predictions_sealed",
                "pretest_lock_sha256": pretest_hash,
                "target_artifact_opened_before_seal": False,
                "target_join_authorized": True,
                "unit_count": 18,
                "outer_folds": list(EVAL.FOLDS),
                "units": prediction_units,
            },
        )
        self.archive = self.root / "locked_hcs_uncertainty_inputs.npz"
        self.uncertainty_seal = self.root / "uncertainty_inputs_seal.json"
        EVAL.SEALER.seal_uncertainty_inputs(
            root=self.root,
            calibration_path=self.calibration,
            output_path=self.archive,
            seal_path=self.uncertainty_seal,
        )

    def validate(self) -> tuple[Any, ...]:
        return EVAL._validate_pre_target_inputs(
            uncertainty_evaluation_spec=self.uncertainty_evaluation_spec,
            evaluation_spec=self.spec,
            calibration_path=self.calibration,
            predictions_seal_path=self.predictions,
            uncertainty_seal_path=self.uncertainty_seal,
            locked_oof_root=self.root,
        )


def test_pre_target_validator_rehashes_exact_18_unit_calibration_predictions_and_arrays(
    tmp_path: Path,
) -> None:
    fixture = PreTargetFixture(tmp_path)
    spec, calibration, predictions, arrays, audit = fixture.validate()
    assert spec["population"]["fixed_seeds"] == list(EVAL.PRIMARY.FIXED_SEEDS)
    assert calibration["unit_count"] == predictions["unit_count"] == 18
    assert set(arrays) == EVAL.EXPECTED_UNCERTAINTY_FIELDS
    assert audit["all_target_free_inputs_verified_before_evaluation_lock_access"] is True
    assert audit["all_uncertainty_array_schema_and_hashes_verified"] is True
    assert audit[
        "dedicated_secondary_spec_verified_before_prediction_uncertainty_or_target_access"
    ] is True
    assert audit["primary_uncertainty_contract_overridden"] is False
    assert audit["secondary_uncertainty_evaluation_spec"] == _binding(
        fixture.uncertainty_evaluation_spec
    )
    assert audit["calibration_declared_bindings_rehashed"] >= 90


def test_pre_target_validator_rejects_resealed_array_hash_mismatch(tmp_path: Path) -> None:
    fixture = PreTargetFixture(tmp_path)
    fixture.uncertainty_seal.chmod(0o644)
    document = json.loads(fixture.uncertainty_seal.read_text(encoding="utf-8"))
    document.pop("content_sha256")
    document["array_schema"]["fallback_std_bpm"]["sha256"] = "0" * 64
    _json(fixture.uncertainty_seal, document, content_hash=True)
    with pytest.raises(EVAL.LockedUncertaintyEvaluationError, match="array schema/hash"):
        fixture.validate()


def test_pre_target_validator_rederives_archive_not_just_its_self_consistent_seal(
    tmp_path: Path,
) -> None:
    fixture = PreTargetFixture(tmp_path)
    fixture.archive.chmod(0o644)
    with np.load(fixture.archive, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    arrays["fallback_std_bpm"][0] += np.float32(0.125)
    np.savez_compressed(fixture.archive, **arrays)
    fixture.uncertainty_seal.chmod(0o644)
    seal = json.loads(fixture.uncertainty_seal.read_text(encoding="utf-8"))
    seal.pop("content_sha256")
    seal["uncertainty_archive"] = _binding(fixture.archive)
    seal["array_schema"]["fallback_std_bpm"]["sha256"] = EVAL.PRIMARY.array_sha256(
        arrays["fallback_std_bpm"]
    )
    _json(fixture.uncertainty_seal, seal, content_hash=True)
    with pytest.raises(EVAL.LockedUncertaintyEvaluationError, match="rederived 18-unit"):
        fixture.validate()


def test_pre_target_validator_rejects_calibration_source_tamper(tmp_path: Path) -> None:
    fixture = PreTargetFixture(tmp_path)
    calibration = json.loads(fixture.calibration.read_text(encoding="utf-8"))
    source = Path(calibration["units"][0]["source_units"][0]["checkpoint"]["path"])
    with source.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(EVAL.LockedUncertaintyEvaluationError, match="binding mismatch"):
        fixture.validate()


def test_evaluator_never_touches_target_context_when_pretarget_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fail_before_target(**_: Any) -> Any:
        calls.append("pretarget")
        raise EVAL.LockedUncertaintyEvaluationError("sealed boundary failed")

    def forbidden_target(**_: Any) -> Any:
        calls.append("target")
        raise AssertionError("target context must remain unopened")

    monkeypatch.setattr(EVAL, "_validate_pre_target_inputs", fail_before_target)
    monkeypatch.setattr(EVAL.PRIMARY, "_validate_context", forbidden_target)
    output = tmp_path / "out"
    with pytest.raises(EVAL.LockedUncertaintyEvaluationError, match="sealed boundary"):
        EVAL.evaluate_locked_uncertainty(
            locked_oof_root=tmp_path,
            uncertainty_evaluation_spec=tmp_path / "uncertainty_spec",
            evaluation_spec=tmp_path / "spec",
            calibration_path=tmp_path / "calibration",
            predictions_seal=tmp_path / "predictions",
            uncertainty_seal=tmp_path / "uncertainty",
            evaluation_lock=tmp_path / "evaluation_lock",
            target_receipt=tmp_path / "target_receipt",
            output_dir=output,
            report_output=output / "report.json",
            csv_output=output / "metrics.csv",
            receipt_output=output / "receipt.json",
        )
    assert calls == ["pretarget"]


def test_dedicated_secondary_spec_is_validated_before_primary_prediction_or_uncertainty_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def reject_secondary(*_: Any, **__: Any) -> Any:
        calls.append("secondary_spec")
        raise EVAL.FREEZER.UncertaintyEvaluationSpecError("secondary drift")

    def forbidden(*_: Any, **__: Any) -> Any:
        calls.append("forbidden_later_input")
        raise AssertionError("later input was opened before secondary spec authorization")

    monkeypatch.setattr(EVAL.FREEZER, "load_uncertainty_evaluation_spec", reject_secondary)
    monkeypatch.setattr(EVAL.PRIMARY, "_load_evaluation_spec", forbidden)
    monkeypatch.setattr(EVAL, "_validate_calibration", forbidden)
    monkeypatch.setattr(EVAL, "_validate_predictions_seal", forbidden)
    monkeypatch.setattr(EVAL, "_validate_uncertainty_seal", forbidden)
    with pytest.raises(EVAL.LockedUncertaintyEvaluationError, match="secondary drift"):
        EVAL._validate_pre_target_inputs(
            uncertainty_evaluation_spec=tmp_path / "secondary.json",
            evaluation_spec=tmp_path / "primary.json",
            calibration_path=tmp_path / "calibration.json",
            predictions_seal_path=tmp_path / "predictions.json",
            uncertainty_seal_path=tmp_path / "uncertainty.json",
            locked_oof_root=tmp_path,
        )
    assert calls == ["secondary_spec"]


def _frame() -> dict[str, np.ndarray]:
    rows = 16
    target = np.linspace(12.0, 27.0, rows, dtype=np.float32)
    prediction = target + np.asarray([0.1, -0.4, 0.8, -1.2] * 4, dtype=np.float32)
    return {
        "cache_index": np.arange(rows, dtype=np.int64),
        "outer_fold": np.arange(rows, dtype=np.int16) % 6,
        "seed": np.full(rows, EVAL.PRIMARY.FIXED_SEEDS[0], dtype=np.int64),
        "target_rr_bpm": target,
        "final_rr_bpm": prediction,
        "identity": np.asarray(["A"] * 8 + ["B"] * 8),
        "window_number": np.arange(rows, dtype=np.int32),
    }


def _uncertainty(frame: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rows = len(frame["cache_index"])
    return {
        "cache_index": frame["cache_index"].copy(),
        "outer_fold": frame["outer_fold"].copy(),
        "seed": frame["seed"].copy(),
        "final_rr_bpm": frame["final_rr_bpm"].copy(),
        "fallback_std_bpm": np.linspace(0.5, 1.0, rows, dtype=np.float32),
    }


def _lookup(seed: int) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (fold, seed): {
            "interval_calibration": {
                f"{coverage:.2f}": {"normalized_absolute_error_quantile": 2.0}
                for coverage in EVAL.INTERVAL_COVERAGES
            },
            "selective_thresholds": {
                f"{coverage:.2f}": {"rr_std_threshold_bpm": 0.8 + coverage}
                for coverage in EVAL.SELECTIVE_COVERAGES
            },
        }
        for fold in EVAL.FOLDS
    }


def test_interval_and_selective_protocol_reports_all_levels_and_all_fixed_phases() -> None:
    frame = _frame()
    seed = int(frame["seed"][0])
    uncertainty = _uncertainty(frame)
    intervals, raw = EVAL._interval_evaluation(
        frame=frame,
        uncertainty=uncertainty,
        calibration_lookup=_lookup(seed),
        seed=seed,
    )
    assert set(intervals) == {"normal_uncalibrated", "normalized_conformal"}
    assert set(intervals["normalized_conformal"]) == {"0.50", "0.80", "0.90", "0.95"}
    assert set(intervals["normalized_conformal"]["0.90"]["fixed_phases"]) == {
        str(value) for value in range(8)
    }
    assert intervals["normalized_conformal"]["0.90"]["fixed_phase_0"] == intervals[
        "normalized_conformal"
    ]["0.90"]["fixed_phases"]["0"]
    assert len(raw) == 8
    selective, masks = EVAL._selective_evaluation(
        frame=frame,
        uncertainty=uncertainty,
        calibration_lookup=_lookup(seed),
        seed=seed,
    )
    assert set(selective) == {"0.50", "0.80", "0.90", "1.00"}
    assert set(masks) == set(EVAL.SELECTIVE_COVERAGES)
    assert all(entry["total_rows"] == 16 for entry in selective.values())


def test_alignment_requires_bit_exact_locked_final_rr() -> None:
    frame = _frame()
    seed = int(frame["seed"][0])
    uncertainty = _uncertainty(frame)
    aligned = EVAL._align_uncertainty(uncertainty, {seed: frame})
    assert np.array_equal(aligned[seed]["cache_index"], frame["cache_index"])
    uncertainty["final_rr_bpm"][0] += np.float32(0.01)
    with pytest.raises(EVAL.LockedUncertaintyEvaluationError, match="bit parity"):
        EVAL._align_uncertainty(uncertainty, {seed: frame})


def test_identity_cluster_bootstrap_is_fixed_per_seed_and_has_interval_and_selective_cis() -> None:
    frame = _frame()
    seed = int(frame["seed"][0])
    uncertainty = _uncertainty(frame)
    lookup = _lookup(seed)
    intervals, raw = EVAL._interval_evaluation(
        frame=frame, uncertainty=uncertainty, calibration_lookup=lookup, seed=seed
    )
    selective, masks = EVAL._selective_evaluation(
        frame=frame, uncertainty=uncertainty, calibration_lookup=lookup, seed=seed
    )
    first = EVAL._cluster_bootstrap(
        frame=frame,
        interval_raw=raw,
        interval_report=intervals,
        selective_masks=masks,
        selective_report=selective,
        seed=seed,
        samples=40,
        base_seed=99,
        confidence=0.90,
    )
    second = EVAL._cluster_bootstrap(
        frame=frame,
        interval_raw=raw,
        interval_report=intervals,
        selective_masks=masks,
        selective_report=selective,
        seed=seed,
        samples=40,
        base_seed=99,
        confidence=0.90,
    )
    assert first == second
    assert first["fixed_spec"]["unit"] == "physical_identity"
    assert first["intervals"]["normalized_conformal"]["0.90"][
        "empirical_coverage"
    ]["samples_finite"] == 40
    assert first["selective"]["0.80"]["mae"]["samples_finite"] == 40


def test_gate_report_applies_frozen_values_without_selection() -> None:
    frame = _frame()
    seed = int(frame["seed"][0])
    uncertainty = _uncertainty(frame)
    intervals, _ = EVAL._interval_evaluation(
        frame=frame,
        uncertainty=uncertainty,
        calibration_lookup=_lookup(seed),
        seed=seed,
    )
    selective, _ = EVAL._selective_evaluation(
        frame=frame,
        uncertainty=uncertainty,
        calibration_lookup=_lookup(seed),
        seed=seed,
    )
    gates = _fixed_gates()
    result = EVAL._gate_report(intervals, selective, gates)
    assert set(result["gates"]) == EVAL.GATE_NAMES - {"all_seeds_required"}
    assert result["gates"]["selective_80_mae_bpm_max"]["threshold"] == gates[
        "selective_80_mae_bpm_max"
    ]
    assert result["gates"]["conformal_90_marginal_coverage_min"]["operator"] == ">="
