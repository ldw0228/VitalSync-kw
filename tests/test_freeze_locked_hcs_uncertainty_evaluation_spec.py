from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/freeze_locked_hcs_uncertainty_evaluation_spec.py"
SPEC = importlib.util.spec_from_file_location(
    "freeze_locked_hcs_uncertainty_evaluation_spec", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
FREEZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FREEZE)


def _write_json(path: Path, value: dict[str, Any], *, content_hash: bool = False) -> None:
    document = dict(value)
    if content_hash:
        document["content_sha256"] = FREEZE.canonical_json_sha256(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _gates() -> dict[str, Any]:
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


class Fixture:
    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path / "locked"
        self.primary = tmp_path / "primary_spec.json"
        primary = FREEZE.PRIMARY.evaluation_spec_document(
            expected_rows=12,
            expected_identities=2,
            bootstrap_samples=25,
            bootstrap_seed=73,
            bootstrap_confidence=0.90,
        )
        _write_json(self.primary, primary, content_hash=True)
        units = []
        for fold in FREEZE.FOLDS:
            for seed in FREEZE.PRIMARY.FIXED_SEEDS:
                units.append(
                    {
                        "outer_fold": fold,
                        "seed": seed,
                        "interval_calibration": {
                            f"{coverage:.2f}": {
                                "nominal_coverage": coverage,
                                "normalized_absolute_error_quantile": 1.25,
                            }
                            for coverage in FREEZE.INTERVAL_COVERAGES
                        },
                        "selective_thresholds": {
                            f"{coverage:.2f}": {
                                "intended_acceptance_coverage": coverage,
                                "rr_std_threshold_bpm": 1.0,
                            }
                            for coverage in FREEZE.SELECTIVE_COVERAGES
                        },
                    }
                )
        self.calibration = tmp_path / "calibration.json"
        _write_json(
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
                "folds": list(FREEZE.FOLDS),
                "seeds": list(FREEZE.PRIMARY.FIXED_SEEDS),
                "unit_count": 18,
                "fixed_method": {
                    "phase_modulus": 8,
                    "phase_value": 0,
                    "std_floor_bpm": 0.25,
                    "interval_coverages": list(FREEZE.INTERVAL_COVERAGES),
                    "selective_coverages": list(FREEZE.SELECTIVE_COVERAGES),
                    "no_test_time_fit_or_threshold_selection": True,
                    "formal_exchangeability_claim": False,
                },
                "fixed_evaluation_gates": _gates(),
                "units": units,
            },
            content_hash=True,
        )
        self.output = tmp_path / "uncertainty_spec.json"

    def freeze(self, *, source_paths: dict[str, Path] | None = None) -> dict[str, Any]:
        return FREEZE.freeze_uncertainty_evaluation_spec(
            output_path=self.output,
            locked_oof_root=self.root,
            primary_spec_path=self.primary,
            calibration_path=self.calibration,
            source_paths=source_paths,
        )

    def load(self, *, source_paths: dict[str, Path] | None = None) -> tuple[Any, ...]:
        return FREEZE.load_uncertainty_evaluation_spec(
            self.output,
            expected_primary_spec_path=self.primary,
            expected_calibration_path=self.calibration,
            source_paths=source_paths,
        )


def test_freezes_create_once_0444_separate_secondary_protocol_with_exact_methods(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    binding = fixture.freeze()
    document, loaded_binding = fixture.load()
    assert binding == loaded_binding
    assert fixture.output.stat().st_mode & 0o777 == 0o444
    assert document["protocol_relationship"] == {
        "role": "separate_secondary_retrospective_engineering_protocol",
        "primary_uncertainty_contract_overridden": False,
        "primary_point_evaluation_or_gates_modified": False,
        "primary_diagnostic_only_uncertainty_claim_preserved": True,
        "secondary_interval_results_are_part_of_primary_evaluation": False,
        "formal_conformal_exchangeability_or_coverage_guarantee_claimed": False,
        "reason": "all interval scales and selective thresholds were frozen in a separate identity-disjoint pretest calibration before target access",
    }
    assert document["target_or_target_bearing_artifact_opened_to_build_spec"] is False
    assert document["outer_test_prediction_or_uncertainty_artifact_opened_to_build_spec"] is False
    assert document["fixed_methods"]["normal_uncalibrated"]["coverages"] == [
        0.5,
        0.8,
        0.9,
        0.95,
    ]
    assert document["fixed_methods"]["normalized_conformal"]["std_floor_bpm"] == 0.25
    assert document["fixed_methods"]["selective"][
        "intended_acceptance_coverages"
    ] == [0.5, 0.8, 0.9, 1.0]
    assert document["fixed_phases"]["phases"] == list(range(8))
    assert document["fixed_evaluation_gates"] == _gates()
    assert set(document["bound_inputs"]["implementation_sources"]) == set(
        FREEZE.DEFAULT_SOURCE_PATHS
    )
    with pytest.raises(FREEZE.UncertaintyEvaluationSpecError, match="already exists"):
        fixture.freeze()


def test_refuses_freeze_after_any_canonical_target_boundary_exists(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    fixture.root.mkdir(parents=True)
    (fixture.root / "evaluation_lock.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FREEZE.UncertaintyEvaluationSpecError, match="before target"):
        fixture.freeze()
    assert not fixture.output.exists()


def test_load_rejects_completed_calibration_drift(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    fixture.freeze()
    fixture.calibration.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FREEZE.UncertaintyEvaluationSpecError):
        fixture.load()


def test_load_rejects_bound_implementation_source_drift(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    source_root = tmp_path / "sources"
    source_paths: dict[str, Path] = {}
    for name, source in FREEZE.DEFAULT_SOURCE_PATHS.items():
        destination = source_root / f"{name}{source.suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        source_paths[name] = destination
    fixture.freeze(source_paths=source_paths)
    with source_paths["uncertainty_evaluator"].open("ab") as stream:
        stream.write(b"\n# drift\n")
    with pytest.raises(FREEZE.UncertaintyEvaluationSpecError, match="drifted"):
        fixture.load(source_paths=source_paths)


def test_load_rejects_method_or_gate_drift_even_with_recomputed_content_hash(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.freeze()
    fixture.output.chmod(0o644)
    document = json.loads(fixture.output.read_text(encoding="utf-8"))
    document.pop("content_sha256")
    document["fixed_methods"]["normal_uncalibrated"]["coverages"] = [0.9]
    document["fixed_evaluation_gates"]["selective_80_mae_bpm_max"] = 99.0
    _write_json(fixture.output, document, content_hash=True)
    fixture.output.chmod(0o444)
    with pytest.raises(FREEZE.UncertaintyEvaluationSpecError, match="drifted"):
        fixture.load()
