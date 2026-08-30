from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/seal_locked_hcs_uncertainty_inputs.py"
SPEC = importlib.util.spec_from_file_location("seal_locked_hcs_uncertainty_inputs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SEAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEAL)


def _write_json(path: Path, value: dict[str, Any], *, content_hash: bool = False) -> None:
    document = dict(value)
    if content_hash:
        document["content_sha256"] = SEAL.canonical_content_sha256(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class Fixture:
    def __init__(self, tmp_path: Path, *, add_forbidden: bool = False) -> None:
        self.root = tmp_path / "locked"
        self.seeds = (101, 102, 103)
        calibration_units = [
            {"outer_fold": fold, "seed": seed}
            for fold in SEAL.FOLDS
            for seed in self.seeds
        ]
        self.calibration = tmp_path / "calibration.json"
        _write_json(
            self.calibration,
            {
                "schema_version": 1,
                "classification": "locked_pretest_cross_fitted_proposer_uncertainty_calibration",
                "commercial_claim_authorized": False,
                "prospective_confirmation_required": True,
                "outer_test_opened": False,
                "target_artifact_opened": False,
                "point_prediction_modified": False,
                "unit_count": 18,
                "units": calibration_units,
            },
            content_hash=True,
        )
        prediction_units = []
        self.raw_paths: list[Path] = []
        for seed in self.seeds:
            for fold in SEAL.FOLDS:
                unit_root = self.root / "units" / f"outer_{fold}_seed_{seed}"
                raw_path = unit_root / "raw_hcs_prediction.npz"
                point_path = unit_root / "sealed_label_free_predictions.npz"
                index = np.arange(fold * 10, fold * 10 + 4, dtype=np.int64)
                fallback = np.asarray([12.0, 13.0, 14.0, 15.0], dtype=np.float32) + fold
                raw_arrays: dict[str, Any] = {
                    "cache_index": index,
                    "fallback_rr_bpm": fallback,
                    "fallback_std_bpm": np.full(4, 0.5 + fold / 10, dtype=np.float32),
                    "fallback_available": np.ones(4, dtype=bool),
                    "source_rr_bpm": fallback + np.float32(0.25),
                    "source_scale_bpm": np.full(4, 0.8, dtype=np.float32),
                    "source_available": np.ones(4, dtype=bool),
                    "selected_probability": np.full(4, 0.7, dtype=np.float32),
                    "margin": np.full(4, 0.2, dtype=np.float32),
                    "entropy": np.full(4, 0.4, dtype=np.float32),
                    "quality": np.full(4, 0.9, dtype=np.float32),
                    "valid_candidate_count": np.full(4, 3, dtype=np.int16),
                    "normalized_entropy": np.full(4, 0.3, dtype=np.float32),
                }
                if add_forbidden and seed == self.seeds[0] and fold == 0:
                    raw_arrays["target_rr_bpm"] = fallback.copy()
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(raw_path, **raw_arrays)
                np.savez_compressed(
                    point_path,
                    cache_index=index,
                    outer_fold=np.asarray(fold, dtype=np.int16),
                    seed=np.asarray(seed, dtype=np.int64),
                    fallback_rr_bpm=fallback,
                    source_rr_bpm=fallback + np.float32(0.25),
                    final_rr_bpm=fallback.copy(),
                    applied_pull=np.zeros(4, dtype=np.float32),
                    target_joined=np.asarray(False),
                )
                derived_path = unit_root / "derived_inference_lock.json"
                _write_json(
                    derived_path,
                    {
                        "schema_version": 1,
                        "classification": "locked_hcs_oof_derived_test_inference",
                        "outer_fold": fold,
                        "seed": seed,
                        "target_artifact_opened": False,
                        "frozen_policy_status": "fail_closed_no_action",
                        "no_action_bit_exact_float32_fallback": True,
                        "derived_artifacts": {
                            "raw_hcs_prediction": SEAL.bind_file(raw_path)
                        },
                        "sealed_prediction": SEAL.bind_file(point_path),
                    },
                )
                prediction_units.append(
                    {
                        "outer_fold": fold,
                        "seed": seed,
                        "derived_lock": SEAL.bind_file(derived_path),
                        "prediction": SEAL.bind_file(point_path),
                    }
                )
                self.raw_paths.append(raw_path)
        _write_json(
            self.root / "predictions_seal.json",
            {
                "schema_version": 1,
                "classification": "locked_hcs_oof_all_label_free_predictions_sealed",
                "target_artifact_opened_before_seal": False,
                "target_join_authorized": True,
                "unit_count": 18,
                "units": prediction_units,
            },
        )
        self.output = self.root / "locked_hcs_uncertainty_inputs.npz"
        self.seal = self.root / "uncertainty_inputs_seal.json"

    def run(self) -> dict[str, Any]:
        return SEAL.seal_uncertainty_inputs(
            root=self.root,
            calibration_path=self.calibration,
            output_path=self.output,
            seal_path=self.seal,
        )


def test_seals_exact_18_unit_target_free_cover_and_bit_parity(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    result = fixture.run()
    assert result["unit_count"] == 18
    assert result["row_count"] == 72
    assert result["rows_per_seed"] == 24
    assert result["target_fields_present"] is False
    assert result["no_action_primary_bit_exact_verified"] is True
    assert result["content_sha256"] == SEAL.canonical_content_sha256(result)
    assert fixture.output.stat().st_mode & 0o777 == 0o444
    assert fixture.seal.stat().st_mode & 0o777 == 0o444
    with np.load(fixture.output, allow_pickle=False) as archive:
        assert set(archive.files) == set(result["array_schema"])
        assert np.array_equal(
            archive["final_rr_bpm"].view(np.uint32),
            np.repeat(
                np.concatenate(
                    [
                        (np.asarray([12, 13, 14, 15], dtype=np.float32) + fold)
                        for fold in SEAL.FOLDS
                    ]
                )[None, :],
                3,
                axis=0,
            ).reshape(-1).view(np.uint32),
        )


def test_existing_seal_is_idempotent_and_rehashes_archive(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    first = fixture.run()
    second = fixture.run()
    assert first == second
    fixture.output.chmod(0o644)
    with fixture.output.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(SEAL.UncertaintySealError, match="hash mismatch"):
        fixture.run()


def test_recovers_exact_archive_after_crash_before_seal_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Fixture(tmp_path)
    publish_json = SEAL._atomic_json

    def crash_before_seal(path: Path, value: dict[str, Any]) -> None:
        del path, value
        raise RuntimeError("synthetic crash before seal publication")

    monkeypatch.setattr(SEAL, "_atomic_json", crash_before_seal)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        fixture.run()
    assert fixture.output.exists()
    assert fixture.output.stat().st_mode & 0o777 == 0o444
    assert not fixture.seal.exists()

    monkeypatch.setattr(SEAL, "_atomic_json", publish_json)
    recovered = fixture.run()
    assert recovered["uncertainty_archive"] == SEAL.bind_file(fixture.output)
    assert fixture.seal.exists()


def test_rejects_mismatched_unsealed_archive_instead_of_attesting_it(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fixture.output, forged=np.asarray([1], dtype=np.int64))
    fixture.output.chmod(0o444)
    with pytest.raises(SEAL.UncertaintySealError, match="field topology differs"):
        fixture.run()
    assert not fixture.seal.exists()


def test_rejects_symlink_uncertainty_output_before_publication(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    real = tmp_path / "real_archive.npz"
    np.savez_compressed(real, forged=np.asarray([1], dtype=np.int64))
    fixture.output.parent.mkdir(parents=True, exist_ok=True)
    fixture.output.symlink_to(real)
    with pytest.raises(SEAL.UncertaintySealError, match="must not be a symlink"):
        fixture.run()
    assert not fixture.seal.exists()


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("schema_version", 2),
        ("classification", "wrong_uncertainty_seal"),
        ("commercial_claim_authorized", True),
        ("prospective_confirmation_required", False),
        ("target_artifact_opened_before_seal", True),
        ("target_fields_present", True),
        ("point_prediction_modified", True),
        ("no_action_primary_bit_exact_verified", False),
    ),
)
def test_existing_seal_rejects_changed_safety_and_policy_flags(
    tmp_path: Path, field: str, invalid: Any
) -> None:
    fixture = Fixture(tmp_path)
    fixture.run()
    document = json.loads(fixture.seal.read_text(encoding="utf-8"))
    document[field] = invalid
    document["content_sha256"] = SEAL.canonical_content_sha256(document)
    fixture.seal.chmod(0o644)
    _write_json(fixture.seal, document)
    fixture.seal.chmod(0o444)
    with pytest.raises(SEAL.UncertaintySealError, match="invariants are invalid"):
        fixture.run()


def test_existing_seal_revalidates_live_derived_no_action_policy(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    fixture.run()
    derived_path = fixture.root / "units/outer_0_seed_101/derived_inference_lock.json"
    derived = json.loads(derived_path.read_text(encoding="utf-8"))
    derived["frozen_policy_status"] = "action_allowed"
    _write_json(derived_path, derived)

    predictions_path = fixture.root / "predictions_seal.json"
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    target_unit = next(
        unit
        for unit in predictions["units"]
        if unit["outer_fold"] == 0 and unit["seed"] == 101
    )
    target_unit["derived_lock"] = SEAL.bind_file(derived_path)
    _write_json(predictions_path, predictions)

    uncertainty = json.loads(fixture.seal.read_text(encoding="utf-8"))
    uncertainty["predictions_seal"] = SEAL.bind_file(predictions_path)
    uncertainty["content_sha256"] = SEAL.canonical_content_sha256(uncertainty)
    fixture.seal.chmod(0o644)
    _write_json(fixture.seal, uncertainty)
    fixture.seal.chmod(0o444)
    with pytest.raises(SEAL.UncertaintySealError, match="derived lock invariants"):
        fixture.run()


def test_rejects_any_raw_target_field_before_publication(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path, add_forbidden=True)
    with pytest.raises(SEAL.UncertaintySealError, match="forbidden"):
        fixture.run()
    assert not fixture.output.exists()
    assert not fixture.seal.exists()


def test_rejects_raw_artifact_tamper(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    with fixture.raw_paths[0].open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(SEAL.UncertaintySealError, match="binding hash mismatch"):
        fixture.run()


def test_rejects_point_prediction_not_bit_exact_fallback(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    prediction = fixture.root / "units/outer_0_seed_101/sealed_label_free_predictions.npz"
    with np.load(prediction, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    arrays["final_rr_bpm"][0] += np.float32(0.1)
    np.savez_compressed(prediction, **arrays)
    # Update both bindings so this test reaches the semantic parity gate.
    derived_path = fixture.root / "units/outer_0_seed_101/derived_inference_lock.json"
    derived = json.loads(derived_path.read_text(encoding="utf-8"))
    derived["sealed_prediction"] = SEAL.bind_file(prediction)
    _write_json(derived_path, derived)
    seal_path = fixture.root / "predictions_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    for unit in seal["units"]:
        if unit["outer_fold"] == 0 and unit["seed"] == 101:
            unit["prediction"] = SEAL.bind_file(prediction)
            unit["derived_lock"] = SEAL.bind_file(derived_path)
    _write_json(seal_path, seal)
    with pytest.raises(SEAL.UncertaintySealError, match="bit exact"):
        fixture.run()


def test_rejects_if_canonical_target_already_exists(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    target = fixture.root / "canonical_locked_hcs_targets.npz"
    target.write_bytes(b"already opened")
    with pytest.raises(SEAL.UncertaintySealError, match="before target"):
        fixture.run()
