from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_locked_hcs_targets.py"
SPEC = importlib.util.spec_from_file_location("build_locked_hcs_targets", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)
RUN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_locked_hcs_oof.py"
RUN_SPEC = importlib.util.spec_from_file_location("run_locked_hcs_oof_for_targets", RUN_SCRIPT)
assert RUN_SPEC is not None and RUN_SPEC.loader is not None
RUN = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(RUN)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha(path), "bytes": path.stat().st_size}


def _write(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _binding(path)


def _write_json(
    path: Path, document: dict[str, Any], *, content_hash: bool = False
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(document)
    if content_hash:
        payload["content_sha256"] = BUILD.canonical_json_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return _binding(path)


def _write_prediction(
    path: Path, *, fold: int, seed: int, duplicate_index: bool = False
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    index = np.asarray([2 * fold, 2 * fold + 1], dtype=np.int64)
    if duplicate_index and fold == 0 and seed == 101:
        index[1] = index[0]
    fallback = np.asarray([18.0 + fold, 18.5 + fold], dtype=np.float32)
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            cache_index=index,
            outer_fold=np.asarray(fold, dtype=np.int16),
            seed=np.asarray(seed, dtype=np.int64),
            fallback_rr_bpm=fallback,
            source_rr_bpm=fallback + np.float32(0.25),
            final_rr_bpm=fallback,
            applied_pull=np.zeros(2, dtype=np.float32),
            target_joined=np.asarray(False),
        )
    return _binding(path)


def _metadata_header() -> list[str]:
    return [
        "session_id",
        "session_number",
        "identity",
        "protocol",
        "window_number",
        "window_start_s",
        "window_end_s",
        "rr_bpm",
        "reference_valid",
        "reference_quality",
        "reference_sigma_bpm",
        "spectral_concentration",
        "periodicity",
        "estimator_disagreement_bpm",
        "phase_residual_rad",
        "clip_fraction",
        "guard_clip_fraction",
        "plateau_fraction",
        "breath_count",
        "radar_observable",
        "classical_confidence",
        "radar_peak_spread_bpm",
    ]


def _metadata_csv(session: str, identity: str, fold: int, *, duplicate_window: bool) -> str:
    header = _metadata_header()
    rows = []
    for local in range(2):
        row = {
            "session_id": session,
            "session_number": fold + 1,
            "identity": identity,
            "protocol": "paced" if fold % 2 else "rest",
            "window_number": 0 if duplicate_window and local == 1 else local,
            "window_start_s": float(local * 32),
            "window_end_s": float((local + 1) * 32),
            "rr_bpm": 12.0 + fold + local,
            "reference_valid": local == 0,
            "reference_quality": 0.9,
            "reference_sigma_bpm": 0.2,
            "spectral_concentration": 0.8,
            "periodicity": 0.7,
            "estimator_disagreement_bpm": 0.3,
            "phase_residual_rad": 0.1,
            "clip_fraction": 0.0,
            "guard_clip_fraction": 0.0,
            "plateau_fraction": 0.0,
            "breath_count": 8,
            "radar_observable": True,
            "classical_confidence": 0.85,
            "radar_peak_spread_bpm": 0.4,
        }
        rows.append(",".join(str(row[name]) for name in header))
    return ",".join(header) + "\n" + "\n".join(rows) + "\n"


def _fixture(
    tmp_path: Path,
    *,
    duplicate_prediction: bool = False,
    duplicate_window: bool = False,
    bad_manifest_content: bool = False,
    metadata_fold_mismatch: bool = False,
) -> dict[str, Path]:
    root = tmp_path / "locked"
    cache = tmp_path / "cache"
    fold_path = tmp_path / "fold_assignments.json"
    identity_to_fold = {f"id{fold}": fold for fold in range(6)}
    _write_json(
        fold_path,
        {"identity_to_fold": identity_to_fold, "validation_rule": "synthetic"},
    )

    sessions = []
    for fold in range(6):
        session = f"S{fold:02d}_id{fold}"
        identity_fold = (fold + 1) % 6 if metadata_fold_mismatch and fold == 0 else fold
        identity = f"id{identity_fold}"
        metadata = cache / session / "metadata.csv"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(
            _metadata_csv(
                session,
                identity,
                fold,
                duplicate_window=duplicate_window and fold == 0,
            ),
            encoding="utf-8",
        )
        sessions.append(
            {
                "session_id": session,
                "status": "ok",
                "window_count": 2,
                "valid_reference_count": 1,
            }
        )
    cache_manifest = _write_json(cache / "manifest.json", {"sessions": sessions})

    effective_source = _write(tmp_path / "source.py", b"# frozen source\n")
    test_manifests: dict[int, dict[str, Any]] = {}
    for fold in range(6):
        document = {
            "schema_version": 1,
            "fold_id": 100 * fold + 60,
            "fold_assignments": _binding(fold_path),
            "cache": {
                "manifest_path": cache_manifest["path"],
                "manifest_sha256": cache_manifest["sha256"],
            },
            "identities": {
                "train": [f"id{(fold + 2) % 6}"],
                "validation": [f"id{(fold + 1) % 6}"],
                "prediction": [f"id{fold}"],
                "excluded": [],
                "scaler": [f"id{(fold + 2) % 6}"],
            },
        }
        binding = _write_json(
            tmp_path / "test_manifests" / f"outer_{fold}.json",
            document,
            content_hash=True,
        )
        if bad_manifest_content and fold == 5:
            path = Path(binding["path"])
            tampered = json.loads(path.read_text(encoding="utf-8"))
            tampered["content_sha256"] = "0" * 64
            path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
            binding = _binding(path)
        test_manifests[fold] = binding

    plan_units = []
    for seed in (101, 102, 103):
        for fold in range(6):
            unit_root = root / "units" / f"outer_{fold}_seed_{seed}"
            stages = []
            derived_artifacts = {}
            for position, stage in enumerate(BUILD.FAST_STAGES):
                output = unit_root / "work" / f"{position}_{stage}.bin"
                argv = ["python", "synthetic.py", stage, str(fold), str(seed)]
                stages.append({"name": stage, "argv": argv, "outputs": [str(output)]})
                derived_artifacts[f"artifact_{position}"] = str(output)
            plan_units.append(
                {
                    "outer_fold": fold,
                    "seed": seed,
                    "test_manifest": test_manifests[fold],
                    "stages": stages,
                    "derived_artifacts": derived_artifacts,
                }
            )
    plan = {
        "schema_version": 1,
        "classification": "locked_hcs_oof_inference_plan",
        "folds": list(range(6)),
        "seeds": [101, 102, 103],
        "rf_cache_manifest": cache_manifest,
        "effective_sources": {"synthetic_source": effective_source},
        "units": plan_units,
    }
    plan_binding = _write_json(root / "plan.json", plan, content_hash=True)
    pretest = {
        "schema_version": 1,
        "classification": "locked_hcs_oof_all_pretest_assets_sealed",
        "outer_test_opened_before_lock": False,
        "target_artifact_opened": False,
        "unit_count": 18,
        "folds": list(range(6)),
        "seeds": [101, 102, 103],
        "plan": plan_binding,
    }
    _write_json(root / "pretest_lock.json", pretest)
    pretest_sha = _sha(root / "pretest_lock.json")

    seal_units = []
    for seed in (101, 102, 103):
        for fold in range(6):
            unit_root = root / "units" / f"outer_{fold}_seed_{seed}"
            prediction = _write_prediction(
                unit_root / "sealed_label_free_predictions.npz",
                fold=fold,
                seed=seed,
                duplicate_index=duplicate_prediction,
            )
            stage_receipts = []
            commands = []
            artifacts = {}
            for position, stage in enumerate(BUILD.FAST_STAGES):
                output = _write(
                    unit_root / "work" / f"{position}_{stage}.bin",
                    f"{fold}/{seed}/{stage}".encode(),
                )
                log = _write(
                    unit_root / "logs" / f"{position}_{stage}.log", b"complete\n"
                )
                argv = ["python", "synthetic.py", stage, str(fold), str(seed)]
                receipt = {
                    "schema_version": 1,
                    "classification": "locked_hcs_oof_stage_receipt",
                    "stage": stage,
                    "argv": argv,
                    "outputs": [output],
                    "stdout_stderr_log": log,
                }
                stage_receipts.append(
                    _write_json(
                        unit_root / "receipts" / f"{position:02d}_{stage}.json",
                        receipt,
                    )
                )
                commands.append({"stage": stage, "argv": argv})
                artifacts[f"artifact_{position}"] = output
            derived = {
                "schema_version": 1,
                "classification": "locked_hcs_oof_derived_test_inference",
                "outer_fold": fold,
                "seed": seed,
                "target_artifact_opened": False,
                "pretest_lock_sha256": pretest_sha,
                "stage_receipts": stage_receipts,
                "commands": commands,
                "derived_artifacts": artifacts,
                "sealed_prediction": prediction,
                "test_manifest": test_manifests[fold],
            }
            derived_binding = _write_json(
                unit_root / "derived_inference_lock.json", derived
            )
            seal_units.append(
                {
                    "outer_fold": fold,
                    "seed": seed,
                    "derived_lock": derived_binding,
                    "prediction": prediction,
                }
            )
    seal = {
        "schema_version": 1,
        "classification": "locked_hcs_oof_all_label_free_predictions_sealed",
        "pretest_lock_sha256": pretest_sha,
        "unit_count": 18,
        "outer_folds": list(range(6)),
        "target_artifact_opened_before_seal": False,
        "target_join_authorized": True,
        "units": seal_units,
    }
    _write_json(root / "predictions_seal.json", seal)
    return {
        "root": root,
        "cache": cache,
        "folds": fold_path,
        "output": tmp_path / "published" / "targets.npz",
        "receipt": tmp_path / "published" / "targets_receipt.json",
    }


def _build(paths: dict[str, Path]) -> dict[str, Any]:
    return BUILD.build_targets(
        locked_oof_root=paths["root"],
        cache_dir=paths["cache"],
        fold_assignments=paths["folds"],
        output=paths["output"],
        receipt=paths["receipt"],
        orchestrator_command=["synthetic-build"],
        _release_capability=BUILD._RELEASE_AUTHORIZATION_CAPABILITY,
    )


def test_direct_target_builder_is_disabled_before_any_target_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    opened = False

    def forbidden_target_load(**_kwargs: Any) -> Any:
        nonlocal opened
        opened = True
        raise AssertionError("target metadata must not be opened")

    monkeypatch.setattr(BUILD, "_load_cache_targets", forbidden_target_load)
    with pytest.raises(BUILD.TargetBuildError, match="release_lock"):
        BUILD.build_targets(
            locked_oof_root=paths["root"],
            cache_dir=paths["cache"],
            fold_assignments=paths["folds"],
            output=paths["output"],
            receipt=paths["receipt"],
        )
    assert opened is False
    assert not paths["output"].exists()


def test_builds_join_compatible_immutable_targets_with_qc_lineage(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    receipt = _build(paths)

    assert receipt["commercial_claim_authorized"] is False
    assert receipt["prospective_confirmation_required"] is True
    assert receipt["prediction_seal_verified_before_any_target_metadata_access"] is True
    assert receipt["target_metadata_opened_only_after_complete_prediction_seal_verification"] is True
    assert receipt["row_count"] == 12
    assert receipt["valid_reference_rows"] == 6
    assert receipt["content_sha256"] == BUILD.canonical_json_sha256(
        {key: value for key, value in receipt.items() if key != "content_sha256"}
    )
    assert len(receipt["prediction_inventory"]) == 18
    assert len(receipt["source_bindings"]["metadata_files"]) == 6
    assert (paths["output"].stat().st_mode & 0o777) == 0o444
    assert (paths["receipt"].stat().st_mode & 0o777) == 0o444

    with np.load(paths["output"], allow_pickle=False) as archive:
        required = {
            "cache_index",
            "outer_fold",
            "target_rr_bpm",
            "identity",
            "reference_valid",
        }
        assert required.issubset(archive.files)
        assert {
            "session_id",
            "window_number",
            "protocol",
            "window_start_s",
            "window_end_s",
            "reference_quality",
            "reference_sigma_bpm",
            "radar_observable",
            "cache_session_position",
            "cache_session_row",
        }.issubset(archive.files)
        assert archive["cache_index"].tolist() == list(range(12))
        assert archive["outer_fold"].tolist() == [fold for fold in range(6) for _ in range(2)]
        assert archive["reference_valid"].tolist() == [True, False] * 6
        assert np.isfinite(archive["target_rr_bpm"]).all()
        for name in archive.files:
            schema = receipt["target_schema"][name]
            assert schema["dtype"] == archive[name].dtype.str
            assert schema["shape"] == list(archive[name].shape)
            assert schema["array_sha256"] == BUILD.array_sha256(archive[name])

    evaluation = RUN.join_and_evaluate(paths["root"], paths["output"])
    assert evaluation["target_join_count"] == 1
    assert evaluation["commercial_claim_authorized"] is False


def test_missing_or_incomplete_seal_never_opens_target_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    seal_path = paths["root"] / "predictions_seal.json"
    seal_path.unlink()
    opened = False

    def forbidden(**_: Any) -> Any:
        nonlocal opened
        opened = True
        raise AssertionError("target metadata was opened")

    monkeypatch.setattr(BUILD, "_load_cache_targets", forbidden)
    with pytest.raises(BUILD.TargetBuildError, match="predictions_seal"):
        _build(paths)
    assert opened is False
    assert not paths["output"].exists()


def test_late_prediction_tamper_fails_before_metadata_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    prediction = sorted(paths["root"].glob("units/*/sealed_label_free_predictions.npz"))[-1]
    prediction.chmod(0o644)
    prediction.write_bytes(b"tampered")
    opened = False

    def forbidden(**_: Any) -> Any:
        nonlocal opened
        opened = True
        raise AssertionError("target metadata was opened")

    monkeypatch.setattr(BUILD, "_load_cache_targets", forbidden)
    with pytest.raises(BUILD.TargetBuildError, match="binding mismatch"):
        _build(paths)
    assert opened is False


def test_declared_manifest_content_hash_is_verified_before_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, bad_manifest_content=True)
    monkeypatch.setattr(
        BUILD,
        "_load_cache_targets",
        lambda **_: pytest.fail("target metadata must remain unopened"),
    )
    with pytest.raises(BUILD.TargetBuildError, match="content_sha256 mismatch"):
        _build(paths)


def test_duplicate_prediction_index_is_rejected_before_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, duplicate_prediction=True)
    monkeypatch.setattr(
        BUILD,
        "_load_cache_targets",
        lambda **_: pytest.fail("target metadata must remain unopened"),
    )
    with pytest.raises(BUILD.TargetBuildError, match="unique, and sorted"):
        _build(paths)


def test_duplicate_metadata_semantics_and_fold_mismatch_fail_closed(tmp_path: Path) -> None:
    duplicate = _fixture(tmp_path / "duplicate", duplicate_window=True)
    with pytest.raises(BUILD.TargetBuildError, match="duplicate session/window"):
        _build(duplicate)
    assert not duplicate["output"].exists()

    mismatch = _fixture(tmp_path / "mismatch", metadata_fold_mismatch=True)
    with pytest.raises(BUILD.TargetBuildError, match="identity cover mismatch|ownership mismatch"):
        _build(mismatch)
    assert not mismatch["output"].exists()


def test_receipt_output_hash_and_immutable_no_overwrite_are_enforced(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    unit = sorted(paths["root"].glob("units/*"))[0]
    stage_output = sorted((unit / "work").glob("*.bin"))[0]
    stage_output.write_bytes(b"tampered stage output")
    with pytest.raises(BUILD.TargetBuildError, match="binding mismatch"):
        _build(paths)
    assert not paths["output"].exists()

    clean = _fixture(tmp_path / "clean")
    first = _build(clean)
    first_target = clean["output"].read_bytes()
    first_receipt = clean["receipt"].read_bytes()
    with pytest.raises(BUILD.TargetBuildError, match="overwrite forbidden"):
        _build(clean)
    assert clean["output"].read_bytes() == first_target
    assert clean["receipt"].read_bytes() == first_receipt
    assert first["target_artifact"]["sha256"] == _sha(clean["output"])


def test_surplus_stage_receipt_output_is_rejected_before_target_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    unit_root = sorted(paths["root"].glob("units/*"))[0]
    receipt_path = sorted((unit_root / "receipts").glob("*.json"))[0]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    surplus_path = unit_root / "work/surplus.bin"
    surplus_path.write_bytes(b"surplus")
    receipt["outputs"].append(_binding(surplus_path))
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    derived_path = unit_root / "derived_inference_lock.json"
    derived = json.loads(derived_path.read_text(encoding="utf-8"))
    derived["stage_receipts"][0] = _binding(receipt_path)
    derived_path.write_text(json.dumps(derived, sort_keys=True), encoding="utf-8")
    prediction_seal_path = paths["root"] / "predictions_seal.json"
    prediction_seal = json.loads(prediction_seal_path.read_text(encoding="utf-8"))
    matching = next(
        item
        for item in prediction_seal["units"]
        if Path(item["derived_lock"]["path"]) == derived_path.resolve()
    )
    matching["derived_lock"] = _binding(derived_path)
    prediction_seal_path.write_text(
        json.dumps(prediction_seal, sort_keys=True), encoding="utf-8"
    )
    monkeypatch.setattr(
        BUILD,
        "_load_cache_targets",
        lambda **_: pytest.fail("target metadata must remain unopened"),
    )
    with pytest.raises(BUILD.TargetBuildError, match="output topology"):
        _build(paths)
    assert not paths["output"].exists()


def test_preexisting_partial_destination_is_never_replaced(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    paths["output"].write_bytes(b"operator-owned partial")
    before = paths["output"].read_bytes()
    with pytest.raises(BUILD.TargetBuildError, match="overwrite forbidden"):
        _build(paths)
    assert paths["output"].read_bytes() == before
    assert not paths["receipt"].exists()
