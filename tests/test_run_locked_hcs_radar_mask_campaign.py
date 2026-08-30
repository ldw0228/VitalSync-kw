from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_locked_hcs_radar_mask_campaign.py"
)
SPEC = importlib.util.spec_from_file_location("run_locked_hcs_radar_mask_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


SEEDS = [20260828, 20260829, 20260830]


def test_python_launcher_binding_preserves_virtualenv_symlink(tmp_path: Path) -> None:
    launcher = tmp_path / "venv-python"
    launcher.symlink_to(Path(sys.executable))
    binding = RUN.bind_python_launcher(launcher)
    assert binding["path"] == str(launcher.absolute())
    assert Path(binding["path"]) != launcher.resolve()
    assert binding["sha256"] == RUN.sha256_file(launcher.resolve())


def _write(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return RUN.bind_file(path)


def _write_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return RUN.bind_file(path)


def _write_npz(path: Path, arrays: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    return RUN.bind_file(path)


def _proposer_arrays(fold: int, seed: int, mask: str) -> dict[str, Any]:
    offset = 0.0 if mask == "radars_123" else (list(RUN.MASKS).index(mask) / 10.0)
    prediction = np.asarray(
        [np.float32(20.0 + fold + (seed - SEEDS[0]) / 1000.0 + offset)],
        dtype=np.float32,
    )
    return {
        "cache_index": np.asarray([fold], dtype=np.int64),
        "prediction": prediction,
        "rr_std": np.asarray([1.25], dtype=np.float32),
        "quality": np.asarray([0.75], dtype=np.float32),
        "target_fields_present": np.asarray(False),
        "radar_mask_name": np.asarray(mask),
        "radar_mask_pattern": np.asarray(RUN.MASKS[mask], dtype=bool),
    }


def _raw_arrays(proposer: dict[str, np.ndarray], fold: int, seed: int) -> dict[str, Any]:
    index = np.asarray(proposer["cache_index"], dtype=np.int64)
    prediction = np.asarray(proposer["prediction"], dtype=np.float32)
    rr_std = np.asarray(proposer["rr_std"], dtype=np.float32)
    rows = len(index)
    return {
        "cache_index": index,
        "fallback_rr_bpm": prediction.copy(),
        "fallback_std_bpm": rr_std.copy(),
        "fallback_available": np.ones(rows, dtype=bool),
        "source_rr_bpm": prediction.copy(),
        "source_scale_bpm": rr_std.copy(),
        "source_available": np.ones(rows, dtype=bool),
        "selected_probability": np.zeros(rows, dtype=np.float32),
        "margin": np.zeros(rows, dtype=np.float32),
        "entropy": np.zeros(rows, dtype=np.float32),
        "quality": np.zeros(rows, dtype=np.float32),
        "valid_candidate_count": np.ones(rows, dtype=np.int16),
        "normalized_entropy": np.zeros(rows, dtype=np.float32),
        "outer_fold": np.asarray(fold, dtype=np.int16),
        "seed": np.asarray(seed, dtype=np.int64),
        "target_fields_present": np.asarray(False),
        "source_is_no_action_placeholder": np.asarray(True),
    }


def _policy_payload() -> dict[str, Any]:
    return {
        "selection_status": "fail_closed_no_action",
        "correction_pull": 0.0,
        "probability_threshold": 1.1,
        "margin_threshold": 1.1,
        "entropy_threshold": 0.0,
        "quality_threshold": 1.1,
        "minimum_valid_candidates": 1,
        "base_std_max": None,
        "source_scale_max": None,
        "disagreement_min": None,
        "disagreement_max": None,
    }


def _fixture(tmp_path: Path) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    primary = tmp_path / "primary"
    output = tmp_path / "radar_campaign"
    helper = Path(RUN.SAFE_INPUTS.__file__).resolve()
    trainer = tmp_path / "sealed_train.py"
    trainer.write_text("# sealed trainer dependency\n", encoding="utf-8")
    rf_cache = tmp_path / "rf_cache"
    rf_manifest = _write_json(rf_cache / "manifest.json", {"target_fields_present": False})
    policy_payload = _policy_payload()
    policy = _write_json(
        tmp_path / "policy.json",
        {"schema_version": 1, "outer_test_opened": False, "policy": policy_payload},
    )

    units = []
    artifact_rows: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for seed in SEEDS:
        for fold in range(6):
            unit_root = primary / "units" / f"outer_{fold}_seed_{seed}"
            checkpoint = _write(unit_root / "work/proposer.pt", f"{fold}/{seed}".encode())
            run_config = _write_json(
                unit_root / "work/run_config.json",
                {"fold": fold, "seed": seed, "target_fields_present": False},
            )
            manifest = _write_json(
                tmp_path / "manifests" / f"outer_{fold}_seed_{seed}.json",
                {"fold": fold, "seed": seed, "prediction_partition_only": True},
            )
            proposer_arrays = _proposer_arrays(fold, seed, "radars_123")
            proposer = _write_npz(unit_root / "work/test_proposer_safe.npz", proposer_arrays)
            raw_arrays = _raw_arrays(proposer_arrays, fold, seed)
            raw = _write_npz(unit_root / "work/no_action_raw_hcs.npz", raw_arrays)
            sealed_arrays = RUN.LOCKED_OOF._sealed_prediction_arrays(
                raw_arrays, fold=fold, seed=seed, policy=policy_payload
            )
            sealed = _write_npz(
                unit_root / "sealed_label_free_predictions.npz", sealed_arrays
            )
            proposer_argv = [
                str(Path(sys.executable).resolve()),
                str(helper.resolve()),
                "proposer-predict",
                "--cache-dir",
                str(rf_cache.resolve()),
                "--checkpoint",
                checkpoint["path"],
                "--run-config",
                run_config["path"],
                "--test-manifest",
                manifest["path"],
                "--output",
                proposer["path"],
                "--device",
                "cpu",
                "--batch-size",
                "128",
            ]
            units.append(
                {
                    "outer_fold": fold,
                    "seed": seed,
                    "stages": [
                        {
                            "name": "test_proposer_bind",
                            "argv": ["synthetic-bind", str(fold), str(seed)],
                            "outputs": [checkpoint["path"], run_config["path"]],
                        },
                        {
                            "name": "test_proposer_predict",
                            "argv": proposer_argv,
                            "outputs": [proposer["path"]],
                        },
                        {
                            "name": "no_action_fallback_adapter",
                            "argv": ["synthetic"],
                            "outputs": [raw["path"]],
                        },
                    ],
                    "derived_artifacts": {
                        "test_proposer_checkpoint": checkpoint["path"],
                        "test_proposer_prediction": proposer["path"],
                        "raw_hcs_prediction": raw["path"],
                    },
                }
            )
            artifact_rows[(fold, seed)] = {
                "checkpoint": checkpoint,
                "run_config": run_config,
                "manifest": manifest,
                "proposer": proposer,
                "raw": raw,
                "sealed": sealed,
            }

    oof_source = Path(RUN.LOCKED_OOF.__file__).resolve()
    plan_document = {
        "schema_version": 1,
        "classification": "locked_hcs_oof_inference_plan",
        "folds": list(range(6)),
        "seeds": SEEDS,
        "common": {"policy": policy},
        "rf_cache_manifest": rf_manifest,
        "effective_sources": {
            "plan_builder": RUN.bind_file(oof_source),
            "safe_test_input_helper": RUN.bind_file(helper),
            "python_executable": RUN.bind_file(Path(sys.executable)),
            "proposer_trainer": RUN.bind_file(trainer),
        },
        "units": units,
    }
    plan_path = primary / "locked_oof_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan_document, sort_keys=True), encoding="utf-8")
    pretest = {
        "schema_version": 1,
        "classification": "locked_hcs_oof_all_pretest_assets_sealed",
        "plan": RUN.bind_file(plan_path),
        "target_artifact_opened": False,
    }
    pretest_binding = _write_json(primary / "pretest_lock.json", pretest)

    seal_units = []
    for seed in SEEDS:
        for fold in range(6):
            artifacts = artifact_rows[(fold, seed)]
            unit_root = primary / "units" / f"outer_{fold}_seed_{seed}"
            plan_unit = next(
                unit
                for unit in units
                if unit["outer_fold"] == fold and unit["seed"] == seed
            )
            stage_receipts = []
            for position, stage in enumerate(plan_unit["stages"]):
                log = _write(
                    unit_root / "logs" / f"{position:02d}_{stage['name']}.log",
                    b"synthetic complete\n",
                )
                receipt = {
                    "schema_version": 1,
                    "classification": "locked_hcs_oof_stage_receipt",
                    "stage": stage["name"],
                    "argv": stage["argv"],
                    "outputs": [RUN.bind_file(Path(value)) for value in stage["outputs"]],
                    "stdout_stderr_log": log,
                }
                stage_receipts.append(
                    _write_json(
                        unit_root
                        / "receipts"
                        / f"{position:02d}_{stage['name']}.json",
                        receipt,
                    )
                )
            derived = {
                "schema_version": 1,
                "classification": "locked_hcs_oof_derived_test_inference",
                "outer_fold": fold,
                "seed": seed,
                "target_artifact_opened": False,
                "pretest_lock_sha256": pretest_binding["sha256"],
                "frozen_policy_status": "fail_closed_no_action",
                "stage_receipts": stage_receipts,
                "test_manifest": artifacts["manifest"],
                "derived_artifacts": {
                    "test_proposer_checkpoint": artifacts["checkpoint"],
                    "test_proposer_prediction": artifacts["proposer"],
                    "raw_hcs_prediction": artifacts["raw"],
                },
                "sealed_prediction": artifacts["sealed"],
            }
            derived_binding = _write_json(
                unit_root / "derived_inference_lock.json",
                derived,
            )
            seal_units.append(
                {
                    "outer_fold": fold,
                    "seed": seed,
                    "derived_lock": derived_binding,
                    "prediction": artifacts["sealed"],
                }
            )
    prediction_seal = {
        "schema_version": 1,
        "classification": "locked_hcs_oof_all_label_free_predictions_sealed",
        "pretest_lock_sha256": pretest_binding["sha256"],
        "unit_count": 18,
        "outer_folds": list(range(6)),
        "target_artifact_opened_before_seal": False,
        "target_join_authorized": True,
        "units": seal_units,
    }
    _write_json(primary / "predictions_seal.json", prediction_seal)
    return {
        "primary": primary,
        "plan": plan_path,
        "output": output,
        "helper": helper,
        "oof_source": oof_source,
        "artifacts": artifact_rows,
    }


def _argument(argv: list[str], name: str) -> str:
    position = argv.index(name)
    return argv[position + 1]


def _fake_runner(
    calls: list[list[str]], *, full_mask_delta: float = 0.0, inject_target: bool = False
) -> Callable[..., None]:
    def run(argv: list[str], *, cwd: Path, log_path: Path) -> None:
        del cwd
        calls.append(list(argv))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("synthetic CPU inference\n", encoding="utf-8")
        if argv[2] == "proposer-predict":
            manifest = json.loads(Path(_argument(argv, "--test-manifest")).read_text())
            mask = _argument(argv, "--radar-mask")
            arrays = _proposer_arrays(int(manifest["fold"]), int(manifest["seed"]), mask)
            if mask == "radars_123" and full_mask_delta:
                arrays["prediction"] = arrays["prediction"] + np.float32(full_mask_delta)
            if inject_target:
                arrays["target_rr_bpm"] = np.asarray([99.0], dtype=np.float32)
            output = Path(_argument(argv, "--output"))
            with output.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
        elif argv[2] == "no-action-adapter":
            proposer_path = Path(_argument(argv, "--proposer"))
            with np.load(proposer_path, allow_pickle=False) as archive:
                proposer = {name: np.asarray(archive[name]).copy() for name in archive.files}
            arrays = _raw_arrays(
                proposer,
                int(_argument(argv, "--outer-fold")),
                int(_argument(argv, "--seed")),
            )
            output = Path(_argument(argv, "--output"))
            with output.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
        else:
            raise AssertionError(f"unexpected command: {argv}")

    return run


def _run(paths: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return RUN.run_campaign(
        primary_plan_path=paths["plan"],
        primary_output_root=paths["primary"],
        output_root=paths["output"],
        python_executable=Path(sys.executable),
        safe_helper=paths["helper"],
        locked_oof_source=paths["oof_source"],
        **kwargs,
    )


def test_dry_run_materializes_deterministic_126_unit_cpu_plan_without_predictions(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    result = _run(paths, dry_run=True)
    assert result["status"] == "dry_run"
    assert result["completed_units"] == result["new_units"] == 0
    assert result["evaluation_authorized"] is False
    plan_path = paths["output"] / "control/plan.json"
    first_bytes = plan_path.read_bytes()
    plan = json.loads(first_bytes)
    assert plan["unit_count"] == 126
    assert [unit["radar_mask"] for unit in plan["units"][:7]] == list(RUN.MASKS)
    assert len({unit["unit_id"] for unit in plan["units"]}) == 126
    assert plan["target_or_label_artifact_bound"] is False
    assert plan["mask_selection_contract"]["best_mask_selection_allowed"] is False
    assert plan["mask_selection_contract"]["radars_123_primary_parity"]["required"] is True
    assert all(
        command["argv"][command["argv"].index("--device") + 1] == "cpu"
        and "--amp" not in command["argv"]
        for unit in plan["units"]
        for command in unit["commands"]
        if command["stage"] == "proposer_predict"
    )
    assert not (paths["output"] / "units").exists()
    assert not (paths["output"] / "complete_seal.json").exists()
    _run(paths, dry_run=True)
    assert plan_path.read_bytes() == first_bytes


def test_max_new_units_resume_and_126_only_complete_seal_are_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(RUN, "_run_command", _fake_runner(calls))
    first = _run(paths, max_new_units=1)
    assert first["completed_units"] == first["new_units"] == 1
    assert first["evaluation_authorized"] is False
    assert len(calls) == 2
    second = _run(paths, max_new_units=0)
    assert second["completed_units"] == 1
    assert second["new_units"] == 0
    assert len(calls) == 2
    partial = _run(paths, max_new_units=124)
    assert partial["completed_units"] == 125
    assert partial["evaluation_authorized"] is False
    assert not (paths["output"] / "complete_seal.json").exists()
    complete = _run(paths, max_new_units=1)
    assert complete["completed_units"] == 126
    assert complete["new_units"] == 1
    assert complete["evaluation_authorized"] is True
    seal_path = paths["output"] / "complete_seal.json"
    seal = json.loads(seal_path.read_text())
    assert seal["complete_matrix"] is True
    assert seal["unit_count"] == 126
    assert len(seal["units"]) == 126
    assert seal["target_or_label_artifact_opened_before_seal"] is False
    assert seal["best_mask_selection_performed"] is False
    assert seal["effective_sources"]
    seal_bytes = seal_path.read_bytes()
    prior_call_count = len(calls)
    resumed = _run(paths)
    assert resumed["new_units"] == 0
    assert len(calls) == prior_call_count
    assert seal_path.read_bytes() == seal_bytes


def test_radars_123_must_be_bit_exact_to_primary_and_label_fields_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mismatch = _fixture(tmp_path / "mismatch")
    mismatch_calls: list[list[str]] = []
    monkeypatch.setattr(
        RUN, "_run_command", _fake_runner(mismatch_calls, full_mask_delta=0.25)
    )
    with pytest.raises(RUN.RadarMaskCampaignError, match="not bit-exact"):
        _run(mismatch, max_new_units=1)
    assert not (
        mismatch["output"]
        / "units"
        / f"outer_0_seed_{SEEDS[0]}"
        / "radars_123"
        / "receipt.json"
    ).exists()
    assert not (mismatch["output"] / "complete_seal.json").exists()

    labelled = _fixture(tmp_path / "labelled")
    label_calls: list[list[str]] = []
    monkeypatch.setattr(RUN, "_run_command", _fake_runner(label_calls, inject_target=True))
    with pytest.raises(RUN.RadarMaskCampaignError, match="target/label fields"):
        _run(labelled, max_new_units=1)
    assert len(label_calls) == 1
    assert not (labelled["output"] / "complete_seal.json").exists()


def test_input_and_receipt_output_drift_stop_before_new_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(RUN, "_run_command", _fake_runner(calls))
    _run(paths, max_new_units=1)
    assert len(calls) == 2
    first_unit = (
        paths["output"]
        / "units"
        / f"outer_0_seed_{SEEDS[0]}"
        / "radars_123"
        / "sealed_label_free_predictions.npz"
    )
    first_unit.chmod(0o644)
    first_unit.write_bytes(b"tampered")
    with pytest.raises(
        RUN.RadarMaskCampaignError,
        match="hash mismatch|stage receipt output differs",
    ):
        _run(paths, max_new_units=1)
    assert len(calls) == 2

    fresh = _fixture(tmp_path / "input_drift")
    checkpoint = fresh["artifacts"][(0, SEEDS[0])]["checkpoint"]
    checkpoint_path = Path(checkpoint["path"])
    checkpoint_path.write_bytes(b"drifted checkpoint")
    fresh_calls: list[list[str]] = []
    monkeypatch.setattr(RUN, "_run_command", _fake_runner(fresh_calls))
    with pytest.raises(
        RUN.RadarMaskCampaignError,
        match="hash mismatch|stage receipt output differs",
    ):
        _run(fresh, max_new_units=1)
    assert fresh_calls == []


def _rewrite_receipt(path: Path, document: dict[str, Any]) -> None:
    payload = dict(document)
    payload.pop("content_sha256", None)
    payload["content_sha256"] = RUN.canonical_json_sha256(payload)
    path.chmod(0o644)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_resume_rejects_surplus_radar_receipt_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(RUN, "_run_command", _fake_runner(calls))
    _run(paths, max_new_units=1)
    receipt_path = (
        paths["output"]
        / "units"
        / f"outer_0_seed_{SEEDS[0]}"
        / "radars_123/receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["surplus_attestation"] = True
    _rewrite_receipt(receipt_path, receipt)
    with pytest.raises(RUN.RadarMaskCampaignError, match="identity/label-free"):
        _run(paths, max_new_units=0)
    assert len(calls) == 2


def test_resume_rejects_radar_receipt_output_outside_unit_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(RUN, "_run_command", _fake_runner(calls))
    _run(paths, max_new_units=1)
    receipt_path = (
        paths["output"]
        / "units"
        / f"outer_0_seed_{SEEDS[0]}"
        / "radars_123/receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    outside = tmp_path / "outside_proposer.npz"
    source = Path(receipt["outputs"]["proposer_prediction"]["path"])
    outside.write_bytes(source.read_bytes())
    receipt["outputs"]["proposer_prediction"] = RUN.bind_file(outside)
    _rewrite_receipt(receipt_path, receipt)
    with pytest.raises(RUN.RadarMaskCampaignError, match="unit output path differs"):
        _run(paths, max_new_units=0)
    assert len(calls) == 2


def test_cli_has_no_target_or_evaluation_argument() -> None:
    parser = RUN.build_parser()
    destinations = {action.dest for action in parser._actions}
    assert not {"targets", "target", "labels", "evaluate"} & destinations
    assert {"dry_run", "max_new_units", "primary_plan", "primary_output_root"} <= destinations
