from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OOF = _load("run_runtime_sealed_locked_hcs_oof_test", ROOT / "scripts/run_runtime_sealed_locked_hcs_oof.py")
FIXTURE = _load("fixed_completion_fixture", ROOT / "tests/test_seal_fixed_i3_pretest_completion.py")


def _argument(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


def _plan(pretest: Path, plan_path: Path, output_root: Path) -> dict[str, Any]:
    units = []
    for seed in FIXTURE.SEAL.SEEDS:
        for fold in FIXTURE.SEAL.FOLDS:
            unit_root = output_root / "units" / f"outer_{fold}_seed_{seed}"
            units.append(
                {
                    "outer_fold": fold,
                    "seed": seed,
                    "no_action_fast_path": True,
                    "stages": [
                        {"name": "test_proposer_bind", "argv": [sys.executable, "helper", "bind"]},
                        {
                            "name": "test_proposer_predict",
                            "argv": [sys.executable, "helper", "predict", "--device", "cpu"],
                        },
                        {"name": "no_action_fallback_adapter", "argv": [sys.executable, "helper", "adapt"]},
                    ],
                    "derived_artifacts": {
                        "test_proposer_checkpoint": str(unit_root / "work/checkpoint.pt"),
                        "test_proposer_prediction": str(unit_root / "work/proposer.npz"),
                        "raw_hcs_prediction": str(unit_root / "work/raw.npz"),
                    },
                }
            )
    return {
        "schema_version": 1,
        "classification": "locked_hcs_oof_inference_plan",
        "folds": list(FIXTURE.SEAL.FOLDS),
        "seeds": list(FIXTURE.SEAL.SEEDS),
        "pretest_index": OOF.bind_file(pretest),
        "common": {},
        "units": units,
    }


def _write_unit(root: Path, fold: int, seed: int) -> dict[str, Any]:
    unit_root = root / "units" / f"outer_{fold}_seed_{seed}"
    unit_root.mkdir(parents=True, exist_ok=True)
    prediction = unit_root / "sealed_label_free_predictions.npz"
    with prediction.open("wb") as stream:
        np.savez_compressed(
            stream,
            cache_index=np.asarray([fold], dtype=np.int64),
            outer_fold=np.asarray([fold], dtype=np.int16),
            seed=np.asarray([seed], dtype=np.int64),
            fallback_rr_bpm=np.asarray([20.0 + fold], dtype=np.float32),
            source_rr_bpm=np.asarray([20.0 + fold], dtype=np.float32),
            final_rr_bpm=np.asarray([20.0 + fold], dtype=np.float32),
            applied_pull=np.asarray([False]),
            target_joined=np.asarray(False),
        )
    lock_path = unit_root / "derived_inference_lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "outer_fold": fold,
                "seed": seed,
                "prediction": OOF.bind_file(prediction),
                "target_artifact_opened": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "outer_fold": fold,
        "seed": seed,
        "derived_lock": OOF.bind_file(lock_path),
        "prediction": OOF.bind_file(prediction),
    }


def _fake_runner(calls: list[list[str]]):
    def run(argv: list[str], *, cwd: Path) -> dict[str, Any]:
        del cwd
        calls.append(list(argv))
        mode = argv[2]
        if mode == "prepare":
            plan_path = Path(_argument(argv, "--plan-output"))
            output = Path(_argument(argv, "--output-root"))
            pretest = Path(_argument(argv, "--pretest-index"))
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(
                json.dumps(_plan(pretest, plan_path, output), sort_keys=True), encoding="utf-8"
            )
        elif mode == "infer":
            output = Path(_argument(argv, "--output-root"))
            plan_path = Path(_argument(argv, "--plan"))
            limit = int(_argument(argv, "--max-units"))
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            pretest_lock = output / "pretest_lock.json"
            pretest_lock.parent.mkdir(parents=True, exist_ok=True)
            if not pretest_lock.exists():
                pretest_lock.write_text(
                    json.dumps(
                        {
                            "classification": "locked_hcs_oof_all_pretest_assets_sealed",
                            "plan": OOF.bind_file(plan_path),
                            "target_artifact_opened": False,
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
            records = []
            ordered = sorted(plan["units"], key=lambda item: (item["seed"], item["outer_fold"]))
            for unit in ordered[:limit]:
                records.append(_write_unit(output, unit["outer_fold"], unit["seed"]))
            if limit == 18:
                (output / "predictions_seal.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "classification": "locked_hcs_oof_all_label_free_predictions_sealed",
                            "pretest_lock_sha256": OOF.sha256_file(pretest_lock),
                            "unit_count": 18,
                            "outer_folds": list(range(6)),
                            "target_artifact_opened_before_seal": False,
                            "target_join_authorized": True,
                            "units": records,
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
        else:
            raise AssertionError(argv)
        return {
            "argv": list(argv),
            "cwd": str(ROOT),
            "returncode": 0,
            "stdout_sha256": "0" * 64,
            "stderr_sha256": "0" * 64,
        }

    return run


def _fixture(tmp_path: Path) -> dict[str, Path]:
    paths = FIXTURE._fixture(tmp_path)
    FIXTURE.SEAL.seal_completion(
        merged_index=paths["merged"],
        runtime_input_seal=paths["runtime"],
        pretest_index=paths["pretest"],
        output=paths["output"],
        python_executable=Path(sys.executable),
    )
    test_manifests = tmp_path / "test_manifests"
    test_manifests.mkdir()
    rf = tmp_path / "rf"
    rf.mkdir()
    (rf / "manifest.json").write_text("{}", encoding="utf-8")
    return {
        **paths,
        "test_manifests": test_manifests,
        "rf": rf,
        "oof": tmp_path / "locked_oof",
    }


def _run(paths: dict[str, Path], **kwargs: Any) -> dict[str, Any]:
    python_executable = kwargs.pop("python_executable", Path(sys.executable))
    return OOF.run_supervisor(
        runtime_input_seal=paths["runtime"],
        completion_attestation=paths["output"],
        pretest_index=paths["pretest"],
        test_manifest_root=paths["test_manifests"],
        output_root=paths["oof"],
        underlying_source=ROOT / "scripts/run_locked_hcs_oof.py",
        python_executable=python_executable,
        rf_cache=paths["rf"],
        **kwargs,
    )


def test_prefix_limited_serial_resume_closes_all_18_units(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(OOF, "_run_subprocess", _fake_runner(calls))
    partial = _run(paths, max_new_units=2)
    assert partial["completed_units"] == 2
    assert not (paths["oof"] / "predictions_seal.json").exists()
    final = _run(paths)
    assert final["status"] == "locked_hcs_oof_runtime_guard_complete"
    limits = [int(_argument(argv, "--max-units")) for argv in calls if argv[2] == "infer"]
    assert limits == list(range(1, 19))
    attestation = json.loads(
        (paths["oof"] / "postlock_runtime_guard_attestation.json").read_text(encoding="utf-8")
    )
    assert attestation["classification"] == "locked_hcs_oof_runtime_guard_attestation"
    assert attestation["completed_units"] == 18
    assert len(attestation["unit_runtime_guard_receipts"]) == 18
    assert attestation["runtime_seal_verified_before_and_after_every_unit"] is True
    assert attestation["target_artifact_opened"] is False
    assert attestation["gpu_execution_performed"] is False


def test_target_artifact_blocks_before_prepare_subprocess(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["oof"].mkdir()
    target = paths["oof"] / "canonical_locked_hcs_targets_receipt.json"
    target.write_bytes(b"must-not-open")
    calls: list[list[str]] = []
    monkeypatch.setattr(OOF, "_run_subprocess", _fake_runner(calls))
    with pytest.raises(OOF.RuntimeGuardError, match="must be absent"):
        _run(paths)
    assert calls == []
    assert target.read_bytes() == b"must-not-open"


def test_unguarded_existing_plan_and_subprocess_shell_contract_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _fixture(tmp_path)
    plan_path = paths["oof"] / "locked_oof_plan.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text("{}", encoding="utf-8")
    calls: list[list[str]] = []
    monkeypatch.setattr(OOF, "_run_subprocess", _fake_runner(calls))
    with pytest.raises(OOF.RuntimeGuardError, match="predates this runtime guard"):
        _run(paths)
    assert calls == []


def test_virtualenv_launcher_symlink_is_preserved_in_child_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _fixture(tmp_path)
    launcher = tmp_path / "venv-python"
    launcher.symlink_to(Path(sys.executable))
    calls: list[list[str]] = []
    monkeypatch.setattr(OOF, "_run_subprocess", _fake_runner(calls))
    _run(paths, max_new_units=1, python_executable=launcher)
    expected = str(launcher.absolute())
    assert calls
    assert all(argv[0] == expected for argv in calls)
    prepare = next(argv for argv in calls if argv[2] == "prepare")
    assert _argument(prepare, "--python-executable") == expected
