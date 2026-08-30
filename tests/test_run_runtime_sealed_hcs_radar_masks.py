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


MASKS = _load(
    "run_runtime_sealed_hcs_radar_masks_test",
    ROOT / "scripts/run_runtime_sealed_hcs_radar_masks.py",
)
OOFTEST = _load(
    "runtime_oof_fixture_for_radar",
    ROOT / "tests/test_run_runtime_sealed_locked_hcs_oof.py",
)
RELEASE = _load(
    "create_locked_hcs_pretarget_release_lock_runtime_abi_test",
    ROOT / "scripts/create_locked_hcs_pretarget_release_lock.py",
)


def _argument(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


def _radar_plan(output_root: Path, *, device: str = "cpu") -> dict[str, Any]:
    units = []
    mask_items = list(MASKS.radar.MASKS.items())
    for seed in MASKS.radar.LOCKED_SEEDS:
        for fold in MASKS.radar.FOLDS:
            for mask, pattern in mask_items:
                root = output_root / "units" / f"outer_{fold}_seed_{seed}" / mask
                proposer = root / "proposer_prediction.npz"
                raw = root / "raw_source_prediction.npz"
                sealed = root / "sealed_label_free_predictions.npz"
                checkpoint = output_root / "synthetic_inputs/checkpoint.pt"
                run_config = output_root / "synthetic_inputs/run_config.json"
                manifest = output_root / "synthetic_inputs/test_manifest.json"
                helper = str((ROOT / "scripts/build_locked_hcs_test_inputs.py").resolve())
                python = str(Path(sys.executable).resolve())
                units.append(
                    {
                        "unit_id": f"outer_{fold}_seed_{seed}__{mask}",
                        "outer_fold": fold,
                        "seed": seed,
                        "radar_mask": mask,
                        "radar_mask_pattern": list(pattern),
                        "inputs": {
                            "checkpoint": {"path": str(checkpoint)},
                            "run_config": {"path": str(run_config)},
                            "test_manifest": {"path": str(manifest)},
                        },
                        "commands": [
                            {
                                "stage": "proposer_predict",
                                "argv": [
                                    python,
                                    helper,
                                    "proposer-predict",
                                    "--device",
                                    device,
                                    "--checkpoint",
                                    str(checkpoint),
                                    "--run-config",
                                    str(run_config),
                                    "--test-manifest",
                                    str(manifest),
                                    "--output",
                                    str(proposer),
                                    "--radar-mask",
                                    mask,
                                ],
                            },
                            {
                                "stage": "no_action_source_adapter",
                                "argv": [
                                    python,
                                    helper,
                                    "no-action-adapter",
                                    "--proposer",
                                    str(proposer),
                                    "--outer-fold",
                                    str(fold),
                                    "--seed",
                                    str(seed),
                                    "--output",
                                    str(raw),
                                ],
                            },
                        ],
                        "outputs": {
                            "proposer_prediction": str(proposer),
                            "raw_source_prediction": str(raw),
                            "sealed_prediction": str(sealed),
                            "receipt": str(root / "receipt.json"),
                        },
                    }
                )
    return {
        "schema_version": 1,
        "classification": "locked_hcs_seven_radar_mask_label_free_plan",
        "folds": list(MASKS.radar.FOLDS),
        "seeds": list(MASKS.radar.LOCKED_SEEDS),
        "radar_masks": [
            {"name": name, "pattern": list(pattern)} for name, pattern in mask_items
        ],
        "unit_count": 126,
        "target_or_label_artifact_bound": False,
        "execution": {"device": "cpu", "amp": False, "shell": False, "batch_size": 128},
        "units": units,
    }


def _write_mask_unit(output: Path, unit: dict[str, Any]) -> None:
    root = (
        output
        / "units"
        / f"outer_{unit['outer_fold']}_seed_{unit['seed']}"
        / unit["radar_mask"]
    )
    root.mkdir(parents=True, exist_ok=True)
    for name in (
        "proposer_prediction.npz",
        "raw_source_prediction.npz",
        "sealed_label_free_predictions.npz",
    ):
        with (root / name).open("wb") as stream:
            np.savez_compressed(
                stream,
                cache_index=np.asarray([unit["outer_fold"]], dtype=np.int64),
                outer_fold=np.asarray([unit["outer_fold"]], dtype=np.int16),
                seed=np.asarray([unit["seed"]], dtype=np.int64),
                final_rr_bpm=np.asarray([20.0], dtype=np.float32),
                target_joined=np.asarray(False),
            )
    (root / "receipt.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "classification": "locked_hcs_radar_mask_unit_receipt",
                "unit_id": unit["unit_id"],
                "target_or_label_artifact_opened": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _fake_runner(calls: list[list[str]], *, device: str = "cpu"):
    def run(argv: list[str], *, cwd: Path) -> dict[str, Any]:
        del cwd
        calls.append(list(argv))
        output = Path(_argument(argv, "--output-root"))
        plan_path = output / "control/plan.json"
        if "--dry-run" in argv:
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(
                json.dumps(_radar_plan(output, device=device), sort_keys=True),
                encoding="utf-8",
            )
        else:
            assert _argument(argv, "--max-new-units") == "1"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            completed = 0
            for unit in plan["units"]:
                root = (
                    output
                    / "units"
                    / f"outer_{unit['outer_fold']}_seed_{unit['seed']}"
                    / unit["radar_mask"]
                )
                if (root / "receipt.json").is_file():
                    completed += 1
                else:
                    _write_mask_unit(output, unit)
                    completed += 1
                    break
            if completed == 126:
                (output / "complete_seal.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "classification": "locked_hcs_all_seven_radar_mask_predictions_sealed",
                            "unit_count": 126,
                            "complete_matrix": True,
                            "target_or_label_artifact_opened_before_seal": False,
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
        return {
            "argv": list(argv),
            "cwd": str(ROOT),
            "returncode": 0,
            "stdout_sha256": "0" * 64,
            "stderr_sha256": "0" * 64,
        }

    return run


def _fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    paths = OOFTEST._fixture(tmp_path)
    oof_calls: list[list[str]] = []
    monkeypatch.setattr(OOFTEST.OOF, "_run_subprocess", OOFTEST._fake_runner(oof_calls))
    result = OOFTEST._run(paths)
    assert result["completed_units"] == 18
    return {
        **paths,
        "postlock": paths["oof"] / "postlock_runtime_guard_attestation.json",
        "radar": tmp_path / "radar_masks",
    }


def _run(paths: dict[str, Path], **kwargs: Any) -> dict[str, Any]:
    python_executable = kwargs.pop("python_executable", Path(sys.executable))
    return MASKS.run_supervisor(
        runtime_input_seal=paths["runtime"],
        completion_attestation=paths["output"],
        pretest_index=paths["pretest"],
        postlock_guard=paths["postlock"],
        primary_output_root=paths["oof"],
        output_root=paths["radar"],
        underlying_source=ROOT / "scripts/run_locked_hcs_radar_mask_campaign.py",
        python_executable=python_executable,
        **kwargs,
    )


def test_serial_one_unit_resume_closes_exact_126_mask_matrix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _fixture(monkeypatch, tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(MASKS.primary_guard, "_run_subprocess", _fake_runner(calls))
    partial = _run(paths, max_new_units=2)
    assert partial["completed_units"] == 2
    final = _run(paths)
    assert final["status"] == "locked_hcs_radar_mask_runtime_guard_complete"
    unit_calls = [argv for argv in calls if "--max-new-units" in argv]
    assert len(unit_calls) == 126
    assert all(_argument(argv, "--max-new-units") == "1" for argv in unit_calls)
    attestation = json.loads(
        (paths["radar"] / "radar_mask_runtime_guard_attestation.json").read_text(
            encoding="utf-8"
        )
    )
    assert attestation["classification"] == "locked_hcs_radar_mask_runtime_guard_attestation"
    assert attestation["completed_units"] == 126
    assert attestation["complete_seal"] == final["complete_seal"]
    assert len(attestation["unit_runtime_guard_receipts"]) == 126
    assert attestation["runtime_seal_verified_before_and_after_every_unit"] is True
    assert attestation["postlock_guard_verified_before_and_after_every_unit"] is True
    assert attestation["target_artifact_opened"] is False
    assert attestation["gpu_execution_performed"] is False
    release_summary = RELEASE._runtime_guard_summary(
        paths["runtime"],
        paths["output"],
        paths["postlock"],
        paths["radar"] / "radar_mask_runtime_guard_attestation.json",
        {
            "inference_plan": RELEASE.bind_file(paths["oof"] / "locked_oof_plan.json"),
            "pretest_lock": RELEASE.bind_file(paths["oof"] / "pretest_lock.json"),
            "predictions_seal": RELEASE.bind_file(paths["oof"] / "predictions_seal.json"),
        },
        {"complete_seal": RELEASE.bind_file(paths["radar"] / "complete_seal.json")},
    )
    assert (
        release_summary["classification"]
        == "fixed_i3_and_postlock_runtime_payload_closure_revalidated"
    )
    assert release_summary["completed_primary_units"] == 18
    assert release_summary["completed_radar_mask_units"] == 126


def test_primary_target_artifact_blocks_before_radar_initialization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _fixture(monkeypatch, tmp_path)
    target = paths["oof"] / "evaluation_lock.json"
    target.write_bytes(b"must-not-open")
    calls: list[list[str]] = []
    monkeypatch.setattr(MASKS.primary_guard, "_run_subprocess", _fake_runner(calls))
    with pytest.raises(MASKS.RadarRuntimeGuardError, match="must be absent"):
        _run(paths)
    assert calls == []
    assert target.read_bytes() == b"must-not-open"


def test_gpu_plan_is_rejected_before_first_mask_unit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _fixture(monkeypatch, tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        MASKS.primary_guard, "_run_subprocess", _fake_runner(calls, device="cuda")
    )
    with pytest.raises(MASKS.RadarRuntimeGuardError, match="CPU-only"):
        _run(paths)
    assert len(calls) == 1
    assert "--dry-run" in calls[0]
    assert not list(paths["radar"].glob("units/*/*/receipt.json"))


def test_virtualenv_launcher_symlink_is_preserved_in_mask_child_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _fixture(monkeypatch, tmp_path)
    launcher = tmp_path / "venv-python"
    launcher.symlink_to(Path(sys.executable))
    calls: list[list[str]] = []
    monkeypatch.setattr(MASKS.primary_guard, "_run_subprocess", _fake_runner(calls))
    _run(paths, max_new_units=1, python_executable=launcher)
    expected = str(launcher.absolute())
    assert calls
    assert all(argv[0] == expected for argv in calls)
    assert all(_argument(argv, "--python-executable") == expected for argv in calls)
