from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts/run_release_locked_hcs_evaluation.py"


def _load_module():
    name = "run_release_locked_hcs_evaluation_under_test"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SUP = _load_module()


def _write_bytes(path: Path, payload: bytes = b"fixture\n", *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)


def _write_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    content_hash: bool = True,
    mode: int = 0o444,
) -> dict[str, Any]:
    document = dict(value)
    if content_hash:
        document["content_sha256"] = SUP.canonical_sha256(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    path.chmod(mode)
    return document


def _fixture_authorization(tmp_path: Path):
    paths = SUP.release_paths(tmp_path)
    for source in (
        paths.raw_join_source,
        paths.primary_source,
        paths.radar_source,
        paths.uncertainty_source,
        paths.execution_source,
    ):
        _write_bytes(source, f"# {source.name}\n".encode())
    _write_json(paths.release_readiness_spec, {"classification": "synthetic_readiness"})
    _write_json(paths.pretarget_release_lock, {"classification": "synthetic_release"})
    _write_json(paths.target_release_receipt, {"classification": "synthetic_wrapper"})
    _write_bytes(paths.canonical_target, b"synthetic canonical target")
    _write_json(paths.canonical_target_receipt, {"classification": "synthetic_target_receipt"})
    _write_json(paths.predictions_seal, {"classification": "synthetic_predictions"})
    _write_json(paths.primary_evaluation_spec, {"classification": "synthetic_primary_spec"})
    _write_json(
        paths.uncertainty_evaluation_spec,
        {"classification": "synthetic_uncertainty_spec"},
    )
    _write_json(paths.radar_mask_complete_seal, {"classification": "synthetic_radar_seal"})
    radar_mask_plan = paths.mask_root / "mask_plan.json"
    radar_mask_preexecution_lock = paths.mask_root / "preexecution_lock.json"
    uncertainty_archive = paths.primary_root / "locked_hcs_uncertainty_inputs.npz"
    _write_json(radar_mask_plan, {"classification": "synthetic_radar_mask_plan"})
    _write_json(
        radar_mask_preexecution_lock,
        {"classification": "synthetic_radar_mask_preexecution_lock"},
    )
    _write_bytes(uncertainty_archive, b"synthetic uncertainty archive")
    _write_json(
        paths.uncertainty_inputs_seal,
        {"classification": "synthetic_uncertainty_seal"},
    )
    _write_json(paths.uncertainty_calibration, {"classification": "synthetic_calibration"})
    authorization = {
        name: SUP.bind_file(path)
        for name, path in {
            "release_readiness_spec": paths.release_readiness_spec,
            "pretarget_release_lock": paths.pretarget_release_lock,
            "target_release_receipt": paths.target_release_receipt,
            "canonical_target": paths.canonical_target,
            "canonical_target_receipt": paths.canonical_target_receipt,
            "predictions_seal": paths.predictions_seal,
            "primary_evaluation_spec": paths.primary_evaluation_spec,
            "uncertainty_evaluation_spec": paths.uncertainty_evaluation_spec,
            "radar_mask_complete_seal": paths.radar_mask_complete_seal,
            "radar_mask_plan": radar_mask_plan,
            "radar_mask_preexecution_lock": radar_mask_preexecution_lock,
            "uncertainty_inputs_seal": paths.uncertainty_inputs_seal,
            "uncertainty_calibration": paths.uncertainty_calibration,
            "uncertainty_archive": uncertainty_archive,
        }.items()
    }
    authorization.update(
        {
            "readiness_spec": {},
            "release_lock": {},
            "release_receipt": {},
            "target_receipt": {},
        }
    )
    authorization["uncertainty_pretarget_audit"] = {
        "secondary_uncertainty_evaluation_spec": authorization[
            "uncertainty_evaluation_spec"
        ],
        "evaluation_spec": authorization["primary_evaluation_spec"],
        "calibration": authorization["uncertainty_calibration"],
        "predictions_seal": authorization["predictions_seal"],
        "uncertainty_inputs_seal": authorization["uncertainty_inputs_seal"],
        "uncertainty_archive": authorization["uncertainty_archive"],
        "calibration_declared_bindings_rehashed": 1,
        "prediction_declared_bindings_rehashed": 1,
        "uncertainty_declared_bindings_rehashed": 1,
        "secondary_protocol_role": "separate_secondary_retrospective_engineering_protocol",
        "primary_uncertainty_contract_overridden": False,
        "dedicated_secondary_spec_verified_before_prediction_uncertainty_or_target_access": True,
        "all_target_free_inputs_verified_before_evaluation_lock_access": True,
        "all_uncertainty_array_schema_and_hashes_verified": True,
    }
    return paths, authorization


def _publish_join(paths, authorization, command: Sequence[str]) -> None:
    _write_bytes(paths.joined_output, b"joined")
    _write_json(
        paths.joined_metrics,
        {"classification": "retrospective_locked_hcs_oof_evaluation"},
        content_hash=False,
    )
    _write_json(
        paths.evaluation_lock,
        {
            "schema_version": 1,
            "classification": "locked_hcs_oof_single_target_join_seal",
            "predictions_seal": authorization["predictions_seal"],
            "target_artifact": authorization["canonical_target"],
            "target_join_count": 1,
            "orchestrator_command": list(command[1:]),
            "outputs": {
                "joined_oof": SUP.bind_file(paths.joined_output),
                "metrics": SUP.bind_file(paths.joined_metrics),
            },
            "commercial_claim_authorized": False,
            "prospective_confirmation_required": True,
        },
        content_hash=False,
    )


def _publish_evaluation(
    name: str,
    files: Sequence[Path],
    paths,
    authorization,
    command: Sequence[str],
) -> None:
    report_path, csv_path, receipt_path = files
    report_class, receipt_class = SUP._EVALUATION_CLASSES[name]
    if name == "uncertainty":
        recorded = list(command[1:])
    else:
        recorded = list(command)
    report: dict[str, Any] = {
        "schema_version": 1,
        "classification": report_class,
        "commercial_claim_authorized": False,
        "prospective_confirmation_required": True,
        "orchestrator_command": recorded,
    }
    if name in {"primary", "radar_masks"}:
        report["evaluation_specification"] = authorization["primary_evaluation_spec"]
    else:
        report["uncertainty_evaluation_specification"] = authorization[
            "uncertainty_evaluation_spec"
        ]
    _write_json(report_path, report)
    _write_bytes(csv_path, b"metric,value\nsynthetic,1\n")
    inputs = {
        "evaluation_lock": SUP.bind_file(paths.evaluation_lock),
        "predictions_seal": authorization["predictions_seal"],
        "target_receipt": authorization["canonical_target_receipt"],
        "target_artifact": authorization["canonical_target"],
        "joined_oof": SUP.bind_file(paths.joined_output),
        "locked_metrics": SUP.bind_file(paths.joined_metrics),
    }
    if name == "primary":
        inputs["evaluation_spec"] = authorization["primary_evaluation_spec"]
    elif name == "radar_masks":
        inputs.update(
            {
                "evaluation_spec": authorization["primary_evaluation_spec"],
                "primary_predictions_seal": authorization["predictions_seal"],
                "radar_mask_complete_seal": authorization["radar_mask_complete_seal"],
                "radar_mask_plan": authorization["radar_mask_plan"],
                "radar_mask_preexecution_lock": authorization[
                    "radar_mask_preexecution_lock"
                ],
            }
        )
    else:
        inputs.update(authorization["uncertainty_pretarget_audit"])
    _write_json(
        receipt_path,
        {
            "schema_version": 1,
            "classification": receipt_class,
            "commercial_claim_authorized": False,
            "prospective_confirmation_required": True,
            "outputs_create_once": True,
            "output_overwrite_allowed": False,
            "inputs": inputs,
            "outputs": {
                "report": SUP.bind_file(report_path),
                "metrics_csv": SUP.bind_file(csv_path),
            },
            "orchestrator_command": recorded,
        },
    )


def _successful_runner(paths, authorization, calls: list[list[str]]):
    commands = SUP.build_frozen_argv(paths)

    def run(command: Sequence[str], cwd: Path):
        del cwd
        observed = list(command)
        calls.append(observed)
        position = commands.index(observed)
        if position == 0:
            _publish_join(paths, authorization, observed)
        else:
            _publish_evaluation(
                ("primary", "radar_masks", "uncertainty")[position - 1],
                SUP._evaluation_outputs(paths)[position - 1],
                paths,
                authorization,
                observed,
            )
        return SimpleNamespace(returncode=0, stderr="", stdout="ok")

    return run


def test_success_exact_bindings_0444_and_exact_idempotent_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, authorization = _fixture_authorization(tmp_path)
    validations: list[str] = []

    def authorize(_paths):
        validations.append("authorization")
        return authorization

    monkeypatch.setattr(SUP, "verify_release_authorization", authorize)
    calls: list[list[str]] = []
    runner = _successful_runner(paths, authorization, calls)

    first = SUP.run_release_locked_evaluation(tmp_path, command_runner=runner)
    assert len(calls) == 4
    assert first["classification"] == SUP.CLASSIFICATION
    assert first["frozen_argv"] == SUP.build_frozen_argv(paths)
    assert first["executed_commands"] == first["frozen_argv"]
    assert first["all_inputs_and_outputs_live_rehashed"] is True
    assert first["target_re_evaluation_performed"] is False
    assert first["all_steps_executed_once"] is True
    assert first["commercial_claim_authorized"] is False
    assert stat.S_IMODE(paths.attestation.stat().st_mode) == 0o444
    payload = dict(first)
    content = payload.pop("content_sha256")
    assert content == SUP.canonical_sha256(payload)
    for field, path in {
        "canonical_target": paths.canonical_target,
        "canonical_target_receipt": paths.canonical_target_receipt,
        "evaluation_lock": paths.evaluation_lock,
        "predictions_seal": paths.predictions_seal,
    }.items():
        assert first[field] == SUP.bind_file(path)

    second = SUP.run_release_locked_evaluation(tmp_path, command_runner=runner)
    assert second == first
    assert len(calls) == 4
    assert len(validations) > 4  # every subprocess had a fresh pre-call drift gate


def test_drift_fails_before_first_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, authorization = _fixture_authorization(tmp_path)
    count = 0

    def authorize(_paths):
        nonlocal count
        count += 1
        if count >= 2:
            raise SUP.ReleaseEvaluationError("synthetic source drift")
        return authorization

    monkeypatch.setattr(SUP, "verify_release_authorization", authorize)
    calls: list[list[str]] = []

    with pytest.raises(SUP.ReleaseEvaluationError, match="drift"):
        SUP.run_release_locked_evaluation(
            tmp_path,
            command_runner=lambda command, cwd: calls.append(list(command)),
        )
    assert calls == []
    assert not paths.attestation.exists()


def test_evaluator_failure_never_publishes_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, authorization = _fixture_authorization(tmp_path)
    monkeypatch.setattr(SUP, "verify_release_authorization", lambda _paths: authorization)
    calls: list[list[str]] = []
    successful = _successful_runner(paths, authorization, calls)

    def fail_uncertainty(command: Sequence[str], cwd: Path):
        if Path(command[1]) == paths.uncertainty_source:
            calls.append(list(command))
            return SimpleNamespace(returncode=9, stderr="synthetic evaluator failure")
        return successful(command, cwd)

    with pytest.raises(SUP.ReleaseEvaluationError, match="exit 9"):
        SUP.run_release_locked_evaluation(tmp_path, command_runner=fail_uncertainty)
    assert len(calls) == 4
    assert not paths.attestation.exists()


def test_partial_or_out_of_order_output_is_quarantined_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, authorization = _fixture_authorization(tmp_path)
    monkeypatch.setattr(SUP, "verify_release_authorization", lambda _paths: authorization)
    _write_bytes(paths.joined_output, b"partial")
    calls: list[list[str]] = []
    with pytest.raises(SUP.ReleaseEvaluationError, match="partial immutable target join"):
        SUP.run_release_locked_evaluation(
            tmp_path,
            command_runner=lambda command, cwd: calls.append(list(command)),
        )
    assert calls == []
    assert not paths.attestation.exists()


def test_alternate_target_and_argv_are_rejected_without_authority_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = SUP.release_paths(tmp_path)
    monkeypatch.setattr(
        SUP,
        "verify_release_authorization",
        lambda _paths: pytest.fail("authorization should not be reached"),
    )
    with pytest.raises(SUP.ReleaseEvaluationError, match="alternate canonical target"):
        SUP.run_release_locked_evaluation(
            tmp_path, canonical_target=tmp_path / "other-target.npz"
        )
    changed = SUP.build_frozen_argv(paths)
    changed[0] = [*changed[0], "--not-frozen"]
    with pytest.raises(SUP.ReleaseEvaluationError, match="argv differs"):
        SUP.run_release_locked_evaluation(tmp_path, requested_argv=changed)
    with pytest.raises(SystemExit):
        SUP.build_parser().parse_args(["--targets", "other.npz"])


@pytest.mark.parametrize(
    ("name", "edge", "message"),
    [
        ("radar_masks", "radar_mask_complete_seal", "radar_mask_complete_seal"),
        ("radar_masks", "radar_mask_plan", "radar_mask_plan"),
        (
            "radar_masks",
            "radar_mask_preexecution_lock",
            "radar_mask_preexecution_lock",
        ),
        ("uncertainty", "calibration", "calibration"),
        ("uncertainty", "uncertainty_archive", "uncertainty_archive"),
        (
            "uncertainty",
            "secondary_uncertainty_evaluation_spec",
            "secondary_uncertainty_evaluation_spec",
        ),
    ],
)
def test_evaluator_specific_receipt_edge_tamper_is_rejected(
    tmp_path: Path,
    name: str,
    edge: str,
    message: str,
) -> None:
    paths, authorization = _fixture_authorization(tmp_path)
    commands = SUP.build_frozen_argv(paths)
    _publish_join(paths, authorization, commands[0])
    evaluation_authorization = SUP._evaluation_authorization(paths, authorization)
    position = {"radar_masks": 2, "uncertainty": 3}[name]
    files = SUP._evaluation_outputs(paths)[position - 1]
    _publish_evaluation(name, files, paths, authorization, commands[position])
    receipt_path = files[2]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("content_sha256")
    receipt["inputs"][edge] = authorization["canonical_target"]
    receipt_path.chmod(0o644)
    _write_json(receipt_path, receipt)
    with pytest.raises(SUP.ReleaseEvaluationError, match=message):
        SUP._validate_evaluation(
            name=name,
            files=files,
            authorization=evaluation_authorization,
            evaluation_lock_binding=SUP.bind_file(paths.evaluation_lock),
            frozen_command=commands[position],
        )


@pytest.mark.parametrize(
    ("edge", "tampered"),
    [
        ("calibration_declared_bindings_rehashed", 2),
        ("prediction_declared_bindings_rehashed", 2),
        ("uncertainty_declared_bindings_rehashed", 2),
        ("secondary_protocol_role", "wrong_protocol"),
        ("primary_uncertainty_contract_overridden", True),
        (
            "dedicated_secondary_spec_verified_before_prediction_uncertainty_or_target_access",
            False,
        ),
        ("all_target_free_inputs_verified_before_evaluation_lock_access", False),
        ("all_uncertainty_array_schema_and_hashes_verified", False),
    ],
)
def test_uncertainty_scalar_receipt_edge_tamper_is_rejected(
    tmp_path: Path, edge: str, tampered: Any
) -> None:
    paths, authorization = _fixture_authorization(tmp_path)
    commands = SUP.build_frozen_argv(paths)
    _publish_join(paths, authorization, commands[0])
    files = SUP._evaluation_outputs(paths)[2]
    _publish_evaluation("uncertainty", files, paths, authorization, commands[3])
    receipt_path = files[2]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("content_sha256")
    receipt["inputs"][edge] = tampered
    receipt_path.chmod(0o644)
    _write_json(receipt_path, receipt)
    with pytest.raises(SUP.ReleaseEvaluationError, match=edge):
        SUP._validate_evaluation(
            name="uncertainty",
            files=files,
            authorization=SUP._evaluation_authorization(paths, authorization),
            evaluation_lock_binding=SUP.bind_file(paths.evaluation_lock),
            frozen_command=commands[3],
        )


def test_surplus_evaluator_receipt_input_is_rejected(tmp_path: Path) -> None:
    paths, authorization = _fixture_authorization(tmp_path)
    commands = SUP.build_frozen_argv(paths)
    _publish_join(paths, authorization, commands[0])
    files = SUP._evaluation_outputs(paths)[0]
    _publish_evaluation("primary", files, paths, authorization, commands[1])
    receipt_path = files[2]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("content_sha256")
    receipt["inputs"]["surplus_unfrozen_input"] = authorization["canonical_target"]
    receipt_path.chmod(0o644)
    _write_json(receipt_path, receipt)

    with pytest.raises(
        SUP.ReleaseEvaluationError,
        match=r"extra=\['surplus_unfrozen_input'\]",
    ):
        SUP._validate_evaluation(
            name="primary",
            files=files,
            authorization=SUP._evaluation_authorization(paths, authorization),
            evaluation_lock_binding=SUP.bind_file(paths.evaluation_lock),
            frozen_command=commands[1],
        )


def test_frozen_child_interpreter_preserves_venv_symlink(tmp_path: Path) -> None:
    launcher = tmp_path / "synthetic-venv/bin/python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(Path(sys.executable))
    commands = SUP.build_frozen_argv(
        SUP.release_paths(tmp_path), python_executable=launcher
    )
    expected = os.path.abspath(str(launcher))
    assert all(command[0] == expected for command in commands)
    assert command_realpath(expected) != expected


def command_realpath(path: str) -> str:
    # Kept outside the production module: this is the regression oracle for
    # proving the frozen argv did not dereference the venv launcher.
    return os.path.realpath(path)


def _synthetic_binding(path: Path, marker: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": (marker.encode().hex() + "0" * 64)[:64],
        "bytes": len(marker),
    }


def _content(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["content_sha256"] = SUP.canonical_sha256(result)
    return result


def test_public_verifiers_and_wrapper_precede_any_target_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = SUP.release_paths(tmp_path)
    radar_plan = paths.mask_root / "control/plan.json"
    radar_preexecution = paths.mask_root / "control/preexecution_lock.json"
    uncertainty_archive = paths.primary_root / "locked_hcs_uncertainty_inputs.npz"
    events: list[str] = []
    bindings = {
        path: _synthetic_binding(path, str(position))
        for position, path in enumerate(
            (
                paths.release_readiness_spec,
                paths.pretarget_release_lock,
                paths.target_release_receipt,
                paths.canonical_target,
                paths.canonical_target_receipt,
                paths.predictions_seal,
                paths.primary_evaluation_spec,
                paths.uncertainty_evaluation_spec,
                paths.radar_mask_complete_seal,
                paths.uncertainty_inputs_seal,
                paths.uncertainty_calibration,
                radar_plan,
                radar_preexecution,
                uncertainty_archive,
            ),
            start=1,
        )
    }
    roles = {
        "target_release_receipt": str(paths.target_release_receipt),
        "pretarget_release_lock": str(paths.pretarget_release_lock),
        "primary_evaluation_lock": str(paths.evaluation_lock),
        "canonical_target": str(paths.canonical_target),
        "canonical_target_receipt": str(paths.canonical_target_receipt),
        "joined_output": str(paths.joined_output),
        "predictions_seal": str(paths.predictions_seal),
        "release_evaluation_execution_attestation": str(paths.attestation),
        "primary_report": str(paths.primary_report),
        "primary_receipt": str(paths.primary_receipt),
        "radar_report": str(paths.radar_report),
        "radar_receipt": str(paths.radar_receipt),
        "uncertainty_spec": str(paths.uncertainty_evaluation_spec),
        "primary_evaluation_spec": str(paths.primary_evaluation_spec),
        "uncertainty_report": str(paths.uncertainty_report),
        "uncertainty_receipt": str(paths.uncertainty_receipt),
        "radar_mask_complete_seal": str(paths.radar_mask_complete_seal),
        "uncertainty_inputs_seal": str(paths.uncertainty_inputs_seal),
    }
    readiness = {
        "input_roles": roles,
        "content_sha256": "a" * 64,
    }
    locations = {
        name: str(path)
        for name, path in SUP._expected_lock_locations(paths).items()
    }
    release_lock = {
        "content_sha256": "b" * 64,
        "locations": locations,
        "frozen_specs": {
            "release_readiness": {
                "binding": bindings[paths.release_readiness_spec],
                "content_sha256": readiness["content_sha256"],
                "target_or_target_bearing_artifact_opened": False,
                "commercial_release_ready_must_equal": False,
                "prospective_confirmation_required": True,
            },
            "primary_evaluation": {
                "binding": bindings[paths.primary_evaluation_spec]
            },
            "secondary_uncertainty_evaluation": {
                "binding": bindings[paths.uncertainty_evaluation_spec],
                "calibration": bindings[paths.uncertainty_calibration],
            },
        },
        "boundaries": {
            "primary_predictions": {
                "predictions_seal": bindings[paths.predictions_seal]
            },
            "radar_masks": {
                "complete_seal": bindings[paths.radar_mask_complete_seal],
                "plan": bindings[radar_plan],
                "preexecution_lock": bindings[radar_preexecution],
            },
            "uncertainty": {
                "uncertainty_inputs_seal": bindings[paths.uncertainty_inputs_seal],
                "pretest_calibration": bindings[paths.uncertainty_calibration],
                "uncertainty_archive": bindings[uncertainty_archive],
            },
        },
    }
    target_receipt = _content(
        {
            "schema_version": 1,
            "classification": "retrospective_locked_hcs_canonical_target_artifact_receipt",
            "target_artifact_created_once": True,
            "target_artifact_overwrite_allowed": False,
            "pretarget_release_capability_verified": True,
            "commercial_claim_authorized": False,
            "prospective_confirmation_required": True,
            "target_artifact": bindings[paths.canonical_target],
            "source_bindings": {
                "predictions_seal": bindings[paths.predictions_seal]
            },
            "orchestrator_command": [
                sys.executable,
                str(tmp_path / "scripts/build_locked_hcs_targets_after_release_lock.py"),
            ],
        }
    )
    wrapper = _content(
        {
            "schema_version": 1,
            "classification": "locked_hcs_canonical_targets_built_after_pretarget_release",
            "commercial_claim_authorized": False,
            "prospective_confirmation_required": True,
            "release_lock_revalidated_before_target_builder_call": True,
            "all_release_bound_artifacts_rehashed_before_target_builder_call": True,
            "target_metadata_access_before_release_validation": False,
            "canonical_target_builder_called_only_after_release_authorization": True,
            "pretarget_release_lock": bindings[paths.pretarget_release_lock],
            "pretarget_release_content_sha256": release_lock["content_sha256"],
            "canonical_target": bindings[paths.canonical_target],
            "canonical_target_receipt": bindings[paths.canonical_target_receipt],
            "canonical_target_receipt_content_sha256": target_receipt["content_sha256"],
        }
    )

    def readiness_loader(path):
        assert path == paths.release_readiness_spec
        events.append("readiness-public-verifier")
        return readiness, bindings[path]

    def release_validator(path, *, require_target_absence):
        assert path == paths.pretarget_release_lock
        assert require_target_absence is False
        events.append("release-public-verifier")
        return release_lock

    def fake_read(path, label):
        del label
        if path == paths.target_release_receipt:
            events.append("wrapper-receipt-read")
            return wrapper
        if path == paths.canonical_target_receipt:
            events.append("target-receipt-read")
            return target_receipt
        raise AssertionError(path)

    def fake_bind(path):
        path = Path(path)
        if path == paths.canonical_target:
            events.append("TARGET-OPEN")
        elif path == paths.canonical_target_receipt:
            events.append("TARGET-RECEIPT-OPEN")
        return bindings[path]

    def prediction_verifier(root):
        assert root == paths.primary_root
        events.append("prediction-public-verifier")
        return {"predictions_seal": bindings[paths.predictions_seal]}

    monkeypatch.setattr(SUP.READINESS, "load_release_readiness_spec", readiness_loader)
    monkeypatch.setattr(SUP.RELEASE, "validate_release_lock", release_validator)
    monkeypatch.setattr(SUP, "_require_0444", lambda path, label: None)
    monkeypatch.setattr(SUP, "_read_json", fake_read)
    monkeypatch.setattr(SUP, "bind_file", fake_bind)
    monkeypatch.setattr(SUP.TARGETS, "verify_prediction_seal", prediction_verifier)
    uncertainty_audit = {
        "secondary_uncertainty_evaluation_spec": bindings[
            paths.uncertainty_evaluation_spec
        ],
        "evaluation_spec": bindings[paths.primary_evaluation_spec],
        "calibration": bindings[paths.uncertainty_calibration],
        "predictions_seal": bindings[paths.predictions_seal],
        "uncertainty_inputs_seal": bindings[paths.uncertainty_inputs_seal],
        "uncertainty_archive": bindings[uncertainty_archive],
        "calibration_declared_bindings_rehashed": 1,
        "prediction_declared_bindings_rehashed": 2,
        "uncertainty_declared_bindings_rehashed": 3,
        "secondary_protocol_role": "separate_secondary_retrospective_engineering_protocol",
        "primary_uncertainty_contract_overridden": False,
        "dedicated_secondary_spec_verified_before_prediction_uncertainty_or_target_access": True,
        "all_target_free_inputs_verified_before_evaluation_lock_access": True,
        "all_uncertainty_array_schema_and_hashes_verified": True,
    }
    monkeypatch.setattr(
        SUP.UNCERTAINTY,
        "_validate_pre_target_inputs",
        lambda **kwargs: ({}, {}, {}, {}, uncertainty_audit),
    )
    monkeypatch.setattr(
        SUP.RELEASED_TARGETS,
        "_release_receipt_document",
        lambda **kwargs: wrapper,
    )

    result = SUP.verify_release_authorization(paths)
    assert result["canonical_target"] == bindings[paths.canonical_target]
    assert events.index("readiness-public-verifier") < events.index("TARGET-OPEN")
    assert events.index("release-public-verifier") < events.index("TARGET-OPEN")
    assert events.index("wrapper-receipt-read") < events.index("TARGET-OPEN")
    assert events.index("TARGET-OPEN") < events.index("prediction-public-verifier")
