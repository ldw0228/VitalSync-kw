from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import stat
import sys
import threading
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_locked_hfr_v3r1_test_inputs as builder  # noqa: E402
import run_fixed_hfr_v3r1_oof_campaign as fixed  # noqa: E402
import run_hfr_v3r1_discovery_campaign as campaign  # noqa: E402


def test_fixed_campaign_uses_shared_venv_symlink_preserving_path(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base-python"
    base.write_bytes(b"python")
    link = tmp_path / ".venv" / "bin" / "python"
    link.parent.mkdir(parents=True)
    link.symlink_to(base)
    observed = campaign.executable_path_without_symlink_dereference(
        tmp_path, Path(".venv/bin/python")
    )
    assert observed == link
    assert observed.is_symlink()


@pytest.mark.parametrize(
    "option",
    (
        "--selection-lock",
        "--promotion-authorization",
        "--trainer",
        "--gpu-wrapper",
        "--gpu-lock",
        "--gpu-ledger",
        "--usage-ledger",
    ),
)
def test_fixed_main_rejects_runtime_path_overrides_before_artifact_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    option: str,
) -> None:
    artifact_read_attempted = False

    def forbidden_artifact_read(*args: Any, **kwargs: Any) -> Any:
        nonlocal artifact_read_attempted
        artifact_read_attempted = True
        raise AssertionError("promotion governance must not be read")

    monkeypatch.setattr(
        fixed.locked_inputs,
        "validate_promotion_authorization",
        forbidden_artifact_read,
    )
    alternate_path = tmp_path / f"alternate-{option.removeprefix('--')}"
    assert (
        fixed.main(
            [
                "--project-root",
                str(tmp_path),
                option,
                str(alternate_path),
            ]
        )
        == 2
    )
    assert artifact_read_attempted is False
    assert not alternate_path.exists()


@pytest.mark.parametrize("override", ("selection", "authorization"))
def test_builder_rejects_governance_path_override_before_selection_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: str,
) -> None:
    authority_called = False

    def forbidden_authority(*args: Any, **kwargs: Any) -> Any:
        nonlocal authority_called
        authority_called = True
        raise AssertionError("selection authority must not read artifacts")

    monkeypatch.setattr(
        builder.selection_authority,
        "validate_locked_selection_authorization",
        forbidden_authority,
    )
    keyword: dict[str, Any] = {}
    if override == "selection":
        keyword["selection_lock_path"] = tmp_path / "alternate-selection.json"
    else:
        keyword["authorization_path"] = tmp_path / "alternate-authorization.json"
    with pytest.raises(campaign.CampaignError, match="canonical paths"):
        builder.validate_promotion_authorization(tmp_path, **keyword)
    assert authority_called is False


def test_completed_training_rejects_recorded_selection_governance_drift(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "training-resume"
    usage_ledger = tmp_path / "usage.jsonl"
    campaign.bind_run_usage_ledger(run_root, usage_ledger, execution_scope="promotion")
    promotion_authorization = tmp_path / "promotion_authorization.json"
    _write_json(promotion_authorization, {"promotion_authorized": True})
    governance = {
        "selection_lock": {"sha256": "c" * 64},
        "pretrain_authorization": {
            "path": str(tmp_path / "pretrain.json"),
            "sha256": "a" * 64,
        },
    }
    completion = (
        run_root / "training/outer_0_seed_20260828/completion_receipt.json"
    )
    campaign.create_once_json(
        completion,
        {
            "schema_version": 1,
            "classification": "adaptive_v3r1_fixed_promotion_training_completion",
            "campaign_id": campaign.CAMPAIGN_ID,
            "campaign_revision": fixed.CAMPAIGN_REVISION,
            "infrastructure_revision": fixed.INFRASTRUCTURE_REVISION,
            "outer_fold": 0,
            "seed": 20260828,
            "variant": "H0_no_factor",
            "outer_test_opened": False,
            "selection_lock_sha256": "d" * 64,
            "promotion_authorization_sha256": campaign.sha256_file(
                promotion_authorization
            ),
            "invocation": {},
            "usage_ledger_path": str(usage_ledger.resolve()),
            "usage_record_sha256": "e" * 64,
            "usage_record_sha256s": ["e" * 64],
            "terminal_results": [],
            "lifecycle_invocations": [],
            "gpu_execution_ledger_path": str((tmp_path / "gpu.jsonl").resolve()),
            "gpu_admission_lock_path": str((tmp_path / "gpu.lock").resolve()),
            "validated_output": {"parameter_count": 1},
            "validation_scores_changed_execution": False,
            "commercial_claim_authorized": False,
        },
    )
    item = campaign.TrainingInput(
        outer_fold=0,
        seed=20260828,
        cache_dir=tmp_path / "cache",
        cache_manifest_sha256="a" * 64,
        proposer_stack=tmp_path / "proposer.npz",
        proposer_stack_sha256="b" * 64,
    )

    with pytest.raises(campaign.CampaignError, match="identity/governance"):
        fixed._run_promotion_training(
            run_root=run_root,
            item=item,
            variant="H0_no_factor",
            selection={
                "selected_release_mode": "raw_anchor",
                "selected_parameter_count": 1,
            },
            governance=governance,
            target_sealed_capability_receipt=(
                tmp_path / campaign.TARGET_SEALED_CAPABILITY_NAME
            ),
            python=tmp_path / "python",
            trainer=tmp_path / "trainer.py",
            wrapper=tmp_path / "wrapper.py",
            promotion_authorization=promotion_authorization,
            gpu_lock=tmp_path / "gpu.lock",
            gpu_ledger=tmp_path / "gpu.jsonl",
            usage_ledger=usage_ledger,
            device="cpu",
            amp=False,
            smoke_test=True,
            command_runner=lambda command, timeout: (_ for _ in ()).throw(
                AssertionError("stale completion must not execute work")
            ),
        )


@pytest.mark.parametrize("drift", ("input_receipt", "model_source"))
def test_completed_prediction_rejects_current_binding_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    keyword, completion_path, _ = _completed_prediction_resume_fixture(tmp_path)
    receipt = campaign.load_json(completion_path, "test prediction completion")
    if drift == "input_receipt":
        receipt["input_receipt"]["sha256"] = "f" * 64
    else:
        receipt["promotion_model_source"]["checkpoint"]["sha256"] = "f" * 64
    _rewrite_content_document(completion_path, receipt)
    monkeypatch.setattr(
        campaign, "validate_completion_receipt_usage", lambda *args, **kwargs: []
    )

    with pytest.raises(campaign.CampaignError, match="identity/governance/input"):
        fixed._run_prediction(
            **keyword,
            command_runner=lambda command, timeout: (_ for _ in ()).throw(
                AssertionError("stale completion must not execute work")
            ),
        )


def test_prediction_re_resolves_v8_model_source_before_completed_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = campaign.TrainingInput(
        outer_fold=0,
        seed=20260828,
        cache_dir=tmp_path / "cache",
        cache_manifest_sha256="a" * 64,
        proposer_stack=tmp_path / "proposer.npz",
        proposer_stack_sha256="b" * 64,
    )
    provided = builder.PromotionModelSource(
        kind="local_training",
        receipt_path=tmp_path / "receipt.json",
        output_dir=tmp_path / "output",
        checkpoint=tmp_path / "output/best.pt",
        scaler=tmp_path / "output/scaler.json",
        scientific_signature_sha256="c" * 64,
        artifacts={},
        receipt={
            "outer_fold": 0,
            "seed": 20260828,
            "variant": "H0_no_factor",
        },
    )
    current = builder.PromotionModelSource(
        **{
            **provided.__dict__,
            "scientific_signature_sha256": "d" * 64,
        }
    )
    monkeypatch.setattr(
        builder,
        "resolve_promotion_model_source",
        lambda **kwargs: current,
    )

    with pytest.raises(campaign.CampaignError, match="live resolution"):
        fixed._run_prediction(
            run_root=tmp_path / "run",
            item=item,
            model_source=provided,
            predict_input=tmp_path / "input.npz",
            input_receipt=tmp_path / "input.receipt.json",
            selection={
                "selected_variant": "H0_no_factor",
                "selected_release_mode": "raw_anchor",
            },
            governance={},
            target_sealed_capability_receipt=(
                tmp_path / campaign.TARGET_SEALED_CAPABILITY_NAME
            ),
            python=tmp_path / "python",
            trainer=tmp_path / "trainer.py",
            wrapper=tmp_path / "wrapper.py",
            promotion_authorization=tmp_path / "authorization.json",
            gpu_lock=tmp_path / "gpu.lock",
            gpu_ledger=tmp_path / "gpu.jsonl",
            usage_ledger=tmp_path / "usage.jsonl",
            device="cpu",
            amp=False,
            command_runner=lambda command, timeout: (_ for _ in ()).throw(
                AssertionError("stale model source must not execute work")
            ),
            project_root=tmp_path,
        )


def test_completed_prediction_rejects_current_invocation_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keyword, completion_path, invocation_path = _completed_prediction_resume_fixture(
        tmp_path
    )
    invocation = campaign.load_json(invocation_path, "test prediction invocation")
    invocation["governance"] = {
        **invocation["governance"],
        "selection_lock": {"sha256": "f" * 64},
    }
    _rewrite_content_document(invocation_path, invocation)
    receipt = campaign.load_json(completion_path, "test prediction completion")
    receipt["attempt_invocation"] = campaign.bind_file(invocation_path)
    _rewrite_content_document(completion_path, receipt)
    monkeypatch.setattr(
        campaign, "validate_completion_receipt_usage", lambda *args, **kwargs: []
    )

    with pytest.raises(campaign.CampaignError, match="current input/model/governance ABI"):
        fixed._run_prediction(
            **keyword,
            command_runner=lambda command, timeout: (_ for _ in ()).throw(
                AssertionError("stale completion must not execute work")
            ),
        )


def _prediction_attempt_publication_inputs() -> tuple[
    dict[str, Any], dict[str, Any], list[str]
]:
    context = {
        "campaign_revision": fixed.CAMPAIGN_REVISION,
        "infrastructure_revision": fixed.INFRASTRUCTURE_REVISION,
        "outer_fold": 0,
        "seed": 20260828,
        "variant": "H0_no_factor",
        "release_mode": "raw_anchor",
        "attempt_number": 0,
    }
    command = ["python", "trainer.py", "--predict-only"]
    invocation = {
        "schema_version": 1,
        "classification": "test_promotion_prediction_invocation",
        **context,
        "trainer_command": command,
        "target_access_authorized": False,
    }
    return invocation, context, command


@pytest.mark.parametrize(
    ("kill_stage", "staged_names"),
    (
        ("prediction_attempt_staged", ()),
        ("prediction_attempt_invocation_durable", ("invocation.json",)),
        (
            "prediction_attempt_invocations_durable",
            ("execution_invocation.json", "invocation.json"),
        ),
    ),
)
def test_prediction_attempt_publication_recovers_exact_killed_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kill_stage: str,
    staged_names: tuple[str, ...],
) -> None:
    root = tmp_path / "attempts"
    invocation, context, command = _prediction_attempt_publication_inputs()

    def kill(stage: str, _path: Path) -> None:
        if stage == kill_stage:
            raise RuntimeError(f"killed-at-{stage}")

    monkeypatch.setattr(campaign, "_PUBLICATION_FAULT_HOOK", kill)
    with pytest.raises(RuntimeError, match=f"killed-at-{kill_stage}"):
        fixed._publish_prediction_attempt_directory(
            root,
            attempt_number=0,
            attempt_invocation=invocation,
            context=context,
            workload_command=command,
        )
    staging = root / ".attempt_000.staging"
    final = root / "attempt_000"
    assert staging.is_dir()
    assert not final.exists()
    assert tuple(sorted(path.name for path in staging.iterdir())) == staged_names

    monkeypatch.setattr(campaign, "_PUBLICATION_FAULT_HOOK", None)
    assert (
        fixed._publish_prediction_attempt_directory(
            root,
            attempt_number=0,
            attempt_invocation=invocation,
            context=context,
            workload_command=command,
        )
        == final
    )
    assert not staging.exists()
    assert fixed._prediction_attempt_directories(root) == [final]
    for name in ("invocation.json", "execution_invocation.json"):
        status = (final / name).stat(follow_symlinks=False)
        assert stat.S_ISREG(status.st_mode)
        assert stat.S_IMODE(status.st_mode) == 0o444
        assert status.st_nlink == 1
    campaign._validate_execution_invocation(
        final / "execution_invocation.json",
        phase="promotion_prediction",
        context=context,
        unit_invocation_path=final / "invocation.json",
        workload_command=command,
    )


def test_prediction_attempt_publication_is_exact_no_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "attempts"
    invocation, context, command = _prediction_attempt_publication_inputs()
    sentinel = b"do-not-replace"

    def collide_before_rename(stage: str, final: Path) -> None:
        if stage == "prediction_attempt_invocations_durable":
            final.mkdir()
            (final / "sentinel").write_bytes(sentinel)

    monkeypatch.setattr(
        campaign, "_PUBLICATION_FAULT_HOOK", collide_before_rename
    )
    with pytest.raises(campaign.CampaignError, match="without replacement"):
        fixed._publish_prediction_attempt_directory(
            root,
            attempt_number=0,
            attempt_invocation=invocation,
            context=context,
            workload_command=command,
        )
    assert (root / "attempt_000/sentinel").read_bytes() == sentinel
    assert (root / ".attempt_000.staging/invocation.json").is_file()
    assert (root / ".attempt_000.staging/execution_invocation.json").is_file()


def test_prediction_attempt_discovery_rejects_nonprefix_staging(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "attempts/.attempt_000.staging"
    staging.mkdir(parents=True)
    campaign.create_once_json(
        staging / "execution_invocation.json", {"classification": "out-of-order"}
    )
    with pytest.raises(campaign.CampaignError, match="exact prefix"):
        fixed._prediction_attempt_directories(staging.parent)


def test_promotion_training_uses_kill_safe_execution_publisher() -> None:
    source = inspect.getsource(fixed._run_promotion_training)
    assert "discovery._publish_execution_directory(" in source
    assert "executions_root / f\"execution_{execution_number:03d}\"" not in source


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_cache(tmp_path: Path) -> Path:
    root = tmp_path / "cache"
    root.mkdir(parents=True)
    rows = 6
    frame = pd.DataFrame(
        {
            "cache_index": [100, 101, 102, 103, 104, 105],
            "fold": [0, 0, 1, 1, 2, 2],
            "window_number": [0, 1, 0, 1, 0, 1],
            "classical_rr_bpm": [18, 19, 20, 21, 22, 23],
            # These values must never be materialized into the sanitized input.
            "identity": ["SECRET"] * rows,
            "protocol": ["SECRET"] * rows,
            "rr_bpm": [999.0] * rows,
            "reference_valid": [True] * rows,
            "reference_quality": [1.0] * rows,
        }
    )
    frame.to_csv(root / "metadata.csv", index=False)
    np.save(root / "node_features.npy", np.ones((rows, 12, 571), np.float32))
    np.save(
        root / "candidate_bpm.npy",
        np.tile(np.linspace(6, 39, 12, dtype=np.float32), (rows, 1)),
    )
    np.save(root / "candidate_mask.npy", np.ones((rows, 12), bool))
    np.save(root / "joint_radar_mask.npy", np.ones((rows, 3), bool))
    _write_json(root / "manifest.json", {"complete": True, "rows": rows})
    authorization = tmp_path / builder.PROMOTION_AUTH_RELATIVE
    _write_json(authorization, {"promotion_authorized": True})
    _publish_outer_pack(root, [100, 101])
    return root


def _publish_outer_pack(
    cache: Path,
    index: list[int],
    *,
    proof_index: list[int] | None = None,
    target_field: bool = False,
    feature_value: float = 1.0,
) -> Path:
    rows = len(index)
    output = cache / "outer_predict_input.npz"
    arrays: dict[str, Any] = {
        "cache_index": np.asarray(index, np.int64),
        "node_features": np.full((rows, 12, 571), feature_value, np.float32),
        "candidate_rr_bpm": np.tile(
            np.linspace(6, 39, 12, dtype=np.float32), (rows, 1)
        ),
        "candidate_mask": np.ones((rows, 12), bool),
        "joint_radar_mask": np.ones((rows, 3), bool),
        "proposer_anchor_bpm": np.asarray(
            [18.5 + number for number in range(rows)], np.float32
        ),
        "proposer_anchor_std_bpm": np.ones(rows, np.float32),
        "proposer_anchor_available": np.ones(rows, bool),
        "classical_rr_bpm": np.asarray(
            [18.0 + number for number in range(rows)], np.float32
        ),
        "session_reset": np.asarray(
            [True] + [False] * max(0, rows - 1), dtype=bool
        ),
    }
    if target_field:
        arrays["target_rr_bpm"] = np.zeros(rows, np.float32)
    np.savez_compressed(output, **arrays)
    authorization = cache.parent / builder.PROMOTION_AUTH_RELATIVE
    proof = np.asarray(index if proof_index is None else proof_index, np.int64)
    opaque = {"path": "opaque", "sha256": "a" * 64, "bytes": 1}
    value = {
        "schema_version": 1,
        "classification": builder.OUTER_PACK_CLASSIFICATION,
        "campaign_id": campaign.CAMPAIGN_ID,
        "campaign_revision": builder.CAMPAIGN_REVISION,
        "outer_fold": 0,
        "seed": 20260828,
        "row_count": rows,
        "fields": list(builder.SAFE_OUTPUT_FIELDS),
        "exact_allowlist": True,
        "forbidden_fields_emitted": False,
        "reference_identity_protocol_quality_decoded": False,
        "legacy_index": dict(opaque),
        "legacy_cache_manifest": dict(opaque),
        "legacy_proposer_stack": dict(opaque),
        "promotion_authorization": campaign.bind_file(authorization),
        "output": campaign.bind_file(output),
        "global_cache_index_sha256": hashlib.sha256(
            np.ascontiguousarray(proof, dtype=np.int64).view(np.uint8)
        ).hexdigest(),
        "object_arrays": False,
        "pickle": False,
        "commercial_or_confirmatory_claim_allowed": False,
    }
    manifest = cache / "OUTER_PREDICTION_PACK_MANIFEST.json"
    if manifest.exists():
        _rewrite_content_document(manifest, value)
        manifest.chmod(0o444)
    else:
        campaign.create_once_json(manifest, value)
    return output


def _anchor(path: Path, index: list[int], *, target: bool = False) -> None:
    arrays: dict[str, Any] = {
        "cache_index": np.asarray(index, np.int64),
        "proposer_anchor_bpm": np.asarray([18.5 + i for i in range(len(index))], np.float32),
        "proposer_anchor_std_bpm": np.ones(len(index), np.float32),
        "proposer_anchor_available": np.ones(len(index), bool),
        "outer_fold": np.asarray(0, np.int16),
        "seed": np.asarray(20260828, np.int64),
    }
    if target:
        arrays["target_rr_bpm"] = np.zeros(len(index), np.float32)
    np.savez_compressed(path, **arrays)


def _governance() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        {
            "selected_variant": "H0_no_factor",
            "selected_release_mode": "raw_anchor",
        },
        {"promotion_authorized": True},
        {
            "contract": {"sha256": "a" * 64},
            "pretrain_authorization": {"sha256": "b" * 64},
            "selection_lock": {"sha256": "c" * 64},
            "promotion_authorization": {"sha256": "d" * 64},
        },
    )


def _rewrite_content_document(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    document = json.loads(json.dumps(value))
    document.pop("content_sha256", None)
    document["content_sha256"] = campaign.semantic_sha256(document)
    path.chmod(0o644)
    _write_json(path, document)
    return document


def _completed_prediction_resume_fixture(
    tmp_path: Path,
) -> tuple[dict[str, Any], Path, Path]:
    run_root = tmp_path / "prediction-resume"
    usage_ledger = tmp_path / "usage.jsonl"
    campaign.bind_run_usage_ledger(run_root, usage_ledger, execution_scope="promotion")
    item = campaign.TrainingInput(
        outer_fold=3,
        seed=20260828,
        cache_dir=tmp_path / "cache",
        cache_manifest_sha256="a" * 64,
        proposer_stack=tmp_path / "proposer.npz",
        proposer_stack_sha256="b" * 64,
    )
    model_root = run_root / "training/outer_3_seed_20260828/attempt_000/output"
    model_root.mkdir(parents=True)
    checkpoint = model_root / "best.pt"
    scaler = model_root / "scaler.json"
    checkpoint.write_bytes(b"checkpoint")
    _write_json(scaler, {"scale": 1})
    predict_input = tmp_path / "sanitized.npz"
    np.savez_compressed(predict_input, cache_index=np.asarray([100], np.int64))
    input_receipt = tmp_path / "sanitized.receipt.json"
    _write_json(input_receipt, {"safe": True})
    promotion_authorization = tmp_path / "promotion_authorization.json"
    _write_json(promotion_authorization, {"promotion_authorized": True})
    pretrain_authorization = tmp_path / "pretrain.json"
    _write_json(pretrain_authorization, {"training_authorized": True})
    selection = {
        "selected_variant": "H0_no_factor",
        "selected_release_mode": "raw_anchor",
    }
    governance = {
        "selection_lock": {"sha256": "c" * 64},
        "pretrain_authorization": {
            "path": str(pretrain_authorization),
            "sha256": campaign.sha256_file(pretrain_authorization),
        },
    }
    python = tmp_path / "python"
    trainer = tmp_path / "trainer.py"
    wrapper = tmp_path / "wrapper.py"
    gpu_lock = tmp_path / "gpu.lock"
    gpu_ledger = tmp_path / "gpu.jsonl"
    attempt_root = run_root / "predictions/outer_3_seed_20260828/attempts/attempt_000"
    output_dir = attempt_root / "output"
    capability = tmp_path / campaign.TARGET_SEALED_CAPABILITY_NAME
    _write_json(capability, {"test_only": True})
    expected_context = {
        "campaign_revision": fixed.CAMPAIGN_REVISION,
        "infrastructure_revision": fixed.INFRASTRUCTURE_REVISION,
        "outer_fold": 3,
        "seed": 20260828,
        "variant": "H0_no_factor",
        "release_mode": "raw_anchor",
        "attempt_number": 0,
    }
    command = fixed._prediction_command(
        python=python,
        trainer=trainer,
        predict_input=predict_input,
        checkpoint=checkpoint,
        scaler=scaler,
        output_dir=output_dir,
        target_sealed_capability_receipt=capability,
        expected_admitted_context=expected_context,
        outer_fold=3,
        seed=20260828,
        variant="H0_no_factor",
        release_mode="raw_anchor",
        promotion_authorization=promotion_authorization,
        device="cpu",
        amp=False,
    )
    model_source_binding = {
        "kind": "legacy_local_training_lifecycle_test",
        "receipt": {"content_sha256": "d" * 64},
        "checkpoint": campaign.bind_file(checkpoint),
        "scaler": campaign.bind_file(scaler),
        "scientific_signature_sha256": "d" * 64,
    }
    invocation_path = attempt_root / "invocation.json"
    campaign.create_once_json(
        invocation_path,
        {
            "schema_version": 1,
            "classification": "adaptive_v3r1_target_free_promotion_prediction_invocation",
            "campaign_id": campaign.CAMPAIGN_ID,
            "campaign_revision": fixed.CAMPAIGN_REVISION,
            "infrastructure_revision": fixed.INFRASTRUCTURE_REVISION,
            "outer_fold": 3,
            "seed": 20260828,
            "variant": "H0_no_factor",
            "release_mode": "raw_anchor",
            "attempt_number": 0,
            "governance": governance,
            "model_source_kind": "legacy_local_training_lifecycle_test",
            "model_source_receipt": {"content_sha256": "d" * 64},
            "scientific_signature_sha256": "d" * 64,
            "predict_input": campaign.bind_file(predict_input),
            "input_receipt": campaign.bind_file(input_receipt),
            "checkpoint": campaign.bind_file(checkpoint),
            "scaler": campaign.bind_file(scaler),
            "trainer_command": command,
            "usage_ledger_identity": campaign.bind_file(
                run_root / "GPU_USAGE_LEDGER_IDENTITY.json"
            ),
            "target_access_authorized": False,
        },
    )
    completion_path = run_root / "predictions/outer_3_seed_20260828/completion_receipt.json"
    campaign.create_once_json(
        completion_path,
        {
            "schema_version": 1,
            "classification": "adaptive_v3r1_target_free_promotion_prediction_completion",
            "campaign_id": campaign.CAMPAIGN_ID,
            "campaign_revision": fixed.CAMPAIGN_REVISION,
            "infrastructure_revision": fixed.INFRASTRUCTURE_REVISION,
            "outer_fold": 3,
            "seed": 20260828,
            "variant": "H0_no_factor",
            "release_mode": "raw_anchor",
            "output_dir": str(output_dir),
            "attempt_invocation": campaign.bind_file(invocation_path),
            "promotion_model_source": model_source_binding,
            "governance": governance,
            "selection_lock_sha256": "c" * 64,
            "promotion_authorization_sha256": campaign.sha256_file(
                promotion_authorization
            ),
            "input_receipt": campaign.bind_file(input_receipt),
            "usage_ledger_path": str(usage_ledger.resolve()),
            "usage_record_sha256": "e" * 64,
            "usage_record_sha256s": ["e" * 64],
            "terminal_results": [],
            "lifecycle_invocations": [],
            "gpu_execution_ledger_path": str(gpu_ledger.resolve()),
            "gpu_admission_lock_path": str(gpu_lock.resolve()),
            "validated_output": {},
            "target_fields_accessed_or_emitted": False,
            "commercial_claim_authorized": False,
        },
    )
    return (
        {
            "run_root": run_root,
            "item": item,
            "training_receipt": {"content_sha256": "d" * 64},
            "predict_input": predict_input,
            "input_receipt": input_receipt,
            "selection": selection,
            "governance": governance,
            "target_sealed_capability_receipt": capability,
            "python": python,
            "trainer": trainer,
            "wrapper": wrapper,
            "promotion_authorization": promotion_authorization,
            "gpu_lock": gpu_lock,
            "gpu_ledger": gpu_ledger,
            "usage_ledger": usage_ledger,
            "device": "cpu",
            "amp": False,
        },
        completion_path,
        invocation_path,
    )


def test_builder_emits_exact_target_identity_protocol_qc_free_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _synthetic_cache(tmp_path)
    anchor = cache / "outer_predict_input.npz"
    monkeypatch.setattr(builder, "validate_promotion_authorization", lambda *a, **k: _governance())
    output = tmp_path / "sanitized.npz"
    receipt_path = tmp_path / "sanitized.receipt.json"
    receipt = builder.build_locked_input(
        project_root=tmp_path,
        cache_dir=cache,
        proposer_anchor=anchor,
        outer_fold=0,
        seed=20260828,
        output=output,
        receipt_path=receipt_path,
    )
    assert receipt["exact_cache_index_cover"] is True
    assert (
        receipt[
            "target_reference_qc_identity_protocol_columns_physically_present"
        ]
        is False
    )
    validation = builder.validate_sanitized_input(
        output, expected_index=np.asarray([100, 101], np.int64)
    )
    assert validation["fields"] == list(builder.SAFE_OUTPUT_FIELDS)
    with np.load(output, allow_pickle=False) as archive:
        assert set(archive.files) == set(builder.SAFE_OUTPUT_FIELDS)
        assert not any(
            token in name.lower()
            for name in archive.files
            for token in ("target", "reference", "identity", "protocol", "quality", "qc")
        )
        assert archive["session_reset"].tolist() == [True, False]


def test_builder_resume_rederives_output_and_rejects_live_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _synthetic_cache(tmp_path)
    anchor = cache / "outer_predict_input.npz"
    monkeypatch.setattr(
        builder, "validate_promotion_authorization", lambda *a, **k: _governance()
    )
    output = tmp_path / "sanitized.npz"
    receipt_path = tmp_path / "sanitized.receipt.json"
    first = builder.build_locked_input(
        project_root=tmp_path,
        cache_dir=cache,
        proposer_anchor=anchor,
        outer_fold=0,
        seed=20260828,
        output=output,
        receipt_path=receipt_path,
    )
    resumed = builder.build_locked_input(
        project_root=tmp_path,
        cache_dir=cache,
        proposer_anchor=anchor,
        outer_fold=0,
        seed=20260828,
        output=output,
        receipt_path=receipt_path,
    )
    assert resumed == first

    _publish_outer_pack(cache, [100, 101], feature_value=9.0)
    with pytest.raises(campaign.CampaignError, match="live source values"):
        builder.build_locked_input(
            project_root=tmp_path,
            cache_dir=cache,
            proposer_anchor=anchor,
            outer_fold=0,
            seed=20260828,
            output=output,
            receipt_path=receipt_path,
        )


def test_builder_resume_rejects_current_governance_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _synthetic_cache(tmp_path)
    anchor = cache / "outer_predict_input.npz"
    current = [_governance()]
    monkeypatch.setattr(
        builder,
        "validate_promotion_authorization",
        lambda *args, **kwargs: current[0],
    )
    output = tmp_path / "sanitized.npz"
    receipt_path = tmp_path / "sanitized.receipt.json"
    builder.build_locked_input(
        project_root=tmp_path,
        cache_dir=cache,
        proposer_anchor=anchor,
        outer_fold=0,
        seed=20260828,
        output=output,
        receipt_path=receipt_path,
    )
    selection, authorization, governance = _governance()
    governance = {
        **governance,
        "selection_lock": {"sha256": "e" * 64},
    }
    current[0] = (selection, authorization, governance)
    with pytest.raises(campaign.CampaignError, match="current exact provenance"):
        builder.build_locked_input(
            project_root=tmp_path,
            cache_dir=cache,
            proposer_anchor=anchor,
            outer_fold=0,
            seed=20260828,
            output=output,
            receipt_path=receipt_path,
        )


@pytest.mark.parametrize("index", ([100], [100, 100], [100, 102]))
def test_builder_rejects_missing_duplicate_or_extra_cache_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    index: list[int],
) -> None:
    cache = _synthetic_cache(tmp_path)
    anchor = _publish_outer_pack(cache, index, proof_index=[100, 101])
    monkeypatch.setattr(builder, "validate_promotion_authorization", lambda *a, **k: _governance())
    with pytest.raises(campaign.CampaignError, match="cache-index proof"):
        builder.build_locked_input(
            project_root=tmp_path,
            cache_dir=cache,
            proposer_anchor=anchor,
            outer_fold=0,
            seed=20260828,
            output=tmp_path / "sanitized.npz",
            receipt_path=tmp_path / "receipt.json",
        )


def test_builder_rejects_target_bearing_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _synthetic_cache(tmp_path)
    anchor = _publish_outer_pack(cache, [100, 101], target_field=True)
    monkeypatch.setattr(builder, "validate_promotion_authorization", lambda *a, **k: _governance())
    with pytest.raises(campaign.CampaignError, match="allow-list drifted"):
        builder.build_locked_input(
            project_root=tmp_path,
            cache_dir=cache,
            proposer_anchor=anchor,
            outer_fold=0,
            seed=20260828,
            output=tmp_path / "sanitized.npz",
            receipt_path=tmp_path / "receipt.json",
        )


def test_safe_anchor_accepts_only_canonical_v2_owner_chain_and_resumes(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        campaign.CampaignError, match="V8R4 forbids every legacy V2 safe-anchor path"
    ):
        fixed._safe_anchor_from_locked_v2(
            source=tmp_path / "legacy.npz",
            output=tmp_path / "safe_anchor.npz",
            receipt_path=tmp_path / "safe_anchor.receipt.json",
            outer_fold=0,
            seed=20260828,
            project_root=tmp_path,
        )
    return

    source = (
        ROOT
        / fixed.DEFAULT_V2_LOCKED_ROOT
        / "outer_0_seed_20260828/work/no_action_raw_hcs.npz"
    )
    output = tmp_path / "safe_anchor.npz"
    receipt_path = tmp_path / "safe_anchor.receipt.json"
    receipt = fixed._safe_anchor_from_locked_v2(
        source=source,
        output=output,
        receipt_path=receipt_path,
        outer_fold=0,
        seed=20260828,
        project_root=ROOT,
    )
    assert set(receipt["source_owner_chain"]) == {
        "pretarget_release_lock",
        "pretest_lock",
        "predictions_seal",
        "derived_inference_lock",
        "no_action_stage_receipt",
    }
    assert receipt["target_reference_identity_protocol_qc_fields_emitted"] is False
    assert (
        fixed._safe_anchor_from_locked_v2(
            source=source,
            output=output,
            receipt_path=receipt_path,
            outer_fold=0,
            seed=20260828,
            project_root=ROOT,
        )
        == receipt
    )


def test_safe_anchor_rejects_stripped_target_derived_looking_alternate_root(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "forged_v2/units/outer_0_seed_20260828/work/no_action_raw_hcs.npz"
    )
    source.parent.mkdir(parents=True)
    np.savez_compressed(
        source,
        cache_index=np.asarray([100, 101], np.int64),
        fallback_rr_bpm=np.asarray([30.0, 31.0], np.float32),
        fallback_std_bpm=np.ones(2, np.float32),
        fallback_available=np.ones(2, bool),
        outer_fold=np.asarray(0, np.int16),
        seed=np.asarray(20260828, np.int64),
        target_fields_present=np.asarray(False),
    )
    output = tmp_path / "safe_anchor.npz"
    receipt_path = tmp_path / "safe_anchor.receipt.json"
    with pytest.raises(
        campaign.CampaignError, match="V8R4 forbids every legacy V2 safe-anchor path"
    ):
        fixed._safe_anchor_from_locked_v2(
            source=source,
            output=output,
            receipt_path=receipt_path,
            outer_fold=0,
            seed=20260828,
            project_root=ROOT,
        )
    assert not output.exists()
    assert not receipt_path.exists()


def test_safe_anchor_rejects_self_rehashed_forged_canonical_release_root(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        campaign.CampaignError, match="V8R4 forbids every legacy V2 safe-anchor path"
    ):
        fixed._safe_anchor_from_locked_v2(
            source=tmp_path / "forged.npz",
            output=tmp_path / "safe_anchor.npz",
            receipt_path=tmp_path / "safe_anchor.receipt.json",
            outer_fold=0,
            seed=20260828,
            project_root=tmp_path,
        )
    return

    canonical_root = tmp_path / fixed.DEFAULT_V2_LOCKED_ROOT
    source = canonical_root / "outer_0_seed_20260828/work/no_action_raw_hcs.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"forged target-derived-looking archive")
    source.chmod(0o600)
    release_path = canonical_root.parent / "pretarget_release_lock.json"
    forged_release: dict[str, Any] = {
        "schema_version": 1,
        "classification": "locked_hcs_pretarget_release_lock",
        "status": "all_target_free_boundaries_complete",
        "boundaries": {"primary_predictions": {}},
    }
    forged_release["content_sha256"] = campaign.semantic_sha256(forged_release)
    _write_json(release_path, forged_release)
    release_path.chmod(0o444)
    assert campaign.canonical_content_sha256(forged_release) == forged_release[
        "content_sha256"
    ]
    assert campaign.sha256_file(release_path) != (
        fixed.V2_PRETARGET_RELEASE_LOCK_FILE_SHA256
    )

    with pytest.raises(campaign.CampaignError, match="release lock SHA-256 drifted"):
        fixed._safe_anchor_from_locked_v2(
            source=source,
            output=tmp_path / "safe_anchor.npz",
            receipt_path=tmp_path / "safe_anchor.receipt.json",
            outer_fold=0,
            seed=20260828,
            project_root=tmp_path,
        )


def test_verified_v2_source_reader_rejects_alias_and_in_place_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aliased = tmp_path / "aliased.npz"
    aliased.write_bytes(b"immutable raw anchor")
    aliased.chmod(0o600)
    alias = tmp_path / "hardlink.npz"
    alias.hardlink_to(aliased)
    with pytest.raises(campaign.CampaignError, match="single-link regular file"):
        fixed._read_verified_single_link_file(
            aliased,
            expected_sha256=campaign.sha256_file(aliased),
            expected_bytes=aliased.stat().st_size,
            expected_mode=0o600,
            label="test raw anchor",
        )

    racing = tmp_path / "racing.npz"
    racing.write_bytes(b"original raw anchor")
    racing.chmod(0o600)
    expected_sha256 = campaign.sha256_file(racing)
    expected_bytes = racing.stat().st_size
    original_read = fixed.os.read
    mutated = False

    def mutating_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        block = original_read(descriptor, count)
        if block and not mutated:
            mutated = True
            racing.write_bytes(b"mutated raw anchor")
        return block

    monkeypatch.setattr(fixed.os, "read", mutating_read)
    with pytest.raises(campaign.CampaignError, match="changed while it was verified"):
        fixed._read_verified_single_link_file(
            racing,
            expected_sha256=expected_sha256,
            expected_bytes=expected_bytes,
            expected_mode=0o600,
            label="test raw anchor",
        )
    assert mutated is True


def _prediction_arrays(index: np.ndarray) -> dict[str, np.ndarray]:
    rows = len(index)
    raw = np.arange(rows, dtype=np.float32) + 18.0
    hard = raw + 0.25
    return {
        "cache_index": index.astype(np.int64),
        "prediction_bpm": raw.copy(),
        "prediction_available": np.ones(rows, bool),
        "raw_anchor_bpm": raw,
        "raw_anchor_available": np.ones(rows, bool),
        "hard_source_bpm": hard,
        "hard_source_available": np.ones(rows, bool),
        "selected_source_probability": np.full(rows, 0.5, np.float32),
        "selected_source_code": np.zeros(rows, np.int16),
        "source_scale_bpm": np.ones(rows, np.float32),
        "quality": np.ones(rows, np.float32),
        "factor_probabilities": np.tile(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], np.float32), (rows, 1)
        ),
        "spike_rate": np.full(rows, 0.1, np.float32),
    }


def _prediction_fixture(tmp_path: Path, index: np.ndarray) -> tuple[Path, Path, Path, Path]:
    output = tmp_path / "prediction"
    output.mkdir(parents=True)
    predict_input = tmp_path / "input.npz"
    np.savez_compressed(predict_input, cache_index=index)
    checkpoint = tmp_path / "best.pt"
    scaler = tmp_path / "scaler.json"
    checkpoint.write_bytes(b"checkpoint")
    _write_json(scaler, {"scale": [1]})
    np.savez_compressed(output / "predictions.npz", **_prediction_arrays(index))
    _write_json(
        output / "prediction_manifest.json",
        {
            "outer_fold": 0,
            "seed": 20260828,
            "variant": "H0_no_factor",
            "release_mode": "raw_anchor",
            "target_fields_emitted": False,
            "commercial_claim_authorized": False,
            "predict_input_sha256": campaign.sha256_file(predict_input),
            "checkpoint_sha256": campaign.sha256_file(checkpoint),
            "scaler_sha256": campaign.sha256_file(scaler),
        },
    )
    return output, predict_input, checkpoint, scaler


def test_prediction_validator_is_exact_cover_and_no_target(tmp_path: Path) -> None:
    index = np.asarray([100, 101], np.int64)
    output, predict_input, checkpoint, scaler = _prediction_fixture(tmp_path, index)
    result = fixed.validate_target_free_prediction(
        output,
        expected_index=index,
        outer_fold=0,
        seed=20260828,
        variant="H0_no_factor",
        release_mode="raw_anchor",
        predict_input=predict_input,
        checkpoint=checkpoint,
        scaler=scaler,
    )
    assert result["target_fields_present"] is False

    arrays = _prediction_arrays(index)
    arrays["target_rr_bpm"] = np.zeros(2, np.float32)
    np.savez_compressed(output / "predictions.npz", **arrays)
    with pytest.raises(campaign.CampaignError, match="allow-list mismatch"):
        fixed.validate_target_free_prediction(
            output,
            expected_index=index,
            outer_fold=0,
            seed=20260828,
            variant="H0_no_factor",
            release_mode="raw_anchor",
            predict_input=predict_input,
            checkpoint=checkpoint,
            scaler=scaler,
        )


def _scientific_training_output(
    root: Path, *, outer_fold: int, seed: int, variant: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    signature = {
        "schema_version": 1,
        "campaign_id": campaign.CAMPAIGN_ID,
        "campaign_revision": builder.CAMPAIGN_REVISION,
        "classification": "synthetic_implementation_smoke_test",
        "contract_file_sha256": campaign.CONTRACT_FILE_SHA256,
        "outer_fold": outer_fold,
        "validation_fold": (outer_fold + 1) % 6,
        "seed": seed,
        "variant": variant,
        "model": {"architecture": "fixed"},
        "optimization": {"loss_optimizer_schedule": "fixed"},
        "source_bindings": {"snapshot": "frozen-v8"},
        "input_bindings": {"cache_and_proposer_stack": "exact"},
        "pretrain_authorization": {"authorized": True},
        "population": {"checkpoint_selection_inputs": "fixed"},
        "batching_execution": {
            "training_batch_unit": "physical_session_group",
            "temporal_schedule": "aligned_tbptt_chunk_rounds",
            "padding_inert": True,
            "per_session_cvar_before_group_reduction": True,
            "valid_length_spike_weighting": True,
            "prediction_batch_sessions": 4,
        },
        "checkpoint_selection": {
            "scope": "outer_validation_only_hard_source",
            "commercial_gates": "fixed",
            "lexicographic_key": "fixed",
        },
    }
    signature_sha = campaign.semantic_sha256(signature)
    _write_json(
        root / "run_manifest.json",
        {
            "scientific_signature": signature,
            "scientific_signature_sha256": signature_sha,
        },
    )
    _write_json(root / "scaler.json", {"scale": [1.0]})
    _write_json(root / "history.json", {"epochs": [1]})
    (root / "last.pt").write_bytes(b"last")
    (root / "best.pt").write_bytes(b"best")
    _write_json(
        root / "checkpoint_selection_lock.json",
        {"scientific_signature_sha256": signature_sha},
    )
    np.savez_compressed(root / "validation_predictions.npz", cache_index=[1])
    _write_json(root / "validation_metrics.json", {"target_sealed": True})
    artifacts = {
        name: campaign.bind_file(root / name)
        for name in campaign.REQUIRED_TRAIN_OUTPUTS
    }
    validated = {
        "campaign_revision": builder.CAMPAIGN_REVISION,
        "outer_fold": outer_fold,
        "validation_fold": (outer_fold + 1) % 6,
        "seed": seed,
        "variant": variant,
        "parameter_count": 123,
        "validation_rows": 1,
        "valid_reference_rows": 1,
        "release_metrics": {},
        "physical_boundary": dict(campaign.DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION),
        "row_access_audit": {
            "campaign_revision": builder.CAMPAIGN_REVISION,
            "outer_fold": outer_fold,
            "outer_row_access_attempts": 0,
        },
        "artifacts": artifacts,
        "scientific_signature_sha256": signature_sha,
    }
    return validated, signature


def _discovery_reuse_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[
    Path,
    campaign.TrainingInput,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    fold, seed, variant = 3, 20260828, "H0_no_factor"
    output = tmp_path / "discovery/units/selected/attempt_000/output"
    validated, _ = _scientific_training_output(
        output, outer_fold=fold, seed=seed, variant=variant
    )
    selected_receipt_path = (
        tmp_path / "discovery/units/selected/completion_receipt.json"
    )
    selected_receipt = campaign.create_once_json(
        selected_receipt_path,
        {
            "schema_version": 1,
            "classification": "adaptive_v3r1_v8r4_discovery_unit_completion",
            "campaign_id": campaign.CAMPAIGN_ID,
            "campaign_revision": builder.CAMPAIGN_REVISION,
            "infrastructure_revision": builder.INFRASTRUCTURE_REVISION,
            "outer_test_opened": False,
            "outer_fold": fold,
            "validation_fold": (fold + 1) % 6,
            "seed": seed,
            "variant": variant,
            "invocation": {"path": "opaque", "sha256": "1" * 64, "bytes": 1},
            "usage_ledger_path": str((tmp_path / "usage.jsonl").resolve()),
            "usage_record_sha256": "2" * 64,
            "usage_record_sha256s": ["2" * 64],
            "terminal_results": [{}],
            "lifecycle_invocations": [{}],
            "gpu_execution_ledger_path": str((tmp_path / "execution.jsonl").resolve()),
            "gpu_admission_lock_path": str((tmp_path / "gpu.lock").resolve()),
            "validated_output": validated,
            "commercial_claim_authorized": False,
        },
    )
    units = []
    for key in campaign.EXPECTED_DISCOVERY_UNITS:
        if key == (fold, seed, variant):
            receipt_path = selected_receipt_path
        else:
            receipt_path = (
                tmp_path
                / "discovery/units"
                / f"outer_{key[0]}_seed_{key[1]}_{key[2]}"
                / "completion_receipt.json"
            )
            campaign.create_once_json(
                receipt_path,
                {
                    "schema_version": 1,
                    "campaign_id": campaign.CAMPAIGN_ID,
                    "outer_fold": key[0],
                    "seed": key[1],
                    "variant": key[2],
                },
            )
        units.append(
            {
                "outer_fold": key[0],
                "seed": key[1],
                "variant": key[2],
                "receipt": campaign.bind_file(receipt_path),
            }
        )
    seal_path = tmp_path / "discovery/DISCOVERY_COMPLETION_SEAL.json"
    campaign.create_once_json(
        seal_path,
        {
            "schema_version": 1,
            "classification": "adaptive_v3r1_v8r4_target_sealed_discovery_completion",
            "campaign_id": campaign.CAMPAIGN_ID,
            "campaign_revision": builder.CAMPAIGN_REVISION,
            "infrastructure_revision": builder.INFRASTRUCTURE_REVISION,
            "contract": {"sha256": campaign.CONTRACT_FILE_SHA256},
            "pretrain_authorization": {"sha256": "3" * 64},
            "training_shards": [{"outer_fold": 3}, {"outer_fold": 4}],
            "outer_runs": list(campaign.OUTER_RUNS),
            "seeds": list(campaign.SEEDS),
            "variants": list(campaign.VARIANTS),
            "completed_units": 18,
            "physical_boundary": dict(campaign.DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION),
            "validation_targets_only": True,
            "gpu_elapsed_seconds": 1.0,
            "gpu_hours_hard": campaign.GPU_HOURS_HARD,
            "gpu_usage_ledger": {"sha256": "4" * 64},
            "gpu_usage_ledger_path": str((tmp_path / "usage.jsonl").resolve()),
            "pre_discovery_efficiency_benchmark": {"receipt": {"sha256": "5" * 64}},
            "v8r3_successful_terminal_quarantine": {"sha256": "6" * 64},
            "ready_for_global_discovery_selection": True,
            "cross_outer_validation_reuse_present": True,
            "fully_nested_confirmatory_oof": False,
            "prospective_confirmation_required": True,
            "commercial_claim_authorized": False,
            "units": units,
        },
    )
    selection_path = tmp_path / "DISCOVERY_SELECTION_LOCK.json"
    promotion_path = tmp_path / "PROMOTION_AUTHORIZATION.json"
    _write_json(selection_path, {"selected_variant": variant})
    _write_json(promotion_path, {"promotion_authorized": True})
    selection = {
        "selected_variant": variant,
        "selected_parameter_count": 123,
        "discovery_completion_seal": campaign.bind_file(seal_path),
    }
    governance = {
        "selection_lock": campaign.bind_file(selection_path),
        "promotion_authorization": campaign.bind_file(promotion_path),
    }
    item = campaign.TrainingInput(
        outer_fold=fold,
        seed=seed,
        cache_dir=tmp_path / "cache",
        cache_manifest_sha256="a" * 64,
        proposer_stack=tmp_path / "proposer.npz",
        proposer_stack_sha256="b" * 64,
    )

    def validate_live(
        candidate: Path, *, outer_fold: int, seed: int, variant: str, cache_dir: Path
    ) -> dict[str, Any]:
        del cache_dir
        assert candidate == output
        assert (outer_fold, seed, variant) == (fold, item.seed, "H0_no_factor")
        return validated

    monkeypatch.setattr(campaign, "validate_training_output", validate_live)
    return tmp_path / "promotion", item, selection, governance, selected_receipt


def test_v8_scientific_signature_retains_checkpoint_selection(
    tmp_path: Path,
) -> None:
    _, signature = _scientific_training_output(
        tmp_path / "scientific",
        outer_fold=3,
        seed=20260828,
        variant="H0_no_factor",
    )
    digest = campaign.semantic_sha256(signature)
    assert builder._validate_signature_object(
        signature,
        digest,
        outer_fold=3,
        seed=20260828,
        variant="H0_no_factor",
    ) == digest
    missing = dict(signature)
    missing.pop("checkpoint_selection")
    with pytest.raises(campaign.CampaignError, match="retained-field set drifted"):
        builder._validate_signature_object(
            missing,
            campaign.semantic_sha256(missing),
            outer_fold=3,
            seed=20260828,
            variant="H0_no_factor",
        )


def test_v8_discovery_pointer_is_create_once_exact_and_nonaccounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root, item, selection, governance, source_receipt = _discovery_reuse_fixture(
        tmp_path, monkeypatch
    )
    source = fixed._create_discovery_reuse_pointer(
        project_root=tmp_path,
        run_root=run_root,
        item=item,
        selection=selection,
        governance=governance,
    )
    assert source.kind == "discovery_pointer"
    assert source.receipt["owns_new_gpu_usage"] is False
    assert source.receipt["usage_record_sha256s"] == []
    assert source.receipt["source_training_receipt"]["sha256"] == campaign.sha256_file(
        Path(str(source.receipt["source_training_receipt"]["path"]))
    )
    assert source_receipt["validated_output"]["scientific_signature_sha256"] == (
        source.scientific_signature_sha256
    )
    # Reissuing is an identical create-once verification, not a second artifact.
    repeated = fixed._create_discovery_reuse_pointer(
        project_root=tmp_path,
        run_root=run_root,
        item=item,
        selection=selection,
        governance=governance,
    )
    assert repeated.receipt == source.receipt

    local = (
        run_root
        / "training"
        / f"outer_{item.outer_fold}_seed_{item.seed}"
        / "completion_receipt.json"
    )
    campaign.create_once_json(local, {"ambiguous": True})
    with pytest.raises(campaign.CampaignError, match="exactly one|ambiguous"):
        builder.resolve_promotion_model_source(
            project_root=tmp_path,
            run_root=run_root,
            cache_dir=item.cache_dir,
            outer_fold=item.outer_fold,
            seed=item.seed,
            variant="H0_no_factor",
        )
    local.unlink()
    source.checkpoint.chmod(0o644)
    source.checkpoint.write_bytes(b"drifted")
    with pytest.raises(campaign.CampaignError, match="hash drifted"):
        builder.resolve_promotion_model_source(
            project_root=tmp_path,
            run_root=run_root,
            cache_dir=item.cache_dir,
            outer_fold=item.outer_fold,
            seed=item.seed,
            variant="H0_no_factor",
        )


def test_v8_fixed_matrix_is_twelve_local_plus_six_pointer() -> None:
    assert len(fixed.EXPECTED_NEW_PROMOTION_TRAINING_UNITS) == 12
    assert len(fixed.EXPECTED_REUSE_UNITS) == 6
    assert not (
        fixed.EXPECTED_NEW_PROMOTION_TRAINING_UNITS & fixed.EXPECTED_REUSE_UNITS
    )
    assert fixed.EXPECTED_NEW_PROMOTION_TRAINING_UNITS | fixed.EXPECTED_REUSE_UNITS == {
        (fold, seed) for fold in range(6) for seed in campaign.SEEDS
    }


def test_v8_model_source_missing_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(campaign.CampaignError, match="exactly one"):
        builder.resolve_promotion_model_source(
            project_root=tmp_path,
            run_root=tmp_path / "run",
            cache_dir=tmp_path / "cache",
            outer_fold=0,
            seed=20260828,
            variant="H0_no_factor",
        )


def _cover_receipts(
    tmp_path: Path,
    *,
    duplicate_seed: int | None = None,
    prediction_before_training_key: tuple[int, int] | None = None,
):
    run_root = tmp_path / "run"
    usage = tmp_path / "usage.jsonl"
    campaign.bind_run_usage_ledger(run_root, usage, execution_scope="promotion")
    selection_path = tmp_path / "selection.json"
    authorization_path = tmp_path / "promotion_authorization.json"
    _write_json(selection_path, {"selected_variant": "H0_no_factor"})
    _write_json(authorization_path, {"promotion_authorized": True})
    governance = {
        "selection_lock": campaign.bind_file(selection_path),
        "promotion_authorization": campaign.bind_file(authorization_path),
    }
    discovery_root = tmp_path / "discovery"
    benchmark_identity = {
        "campaign_revision": fixed.CAMPAIGN_REVISION,
        "infrastructure_revision": fixed.INFRASTRUCTURE_REVISION,
        "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
        "outer_fold": 3,
        "seed": 20260828,
        "variant": "H0_no_factor",
    }
    benchmark_record = campaign.append_usage_record(
        usage,
        {
            "schema_version": 1,
            "campaign_id": campaign.CAMPAIGN_ID,
            "phase": "efficiency_benchmark",
            **benchmark_identity,
            "command_sha256": "9" * 64,
            "elapsed_seconds": 1.0,
            "return_code": 0,
            "hard_timeout_reached": False,
        },
    )
    benchmark_receipt_path = discovery_root / "benchmark" / "completion_receipt.json"
    campaign.create_once_json(
        benchmark_receipt_path,
        {
            "schema_version": 1,
            "campaign_id": campaign.CAMPAIGN_ID,
            "phase": "efficiency_benchmark",
            "usage_identity": benchmark_identity,
            **benchmark_identity,
            "usage_ledger_path": str(usage.resolve()),
            "usage_record_sha256": benchmark_record["record_sha256"],
            "usage_record_sha256s": [benchmark_record["record_sha256"]],
        },
    )
    discovery_units = []
    for number, key in enumerate(campaign.EXPECTED_DISCOVERY_UNITS):
        usage_record = campaign.append_usage_record(
            usage,
            {
                "schema_version": 1,
                "campaign_id": campaign.CAMPAIGN_ID,
                "phase": "discovery",
                "campaign_revision": fixed.CAMPAIGN_REVISION,
                "infrastructure_revision": fixed.INFRASTRUCTURE_REVISION,
                "outer_fold": key[0],
                "seed": key[1],
                "variant": key[2],
                "resume": False,
                "command_sha256": f"{number + 1:064x}",
                "elapsed_seconds": 1.0,
                "return_code": 0,
                "hard_timeout_reached": False,
            },
        )
        receipt_path = (
            discovery_root
            / "units"
            / f"outer_{key[0]}_seed_{key[1]}_{key[2]}"
            / "completion_receipt.json"
        )
        campaign.create_once_json(
            receipt_path,
            {
                "schema_version": 1,
                "campaign_id": campaign.CAMPAIGN_ID,
                "campaign_revision": fixed.CAMPAIGN_REVISION,
                "infrastructure_revision": fixed.INFRASTRUCTURE_REVISION,
                "outer_fold": key[0],
                "seed": key[1],
                "variant": key[2],
                "usage_ledger_path": str(usage.resolve()),
                "usage_record_sha256": usage_record["record_sha256"],
                "usage_record_sha256s": [usage_record["record_sha256"]],
            },
        )
        discovery_units.append(
            {
                "outer_fold": key[0],
                "seed": key[1],
                "variant": key[2],
                "receipt": campaign.bind_file(receipt_path),
            }
        )
    discovery_seal_path = discovery_root / "DISCOVERY_COMPLETION_SEAL.json"
    campaign.create_once_json(
        discovery_seal_path,
        {
            "schema_version": 1,
            "campaign_id": campaign.CAMPAIGN_ID,
            "completed_units": 18,
            "gpu_usage_ledger": campaign.bind_file(usage),
            "pre_discovery_efficiency_benchmark": {
                "receipt": campaign.bind_file(benchmark_receipt_path),
                "included_in_gpu_exact_cover": True,
                "excluded_from_selection": True,
                "artifacts_quarantined": True,
            },
            "units": discovery_units,
        },
    )

    early_prediction_records: dict[tuple[int, int], dict[str, Any]] = {}
    if prediction_before_training_key is not None:
        fold, seed = prediction_before_training_key
        early_prediction_records[(fold, seed)] = campaign.append_usage_record(
            usage,
            {
                "schema_version": 1,
                "campaign_id": campaign.CAMPAIGN_ID,
                "phase": "promotion_prediction",
                "campaign_revision": fixed.CAMPAIGN_REVISION,
                "infrastructure_revision": fixed.INFRASTRUCTURE_REVISION,
                "outer_fold": fold,
                "seed": seed,
                "variant": "H0_no_factor",
                "release_mode": "raw_anchor",
                "command_sha256": "8" * 64,
                "elapsed_seconds": 1.0,
                "return_code": 0,
                "hard_timeout_reached": False,
            },
        )

    model_sources: dict[tuple[int, int], builder.PromotionModelSource] = {}
    for number, (fold, seed) in enumerate(
        (fold, seed)
        for fold in builder.NEW_PROMOTION_TRAINING_FOLDS
        for seed in campaign.SEEDS
    ):
        usage_record = campaign.append_usage_record(
            usage,
            {
                "schema_version": 1,
                "campaign_id": campaign.CAMPAIGN_ID,
                "phase": "promotion_training",
                "campaign_revision": fixed.CAMPAIGN_REVISION,
                "infrastructure_revision": fixed.INFRASTRUCTURE_REVISION,
                "outer_fold": fold,
                "seed": seed,
                "variant": "H0_no_factor",
                "resume": False,
                "command_sha256": f"{100 + number:064x}",
                "elapsed_seconds": 1.0,
                "return_code": 0,
                "hard_timeout_reached": False,
            },
        )
        unit_root = run_root / "training" / f"outer_{fold}_seed_{seed}"
        checkpoint = unit_root / "attempt_000/output/best.pt"
        scaler = unit_root / "attempt_000/output/scaler.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{fold}-{seed}".encode())
        _write_json(scaler, {"fold": fold, "seed": seed})
        receipt_path = unit_root / "completion_receipt.json"
        receipt = campaign.create_once_json(
            receipt_path,
            {
                "schema_version": 1,
                "campaign_id": campaign.CAMPAIGN_ID,
                "campaign_revision": fixed.CAMPAIGN_REVISION,
                "infrastructure_revision": fixed.INFRASTRUCTURE_REVISION,
                "outer_fold": fold,
                "seed": seed,
                "variant": "H0_no_factor",
                "usage_ledger_path": str(usage.resolve()),
                "usage_record_sha256": usage_record["record_sha256"],
                "usage_record_sha256s": [usage_record["record_sha256"]],
            },
        )
        model_sources[(fold, seed)] = builder.PromotionModelSource(
            kind="local_training",
            receipt_path=receipt_path,
            output_dir=checkpoint.parent,
            checkpoint=checkpoint,
            scaler=scaler,
            scientific_signature_sha256=f"{300 + number:064x}",
            artifacts={},
            receipt=receipt,
        )

    for number, (fold, seed) in enumerate(sorted(fixed.EXPECTED_REUSE_UNITS)):
        unit_root = run_root / "training" / f"outer_{fold}_seed_{seed}"
        source_root = discovery_root / "selected_sources" / f"outer_{fold}_seed_{seed}"
        checkpoint = source_root / "best.pt"
        scaler = source_root / "scaler.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"discovery-{fold}-{seed}".encode())
        _write_json(scaler, {"fold": fold, "seed": seed})
        pointer_path = unit_root / builder.REUSE_POINTER_FILENAME
        pointer = campaign.create_once_json(
            pointer_path,
            {
                "schema_version": 1,
                "campaign_id": campaign.CAMPAIGN_ID,
                "outer_fold": fold,
                "seed": seed,
                "variant": "H0_no_factor",
                "owns_new_gpu_usage": False,
                "usage_record_sha256s": [],
            },
        )
        model_sources[(fold, seed)] = builder.PromotionModelSource(
            kind="discovery_pointer",
            receipt_path=pointer_path,
            output_dir=source_root,
            checkpoint=checkpoint,
            scaler=scaler,
            scientific_signature_sha256=f"{400 + number:064x}",
            artifacts={},
            receipt=pointer,
        )

    receipts = []
    for number, (fold, seed) in enumerate(
        (fold, seed) for fold in range(6) for seed in campaign.SEEDS
    ):
        unit = run_root / "predictions" / f"outer_{fold}_seed_{seed}"
        unit.mkdir(parents=True)
        index_value = fold
        if duplicate_seed == seed and fold == 5:
            index_value = 4
        work = run_root / "inputs" / f"outer_{fold}_seed_{seed}"
        work.mkdir(parents=True)
        predict_input = work / "sanitized_test_input.npz"
        np.savez_compressed(
            predict_input, cache_index=np.asarray([index_value], np.int64)
        )
        sanitized_receipt_path = work / "sanitized_test_input.receipt.json"
        campaign.create_once_json(
            sanitized_receipt_path,
            {
                "outer_fold": fold,
                "seed": seed,
                "output": campaign.bind_file(predict_input),
                "target_fields_present": False,
                "target_reference_qc_identity_protocol_columns_physically_present": False,
            },
        )
        prediction = unit / "predictions.npz"
        np.savez_compressed(
            prediction,
            cache_index=np.asarray([index_value], np.int64),
            prediction_bpm=np.asarray([20.0], np.float32),
            prediction_available=np.asarray([True]),
            raw_anchor_bpm=np.asarray([20.0], np.float32),
            raw_anchor_available=np.asarray([True]),
            hard_source_bpm=np.asarray([21.0], np.float32),
            hard_source_available=np.asarray([True]),
            selected_source_probability=np.asarray([0.5], np.float32),
            selected_source_code=np.asarray([0], np.int16),
            source_scale_bpm=np.asarray([1.0], np.float32),
            quality=np.asarray([1.0], np.float32),
            factor_probabilities=np.asarray([[0.25] * 4], np.float32),
            spike_rate=np.asarray([0.1], np.float32),
        )
        source = model_sources[(fold, seed)]
        campaign.create_once_json(
            unit / "prediction_manifest.json",
            {
                "outer_fold": fold,
                "seed": seed,
                "variant": "H0_no_factor",
                "release_mode": "raw_anchor",
                "target_fields_emitted": False,
                "commercial_claim_authorized": False,
                "predict_input": campaign.bind_file(predict_input),
                "checkpoint": campaign.bind_file(source.checkpoint),
                "scaler": campaign.bind_file(source.scaler),
            },
        )
        usage_record = early_prediction_records.get((fold, seed))
        if usage_record is None:
            usage_record = campaign.append_usage_record(
                usage,
                {
                    "schema_version": 1,
                    "campaign_id": campaign.CAMPAIGN_ID,
                    "phase": "promotion_prediction",
                    "campaign_revision": fixed.CAMPAIGN_REVISION,
                    "infrastructure_revision": fixed.INFRASTRUCTURE_REVISION,
                    "outer_fold": fold,
                    "seed": seed,
                    "variant": "H0_no_factor",
                    "release_mode": "raw_anchor",
                    "command_sha256": f"{200 + number:064x}",
                    "elapsed_seconds": 1.0,
                    "return_code": 0,
                    "hard_timeout_reached": False,
                },
            )
        validated_output = fixed.validate_target_free_prediction(
            unit,
            expected_index=np.asarray([index_value], np.int64),
            outer_fold=fold,
            seed=seed,
            variant="H0_no_factor",
            release_mode="raw_anchor",
            predict_input=predict_input,
            checkpoint=source.checkpoint,
            scaler=source.scaler,
        )
        receipt = campaign.create_once_json(
            unit / "completion_receipt.json",
            {
                "schema_version": 1,
                "campaign_id": campaign.CAMPAIGN_ID,
                "campaign_revision": fixed.CAMPAIGN_REVISION,
                "infrastructure_revision": fixed.INFRASTRUCTURE_REVISION,
                "outer_fold": fold,
                "seed": seed,
                "variant": "H0_no_factor",
                "release_mode": "raw_anchor",
                "output_dir": str(unit),
                "usage_ledger_path": str(usage.resolve()),
                "usage_record_sha256": usage_record["record_sha256"],
                "usage_record_sha256s": [usage_record["record_sha256"]],
                "promotion_model_source": {
                    "kind": model_sources[(fold, seed)].kind,
                    "receipt": campaign.bind_file(
                        model_sources[(fold, seed)].receipt_path
                    ),
                    "checkpoint": campaign.bind_file(
                        model_sources[(fold, seed)].checkpoint
                    ),
                    "scaler": campaign.bind_file(model_sources[(fold, seed)].scaler),
                    "scientific_signature_sha256": model_sources[
                        (fold, seed)
                    ].scientific_signature_sha256,
                },
                "validated_output": validated_output,
            },
        )
        receipts.append(receipt)
    training = {
        (fold, seed): campaign.TrainingInput(
            outer_fold=fold,
            seed=seed,
            cache_dir=tmp_path / "unused-cache",
            cache_manifest_sha256="a" * 64,
            proposer_stack=tmp_path / "unused-proposer.npz",
            proposer_stack_sha256="b" * 64,
        )
        for fold in range(6)
        for seed in campaign.SEEDS
    }
    return (
        run_root,
        receipts,
        usage,
        campaign.bind_file(discovery_seal_path),
        training,
        model_sources,
        governance,
    )


def _install_cover_input_revalidator(
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_root: Path,
    usage: Path,
    model_sources: Mapping[tuple[int, int], builder.PromotionModelSource],
) -> list[tuple[int, int]]:
    calls: list[tuple[int, int]] = []

    def revalidate(**kwargs: Any) -> dict[str, Any]:
        key = (int(kwargs["outer_fold"]), int(kwargs["seed"]))
        calls.append(key)
        assert kwargs["model_source"] is model_sources[key]
        expected_work = run_root / "inputs" / f"outer_{key[0]}_seed_{key[1]}"
        assert kwargs["output"] == expected_work / "sanitized_test_input.npz"
        assert kwargs["receipt_path"] == (
            expected_work / "sanitized_test_input.receipt.json"
        )
        receipt = campaign.load_json(
            kwargs["receipt_path"], "stub sanitized input receipt"
        )
        assert campaign.canonical_content_sha256(receipt) == receipt["content_sha256"]
        assert receipt["output"] == campaign.bind_file(kwargs["output"])
        return receipt

    monkeypatch.setattr(builder, "build_locked_input", revalidate)

    original_locked_snapshot = campaign.gpu_budget_ledger.locked_closed_snapshot

    class V8R4Snapshot:
        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped

        def __enter__(self) -> Any:
            actual = self.wrapped.__enter__()
            converted: list[dict[str, Any]] = []
            for number, record in enumerate(actual.records):
                phase = str(record.get("phase", ""))
                names = {
                    "efficiency_benchmark": (
                        "campaign_revision", "infrastructure_revision",
                        "benchmark_id", "outer_fold", "seed", "variant",
                    ),
                    "discovery": (
                        "campaign_revision", "infrastructure_revision",
                        "outer_fold", "seed", "variant",
                    ),
                    "promotion_training": (
                        "campaign_revision", "infrastructure_revision",
                        "outer_fold", "seed", "variant",
                    ),
                    "promotion_prediction": (
                        "campaign_revision", "infrastructure_revision",
                        "outer_fold", "seed", "variant", "release_mode",
                    ),
                }.get(phase)
                if names is None:
                    converted.append(dict(record))
                    continue
                converted.append(
                    {
                        "schema_version": 2,
                        "campaign_id": campaign.CAMPAIGN_ID,
                        "event": "terminal",
                        "phase": phase,
                        "context": {name: record.get(name) for name in names},
                        "command_sha256": record["command_sha256"],
                        "invocation_sha256": f"{10_000 + number:064x}",
                        "charged_usage_ns": int(
                            round(float(record["elapsed_seconds"]) * 1_000_000_000)
                        ),
                        "return_code": int(record["return_code"]),
                        "hard_timeout_reached": bool(
                            record["hard_timeout_reached"]
                        ),
                        "reuse_eligible": True,
                        "record_sha256": record["record_sha256"],
                    }
                )
            return type("SyntheticClosedSnapshot", (), {
                "records": tuple(converted),
                "raw_bytes": actual.raw_bytes,
                "open_reservations": {},
                "tail_sha256": actual.tail_sha256,
                "settled_usage_ns": actual.settled_usage_ns,
            })()

        def __exit__(self, *args: Any) -> Any:
            return self.wrapped.__exit__(*args)

    def locked_snapshot(path: Path, *args: Any, **kwargs: Any) -> Any:
        assert path.resolve() == usage.resolve()
        return V8R4Snapshot(original_locked_snapshot(path, *args, **kwargs))

    monkeypatch.setattr(
        campaign.gpu_budget_ledger, "locked_closed_snapshot", locked_snapshot
    )
    # Lifecycle invocation/result byte replay has dedicated regressions.  This
    # adapter keeps this fixture focused on the 49-owner ordering/exact-cover
    # proof while still exercising the production V8R4 identity matcher.
    monkeypatch.setattr(
        campaign,
        "_validate_terminal_result_bindings",
        lambda *args, **kwargs: None,
    )
    return calls


def test_global_prediction_seal_requires_exact_18_and_equal_seed_covers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        run_root,
        receipts,
        usage,
        discovery_seal,
        training,
        model_sources,
        governance,
    ) = _cover_receipts(tmp_path)
    source_revalidations: list[tuple[int, int]] = []

    def resolve_source(**kwargs: Any) -> builder.PromotionModelSource:
        key = (int(kwargs["outer_fold"]), int(kwargs["seed"]))
        source_revalidations.append(key)
        return model_sources[key]

    monkeypatch.setattr(
        builder,
        "resolve_promotion_model_source",
        resolve_source,
    )
    input_revalidations = _install_cover_input_revalidator(
        monkeypatch, run_root=run_root, usage=usage, model_sources=model_sources
    )
    live_prediction_revalidations: list[tuple[int, int]] = []
    original_prediction_validator = fixed.validate_target_free_prediction

    def validate_live_prediction(*args: Any, **kwargs: Any) -> dict[str, Any]:
        live_prediction_revalidations.append(
            (int(kwargs["outer_fold"]), int(kwargs["seed"]))
        )
        return original_prediction_validator(*args, **kwargs)

    monkeypatch.setattr(
        fixed, "validate_target_free_prediction", validate_live_prediction
    )
    original_create_once = campaign.create_once_json
    append_started = threading.Event()
    append_finished = threading.Event()
    writer: threading.Thread | None = None

    def append_unexplained_terminal() -> None:
        append_started.set()
        campaign.gpu_budget_ledger.append_record(
            usage,
            {
                "schema_version": 1,
                "campaign_id": campaign.CAMPAIGN_ID,
                "phase": "promotion_prediction",
                "outer_fold": 99,
                "seed": 20260828,
                "variant": "H0_no_factor",
                "release_mode": "raw_anchor",
                "command_sha256": "f" * 64,
                "elapsed_seconds": 1.0,
                "return_code": 0,
                "hard_timeout_reached": False,
            },
        )
        append_finished.set()

    def create_with_barrier(path: Path, value: Any) -> dict[str, Any]:
        nonlocal writer
        if path.name == "PROMOTION_PREDICTION_EXACT_COVER_SEAL.json":
            writer = threading.Thread(target=append_unexplained_terminal)
            writer.start()
            assert append_started.wait(timeout=1.0)
            time.sleep(0.05)
            assert not append_finished.is_set()
        return original_create_once(path, value)

    monkeypatch.setattr(campaign, "create_once_json", create_with_barrier)
    seal = fixed.build_exact_cover_seal(
        project_root=tmp_path,
        run_root=run_root,
        receipts=receipts,
        selection={
            "selected_variant": "H0_no_factor",
            "selected_release_mode": "raw_anchor",
            "discovery_completion_seal": discovery_seal,
        },
        governance=governance,
        usage_ledger=usage,
        training=training,
    )
    assert seal["completed_prediction_units"] == 18
    assert seal["completed_new_training_units"] == 12
    assert seal["completed_reused_training_pointer_units"] == 6
    assert seal["gpu_execution_owner_count"] == 49
    assert seal["nonaccounting_pointer_receipt_count"] == 6
    assert seal["reuse_pointer_receipts_own_gpu_records"] is False
    assert seal["all_18_predictions_sealed_before_target_join"] is True
    assert seal["target_join_performed"] is False
    assert set(input_revalidations) == {
        (fold, seed) for fold in range(6) for seed in campaign.SEEDS
    }
    assert len(input_revalidations) == 18
    assert source_revalidations == sorted(source_revalidations)
    assert set(source_revalidations) == set(input_revalidations)
    assert len(source_revalidations) == 18
    assert set(live_prediction_revalidations) == set(input_revalidations)
    assert len(live_prediction_revalidations) == 18
    assert writer is not None
    writer.join(timeout=1.0)
    assert append_finished.is_set()
    assert seal["gpu_usage_ledger"]["bytes"] < usage.stat().st_size

    with pytest.raises(campaign.CampaignError, match="exact 6x3 unit cover"):
        fixed.build_exact_cover_seal(
            project_root=tmp_path,
            run_root=tmp_path / "missing",
            receipts=receipts[:-1],
            selection={
                "selected_variant": "H0_no_factor",
                "selected_release_mode": "raw_anchor",
            },
            governance={},
            usage_ledger=usage,
            training=training,
        )


def test_v8r4_fixed_execution_plan_is_six_twelve_eighteen_and_49() -> None:
    plan = fixed.v8r4_fixed_execution_plan()
    assert plan["campaign_revision"] == "V8R4"
    assert len(plan["clean_discovery_reuse_units"]) == 6
    assert {tuple(value) for value in plan["clean_discovery_reuse_units"]} == {
        (fold, seed) for fold in (3, 4) for seed in campaign.SEEDS
    }
    assert len(plan["new_promotion_training_units"]) == 12
    assert {tuple(value) for value in plan["new_promotion_training_units"]} == {
        (fold, seed) for fold in (0, 1, 2, 5) for seed in campaign.SEEDS
    }
    assert len(plan["target_free_prediction_units"]) == 18
    assert plan["active_gpu_owner_count"] == 49
    assert plan["active_gpu_owner_breakdown"] == {
        "benchmark": 1,
        "clean_v8r4_discovery": 18,
        "new_promotion_training": 12,
        "target_free_prediction": 18,
    }
    assert plan["v8r3_quarantine_in_active_owner_count"] is False
    assert plan["fully_nested_confirmatory_oof"] is False


def test_v8r4_fixed_parser_is_capability_sharded_and_has_no_legacy_cache_flags() -> None:
    parser = fixed.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    assert parser.parse_args(["--promotion-model-shard", "3"]).promotion_model_shard == 3
    assert parser.parse_args(["--prediction-shard", "4"]).prediction_shard == 4
    assert parser.parse_args(["--aggregate"]).aggregate is True
    option_strings = {
        option for action in parser._actions for option in action.option_strings
    }
    assert "--training-index" not in option_strings
    assert "--v2-locked-units" not in option_strings


def test_v8r4_fixed_production_blocks_before_any_governance_or_pack_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        fixed,
        "_read_frozen_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("governance/pack must remain unopened")
        ),
    )
    assert fixed.main(["--project-root", str(tmp_path), "--aggregate"]) == 2
    output = capsys.readouterr().out
    assert "cannot load the V8R4A GPU-state validator" in output


def test_v8r4_fixed_legacy_v2_paths_are_explicit_tombstones() -> None:
    with pytest.raises(campaign.CampaignError, match="forbids every legacy V2"):
        fixed._safe_anchor_from_locked_v2()
    with pytest.raises(campaign.CampaignError, match="forbids every legacy V2"):
        fixed._validate_canonical_v2_anchor_source()


def test_v8r4_fixed_rejects_discovery_index_as_promotion_pack() -> None:
    discovery_index = ROOT / campaign.SHARD_TRAINING_INDEX[3]
    with pytest.raises(campaign.CampaignError, match="producer/runtime ABI"):
        fixed.validate_v8r4_promotion_pack_index(
            index_path=discovery_index,
            phase="promotion_training",
            outer_fold=3,
            authorization_binding={
                "path": "PROMOTION_AUTHORIZATION.json",
                "sha256": "a" * 64,
                "bytes": 1,
            },
        )


@pytest.mark.parametrize(
    ("phase", "classification", "artifact_names"),
    (
        (
            "promotion_training",
            fixed.PROMOTION_TRAINING_INDEX_CLASSIFICATION,
            {"cache_manifest", "proposer_stack", "partition_manifest"},
        ),
        (
            "promotion_prediction",
            fixed.PREDICTION_INDEX_CLASSIFICATION,
            {
                "prediction_pack_manifest",
                "model_bound_prediction_pack_manifest",
                "outer_predict_input",
                "model_checkpoint",
                "model_scaler",
                "model_source_capability",
            },
        ),
    ),
)
def test_v8r4_fixed_accepts_only_distinct_exact_shard_index_abis(
    tmp_path: Path,
    phase: str,
    classification: str,
    artifact_names: set[str],
) -> None:
    authorization = {
        "path": str(tmp_path / "PROMOTION_AUTHORIZATION.json"),
        "sha256": "a" * 64,
        "bytes": 123,
    }
    index_path = tmp_path / f"{phase}.json"
    document: dict[str, Any] = {
            "schema_version": 1,
            "classification": classification,
            "campaign_id": campaign.CAMPAIGN_ID,
            "campaign_revision": "V8R4",
            "outer_fold": 2,
            "seeds": list(campaign.SEEDS),
            "unit_count": 3,
            "completed_units": 3,
            "status": "complete",
            "outer_test_opened": False,
            "combined_target_bearing_cache_consumer_access_authorized": False,
            "cross_outer_shard_mounted": False,
            "promotion_authorization": authorization,
            "units": [
                {
                    "outer_fold": 2,
                    "seed": seed,
                    "relative_path": f"units/outer_2_seed_{seed}",
                    "artifacts": {
                        name: {"path": name, "sha256": "b" * 64, "bytes": 1}
                        for name in artifact_names
                    },
                }
                for seed in campaign.SEEDS
            ],
        }
    if phase == "promotion_training":
        document.update(
            {
                "physical_nonouter_training_packs": True,
                "outer_prediction_packs_absent": True,
                "promotion_scope": "promotion_training_pack",
            }
        )
    else:
        document.update(
            {
                "infrastructure_revision": "V8R4A",
                "selected_variant": "H0_no_factor",
                "physical_target_free_input_and_model_packs": True,
                "source_paths_or_peer_outputs_authorized_in_child": False,
                "model_source_shard_seal": {
                    "path": "MODEL_SOURCE_SHARD_SEAL.json",
                    "sha256": "c" * 64,
                    "bytes": 1,
                },
            }
        )
    campaign.create_once_json(index_path, document)
    document = fixed.validate_v8r4_promotion_pack_index(
        index_path=index_path,
        phase=phase,
        outer_fold=2,
        authorization_binding=authorization,
    )
    assert document["classification"] == classification


def test_global_prediction_seal_rejects_duplicate_cache_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        run_root,
        receipts,
        usage,
        discovery_seal,
        training,
        model_sources,
        governance,
    ) = _cover_receipts(tmp_path, duplicate_seed=20260829)
    monkeypatch.setattr(
        builder,
        "resolve_promotion_model_source",
        lambda **kwargs: model_sources[(kwargs["outer_fold"], kwargs["seed"])],
    )
    _install_cover_input_revalidator(
        monkeypatch, run_root=run_root, usage=usage, model_sources=model_sources
    )
    with pytest.raises(campaign.CampaignError, match="duplicated"):
        fixed.build_exact_cover_seal(
            project_root=tmp_path,
            run_root=run_root,
            receipts=receipts,
            selection={
                "selected_variant": "H0_no_factor",
                "selected_release_mode": "raw_anchor",
                "discovery_completion_seal": discovery_seal,
            },
            governance=governance,
            usage_ledger=usage,
            training=training,
        )


def test_global_prediction_seal_rejects_validated_output_claim_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        run_root,
        receipts,
        usage,
        discovery_seal,
        training,
        model_sources,
        governance,
    ) = _cover_receipts(tmp_path)
    monkeypatch.setattr(
        builder,
        "resolve_promotion_model_source",
        lambda **kwargs: model_sources[(kwargs["outer_fold"], kwargs["seed"])],
    )
    _install_cover_input_revalidator(
        monkeypatch, run_root=run_root, usage=usage, model_sources=model_sources
    )
    forged = json.loads(json.dumps(receipts[0]))
    forged["validated_output"]["rows"] = 999
    forged.pop("content_sha256")
    forged["content_sha256"] = campaign.semantic_sha256(forged)
    receipt_path = (
        run_root
        / "predictions"
        / f"outer_{forged['outer_fold']}_seed_{forged['seed']}"
        / "completion_receipt.json"
    )
    receipt_path.chmod(0o644)
    _write_json(receipt_path, forged)
    receipts[0] = forged

    with pytest.raises(campaign.CampaignError, match="live proof"):
        fixed.build_exact_cover_seal(
            project_root=tmp_path,
            run_root=run_root,
            receipts=receipts,
            selection={
                "selected_variant": "H0_no_factor",
                "selected_release_mode": "raw_anchor",
                "discovery_completion_seal": discovery_seal,
            },
            governance=governance,
            usage_ledger=usage,
            training=training,
        )


def test_global_prediction_seal_requires_local_training_before_same_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = sorted(fixed.EXPECTED_NEW_PROMOTION_TRAINING_UNITS)[0]
    (
        run_root,
        receipts,
        usage,
        discovery_seal,
        training,
        model_sources,
        governance,
    ) = _cover_receipts(tmp_path, prediction_before_training_key=key)
    monkeypatch.setattr(
        builder,
        "resolve_promotion_model_source",
        lambda **kwargs: model_sources[(kwargs["outer_fold"], kwargs["seed"])],
    )
    _install_cover_input_revalidator(
        monkeypatch, run_root=run_root, usage=usage, model_sources=model_sources
    )

    with pytest.raises(campaign.CampaignError, match="did not precede"):
        fixed.build_exact_cover_seal(
            project_root=tmp_path,
            run_root=run_root,
            receipts=receipts,
            selection={
                "selected_variant": "H0_no_factor",
                "selected_release_mode": "raw_anchor",
                "discovery_completion_seal": discovery_seal,
            },
            governance=governance,
            usage_ledger=usage,
            training=training,
        )
