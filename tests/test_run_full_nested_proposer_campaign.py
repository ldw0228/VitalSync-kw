from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_full_nested_proposer_campaign.py"
)
SPEC = importlib.util.spec_from_file_location("run_full_nested_proposer_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)

BUILDER_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/build_nested_proposer_manifests.py"
)
BUILDER_SPEC = importlib.util.spec_from_file_location(
    "build_nested_proposer_manifests_non_test", BUILDER_SCRIPT
)
assert BUILDER_SPEC is not None and BUILDER_SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(BUILDER)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _source_files(tmp_path: Path) -> tuple[Path, Path]:
    assignments = tmp_path / "source/fold_assignments.json"
    cache_dir = tmp_path / "source/cache"
    mapping = {
        f"I{fold}_{index}": fold
        for fold in range(6)
        for index in range(2)
    }
    _write_json(assignments, {"identity_to_fold": mapping})
    _write_json(cache_dir / "manifest.json", {"sessions": []})
    return assignments, cache_dir


def test_builder_can_construct_six_fold_plan_without_any_test_unit(
    tmp_path: Path,
) -> None:
    assignments, cache_dir = _source_files(tmp_path)
    records, summary = BUILDER.build_plan(
        assignments_path=assignments,
        cache_manifest=cache_dir / "manifest.json",
        outer_folds=range(6),
        include_outer_test=False,
    )
    assert len(records) == 30
    assert summary["outer_test_opened"] is False
    assert summary["outer_test_unit_count"] == 0
    assert all("test_pred_" not in str(path) for path, _ in records)
    for outer in range(6):
        units = summary["outer_folds"][str(outer)]["units"]
        assert len(units) == 5
        assert {unit["role"] for unit in units} <= RUN.ALLOWED_ROLES
    assert BUILDER.parse_args(["--exclude-outer-test"]).exclude_outer_test is True


def test_dry_run_emits_separate_immutable_90_unit_control_plane(
    tmp_path: Path,
) -> None:
    assignments, cache_dir = _source_files(tmp_path)
    manifest_root = tmp_path / "campaign/full_oof_non_test/manifests"
    control_root = tmp_path / "campaign/full_oof_non_test/control"
    run_root = tmp_path / "runs/nested_proposer"
    legacy_plan = tmp_path / "campaign/nested_proposer/manifests/plan.json"
    legacy_index = run_root / "discovery_index.json"
    _write_json(legacy_plan, {"sentinel": "legacy-plan"})
    _write_json(legacy_index, {"sentinel": "legacy-index"})
    args = RUN.parse_args(
        [
            "--fold-assignments",
            str(assignments),
            "--cache-dir",
            str(cache_dir),
            "--manifest-root",
            str(manifest_root),
            "--control-root",
            str(control_root),
            "--run-root",
            str(run_root),
            "--dry-run",
        ]
    )
    status = RUN.run(args)
    assert status["state"] == "dry_run"
    assert status["requested_units"] == 90
    assert status["completed_units"] == 0
    assert status["outer_test_opened"] is False
    assert status["outer_test_record_count"] == 0

    plan = json.loads((control_root / "plan.json").read_text(encoding="utf-8"))
    index = json.loads((control_root / "index.json").read_text(encoding="utf-8"))
    progress = json.loads(
        (control_root / "progress.json").read_text(encoding="utf-8")
    )
    assert plan["classification"] == (
        "retrospective_fully_nested_non_test_proposer_campaign_plan"
    )
    assert Path(plan["manifest_root"]) == manifest_root.resolve()
    assert len(plan["units"]) == 90
    assert all(Path(unit["manifest"]).is_absolute() for unit in plan["units"])
    assert {unit["role"] for unit in plan["units"]} == RUN.ALLOWED_ROLES
    assert not any("test_pred_" in unit["manifest"] for unit in plan["units"])
    assert index["campaign_plan_content_sha256"] == plan["content_sha256"]
    assert index["campaign_plan"]["sha256"] == RUN.sha256_file(
        control_root / "plan.json"
    )
    assert index["manifest_root"] == str(manifest_root.resolve())
    assert index["records"] == []
    assert len(progress["units"]) == 90
    assert not list(manifest_root.rglob("test_pred_*.json"))
    assert json.loads(legacy_plan.read_text(encoding="utf-8")) == {
        "sentinel": "legacy-plan"
    }
    assert json.loads(legacy_index.read_text(encoding="utf-8")) == {
        "sentinel": "legacy-index"
    }

    # A second scan is byte-stable for immutable manifests/plan and remains
    # safely resumable despite mutable progress timestamps.
    plan_hash = RUN.sha256_file(control_root / "plan.json")
    manifest_hashes = {
        path.relative_to(manifest_root): RUN.sha256_file(path)
        for path in manifest_root.rglob("*.json")
    }
    RUN.run(args)
    assert RUN.sha256_file(control_root / "plan.json") == plan_hash
    assert {
        path.relative_to(manifest_root): RUN.sha256_file(path)
        for path in manifest_root.rglob("*.json")
    } == manifest_hashes


def test_campaign_rejects_forbidden_test_manifest_without_parsing_it(
    tmp_path: Path,
) -> None:
    assignments, cache_dir = _source_files(tmp_path)
    manifest_root = tmp_path / "manifests"
    forbidden = manifest_root / "outer_0/test_pred_0.json"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_text("this is deliberately not JSON", encoding="utf-8")
    with pytest.raises(RuntimeError, match="outer-test manifest is forbidden"):
        RUN.materialize_non_test_manifests(
            assignments_path=assignments,
            cache_manifest=cache_dir / "manifest.json",
            manifest_root=manifest_root,
            outer_folds=range(6),
        )


def test_campaign_fails_closed_if_an_immutable_manifest_changes(
    tmp_path: Path,
) -> None:
    assignments, cache_dir = _source_files(tmp_path)
    manifest_root = tmp_path / "manifests"
    RUN.materialize_non_test_manifests(
        assignments_path=assignments,
        cache_manifest=cache_dir / "manifest.json",
        manifest_root=manifest_root,
        outer_folds=[0],
    )
    path = manifest_root / "outer_0/inner_pred_2.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["fold_id"] = 999
    _write_json(path, value)
    with pytest.raises(RuntimeError, match="existing immutable non-test manifest differs"):
        RUN.materialize_non_test_manifests(
            assignments_path=assignments,
            cache_manifest=cache_dir / "manifest.json",
            manifest_root=manifest_root,
            outer_folds=[0],
        )


def test_cuda_commands_are_serialized_by_the_existing_gpu_admission_wrapper(
    tmp_path: Path,
) -> None:
    args = RUN.parse_args(
        [
            "--gpu-lock",
            str(tmp_path / "gpu.lock"),
            "--gpu-ledger",
            str(tmp_path / "ledger.jsonl"),
        ]
    )
    wrapped = RUN.gpu_admitted_command(["trainer", "--flag"], args=args)
    assert Path(wrapped[1]).name == "run_gpu_admitted.py"
    assert wrapped[2:7] == [
        "--lock-file",
        str(tmp_path / "gpu.lock"),
        "--ledger",
        str(tmp_path / "ledger.jsonl"),
        "--",
    ]
    assert wrapped[7:] == ["trainer", "--flag"]
    train = RUN.build_train_command(
        args=args,
        unit={
            "manifest": tmp_path / "inner_pred_0.json",
            "output_dir": tmp_path / "unit",
            "seed": 20260828,
        },
    )
    assert train.count("--resume") == 1


def test_campaign_source_lock_detects_mid_run_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "launch_source.py"
    source.write_text("version = 1\n", encoding="utf-8")
    monkeypatch.setattr(RUN, "CAMPAIGN_SOURCE_PATHS", (source,))
    plan = {"source_bindings": RUN.campaign_source_bindings()}
    RUN.verify_campaign_source_bindings(plan)
    source.write_text("version = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after plan lock"):
        RUN.verify_campaign_source_bindings(plan)


def test_completed_reusable_unit_requires_all_hash_and_manifest_bindings(
    tmp_path: Path,
) -> None:
    assignments, cache_dir = _source_files(tmp_path)
    manifest_root = tmp_path / "campaign/manifests"
    run_root = tmp_path / "runs"
    records, summary = RUN.materialize_non_test_manifests(
        assignments_path=assignments,
        cache_manifest=cache_dir / "manifest.json",
        manifest_root=manifest_root,
        outer_folds=[0],
    )
    plan = RUN.build_campaign_plan(
        manifest_records=records,
        manifest_summary=summary,
        manifest_root=manifest_root,
        run_root=run_root,
        assignments_path=assignments,
        cache_manifest=cache_dir / "manifest.json",
        outer_folds=[0],
        seeds=[20260828],
        epochs=80,
        patience=12,
    )
    unit = plan["units"][0]
    manifest_path = Path(unit["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    authority = RUN._expected_authority(manifest, manifest_path)
    relative = Path("outer_0") / manifest_path.name
    paths = RUN._unit_paths(
        run_root=run_root,
        seed=20260828,
        outer=0,
        relative_manifest=relative,
        fold_id=manifest["fold_id"],
    )
    paths["fold_dir"].mkdir(parents=True)
    run_arguments = {
        **dict(RUN.CRITICAL_TRAIN_ARGUMENTS),
        "seed": 20260828,
        "identity_split_manifest_sha256": manifest["content_sha256"],
    }
    _write_json(
        paths["run_config"],
        {
            "run_signature": "unit-signature",
            "arguments": run_arguments,
            "split_authority": authority,
        },
    )
    split_authority = {
        "cache_manifest_path": str(cache_dir.resolve() / "manifest.json"),
        "cache_manifest_sha256": authority["cache_manifest_sha256"],
        "excluded_identities": authority["excluded_identities"],
        "fold_assignments_path": str(assignments.resolve()),
        "fold_assignments_sha256": authority["fold_assignments_sha256"],
        "fold_id": authority["fold_id"],
        "mode": "custom_identity_split",
        "prediction_identities": authority["prediction_identities"],
        "scaler_identities": authority["scaler_identities"],
        "schema_version": 1,
        "split_manifest_content_sha256": authority[
            "split_manifest_content_sha256"
        ],
        "split_manifest_file_sha256": authority["split_manifest_file_sha256"],
        "split_manifest_path": str(manifest_path),
        "train_identities": authority["train_identities"],
        "validation_identities": authority["validation_identities"],
    }
    _write_json(paths["split_authority"], split_authority)
    torch.save(
        {
            "format_version": 2,
            "model_type": "snn",
            "fold": authority["fold_id"],
            "run_signature": "unit-signature",
            "split_authority_provenance": authority,
        },
        paths["checkpoint"],
    )
    _write_json(
        paths["metrics"],
        {
            "folds": {
                str(authority["fold_id"]): {
                    "models": {
                        "snn": {"best_checkpoint": str(paths["checkpoint"])}
                    }
                }
            }
        },
    )
    checkpoint_hash = RUN.sha256_file(paths["checkpoint"])
    identities = np.asarray(authority["prediction_identities"])
    provenance = {
        "checkpoint_sha256": checkpoint_hash,
        "split_manifest_file_sha256": authority["split_manifest_file_sha256"],
        "split_manifest_content_sha256": authority[
            "split_manifest_content_sha256"
        ],
        "fold_assignments_sha256": authority["fold_assignments_sha256"],
        "cache_manifest_sha256": authority["cache_manifest_sha256"],
        "strict_nested_role": "prediction",
        "labels_forwarded_to_model": False,
        "prediction_identities": authority["prediction_identities"],
        "split_manifest_path": str(manifest_path),
    }
    np.savez_compressed(
        paths["prediction"],
        cache_index=np.arange(len(identities), dtype=np.int64),
        identity=identities,
        prediction=np.full(len(identities), 18.0, dtype=np.float32),
        fold_id=np.asarray(authority["fold_id"], dtype=np.int16),
        checkpoint_sha256=np.asarray(checkpoint_hash),
        split_manifest_file_sha256=np.asarray(
            authority["split_manifest_file_sha256"]
        ),
        split_manifest_content_sha256=np.asarray(
            authority["split_manifest_content_sha256"]
        ),
        fold_assignments_sha256=np.asarray(authority["fold_assignments_sha256"]),
        cache_manifest_sha256=np.asarray(authority["cache_manifest_sha256"]),
        strict_retrospective=np.asarray(True),
        strict_nested_prediction_role=np.asarray(True),
        provenance_json=np.asarray(json.dumps(provenance)),
    )
    state, record, detail = RUN.inspect_unit(unit, run_root=run_root)
    assert state == "complete"
    assert detail is None
    assert record is not None
    assert record["checkpoint"]["sha256"] == checkpoint_hash
    assert record["all_window_prediction"]["sha256"] == RUN.sha256_file(
        paths["prediction"]
    )

    # A present artifact is never trusted merely because its filename exists.
    np.savez_compressed(
        paths["prediction"],
        cache_index=np.arange(len(identities), dtype=np.int64),
        identity=identities,
        prediction=np.full(len(identities), 18.0, dtype=np.float32),
        fold_id=np.asarray(authority["fold_id"], dtype=np.int16),
        checkpoint_sha256=np.asarray("0" * 64),
        split_manifest_file_sha256=np.asarray(
            authority["split_manifest_file_sha256"]
        ),
        split_manifest_content_sha256=np.asarray(
            authority["split_manifest_content_sha256"]
        ),
        fold_assignments_sha256=np.asarray(authority["fold_assignments_sha256"]),
        cache_manifest_sha256=np.asarray(authority["cache_manifest_sha256"]),
        strict_retrospective=np.asarray(True),
        strict_nested_prediction_role=np.asarray(True),
        provenance_json=np.asarray(json.dumps(provenance)),
    )
    with pytest.raises(RuntimeError, match="prediction provenance mismatch"):
        RUN.inspect_unit(unit, run_root=run_root)
