from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_fixed_i3_pretest_campaign.py"
SPEC = importlib.util.spec_from_file_location("run_fixed_i3_pretest_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def _write_content_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    document = dict(value)
    document["content_sha256"] = RUN.canonical_content_sha256(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return document


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": RUN.sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _non_test_cover(tmp_path: Path, *, completed: int = 90) -> tuple[Path, Path, list[Path]]:
    manifests = tmp_path / "manifests"
    plan_units: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    predictions: list[Path] = []
    for outer in RUN.FOLDS:
        validation = (outer + 1) % 6
        training = sorted(set(RUN.FOLDS) - {outer, validation})
        names = [(f"inner_pred_{fold}", "hcs_train_oof") for fold in training]
        names.append((f"validation_pred_{validation}", "hcs_validation"))
        manifest_documents: dict[str, tuple[Path, dict[str, Any]]] = {}
        for stem, _ in names:
            path = manifests / f"outer_{outer}" / f"{stem}.json"
            document = _write_content_json(
                path,
                {"schema_version": 1, "fold_id": outer * 10 + len(manifest_documents)},
            )
            manifest_documents[stem] = (path, document)
        for seed in RUN.SEEDS:
            for stem, role in names:
                manifest_path, manifest = manifest_documents[stem]
                source = tmp_path / "proposer" / f"outer_{outer}" / f"seed_{seed}" / stem
                checkpoint = source / "checkpoint.pt"
                prediction = source / "prediction.npz"
                source.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(f"checkpoint:{outer}:{seed}:{stem}".encode())
                prediction.write_bytes(f"prediction:{outer}:{seed}:{stem}".encode())
                predictions.append(prediction)
                common = {
                    "unit_id": f"seed_{seed}/outer_{outer}/{stem}",
                    "seed": seed,
                    "outer_fold": outer,
                    "role": role,
                    "manifest": str(manifest_path.resolve()),
                    "manifest_sha256": RUN.sha256_file(manifest_path),
                    "manifest_content_sha256": manifest["content_sha256"],
                    "output_dir": str(source.resolve()),
                }
                plan_units.append(
                    {
                        **common,
                        "checkpoint": str(checkpoint.resolve()),
                        "all_window_prediction": str(prediction.resolve()),
                    }
                )
                records.append(
                    {
                        **common,
                        "checkpoint": _binding(checkpoint),
                        "all_window_prediction": _binding(prediction),
                    }
                )
    plan_path = tmp_path / "control/plan.json"
    plan = _write_content_json(
        plan_path,
        {
            "schema_version": 1,
            "classification": "retrospective_fully_nested_non_test_proposer_campaign_plan",
            "requested_units": 90,
            "outer_folds": list(RUN.FOLDS),
            "seeds": list(RUN.SEEDS),
            "roles": ["hcs_train_oof", "hcs_validation"],
            "manifest_root": str(manifests.resolve()),
            "outer_test_opened": False,
            "outer_test_record_count": 0,
            "units": plan_units,
        },
    )
    index_path = tmp_path / "control/index.json"
    _write_content_json(
        index_path,
        {
            "schema_version": 1,
            "classification": "retrospective_fully_nested_non_test_proposer_index",
            "campaign_plan": {
                "path": str(plan_path.resolve()),
                "sha256": RUN.sha256_file(plan_path),
                "content_sha256": plan["content_sha256"],
            },
            "campaign_plan_content_sha256": plan["content_sha256"],
            "requested_units": 90,
            "completed_units": completed,
            "outer_test_opened": False,
            "outer_test_record_count": 0,
            "records": records[:completed],
        },
    )
    return plan_path, index_path, predictions


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        plan=tmp_path / "plan.json",
        index=tmp_path / "index.json",
        artifact_root=tmp_path / "artifacts",
        cache_root=tmp_path / "cache",
        reuse_root=tmp_path / "reuse",
        rf_cache=tmp_path / "rf",
        svd_cache=tmp_path / "svd",
        fold_assignments=tmp_path / "folds.json",
        runtime_seal=tmp_path / "runtime_seal.json",
        stack_builder=tmp_path / "stack.py",
        fallback_builder=tmp_path / "fallback.py",
        cache_builder=tmp_path / "cache.py",
        trainer=tmp_path / "trainer.py",
        gpu_wrapper=tmp_path / "gpu.py",
        gpu_lock=tmp_path / "gpu.lock",
        gpu_ledger=tmp_path / "gpu.jsonl",
        python_executable=Path(sys.executable),
        force_retrain_units=frozenset(),
    )


def test_synthetic_full_cover_is_exactly_eighteen_ready_groups(tmp_path: Path) -> None:
    plan, index, _ = _non_test_cover(tmp_path)
    groups, bindings = RUN.validate_non_test_plan_index(plan, index)
    assert set(groups) == {(fold, seed) for fold in RUN.FOLDS for seed in RUN.SEEDS}
    assert all(len(group["units"]) == 5 for group in groups.values())
    assert bindings["index"]["completed_units"] == 90


def test_missing_readiness_fails_closed_before_any_dag_work(tmp_path: Path) -> None:
    plan, index, _ = _non_test_cover(tmp_path, completed=89)
    with pytest.raises(RuntimeError, match="90/90 required"):
        RUN.validate_non_test_plan_index(plan, index)


def test_bound_proposer_tampering_is_detected(tmp_path: Path) -> None:
    plan, index, predictions = _non_test_cover(tmp_path)
    predictions[17].write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="prediction hash mismatch"):
        RUN.validate_non_test_plan_index(plan, index)


def test_dry_run_plan_reuses_six_and_fixes_all_twelve_commands(tmp_path: Path) -> None:
    args = _args(tmp_path)
    common = {
        "selection_lock": {"path": "selection", "sha256": "a" * 64},
        "capacity_selection": {"path": "capacity", "sha256": "b" * 64},
        "policy": {"path": "policy", "sha256": "c" * 64},
    }
    plan = RUN._build_static_plan(args, inputs={}, sources={}, common=common)
    reuse = [unit for unit in plan["units"] if unit["mode"].startswith("reuse")]
    trained = [unit for unit in plan["units"] if unit["mode"] == "fixed_i3_default_training"]
    assert len(plan["units"]) == 18
    assert {(unit["outer_fold"], unit["seed"]) for unit in reuse} == {
        (fold, seed) for fold in (3, 4) for seed in RUN.SEEDS
    }
    assert len(trained) == 12
    assert all(unit["outer_fold"] in (0, 1, 2, 5) for unit in trained)
    commands = [unit["initial_attempt_commands"]["trainer"]["argv"] for unit in trained]
    assert all(command[-len(RUN.fixed_trainer_flags()):] == list(RUN.fixed_trainer_flags()) for command in commands)
    assert all("--preset" in command and command[command.index("--preset") + 1] == "default" for command in commands)
    assert plan["validation_scores_control_execution"] is False
    assert plan["capacity_reselection_permitted"] is False
    assert plan["common_policy_reselection_permitted"] is False


def test_replaced_common_stack_is_explicitly_retrained_not_reused(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.force_retrain_units = frozenset({(3, RUN.SEEDS[0])})
    common = {
        "selection_lock": {"path": "selection", "sha256": "a" * 64},
        "capacity_selection": {"path": "capacity", "sha256": "b" * 64},
        "policy": {"path": "policy", "sha256": "c" * 64},
    }
    plan = RUN._build_static_plan(args, inputs={}, sources={}, common=common)
    by_key = {
        (unit["outer_fold"], unit["seed"]): unit for unit in plan["units"]
    }
    assert by_key[(3, RUN.SEEDS[0])]["mode"] == "fixed_i3_default_training"
    assert by_key[(3, RUN.SEEDS[1])]["mode"] == (
        "reuse_common_locked_discovery_unit"
    )
    assert plan["reuse"]["unit_count"] == 5
    assert plan["new_training"]["unit_count"] == 13
    assert plan["new_training"]["forced_common_provenance_replacements"] == [
        {"outer_fold": 3, "seed": RUN.SEEDS[0]}
    ]


def test_force_retrain_unit_parser_is_strict() -> None:
    assert RUN.parse_force_retrain_units("3:20260828,4:20260830") == frozenset(
        {(3, 20260828), (4, 20260830)}
    )
    for raw in (
        "3",
        "x:20260828",
        "2:20260828",
        "3:1",
        "3:20260828,3:20260828",
    ):
        with pytest.raises(ValueError):
            RUN.parse_force_retrain_units(raw)


def _retrain_impact_pair(tmp_path: Path) -> tuple[Path, Path]:
    bindings = {}
    for name in ("full_plan", "main_index", "retrain_plan", "retrain_index"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}", encoding="utf-8")
        bindings[name] = {
            "path": str(path.resolve()),
            "sha256": RUN.sha256_file(path),
            "content_sha256": name[0] * 64,
        }
    merged_path = tmp_path / "merged_index.json"
    _write_content_json(
        merged_path,
        {
            "schema_version": 1,
            "classification": "retrospective_fully_nested_non_test_proposer_index",
            "merge_classification": (
                "retrospective_current_source_uniform_90_unit_proposer_index"
            ),
            "merge_provenance": {
                "full_split_authority_plan": bindings["full_plan"],
                "retrain_plan": bindings["retrain_plan"],
                "source_indexes": {
                    "main": bindings["main_index"],
                    "current_source_retrain_f34": bindings["retrain_index"],
                },
            },
        },
    )
    audit_path = tmp_path / "retrain_impact_audit.json"
    _write_content_json(
        audit_path,
        {
            "schema_version": 1,
            "classification": (
                "retrospective_nested_proposer_retrain_impact_audit"
            ),
            "commercial_claim_authorized": False,
            "outer_test_opened": False,
            "outer_test_record_count": 0,
            "target_or_reference_accessed": False,
            "source_campaigns_hash_complete": True,
            "source_plans_compatible": True,
            "inputs": bindings,
            "comparison": {
                "affected_hcs_units": [
                    {"outer_fold": 3, "seed": RUN.SEEDS[0]}
                ],
                "force_retrain_unit_count": 1,
                "force_retrain_units_cli_value": f"3:{RUN.SEEDS[0]}",
                "checkpoint_change_alone_forces_hcs_retrain": False,
                "prediction_paths_ignored_after_bound_file_validation": True,
            },
        },
    )
    return audit_path, merged_path


def test_retrain_impact_audit_auto_binds_exact_forced_set(tmp_path: Path) -> None:
    audit, merged = _retrain_impact_pair(tmp_path)
    forced, binding = RUN.validate_retrain_impact_audit(audit, merged, None)
    assert forced == frozenset({(3, RUN.SEEDS[0])})
    assert binding["audited_force_retrain_units"] == [
        {"outer_fold": 3, "seed": RUN.SEEDS[0]}
    ]


def test_retrain_impact_audit_rejects_operator_override(tmp_path: Path) -> None:
    audit, merged = _retrain_impact_pair(tmp_path)
    with pytest.raises(RuntimeError, match="differs from the sealed"):
        RUN.validate_retrain_impact_audit(audit, merged, frozenset())


def test_test_manifest_or_test_argument_is_rejected_without_opening_it(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="outer-test manifest"):
        RUN._manifest_path("outer_0/test_pred_0.json", tmp_path / "plan.json")
    with pytest.raises(RuntimeError, match="outer-test argument"):
        RUN._assert_no_test_command([sys.executable, "worker.py", "--test-manifest", "secret.json"])


def test_status_snapshot_tampering_blocks_resume(tmp_path: Path) -> None:
    RUN.publish_current(tmp_path, "pretest_status.json", {"schema_version": 1, "status": "pending"})
    (tmp_path / "pretest_status.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="tampered"):
        RUN.publish_current(tmp_path, "pretest_status.json", {"schema_version": 1, "status": "complete"})


def test_immutable_publications_and_status_snapshots_use_mode_0444(
    tmp_path: Path,
) -> None:
    immutable = tmp_path / "campaign_lock.json"
    RUN.exclusive_write(immutable, b"sealed\n")
    assert immutable.stat().st_mode & 0o777 == 0o444
    RUN.exclusive_write(immutable, b"sealed\n")
    assert immutable.stat().st_mode & 0o777 == 0o444

    status = RUN.publish_current(
        tmp_path,
        "pretest_status.json",
        {"schema_version": 1, "status": "in_progress"},
    )
    snapshot = tmp_path / "pretest_status_snapshots" / f"{status['content_sha256']}.json"
    assert snapshot.stat().st_mode & 0o777 == 0o444


def test_interrupted_partial_attempt_is_preserved(tmp_path: Path) -> None:
    base = tmp_path / "training"
    partial = base / "attempt_000"
    partial.mkdir(parents=True)
    (partial / "run_manifest.json").write_text("{}", encoding="utf-8")
    choice = RUN.choose_training_attempt(base)
    assert choice["action"] == "train_fresh"
    assert choice["attempt"] == 1
    assert choice["path"] == base.resolve() / "attempt_001"
    assert choice["preserved_attempts"] == [partial.resolve()]
    assert (partial / "run_manifest.json").is_file()


def test_valid_prelock_attempt_uses_publication_recovery(tmp_path: Path) -> None:
    base = tmp_path / "training"
    attempt = base / "attempt_000"
    attempt.mkdir(parents=True)
    history = [
        {
            "epoch": epoch,
            "retrospective_selection_key": [0.0] if epoch == 1 else [1.0],
        }
        for epoch in range(1, 21)
    ]
    (attempt / "history.json").write_text(json.dumps(history), encoding="utf-8")
    (attempt / "run_manifest.json").write_text("{}", encoding="utf-8")
    (attempt / "scaler.json").write_text("{}", encoding="utf-8")
    with zipfile.ZipFile(attempt / "best_checkpoint.pt", "w") as archive:
        archive.writestr("checkpoint/data.pkl", b"payload")
    choice = RUN.choose_training_attempt(base)
    assert choice["action"] == "recover_prelock"
    assert choice["attempt"] == 0
    args = _args(tmp_path)
    command = RUN._commands_for_unit(
        args, 0, RUN.SEEDS[0], attempt=0, recover_prelock=True
    )["trainer"]
    assert command[-1] == "--recover-prelock"
    assert command[-len(RUN.fixed_trainer_flags()) - 1:-1] == list(RUN.fixed_trainer_flags())


def test_mid_training_core_artifacts_do_not_trigger_premature_recovery(tmp_path: Path) -> None:
    base = tmp_path / "training"
    attempt = base / "attempt_000"
    attempt.mkdir(parents=True)
    history = [
        {"epoch": epoch, "retrospective_selection_key": [float(20 - epoch)]}
        for epoch in range(1, 8)
    ]
    (attempt / "history.json").write_text(json.dumps(history), encoding="utf-8")
    (attempt / "run_manifest.json").write_text("{}", encoding="utf-8")
    (attempt / "scaler.json").write_text("{}", encoding="utf-8")
    with zipfile.ZipFile(attempt / "best_checkpoint.pt", "w") as archive:
        archive.writestr("checkpoint/data.pkl", b"payload")
    choice = RUN.choose_training_attempt(base)
    assert choice["action"] == "train_fresh"
    assert choice["attempt"] == 1
    assert choice["preserved_attempts"] == [attempt.resolve()]


def test_corrupt_prelock_payload_advances_to_fresh_version(tmp_path: Path) -> None:
    base = tmp_path / "training"
    attempt = base / "attempt_004"
    attempt.mkdir(parents=True)
    (attempt / "history.json").write_text("not-json", encoding="utf-8")
    (attempt / "run_manifest.json").write_text("{}", encoding="utf-8")
    (attempt / "scaler.json").write_text("{}", encoding="utf-8")
    (attempt / "best_checkpoint.pt").write_bytes(b"interrupted")
    choice = RUN.choose_training_attempt(base)
    assert choice["action"] == "train_fresh"
    assert choice["attempt"] == 5
    assert choice["path"].name == "attempt_005"
    assert (attempt / "best_checkpoint.pt").read_bytes() == b"interrupted"


def test_locked_complete_attempt_is_idempotently_reused(tmp_path: Path) -> None:
    base = tmp_path / "training"
    attempt = base / "attempt_002"
    attempt.mkdir(parents=True)
    for name in RUN.REQUIRED_TRAINING_FILES:
        (attempt / name).write_bytes(name.encode())
    first = RUN.choose_training_attempt(base)
    second = RUN.choose_training_attempt(base)
    assert first["action"] == second["action"] == "validate_complete"
    assert first["attempt"] == second["attempt"] == 2
    assert first["path"] == second["path"] == attempt.resolve()
    assert not (base / "attempt_003").exists()
