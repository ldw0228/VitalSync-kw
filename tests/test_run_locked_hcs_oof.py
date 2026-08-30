from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_locked_hcs_oof.py"
SPEC = importlib.util.spec_from_file_location("run_locked_hcs_oof", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def _write(path: Path, data: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": str(path.resolve()), "sha256": RUN.sha256_file(path)}


def _json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": RUN.sha256_file(path)}


def _stub(path: Path) -> None:
    path.write_text(
        """from pathlib import Path
import json, sys
import numpy as np
stage, output, fold, seed, log = sys.argv[1:]
output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
with Path(log).open('a',encoding='utf-8') as stream: stream.write(f'{fold}/{seed}/{stage}\\n')
fold=int(fold); seed=int(seed)
if stage=='train': output.write_bytes(b'test proposer checkpoint')
elif stage=='proposer':
    with output.open('wb') as stream: np.savez_compressed(stream,cache_index=np.asarray([fold],dtype=np.int64),prediction=np.asarray([20+fold],dtype=np.float32))
elif stage=='cache':
    output.write_text(json.dumps({'schema_version':1,'artifact_type':'hcs_test_only_inference_cache','row_scope':'outer_test_only','target_fields_present':False,'outer_fold':fold,'seed':seed},sort_keys=True))
elif stage=='infer':
    fallback=np.asarray([20.25+fold],dtype=np.float32)
    with output.open('wb') as stream: np.savez_compressed(stream,cache_index=np.asarray([fold],dtype=np.int64),fallback_rr_bpm=fallback,fallback_std_bpm=np.asarray([1],dtype=np.float32),fallback_available=np.asarray([True]),source_rr_bpm=np.asarray([40-fold],dtype=np.float32),source_scale_bpm=np.asarray([1],dtype=np.float32),source_available=np.asarray([True]),selected_probability=np.asarray([.99],dtype=np.float32),margin=np.asarray([.9],dtype=np.float32),entropy=np.asarray([.1],dtype=np.float32),quality=np.asarray([1],dtype=np.float32),valid_candidate_count=np.asarray([12],dtype=np.int16))
else: raise SystemExit(2)
""",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    common_root = tmp_path / "common"
    freeze_root = tmp_path / "freeze"
    source_names = (
        "train_harmonic_set_snn.py",
        "harmonic_set_models.py",
        "ADAPTIVE_CAMPAIGN_CONTRACT.json",
        "harmonic_set_v2.yaml",
    )
    source_hashes = {}
    for name in source_names:
        path = freeze_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        source_hashes[name] = RUN.sha256_file(path)
    freeze = {
        "schema_version": 1,
        "declared_before_any_i3_score": True,
        "outer_test_opened": False,
        "files": source_hashes,
    }
    freeze_binding = _json(freeze_root / "MANIFEST.json", freeze)
    capacity = {
        "schema_version": 1,
        "outer_test_opened": False,
        "selected_preset": "default",
        "source_freeze_manifest_sha256": freeze_binding["sha256"],
    }
    capacity_binding = _json(common_root / "capacity_selection.json", capacity)
    policy_payload = {
        "selection_status": "fail_closed_no_action",
        "correction_pull": 0.0,
        "probability_threshold": 1.1,
        "margin_threshold": 1.1,
        "entropy_threshold": 0.1,
        "quality_threshold": 0.75,
        "minimum_valid_candidates": 12,
        "base_std_max": None,
        "source_scale_max": None,
        "disagreement_min": None,
        "disagreement_max": None,
    }
    policy = {
        "schema_version": 1,
        "outer_test_opened": False,
        "selected_preset": "default",
        "policy": policy_payload,
    }
    policy_binding = _json(common_root / "common_fallback_policy.json", policy)
    common_lock = {
        "schema_version": 1,
        "outer_test_opened_before_lock": False,
        "selected_preset": "default",
        "selected_parameter_count": 195603,
        "capacity_selection_sha256": capacity_binding["sha256"],
        "common_fallback_policy_sha256": policy_binding["sha256"],
        "policy_selection_status": "fail_closed_no_action",
        "source_freeze": source_hashes,
    }
    common_lock_binding = _json(common_root / "selection_lock.json", common_lock)

    output_root = tmp_path / "output"
    log = tmp_path / "stage_calls.log"
    stub = tmp_path / "stub.py"
    _stub(stub)
    seeds = [101, 102, 103]
    units = []
    for seed in seeds:
        for fold in range(6):
            source = tmp_path / "locked" / f"outer_{fold}_seed_{seed}"
            checkpoint = _write(source / "best_checkpoint.pt", f"checkpoint {fold} {seed}".encode())
            scaler = _json(source / "scaler.json", {"center": [0], "scale": [1]})
            cache = _json(source / "manifest.json", {"complete": True})
            fallback = _write(source / "fallback.csv", b"cache_index,prediction_bpm\n")
            original_policy = _json(source / "fallback_policy.json", {"diagnostic": True})
            run_manifest_document = {
                "outer_fold": fold,
                "optimization": {"seed": seed},
                "model_config": {"hidden_channels": 64},
                "iteration_effective_configuration": {
                    "model_capacity": {"parameter_count": 195603}
                },
            }
            run_manifest = _json(source / "run_manifest.json", run_manifest_document)
            source_bindings = {
                "trainer": {"sha256": source_hashes["train_harmonic_set_snn.py"]},
                "harmonic_set_model": {"sha256": source_hashes["harmonic_set_models.py"]},
                "adaptive_campaign_contract": {
                    "sha256": source_hashes["ADAPTIVE_CAMPAIGN_CONTRACT.json"]
                },
                "campaign_config": {"sha256": source_hashes["harmonic_set_v2.yaml"]},
            }
            lock_document = {
                "schema_version": 1,
                "outer_fold": fold,
                "seed": seed,
                "adaptive_iteration": 3,
                "outer_test_not_opened_before_this_lock": True,
                "checkpoint_sha256": checkpoint["sha256"],
                "scaler_sha256": scaler["sha256"],
                "cache_manifest_sha256": cache["sha256"],
                "fallback_oof_sha256": fallback["sha256"],
                "run_manifest_sha256": run_manifest["sha256"],
                "policy_sha256": original_policy["sha256"],
                "source_bindings": source_bindings,
            }
            selection = _json(source / "selection_lock.json", lock_document)
            test_manifest = _json(
                tmp_path / "test_manifests" / f"outer_{fold}.json",
                {"fold": fold, "label_free": True},
            )
            unit_root = output_root / "units" / f"outer_{fold}_seed_{seed}"
            derived = {
                "test_proposer_checkpoint": str(unit_root / "work/test_proposer.pt"),
                "test_proposer_prediction": str(unit_root / "work/test_proposer.npz"),
                "derived_cache_manifest": str(unit_root / "work/derived_manifest.json"),
                "raw_hcs_prediction": str(unit_root / "work/raw_hcs.npz"),
            }
            stage_kinds = ("train", "proposer", "cache", "infer")
            stages = []
            for stage_name, kind in zip(RUN.STAGES, stage_kinds, strict=True):
                artifact_name = RUN.ARTIFACT_FOR_STAGE[stage_name]
                output = derived[artifact_name]
                stages.append(
                    {
                        "name": stage_name,
                        "argv": [sys.executable, str(stub), kind, output, str(fold), str(seed), str(log)],
                        "outputs": [output],
                    }
                )
            units.append(
                {
                    "outer_fold": fold,
                    "seed": seed,
                    "selection_lock": selection,
                    "locked_assets": {
                        "checkpoint": checkpoint,
                        "scaler": scaler,
                        "cache_manifest": cache,
                        "fallback_oof": fallback,
                        "run_manifest": run_manifest,
                        "original_policy": original_policy,
                    },
                    "test_manifest": test_manifest,
                    "stages": stages,
                    "derived_artifacts": derived,
                }
            )
    plan = {
        "schema_version": 1,
        "classification": "locked_hcs_oof_inference_plan",
        "folds": list(range(6)),
        "seeds": seeds,
        "common": {
            "selection_lock": common_lock_binding,
            "capacity_selection": capacity_binding,
            "policy": policy_binding,
            "source_freeze_manifest": freeze_binding,
        },
        "units": units,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    target = tmp_path / "targets.npz"
    with target.open("wb") as stream:
        np.savez_compressed(
            stream,
            cache_index=np.arange(6, dtype=np.int64),
            outer_fold=np.arange(6, dtype=np.int16),
            target_rr_bpm=np.arange(20, 26, dtype=np.float32),
            identity=np.asarray([f"id{fold}" for fold in range(6)]),
            reference_valid=np.ones(6, dtype=bool),
        )
    return plan_path, output_root, target, log


def test_no_action_is_bit_exact_and_resume_is_idempotent(tmp_path: Path) -> None:
    plan, output, _, log = _fixture(tmp_path)
    first = RUN.run_inference(plan, output, orchestrator_command=["synthetic"])
    assert first["status"] == "label_free_predictions_sealed"
    assert len(log.read_text(encoding="utf-8").splitlines()) == 72
    for prediction in output.glob("units/*/sealed_label_free_predictions.npz"):
        with np.load(prediction, allow_pickle=False) as archive:
            assert np.array_equal(
                archive["final_rr_bpm"].view(np.uint32),
                archive["fallback_rr_bpm"].view(np.uint32),
            )
            assert not archive["applied_pull"].any()
    first_seal = (output / "predictions_seal.json").read_bytes()
    second = RUN.run_inference(plan, output, orchestrator_command=["synthetic"])
    assert second["status"] == "label_free_predictions_sealed"
    assert (output / "predictions_seal.json").read_bytes() == first_seal
    assert len(log.read_text(encoding="utf-8").splitlines()) == 72


def test_hash_tampering_fails_before_any_post_lock_command(tmp_path: Path) -> None:
    plan, output, _, log = _fixture(tmp_path)
    document = json.loads(plan.read_text(encoding="utf-8"))
    checkpoint = Path(document["units"][0]["locked_assets"]["checkpoint"]["path"])
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(RUN.LockedOOFError, match="hash mismatch"):
        RUN.run_inference(plan, output)
    assert not log.exists()
    assert not (output / "pretest_lock.json").exists()


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("classification", "wrong_receipt_classification", "identity/command"),
        ("stage", "wrong_stage", "identity/command"),
        ("argv", ["wrong", "argv"], "identity/command"),
        ("outputs", "append", "output topology"),
    ),
)
def test_resume_revalidates_exact_stage_receipt_contract(
    tmp_path: Path,
    field: str,
    replacement: Any,
    message: str,
) -> None:
    plan, output, _, _ = _fixture(tmp_path)
    RUN.run_inference(plan, output, max_units=1)
    unit_root = output / "units/outer_0_seed_101"
    receipt_path = unit_root / "receipts/00_test_proposer_train.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if replacement == "append":
        receipt[field].append(dict(receipt[field][0]))
    else:
        receipt[field] = replacement
    receipt_path.chmod(0o644)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    derived_path = unit_root / "derived_inference_lock.json"
    derived = json.loads(derived_path.read_text(encoding="utf-8"))
    derived["stage_receipts"][0] = RUN.bind_file(receipt_path)
    derived_path.chmod(0o644)
    derived_path.write_text(json.dumps(derived, sort_keys=True), encoding="utf-8")

    with pytest.raises(RUN.LockedOOFError, match=message):
        RUN.run_inference(plan, output, max_units=1)


def test_stage_plan_output_must_remain_inside_deterministic_unit_root(
    tmp_path: Path,
) -> None:
    plan, output, _, log = _fixture(tmp_path)
    document = json.loads(plan.read_text(encoding="utf-8"))
    unit = next(
        item
        for item in document["units"]
        if item["outer_fold"] == 0 and item["seed"] == 101
    )
    escaped = tmp_path / "escaped_output.pt"
    unit["stages"][0]["outputs"] = [str(escaped)]
    unit["derived_artifacts"]["test_proposer_checkpoint"] = str(escaped)
    plan.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    with pytest.raises(RUN.LockedOOFError, match="escapes its unit root"):
        RUN.run_inference(plan, output, max_units=1)
    assert not escaped.exists()
    assert not log.exists()


def test_incomplete_seal_and_early_target_join_fail_closed(tmp_path: Path) -> None:
    plan, output, target, _ = _fixture(tmp_path)
    partial = RUN.run_inference(plan, output, max_units=17)
    assert partial["target_join_authorized"] is False
    assert not (output / "predictions_seal.json").exists()
    with pytest.raises(RUN.LockedOOFError, match="18 predictions"):
        RUN.join_and_evaluate(output, target)
    assert not (output / "evaluation_lock.json").exists()


def test_single_join_requires_seal_and_is_idempotent(tmp_path: Path) -> None:
    plan, output, target, _ = _fixture(tmp_path)
    with pytest.raises(RUN.LockedOOFError, match="18 predictions"):
        RUN.join_and_evaluate(output, tmp_path / "must_not_be_opened.npz")
    RUN.run_inference(plan, output)
    first = RUN.join_and_evaluate(output, target, orchestrator_command=["join"])
    assert first["target_join_count"] == 1
    assert first["commercial_claim_authorized"] is False
    first_bytes = (output / "evaluation_lock.json").read_bytes()
    second = RUN.join_and_evaluate(output, target, orchestrator_command=["join"])
    assert second["target_join_count"] == 1
    assert (output / "evaluation_lock.json").read_bytes() == first_bytes


def test_prepare_materializes_concrete_fast_path_from_pretest_index(tmp_path: Path) -> None:
    source_plan, _, _, _ = _fixture(tmp_path)
    source = json.loads(source_plan.read_text(encoding="utf-8"))
    pretest_units = []
    for unit in source["units"]:
        assets = dict(unit["locked_assets"])
        assets["selection_lock"] = unit["selection_lock"]
        fold = unit["outer_fold"]
        seed = unit["seed"]
        proposer_root = tmp_path / "validation_proposer" / f"outer_{fold}_seed_{seed}"
        checkpoint = _write(proposer_root / "fold_1/snn_best.pt", b"validation proposer")
        run_config = _json(proposer_root / "run_config.json", {"run_signature": "x"})
        source_manifest = _json(proposer_root / "validation_manifest.json", {"safe": True})
        strict_stack = tmp_path / "strict_stacks" / f"outer_{fold}_seed_{seed}.npz"
        strict_stack.parent.mkdir(parents=True, exist_ok=True)
        provenance = {
            "source_units": [
                {
                    "role": "hcs_validation",
                    "checkpoint": checkpoint["path"],
                    "run_config": run_config["path"],
                    "run_config_sha256": run_config["sha256"],
                    "manifest": source_manifest["path"],
                }
            ]
        }
        with strict_stack.open("wb") as stream:
            np.savez_compressed(stream, provenance_json=np.asarray(json.dumps(provenance)))
        assets["strict_stack"] = {
            "path": str(strict_stack.resolve()),
            "sha256": RUN.sha256_file(strict_stack),
        }
        pretest_units.append(
            {
                "outer_fold": unit["outer_fold"],
                "seed": unit["seed"],
                "status": "complete",
                "artifacts": assets,
            }
        )
    pretest = {
        "schema_version": 1,
        "classification": "retrospective_fixed_i3_pretest_index",
        "status": "complete",
        "matrix": {"folds": list(range(6)), "seeds": source["seeds"], "unit_count": 18},
        "completed_units": 18,
        "common": source["common"],
        "units": pretest_units,
        "outer_test_opened": False,
        "ready_for_separately_locked_label_free_outer_test_construction": True,
    }
    pretest["content_sha256"] = RUN.canonical_json_sha256(pretest)
    pretest_path = tmp_path / "pretest_index.json"
    pretest_path.write_text(json.dumps(pretest, sort_keys=True), encoding="utf-8")
    manifest_root = tmp_path / "test_manifests"
    for fold in range(6):
        manifest = {
            "schema_version": 1,
            "fold_id": 100 * fold + 60,
            "fold_assignments": {"path": "folds.json", "sha256": "0" * 64},
            "cache": {"manifest_path": "manifest.json", "manifest_sha256": "1" * 64},
            "identities": {
                "train": ["a"],
                "validation": ["b"],
                "prediction": ["c"],
                "excluded": [],
                "scaler": ["a"],
            },
        }
        manifest["content_sha256"] = RUN.canonical_json_sha256(manifest)
        path = manifest_root / f"outer_{fold}" / f"test_pred_{fold}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    rf_cache = tmp_path / "rf"
    _json(rf_cache / "manifest.json", {"sessions": []})
    output = tmp_path / "prepared"
    result = RUN.prepare_plan(
        pretest_index=pretest_path,
        test_manifest_root=manifest_root,
        output_root=output,
        plan_output=output / "plan.json",
        rf_cache=rf_cache,
        python_executable=Path(sys.executable),
    )
    assert result["units"] == 18
    prepared = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    assert all(
        [stage["name"] for stage in unit["stages"]] == list(RUN.FAST_NO_ACTION_STAGES)
        for unit in prepared["units"]
    )
    assert all(unit["bound_but_not_executed_hcs_checkpoint"] for unit in prepared["units"])
