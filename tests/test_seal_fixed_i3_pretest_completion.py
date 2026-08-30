from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/seal_fixed_i3_pretest_completion.py"
SPEC = importlib.util.spec_from_file_location("seal_fixed_i3_pretest_completion", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SEAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SEAL
SPEC.loader.exec_module(SEAL)


def _write(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": str(path.resolve()), "sha256": SEAL.sha256_file(path), "bytes": len(payload)}


def _tree(root: Path) -> dict[str, Any]:
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    records = [
        [str(path.relative_to(root)), SEAL.sha256_file(path), path.stat().st_size]
        for path in paths
    ]
    return {
        "path": str(root.resolve()),
        "files": {
            name: {"sha256": digest, "bytes": size} for name, digest, size in records
        },
        "tree_sha256": SEAL._tree_sha(records),
    }


def _fixture(tmp_path: Path) -> dict[str, Path]:
    merged = tmp_path / "campaign/current_source_merged/index.json"
    _write(merged, b'{"complete":90}')
    source = tmp_path / "runtime/source.py"
    binding = tmp_path / "runtime/binding.json"
    tree_root = tmp_path / "runtime/input_tree"
    tree_root_2 = tmp_path / "runtime/input_tree_2"
    _write(source, b"SOURCE = True\n")
    _write(binding, b"{}")
    _write(tree_root / "payload.bin", b"payload")
    _write(tree_root_2 / "payload.bin", b"payload-2")
    fixed_source = Path(SEAL.PROJECT_ROOT / "scripts/run_fixed_i3_pretest_campaign.py")
    verifier_source = Path(SEAL.PROJECT_ROOT / "scripts/seal_runtime_inputs.py")
    runtime = SEAL.runtime_seal.inventory(
        sources=[source, fixed_source, verifier_source],
        trees=[tree_root, tree_root_2],
        bindings=[binding, merged],
        post_launch_attestation=False,
    )
    runtime["fixed_i3_context"] = {
        "classification": "retrospective_fixed_i3_pretest_runtime_input_context",
        "outer_test_opened": False,
        "outer_test_record_count": 0,
        "target_or_evaluation_artifact_accessed": False,
        "proposer_matrix_groups": 18,
        "proposer_matrix_units": 90,
        "validated_index": {
            "path": str(merged.resolve()),
            "sha256": SEAL.sha256_file(merged),
            "bytes": merged.stat().st_size,
        },
    }
    runtime["content_sha256"] = SEAL.runtime_seal.canonical_sha256(runtime)
    runtime_path = tmp_path / "campaign/current_source_merged/fixed_i3_pretest_runtime_seal.json"
    SEAL.runtime_seal.atomic_json(runtime_path, runtime)

    common = {}
    for name in ("selection_lock", "capacity_selection", "policy", "source_freeze_manifest"):
        common[name] = _write(tmp_path / "pretest/common" / f"{name}.json", name.encode())
    units = []
    artifact_names = (
        "selection_lock",
        "checkpoint",
        "scaler",
        "cache_manifest",
        "fallback_oof",
        "fallback_provenance",
        "run_manifest",
        "original_policy",
        "history",
        "validation_metrics",
        "validation_predictions",
        "strict_stack",
    )
    for seed in SEAL.SEEDS:
        for fold in SEAL.FOLDS:
            root = tmp_path / "pretest/units" / f"outer_{fold}_seed_{seed}"
            artifacts = {
                name: _write(root / f"{name}.bin", f"{fold}/{seed}/{name}".encode())
                for name in artifact_names
            }
            units.append(
                {
                    "outer_fold": fold,
                    "seed": seed,
                    "status": "complete",
                    "outer_test_opened": False,
                    "artifacts": artifacts,
                    "output_tree": _tree(root),
                }
            )
    runtime_binding = SEAL.bind_file(runtime_path)
    runtime_input_binding = {
        **runtime_binding,
        "content_sha256": runtime["content_sha256"],
        "verified_files": 1,
        "attestation_phase": "prelaunch",
    }
    campaign = {
        "schema_version": 1,
        "classification": "retrospective_fixed_i3_pretest_completion_plan",
        "matrix": {"folds": list(SEAL.FOLDS), "seeds": list(SEAL.SEEDS), "unit_count": 18},
        "outer_test_opened": False,
        "inputs": {"runtime_input_seal": runtime_input_binding},
        "sources": {
            "orchestrator": SEAL.bind_file(fixed_source),
            "runtime_seal_verifier": SEAL.bind_file(verifier_source),
        },
    }
    campaign["content_sha256"] = SEAL.canonical_sha256(campaign)
    campaign_path = tmp_path / "pretest/campaign_lock.json"
    campaign_path.write_text(json.dumps(campaign, sort_keys=True), encoding="utf-8")
    campaign_path.chmod(0o444)
    status_root = tmp_path / "pretest/pretest_status_snapshots"
    final_status: dict[str, Any] | None = None
    for count in range(1, 19):
        status = {
            "schema_version": 1,
            "classification": "retrospective_fixed_i3_pretest_status",
            "campaign_lock_sha256": campaign["content_sha256"],
            "matrix_unit_count": 18,
            "completed_units": count,
            "status": "complete" if count == 18 else "in_progress",
            "outer_test_opened": False,
            "units": units[:count],
        }
        status["content_sha256"] = SEAL.canonical_sha256(status)
        status_path = status_root / f"{status['content_sha256']}.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, sort_keys=True), encoding="utf-8")
        status_path.chmod(0o444)
        final_status = status
    assert final_status is not None
    final_status_path = tmp_path / "pretest/pretest_status.json"
    final_status_path.write_text(
        json.dumps(final_status, sort_keys=True), encoding="utf-8"
    )
    final_status_path.chmod(0o444)
    pretest = {
        "schema_version": 1,
        "classification": "retrospective_fixed_i3_pretest_index",
        "status": "complete",
        "matrix": {"folds": list(SEAL.FOLDS), "seeds": list(SEAL.SEEDS), "unit_count": 18},
        "completed_units": 18,
        "campaign_lock_sha256": campaign["content_sha256"],
        "common": common,
        "units": units,
        "outer_test_opened": False,
        "outer_test_artifact_count": 0,
        "ready_for_separately_locked_label_free_outer_test_construction": True,
    }
    pretest["content_sha256"] = SEAL.canonical_sha256(pretest)
    pretest_path = tmp_path / "pretest/pretest_index.json"
    pretest_path.write_text(json.dumps(pretest, sort_keys=True), encoding="utf-8")
    pretest_path.chmod(0o444)
    pretest_snapshot = (
        tmp_path
        / "pretest/pretest_index_snapshots"
        / f"{pretest['content_sha256']}.json"
    )
    pretest_snapshot.parent.mkdir(parents=True, exist_ok=True)
    pretest_snapshot.write_bytes(pretest_path.read_bytes())
    pretest_snapshot.chmod(0o444)
    return {
        "merged": merged,
        "runtime": runtime_path,
        "pretest": pretest_path,
        "output": tmp_path / "pretest/fixed_runtime_completion_attestation.json",
    }


def test_completion_seal_closes_exact_18_unit_runtime_payload_and_is_immutable(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    first = SEAL.seal_completion(
        merged_index=paths["merged"],
        runtime_input_seal=paths["runtime"],
        pretest_index=paths["pretest"],
        output=paths["output"],
        python_executable=Path(sys.executable),
    )
    assert first["classification"] == "fixed_i3_pretest_runtime_completion_attestation"
    assert first["completed_units"] == 18
    assert len(first["unit_payloads"]) == 18
    assert paths["output"].stat().st_mode & 0o777 == 0o444
    before = paths["output"].read_bytes()
    second = SEAL.seal_completion(
        merged_index=paths["merged"],
        runtime_input_seal=paths["runtime"],
        pretest_index=paths["pretest"],
        output=paths["output"],
        python_executable=Path(sys.executable),
    )
    assert second == first
    assert paths["output"].read_bytes() == before
    paths["output"].chmod(0o644)
    with pytest.raises(SEAL.FixedCompletionError, match="mode must be exactly 0444"):
        SEAL.verify_completion_attestation(paths["output"])


def test_completion_rejects_mutable_index_snapshot_and_execution_evidence_modes(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    pretest = json.loads(paths["pretest"].read_text(encoding="utf-8"))
    candidates = [
        paths["pretest"],
        paths["pretest"].parent
        / "pretest_index_snapshots"
        / f"{pretest['content_sha256']}.json",
        paths["pretest"].parent / "campaign_lock.json",
        paths["pretest"].parent / "pretest_status.json",
        sorted((paths["pretest"].parent / "pretest_status_snapshots").glob("*.json"))[0],
    ]
    for candidate in candidates:
        candidate.chmod(0o644)
        with pytest.raises(SEAL.FixedCompletionError, match="mode must be exactly 0444"):
            SEAL.seal_completion(
                merged_index=paths["merged"],
                runtime_input_seal=paths["runtime"],
                pretest_index=paths["pretest"],
                output=paths["output"],
            )
        candidate.chmod(0o444)
        assert not paths["output"].exists()


def test_output_tree_or_artifact_tampering_fails_before_attestation(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    pretest = json.loads(paths["pretest"].read_text(encoding="utf-8"))
    artifact = Path(pretest["units"][0]["artifacts"]["checkpoint"]["path"])
    artifact.write_bytes(b"tampered")
    with pytest.raises(SEAL.FixedCompletionError, match="file hash mismatch"):
        SEAL.seal_completion(
            merged_index=paths["merged"],
            runtime_input_seal=paths["runtime"],
            pretest_index=paths["pretest"],
            output=paths["output"],
        )
    assert not paths["output"].exists()


def test_target_or_runtime_drift_is_rejected_without_opening_target(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    forbidden = paths["pretest"].parent / "canonical_locked_hcs_targets.npz"
    forbidden.write_bytes(b"must-not-open")
    with pytest.raises(SEAL.FixedCompletionError, match="target/evaluation"):
        SEAL.seal_completion(
            merged_index=paths["merged"],
            runtime_input_seal=paths["runtime"],
            pretest_index=paths["pretest"],
            output=paths["output"],
        )
    assert forbidden.read_bytes() == b"must-not-open"
