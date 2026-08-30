from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import sys
import time
import py_compile
from typing import Any, Mapping
import zipfile

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_hfr_v3r1_target_sealed.py"
)
SPEC = importlib.util.spec_from_file_location("v8r4_target_sealed", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)

WORKSPACE = Path(__file__).resolve().parents[3] / "home"  # deliberately unused
REAL_PROJECT = Path("/home/hwiseong/Documents/SnnProject")
RUNTIME_DEPENDENCY_SOURCE = Path(
    os.environ.get("SNN_RR_TEST_RUNTIME_DEPENDENCY_SOURCE", str(REAL_PROJECT))
)
REAL_VENV = REAL_PROJECT / ".venv"
REAL_INTERPRETER = REAL_VENV / "bin/python"
REAL_RUNTIME = REAL_INTERPRETER.resolve().parent.parent
FROZEN_CONTEXT1_LEDGER_RAW = {
    role: (
        RUNTIME_DEPENDENCY_SOURCE / str(binding["path"])
    ).read_bytes()
    for role, binding in runtime.BENCHMARK_ADMITTED_CONTEXT_LEDGER_PREFIXES.items()
}


def _write_frozen(path: Path, raw: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o444)
    return path


def _json_document(**values: Any) -> dict[str, Any]:
    document = dict(values)
    document["content_sha256"] = runtime.semantic_sha256(document)
    return document


def _write_json(path: Path, document: Mapping[str, Any]) -> Path:
    return _write_frozen(
        path,
        json.dumps(
            dict(document), ensure_ascii=False, sort_keys=True, indent=2
        ).encode("utf-8")
        + b"\n",
    )


def _write_exact_governance_json(
    path: Path, role: str, document: Mapping[str, Any]
) -> Path:
    value = dict(document)
    value.pop("content_sha256", None)
    for key in runtime.GOVERNANCE_TOP_LEVEL_KEYS[role]:
        if key != "content_sha256":
            value.setdefault(key, None)
    value["content_sha256"] = runtime.semantic_sha256(value)
    return _write_json(path, value)


def _binding_row(project: Path, path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(project).as_posix(),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "mode": stat.S_IMODE(path.stat().st_mode),
    }


def _artifact_binding(root: Path, path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _authorization_binding(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _make_pack(
    project: Path,
    *,
    phase: str = "discovery",
    outer_fold: int = 3,
    promotion_authorization: Path | None = None,
    selection_lock: Path | None = None,
) -> tuple[Path, Path]:
    root = project / "sealed" / f"{phase}_shard_outer_{outer_fold}"
    units: list[dict[str, Any]] = []
    for seed in runtime.SEEDS:
        relative = Path("units") / f"outer_{outer_fold}_seed_{seed}"
        unit = root / relative
        if phase == "promotion_prediction":
            assert promotion_authorization is not None and selection_lock is not None
            predict = unit / "outer_predict_input.npz"
            predict.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(predict, mode="w", compression=zipfile.ZIP_STORED) as archive:
                for field in runtime.OUTER_PREDICT_FIELDS:
                    archive.writestr(f"{field}.npy", b"npy")
            predict.chmod(0o444)
            legacy = {"path": "legacy/input", "sha256": "1" * 64, "bytes": 1}
            manifest = _write_json(
                unit / "OUTER_PREDICTION_PACK_MANIFEST.json",
                _json_document(
                    schema_version=1,
                    classification="adaptive_v3r1_v8r4_authorized_outer_prediction_pack",
                    campaign_id=runtime.CAMPAIGN_ID,
                    campaign_revision="V8R4",
                    outer_fold=outer_fold,
                    seed=seed,
                    row_count=1,
                    fields=list(runtime.OUTER_PREDICT_FIELDS),
                    exact_allowlist=True,
                    forbidden_fields_emitted=False,
                    reference_identity_protocol_quality_decoded=False,
                    legacy_index=legacy,
                    legacy_cache_manifest=legacy,
                    legacy_proposer_stack=legacy,
                    promotion_authorization=_authorization_binding(
                        promotion_authorization
                    ),
                    output=_artifact_binding(unit, predict),
                    global_cache_index_sha256="2" * 64,
                    object_arrays=False,
                    pickle=False,
                    commercial_or_confirmatory_claim_allowed=False,
                ),
            )
            checkpoint = _write_frozen(
                unit / "model_checkpoint.pt", f"checkpoint-{seed}".encode()
            )
            scaler = _write_frozen(
                unit / "model_scaler.json", f'{{"seed":{seed}}}\n'.encode()
            )
            scientific_signature = hashlib.sha256(
                f"signature-{outer_fold}-{seed}".encode()
            ).hexdigest()
            selection_binding = _authorization_binding(selection_lock)
            authorization_binding = _authorization_binding(
                promotion_authorization
            )
            capability = _write_json(
                unit / runtime.MODEL_SOURCE_CAPABILITY_FILENAME,
                _json_document(
                    schema_version=1,
                    classification=runtime.MODEL_SOURCE_CAPABILITY_CLASSIFICATION,
                    campaign_id=runtime.CAMPAIGN_ID,
                    campaign_revision="V8R4",
                    infrastructure_revision="V8R4A",
                    outer_fold=outer_fold,
                    seed=seed,
                    selected_variant="H1_factor",
                    source_kind="local_training",
                    scientific_signature_sha256=scientific_signature,
                    source_receipt={
                        "path": f"/opaque/source/{seed}/receipt.json",
                        "sha256": "3" * 64,
                        "bytes": 1,
                    },
                    source_checkpoint={
                        **_artifact_binding(unit, checkpoint),
                        "path": f"/opaque/source/{seed}/best.pt",
                    },
                    source_scaler={
                        **_artifact_binding(unit, scaler),
                        "path": f"/opaque/source/{seed}/scaler.json",
                    },
                    packed_checkpoint=_artifact_binding(unit, checkpoint),
                    packed_scaler=_artifact_binding(unit, scaler),
                    selection_lock=selection_binding,
                    promotion_authorization=authorization_binding,
                    source_deep_validated_before_copy=True,
                    source_paths_or_peer_outputs_authorized_in_child=False,
                    target_reference_quality_identity_protocol_present=False,
                    model_bytes_changed=False,
                    commercial_or_confirmatory_claim_allowed=False,
                ),
            )
            successor = _write_json(
                unit / runtime.MODEL_BOUND_PREDICTION_MANIFEST_FILENAME,
                _json_document(
                    schema_version=1,
                    classification=(
                        runtime.MODEL_BOUND_PREDICTION_PACK_CLASSIFICATION
                    ),
                    campaign_id=runtime.CAMPAIGN_ID,
                    campaign_revision="V8R4",
                    infrastructure_revision="V8R4A",
                    outer_fold=outer_fold,
                    seed=seed,
                    selected_variant="H1_factor",
                    row_count=1,
                    global_cache_index_sha256="2" * 64,
                    fields=list(runtime.OUTER_PREDICT_FIELDS),
                    exact_target_free_allowlist=True,
                    selection_lock=selection_binding,
                    promotion_authorization=authorization_binding,
                    base_target_free_manifest=_artifact_binding(unit, manifest),
                    artifacts={
                        "outer_predict_input": _artifact_binding(unit, predict),
                        "model_checkpoint": _artifact_binding(unit, checkpoint),
                        "model_scaler": _artifact_binding(unit, scaler),
                        "model_source_capability": _artifact_binding(
                            unit, capability
                        ),
                    },
                    exact_unit_file_inventory=sorted(
                        runtime.MODEL_BOUND_UNIT_FILES
                    ),
                    prediction_child_reads_model_only_from_this_pack=True,
                    source_paths_or_peer_outputs_authorized_in_child=False,
                    target_reference_quality_identity_protocol_present=False,
                    model_bytes_changed=False,
                    commercial_or_confirmatory_claim_allowed=False,
                ),
            )
            units.append(
                {
                    "outer_fold": outer_fold,
                    "seed": seed,
                    "relative_path": relative.as_posix(),
                    "scientific_signature_sha256": scientific_signature,
                    "row_count": 1,
                    "global_cache_index_sha256": "2" * 64,
                    "source_kind": "local_training",
                    "artifacts": {
                        "prediction_pack_manifest": _artifact_binding(root, manifest),
                        "model_bound_prediction_pack_manifest": _artifact_binding(
                            root, successor
                        ),
                        "outer_predict_input": _artifact_binding(root, predict),
                        "model_checkpoint": _artifact_binding(root, checkpoint),
                        "model_scaler": _artifact_binding(root, scaler),
                        "model_source_capability": _artifact_binding(
                            root, capability
                        ),
                    },
                }
            )
            continue
        proposer = _write_frozen(
            unit / "discovery_proposer_stack.npz", f"npz-{seed}".encode()
        )
        cache_files = {
            "feature_names": _write_frozen(
                unit / "discovery_cache/feature_names.json", b"[]\n"
            ),
            "metadata": _write_frozen(
                unit / "discovery_cache/metadata.csv", b"cache_index\n"
            ),
        }
        for role, name in (
            ("node_features", "node_features.npy"),
            ("candidate_bpm", "candidate_bpm.npy"),
            ("candidate_mask", "candidate_mask.npy"),
            ("joint_radar_mask", "joint_radar_mask.npy"),
            ("local_to_global_cache_index", "local_to_global_cache_index.npy"),
        ):
            cache_files[role] = _write_frozen(
                unit / "discovery_cache" / name, name.encode()
            )
        cache_values: dict[str, Any] = {
            "schema_version": 1,
            "classification": "adaptive_v3r1_v8r4_nonouter_training_cache",
            "campaign_id": runtime.CAMPAIGN_ID,
            "outer_fold": outer_fold,
            "seed": seed,
            "outer_test_rows_physically_present": False,
        }
        if phase == "promotion_training":
            assert promotion_authorization is not None
            cache_values = dict(
                schema_version=1,
                classification="adaptive_v3r1_v8r4_nonouter_training_validation_pack",
                campaign_id=runtime.CAMPAIGN_ID,
                campaign_revision="V8R4",
                format_version=1,
                complete=True,
                outer_fold=outer_fold,
                partition="outer_excluded_training_validation",
                source_combined_cache_open_authorized_by_consumer=False,
                outer_test_rows_physically_present=False,
                outer_prediction_pack_absent=True,
                inputs={},
                outputs={
                    role: {
                        "filename": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bytes": path.stat().st_size,
                    }
                    for role, path in cache_files.items()
                },
                promotion_scope="promotion_training_pack",
                promotion_authorization=_authorization_binding(
                    promotion_authorization
                ),
            )
        cache = _write_json(
            unit / "discovery_cache/manifest.json", _json_document(**cache_values)
        )
        partition_values: dict[str, Any] = {
            "schema_version": 1,
            "classification": "adaptive_v3r1_v8r4_sealed_nonouter_partition",
            "campaign_id": runtime.CAMPAIGN_ID,
            "campaign_revision": "V8R4",
            "outer_fold": outer_fold,
            "seed": seed,
            "outer_test_opened": False,
        }
        if phase == "promotion_training":
            assert promotion_authorization is not None
            partition_values = {
                "schema_version": 1,
                "classification": "adaptive_v3r1_v8r4_sealed_nonouter_partition",
                "campaign_id": runtime.CAMPAIGN_ID,
                "campaign_revision": "V8R4",
                "outer_fold": outer_fold,
                "seed": seed,
                "legacy_row_count": 1,
                "partition": {},
                "legacy_inputs": {},
                "outputs": {
                    "discovery_cache_manifest": _artifact_binding(unit, cache),
                    "discovery_proposer_stack": _artifact_binding(unit, proposer),
                    "discovery_local_to_global_map": _artifact_binding(
                        unit, cache_files["local_to_global_cache_index"]
                    ),
                },
                "integration_interface": {},
                "protected_outer_access": {
                    "forbidden_fields_emitted": False,
                    "outer_reference_decoded": False,
                    "outer_reference_validity_decoded": False,
                    "outer_identity_decoded": False,
                    "outer_protocol_decoded": False,
                    "outer_quality_decoded": False,
                },
                "preselection_prediction_boundary": {
                    "outer_prediction_pack_absent": True,
                    "outer_prediction_path_bound": False,
                    "outer_prediction_values_materialized": False,
                },
                "serialization": {
                    "object_arrays": False,
                    "pickle": False,
                    "outputs_mode": "0444",
                },
                "claim_boundary": {"outer_targets_opened": False},
                "promotion_scope": "promotion_training_pack",
                "promotion_authorization": _authorization_binding(
                    promotion_authorization
                ),
            }
        partition = _write_json(
            unit / "PARTITION_MANIFEST.json",
            _json_document(**partition_values),
        )
        units.append(
            {
                "outer_fold": outer_fold,
                "seed": seed,
                "relative_path": relative.as_posix(),
                "artifacts": {
                    "cache_manifest": _artifact_binding(root, cache),
                    "proposer_stack": _artifact_binding(root, proposer),
                    "partition_manifest": _artifact_binding(root, partition),
                },
            }
        )
    values: dict[str, Any] = {
        "schema_version": 1,
        "classification": (
            runtime.PREDICTION_PACK_INDEX_CLASSIFICATION
            if phase == "promotion_prediction"
            else runtime.PACK_INDEX_CLASSIFICATION
        ),
        "campaign_id": runtime.CAMPAIGN_ID,
        "campaign_revision": "V8R4",
        "outer_fold": outer_fold,
        "seeds": list(runtime.SEEDS),
        "unit_count": 3,
        "completed_units": 3,
        "status": "complete",
        "outer_test_opened": False,
        "combined_target_bearing_cache_consumer_access_authorized": False,
        "cross_outer_shard_mounted": False,
        "units": units,
    }
    if phase == "promotion_prediction":
        assert promotion_authorization is not None and selection_lock is not None
        model_seal = _write_json(
            root / runtime.MODEL_SOURCE_SHARD_SEAL_FILENAME,
            _json_document(
                schema_version=1,
                classification="adaptive_v3r1_v8r4a_model_source_shard_seal",
                campaign_id=runtime.CAMPAIGN_ID,
                campaign_revision="V8R4",
                infrastructure_revision="V8R4A",
                outer_fold=outer_fold,
                seeds=list(runtime.SEEDS),
                selected_variant="H1_factor",
                unit_count=3,
                exact_three_seed_cover=True,
                selection_lock=_authorization_binding(selection_lock),
                promotion_authorization=_authorization_binding(
                    promotion_authorization
                ),
                units=[
                    {
                        "outer_fold": row["outer_fold"],
                        "seed": row["seed"],
                        "source_kind": row["source_kind"],
                        "scientific_signature_sha256": row[
                            "scientific_signature_sha256"
                        ],
                        "row_count": row["row_count"],
                        "global_cache_index_sha256": row[
                            "global_cache_index_sha256"
                        ],
                        "model_bound_prediction_pack_manifest": row[
                            "artifacts"
                        ]["model_bound_prediction_pack_manifest"],
                        "model_checkpoint": row["artifacts"]["model_checkpoint"],
                        "model_scaler": row["artifacts"]["model_scaler"],
                        "model_source_capability": row["artifacts"][
                            "model_source_capability"
                        ],
                    }
                    for row in units
                ],
                target_or_prediction_values_present=False,
                source_paths_or_peer_outputs_authorized_in_child=False,
                commercial_or_confirmatory_claim_allowed=False,
            ),
        )
        values.update(
            {
                "infrastructure_revision": "V8R4A",
                "selected_variant": "H1_factor",
                "physical_target_free_input_and_model_packs": True,
                "source_paths_or_peer_outputs_authorized_in_child": False,
                "promotion_authorization": _authorization_binding(
                    promotion_authorization
                ),
                "model_source_shard_seal": _artifact_binding(root, model_seal),
            }
        )
    else:
        values["physical_nonouter_training_packs"] = True
        values["outer_prediction_packs_absent"] = True
        if phase == "promotion_training":
            assert promotion_authorization is not None
            values.update(
                {
                    "promotion_scope": "promotion_training_pack",
                    "promotion_authorization": _authorization_binding(
                        promotion_authorization
                    ),
                }
            )
    index = _json_document(**values)
    index_name = (
        runtime.PREDICTION_PACK_INDEX_FILENAME
        if phase == "promotion_prediction"
        else "V8R4_NONOUTER_TRAINING_INDEX.json"
    )
    index_path = _write_json(root / index_name, index)
    for directory in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    root.chmod(0o555)
    return root, index_path


def _source_snapshot(
    project: Path,
    *,
    entry: Path,
    launcher: Path,
    migration_module: Path,
    gpu_admission_wrapper: Path,
    gpu_budget_module: Path,
    package_initializer: Path,
    implementation_receipt: Path,
    campaign_contract: Path,
) -> dict[str, Any]:
    source = _write_frozen(project / "src/snn_rr/example.py", b"VALUE = 1\n")
    test = _write_frozen(project / "tests/test_example.py", b"def test_x(): pass\n")
    config = _write_frozen(project / "configs/harmonic_factor_router_v3.yaml", b"x: 1\n")
    pyproject = _write_frozen(project / "pyproject.toml", b"[project]\nname='x'\n")
    return _json_document(
        schema_version=1,
        classification="adaptive_v3r1_v8r4a_source_snapshot",
        campaign_id=runtime.CAMPAIGN_ID,
        authorization_generation="CONTEXT1",
        scientific_campaign_revision="V8R4",
        infrastructure_revision="V8R4A",
        implementation_files=[
            _binding_row(project, entry),
            _binding_row(project, launcher),
            _binding_row(project, migration_module),
            _binding_row(project, gpu_admission_wrapper),
            _binding_row(project, gpu_budget_module),
            _binding_row(project, package_initializer),
            _binding_row(project, source),
            _binding_row(project, test),
            _binding_row(project, config),
            _binding_row(project, campaign_contract),
        ],
        entry_evidence=[],
        read_only_ancestry=[],
        implementation_test_receipt=_binding_row(project, implementation_receipt),
        environment={
            "pyproject": _binding_row(project, pyproject),
            "python_executable_resolved": str(REAL_INTERPRETER.resolve()),
        },
    )


MIGRATION_VALIDATOR_STUB = r'''
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import stat

ROOT = Path("artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/gpu_state_v8r4a")
RECEIPT = Path("artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/GPU_STATE_MIGRATION_RECEIPT_V8R4A.json")
DIRS = {
    "admission": ("gpu_admission_v7.lock",),
    "execution": ("gpu_execution_ledger_v7.jsonl", "gpu_execution_ledger_v7.jsonl.lock"),
    "usage": ("campaign_gpu_usage_chain_v6.jsonl", "campaign_gpu_usage_chain_v6.jsonl.lock"),
}
FILES = {
    "admission_lock": ("admission", "gpu_admission_v7.lock"),
    "execution_ledger": ("execution", "gpu_execution_ledger_v7.jsonl"),
    "execution_ledger_lock": ("execution", "gpu_execution_ledger_v7.jsonl.lock"),
    "usage_ledger": ("usage", "campaign_gpu_usage_chain_v6.jsonl"),
    "usage_ledger_lock": ("usage", "campaign_gpu_usage_chain_v6.jsonl.lock"),
}

def _file(root, path):
    raw = path.read_bytes(); s = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(s.st_mode): raise RuntimeError("state file is not regular")
    return {"bytes":len(raw), "mode":f"{stat.S_IMODE(s.st_mode):04o}",
            "nlink":s.st_nlink, "path":path.relative_to(root).as_posix(),
            "sha256":hashlib.sha256(raw).hexdigest(), "st_dev":s.st_dev,
            "st_ino":s.st_ino}

def _directory(root, path, expected):
    s = path.stat(follow_symlinks=False)
    entries = sorted(os.listdir(path))
    if entries != sorted(expected): raise RuntimeError("state directory inventory drifted")
    if stat.S_IMODE(s.st_mode) != 0o700: raise RuntimeError("state directory mode drifted")
    return {"exact_entries":entries, "mode":"0700",
            "path":path.relative_to(root).as_posix(), "st_dev":s.st_dev,
            "st_ino":s.st_ino}

def validate_migrated_state(project_root, receipt_path, *, require_closed=True):
    project_root = Path(project_root).resolve(strict=True)
    expected_receipt = project_root / RECEIPT
    if Path(receipt_path) != expected_receipt: raise RuntimeError("receipt path drifted")
    receipt = json.loads(expected_receipt.read_text())
    receipt_binding = _file(project_root, expected_receipt)
    receipt_binding["content_sha256"] = receipt["content_sha256"]
    state_root = project_root / ROOT
    directories = {"root": _directory(project_root, state_root, DIRS)}
    directories.update({role:_directory(project_root, state_root/role, names)
                        for role,names in DIRS.items()})
    paths = {role:state_root/directory/name
             for role,(directory,name) in FILES.items()}
    paths.update({f"{role}_directory":state_root/role for role in DIRS})
    files = {role:_file(project_root,path) for role,path in paths.items()
             if role in FILES}
    usage_raw = paths["usage_ledger"].read_bytes()
    execution_raw = paths["execution_ledger"].read_bytes()
    usage_lines = usage_raw.splitlines()
    execution_lines = execution_raw.splitlines()
    try:
        usage_rows = [json.loads(line) for line in usage_lines]
    except (UnicodeDecodeError, json.JSONDecodeError):
        usage_rows = []
    usage_open = {
        row.get("lifecycle_id") for row in usage_rows
        if isinstance(row, dict) and row.get("event") == "reservation"
    }
    usage_open -= {
        row.get("lifecycle_id") for row in usage_rows
        if isinstance(row, dict) and row.get("event") in {"terminal", "reconciled_terminal"}
    }
    try:
        execution_rows = [json.loads(line) for line in execution_lines]
    except (UnicodeDecodeError, json.JSONDecodeError):
        execution_rows = []
    execution_open = {
        row.get("lifecycle_id", row.get("job_id")) for row in execution_rows
        if isinstance(row, dict) and row.get("event") == "start"
    }
    execution_open -= {
        row.get("lifecycle_id", row.get("job_id")) for row in execution_rows
        if isinstance(row, dict) and row.get("event") in {"end", "wrapper_exception"}
    }
    settled_usage_ns = sum(
        int(row.get("charged_usage_ns", 0))
        for row in usage_rows
        if isinstance(row, dict)
        and row.get("event") in {"terminal", "reconciled_terminal"}
    ) + sum(
        int(float(row.get("elapsed_seconds", 0)) * 1_000_000_000)
        for row in usage_rows
        if isinstance(row, dict)
        and row.get("event") == "forced_termination_usage_carry_forward"
    )
    usage = {"open_reservation_count":len(usage_open) + sum(line == b"OPEN_USAGE" for line in usage_lines),
             "record_count":len(usage_lines),
             "settled_usage_ns":settled_usage_ns,
             "tail_record_sha256": usage_rows[-1].get("record_sha256")
             if usage_rows else None}
    execution = {"open_start_count":len(execution_open) + sum(line == b"OPEN_EXECUTION" for line in execution_lines),
                 "record_count":len(execution_lines),
                 "last_line_sha256": hashlib.sha256(execution_raw.splitlines()[-1]).hexdigest()
                 if execution_raw.splitlines() else None}
    return SimpleNamespace(receipt_path=expected_receipt, receipt=receipt,
        receipt_binding=receipt_binding, canonical_paths=paths,
        directory_bindings=directories, current_file_bindings=files,
        usage_state=usage, execution_state=execution)
'''


def _governance_row(project: Path, path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    status = path.stat(follow_symlinks=False)
    return {
        "path": path.relative_to(project).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "mode": f"{stat.S_IMODE(status.st_mode):04o}",
        "nlink": status.st_nlink,
        "st_dev": status.st_dev,
        "st_ino": status.st_ino,
    }


def _authority_legacy_row(project: Path, path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    document = json.loads(raw)
    return {
        "path": path.relative_to(project).as_posix(),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "content_sha256": document["content_sha256"],
    }


def _authority_sha_row(
    project: Path, path: Path, *, content_hash: bool = True
) -> dict[str, Any]:
    raw = path.read_bytes()
    document = json.loads(raw)
    row: dict[str, Any] = {
        "path": path.relative_to(project).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "mode": "0444",
    }
    if content_hash:
        row["content_sha256"] = document["content_sha256"]
    return row


def _make_project(tmp_path: Path, *, phase: str = "efficiency_benchmark") -> dict[str, Any]:
    project = tmp_path / "project"
    for directory in ("scripts", "src", "tests", "configs", "governance"):
        (project / directory).mkdir(parents=True, exist_ok=True)
    entry_name = next(iter(runtime.ENTRY_SCRIPT_BY_PHASE[phase]))
    outer_fold: int | None = (
        None
        if phase in {"discovery_aggregation", "promotion_aggregation"}
        else 3
        if phase in {"efficiency_benchmark", "discovery"}
        else 0
    )
    shard_output = project / runtime._canonical_output_relative(
        phase=phase, outer_fold=outer_fold, entry_name=entry_name
    )
    marker = shard_output / "marker.json"
    entry = _write_frozen(
        project / "scripts" / entry_name,
        (
            "import json,os,pathlib\n"
            f"p=pathlib.Path({str(marker)!r});p.write_text(json.dumps({{'env':dict(os.environ)}}));p.chmod(0o444)\n"
        ).encode(),
    )
    launcher = project / "scripts" / SCRIPT.name
    shutil.copyfile(SCRIPT, launcher)
    launcher.chmod(0o444)
    gpu_admission_wrapper = project / runtime.GPU_ADMISSION_WRAPPER_RELATIVE_PATH
    gpu_admission_wrapper.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        RUNTIME_DEPENDENCY_SOURCE / runtime.GPU_ADMISSION_WRAPPER_RELATIVE_PATH,
        gpu_admission_wrapper,
    )
    gpu_admission_wrapper.chmod(0o444)
    gpu_budget_module = project / runtime.GPU_BUDGET_MODULE_RELATIVE_PATH
    gpu_budget_module.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        RUNTIME_DEPENDENCY_SOURCE / runtime.GPU_BUDGET_MODULE_RELATIVE_PATH,
        gpu_budget_module,
    )
    gpu_budget_module.chmod(0o444)
    package_initializer = project / runtime.SNN_RR_PACKAGE_INIT_RELATIVE_PATH
    shutil.copyfile(
        RUNTIME_DEPENDENCY_SOURCE / runtime.SNN_RR_PACKAGE_INIT_RELATIVE_PATH,
        package_initializer,
    )
    package_initializer.chmod(0o444)
    migration_module = _write_frozen(
        project / runtime.MIGRATION_MODULE_RELATIVE_PATH,
        MIGRATION_VALIDATOR_STUB.encode(),
    )
    unbound_project_files = [
        _write_frozen(project / "scripts/unbound.py", b"SECRET = True\n"),
        _write_frozen(project / "src/snn_rr/unbound.py", b"SECRET = True\n"),
        _write_frozen(project / "configs/unbound.yaml", b"secret: true\n"),
    ]

    state_root = project / runtime.GPU_STATE_ROOT_RELATIVE
    for role, entries in runtime.GPU_STATE_EXACT_ENTRIES.items():
        directory = state_root / role
        directory.mkdir(parents=True, exist_ok=True)
        for name in entries:
            path = directory / name
            if name == "campaign_gpu_usage_chain_v6.jsonl":
                path.write_bytes(FROZEN_CONTEXT1_LEDGER_RAW["usage_ledger"])
            elif name == "gpu_execution_ledger_v7.jsonl":
                path.write_bytes(FROZEN_CONTEXT1_LEDGER_RAW["execution_ledger"])
            else:
                path.write_bytes(b"")
            path.chmod(0o644)
        directory.chmod(0o700)
    state_root.chmod(0o700)

    campaign = (
        project
        / "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1"
    )
    historical_diagnostic = _write_exact_governance_json(
        campaign / "diagnostics/v3r1_v8r3_outer_capability_and_identity_dtype_failure_v8r4.json",
        "failure_diagnostic",
        _json_document(
            schema_version=1,
            classification="posttrain_preselection_v8r4_outer_capability_and_pickle_free_npz_failure_diagnostic",
            campaign_id=runtime.CAMPAIGN_ID,
        ),
    )
    parent_authority = _write_exact_governance_json(
        campaign / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4.json",
        "correction_authorization",
        _json_document(
            schema_version=1,
            classification="posttrain_preselection_adaptive_v3r1_v8r4_physical_target_capability_and_pickle_free_output_correction_authorization",
            campaign_id=runtime.CAMPAIGN_ID,
            diagnostic=_authority_legacy_row(project, historical_diagnostic),
        ),
    )
    infrastructure_diagnostic = _write_exact_governance_json(
        campaign / "diagnostics/v3r1_v8r4_exact_file_atomic_replace_failure_v8r4a.json",
        "infrastructure_failure_diagnostic",
        _json_document(
            schema_version=1,
            classification="pretrain_v8r4a_dedicated_gpu_state_directory_migration_diagnostic",
            campaign_id=runtime.CAMPAIGN_ID,
            scientific_campaign_revision="V8R4",
            infrastructure_revision="V8R4A",
        ),
    )
    diagnostic_raw = infrastructure_diagnostic.read_bytes()
    diagnostic_document = json.loads(diagnostic_raw)
    infrastructure_authority = _write_exact_governance_json(
        campaign / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A.json",
        "infrastructure_correction_authorization",
        _json_document(
            schema_version=1,
            classification="pretrain_adaptive_v3r1_v8r4a_dedicated_gpu_state_directory_migration_correction_authorization",
            campaign_id=runtime.CAMPAIGN_ID,
            scientific_campaign_revision="V8R4",
            infrastructure_revision="V8R4A",
            diagnostic={
                "path": infrastructure_diagnostic.relative_to(project).as_posix(),
                "file_sha256": hashlib.sha256(diagnostic_raw).hexdigest(),
                "bytes": len(diagnostic_raw),
                "content_sha256": diagnostic_document["content_sha256"],
            },
        ),
    )
    source_closure_diagnostic = _write_exact_governance_json(
        campaign
        / "diagnostics/v3r1_v8r4a_pretrain_validator_and_source_closure_failure.json",
        "source_closure_failure_diagnostic",
        _json_document(
            schema_version=1,
            classification="pretrain_adaptive_v3r1_v8r4a_validator_deadlock_and_executable_source_closure_failure_diagnostic",
            campaign_id=runtime.CAMPAIGN_ID,
            scientific_campaign_revision="V8R4",
            infrastructure_revision="V8R4A",
        ),
    )
    source_closure_authority = _write_exact_governance_json(
        campaign / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_SOURCE_CLOSURE.json",
        "source_closure_correction_authorization",
        _json_document(
            schema_version=1,
            classification="pretrain_adaptive_v3r1_v8r4a_validator_and_executable_source_closure_correction_addendum",
            campaign_id=runtime.CAMPAIGN_ID,
            scientific_campaign_revision="V8R4",
            infrastructure_revision="V8R4A",
            authority_basis={
                "diagnostic": _authority_legacy_row(
                    project, source_closure_diagnostic
                ),
                "parent_correction_authorization": _authority_legacy_row(
                    project, infrastructure_authority
                ),
            },
        ),
    )
    source_dependency_authority = _write_exact_governance_json(
        campaign
        / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_SOURCE_CLOSURE_DEPENDENCIES.json",
        "source_closure_dependency_authorization",
        _json_document(
            schema_version=1,
            classification="pretrain_adaptive_v3r1_v8r4a_executable_source_transitive_dependency_closure_addendum",
            campaign_id=runtime.CAMPAIGN_ID,
            scientific_campaign_revision="V8R4",
            infrastructure_revision="V8R4A",
            authority_basis={
                "diagnostic": _authority_legacy_row(
                    project, source_closure_diagnostic
                ),
                "parent_source_closure_addendum": _authority_legacy_row(
                    project, source_closure_authority
                ),
            },
        ),
    )
    kill_safe_diagnostic = _write_exact_governance_json(
        campaign
        / "diagnostics/v3r1_v8r4a_kill_safe_output_and_shared_ledger_resume_failure.json",
        "kill_safe_failure_diagnostic",
        _json_document(
            schema_version=1,
            classification="pretrain_adaptive_v3r1_v8r4a_kill_safe_output_and_shared_ledger_resume_failure_diagnostic",
            campaign_id=runtime.CAMPAIGN_ID,
            scientific_campaign_revision="V8R4",
            infrastructure_revision="V8R4A",
        ),
    )
    kill_safe_authority = _write_exact_governance_json(
        campaign / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_KILL_SAFE_RESUME.json",
        "kill_safe_correction_authorization",
        _json_document(
            schema_version=1,
            classification="pretrain_adaptive_v3r1_v8r4a_kill_safe_atomic_output_and_append_only_completion_correction_authorization",
            campaign_id=runtime.CAMPAIGN_ID,
            scientific_campaign_revision="V8R4",
            infrastructure_revision="V8R4A",
            authority_basis={
                "diagnostic": _authority_legacy_row(project, kill_safe_diagnostic),
                "parent_source_closure_addendum": _authority_legacy_row(
                    project, source_closure_authority
                ),
                "transitive_dependency_addendum": _authority_legacy_row(
                    project, source_dependency_authority
                ),
            },
        ),
    )
    open_lifecycle_diagnostic = _write_exact_governance_json(
        campaign
        / "diagnostics/v3r1_v8r4a_open_gpu_lifecycle_kill_recovery_failure.json",
        "open_lifecycle_recovery_failure_diagnostic",
        _json_document(
            schema_version=1,
            classification="pretrain_adaptive_v3r1_v8r4a_open_gpu_lifecycle_kill_recovery_failure_diagnostic",
            campaign_id=runtime.CAMPAIGN_ID,
            scientific_campaign_revision="V8R4",
            infrastructure_revision="V8R4A",
        ),
    )
    open_lifecycle_authority = _write_exact_governance_json(
        campaign
        / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_OPEN_LIFECYCLE_RECOVERY.json",
        "open_lifecycle_recovery_correction_authorization",
        _json_document(
            schema_version=1,
            classification="pretrain_adaptive_v3r1_v8r4a_open_gpu_lifecycle_kill_recovery_correction_addendum",
            campaign_id=runtime.CAMPAIGN_ID,
            scientific_campaign_revision="V8R4",
            infrastructure_revision="V8R4A",
            authority_basis={
                "authorization_limited_to_open_lifecycle_kill_recovery": True,
                "diagnostic": _authority_legacy_row(
                    project, open_lifecycle_diagnostic
                ),
                "parent_kill_safe_addendum": _authority_legacy_row(
                    project, kill_safe_authority
                ),
                "parent_source_closure_addendum": _authority_legacy_row(
                    project, source_closure_authority
                ),
                "user_goal_scope": "synthetic CPU-only recovery fixture",
            },
        ),
    )
    execution_closure_diagnostic = _write_exact_governance_json(
        campaign / "diagnostics/v3r1_v8r4a_terminal_execution_closure_failure.json",
        "execution_closure_failure_diagnostic",
        _json_document(
            schema_version=1,
            classification="pretrain_adaptive_v3r1_v8r4a_terminal_execution_closure_failure_diagnostic",
            campaign_id=runtime.CAMPAIGN_ID,
            scientific_campaign_revision="V8R4",
            infrastructure_revision="V8R4A",
        ),
    )
    execution_closure_authority = _write_exact_governance_json(
        campaign
        / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_EXECUTION_CLOSURE.json",
        "execution_closure_correction_authorization",
        _json_document(
            schema_version=1,
            classification="pretrain_adaptive_v3r1_v8r4a_kill_safe_capability_and_promotion_execution_closure_correction_addendum",
            campaign_id=runtime.CAMPAIGN_ID,
            scientific_campaign_revision="V8R4",
            infrastructure_revision="V8R4A",
            authority_basis={
                "authorization_limited_to_terminal_execution_closure": True,
                "diagnostic": _authority_legacy_row(
                    project, execution_closure_diagnostic
                ),
                "historical_benchmark_prefix": {
                    "active_output_root": (
                        runtime.LEGACY_EXECUTION_CLOSURE_BENCHMARK_OUTPUT_RELATIVE.as_posix()
                    ),
                    "entries": [
                        {
                            "path": path,
                            "file_sha256": digest,
                            "bytes": size,
                            "mode": mode,
                            "role": role,
                        }
                        for path, digest, size, mode, role in (
                            runtime.EXECUTION_CLOSURE_HISTORICAL_ROWS
                        )
                    ],
                    "historical_root_mounted_or_mutated": False,
                    "known_v8r3_mode_0644_is_read_only_quarantined_evidence": True,
                },
                "parent_kill_safe_addendum": _authority_legacy_row(
                    project, kill_safe_authority
                ),
                "parent_open_lifecycle_recovery_addendum": _authority_legacy_row(
                    project, open_lifecycle_authority
                ),
                "parent_source_closure_addendum": _authority_legacy_row(
                    project, source_closure_authority
                ),
                "user_goal_scope": "synthetic CPU-only execution-closure fixture",
            },
        ),
    )
    migration_source_succession_diagnostic = _write_exact_governance_json(
        campaign
        / "diagnostics/v3r1_v8r4a_migrated_state_source_succession_failure.json",
        "migration_source_succession_failure_diagnostic",
        _json_document(
            schema_version=1,
            classification="pretrain_adaptive_v3r1_v8r4a_migrated_state_source_succession_failure_diagnostic",
            campaign_id=runtime.CAMPAIGN_ID,
            scientific_campaign_revision="V8R4",
            status="diagnosed_not_authorized_by_diagnostic",
        ),
    )
    migration_receipt = _write_exact_governance_json(
        campaign / "GPU_STATE_MIGRATION_RECEIPT_V8R4A.json",
        "gpu_state_migration_receipt",
        _json_document(
            schema_version=1,
            classification="adaptive_v3r1_v8r4a_gpu_state_migration_receipt",
            campaign_id=runtime.CAMPAIGN_ID,
            scientific_campaign_revision="V8R4",
            infrastructure_revision="V8R4A",
            production_runtime_authorized=False,
        ),
    )
    migration_source_succession_authority = _write_exact_governance_json(
        campaign
        / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_MIGRATION_SOURCE_SUCCESSION.json",
        "migration_source_succession_correction_authorization",
        _json_document(
            schema_version=1,
            classification="pretrain_adaptive_v3r1_v8r4a_migrated_state_source_succession_correction_addendum",
            campaign_id=runtime.CAMPAIGN_ID,
            scientific_campaign_revision="V8R4",
            infrastructure_revision="V8R4A",
            authority_basis={
                "diagnostic": _authority_legacy_row(
                    project, migration_source_succession_diagnostic
                ),
                "execution_closure_authority": _authority_legacy_row(
                    project, execution_closure_authority
                ),
                "immutable_migration_receipt": _authority_legacy_row(
                    project, migration_receipt
                ),
                "original_migration_authority": _authority_legacy_row(
                    project, infrastructure_authority
                ),
                "user_goal_scope": "synthetic migrated-state source succession",
            },
        ),
    )
    fd_closure_diagnostic = _write_exact_governance_json(
        campaign
        / "diagnostics/v3r1_v8r4a_outer_guard_urandom_descriptor_failure.json",
        "fd_closure_failure_diagnostic",
        _json_document(
            schema_version=1,
            classification="pretrain_adaptive_v3r1_v8r4a_outer_guard_urandom_descriptor_failure_diagnostic",
            campaign_id=runtime.CAMPAIGN_ID,
            scientific_campaign_revision="V8R4",
            infrastructure_revision="V8R4A",
            status="diagnosed_not_authorized_by_diagnostic",
            failed_attempt={
                "first_phase": "efficiency_benchmark",
                "outer_fold": 3,
                "target_runtime_return_code": 73,
                "coordinator_return_code": 79,
                "gpu_child_launched": False,
            },
            reproduction={
                "descriptor_3_target": "/dev/urandom",
                "descriptor_3_fd_cloexec": True,
                "arbitrary_unexpected_descriptor_rejection_must_remain": True,
            },
            superseded_pretrain_chain={
                "implementation_test_receipt": dict(
                    runtime.FD_CLOSURE_PARENT_BINDINGS[
                        "parent_implementation_test_receipt"
                    ]
                ),
                "source_snapshot": dict(
                    runtime.FD_CLOSURE_PARENT_BINDINGS[
                        "parent_source_snapshot"
                    ]
                ),
                "pretrain_authorization": dict(
                    runtime.FD_CLOSURE_PARENT_BINDINGS[
                        "parent_pretrain_authorization"
                    ]
                ),
                "preserved_as_immutable_audit_evidence": True,
                "may_authorize_retry_without_successor_chain": False,
            },
        ),
    )
    fd_closure_authority = _write_exact_governance_json(
        campaign
        / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_FD_CLOSURE.json",
        "fd_closure_correction_authorization",
        _json_document(
            schema_version=1,
            classification="pretrain_adaptive_v3r1_v8r4a_outer_guard_urandom_descriptor_closure_correction_addendum",
            campaign_id=runtime.CAMPAIGN_ID,
            scientific_campaign_revision="V8R4",
            infrastructure_revision="V8R4A",
            authority_basis={
                "diagnostic": _authority_legacy_row(
                    project, fd_closure_diagnostic
                ),
                "parent_implementation_test_receipt": dict(
                    runtime.FD_CLOSURE_PARENT_BINDINGS[
                        "parent_implementation_test_receipt"
                    ]
                ),
                "parent_source_snapshot": dict(
                    runtime.FD_CLOSURE_PARENT_BINDINGS[
                        "parent_source_snapshot"
                    ]
                ),
                "parent_pretrain_authorization": dict(
                    runtime.FD_CLOSURE_PARENT_BINDINGS[
                        "parent_pretrain_authorization"
                    ]
                ),
                "parent_execution_closure_authority": _authority_legacy_row(
                    project, execution_closure_authority
                ),
                "user_goal_scope": "synthetic CPU-only FD-closure fixture",
            },
            authorized_modifications=[
                {
                    "path": path,
                    "before_sha256": digest,
                    "allowed_change": "synthetic exact authorized change",
                }
                for path, digest in runtime.FD_CLOSURE_AUTHORIZED_BEFORE.items()
            ],
            required_reauthorization={
                "new_test_receipt": "IMPLEMENTATION_TEST_RECEIPT_V8R4A_FD1.json",
                "new_source_snapshot": "V3R1_SOURCE_SNAPSHOT_V8R4A_FD1.json",
                "new_pretrain_authorization": "PRETRAIN_AUTHORIZATION_V8R4A_FD1.json",
                "all_fixed_tests_pass": True,
                "fresh_interpreter_subprocess_regression_passes": True,
                "gpu_retry_only_after_successor_pretrain_validation": True,
            },
            claim_boundary={
                "correction_is_infrastructure_only": True,
                "gpu_execution_authorized_by_this_document": False,
                "successor_pretrain_authorization_required": True,
                "commercial_claim_authorized": False,
            },
        ),
    )
    canary_boundary_diagnostic = _write_exact_governance_json(
        campaign
        / "diagnostics/v3r1_v8r4a_denied_canary_prefix_collision_failure.json",
        "canary_boundary_failure_diagnostic",
        _json_document(
            schema_version=1,
            classification="pretrain_adaptive_v3r1_v8r4a_denied_canary_prefix_collision_failure_diagnostic",
            campaign_id=runtime.CAMPAIGN_ID,
            scientific_campaign_revision="V8R4",
            infrastructure_revision="V8R4A",
            status="diagnosed_not_authorized_by_diagnostic",
            failed_attempt={
                "first_phase": "efficiency_benchmark",
                "outer_fold": 3,
                "target_runtime_return_code": 73,
                "coordinator_return_code": 79,
                "stderr": "child command names a denied capability",
                "capability_receipt_created": False,
                "completion_receipt_created": False,
                "gpu_child_launched": False,
                "gpu_usage_ledger_mutated": False,
                "gpu_execution_ledger_mutated": False,
            },
            root_cause={
                "raw_substring_relation": True,
                "path_component_ancestor_relation": False,
            },
            reproduction={
                "python_substring_result": True,
                "lexical_relative_to_denied_result": False,
                "component_aware_mount_boundary_validation_passed": True,
                "exact_denied_path_must_fail": True,
                "denied_descendant_must_fail": True,
                "path_distinct_prefix_siblings_must_pass": True,
                "embedded_absolute_option_path_must_not_bypass_validation": True,
                "path_traversal_must_fail_before_normalization": True,
            },
            superseded_pretrain_chain={
                "implementation_test_receipt": dict(
                    runtime.CANARY_BOUNDARY_PARENT_BINDINGS[
                        "parent_implementation_test_receipt"
                    ]
                ),
                "source_snapshot": dict(
                    runtime.CANARY_BOUNDARY_PARENT_BINDINGS[
                        "parent_source_snapshot"
                    ]
                ),
                "pretrain_authorization": dict(
                    runtime.CANARY_BOUNDARY_PARENT_BINDINGS[
                        "parent_pretrain_authorization"
                    ]
                ),
                "preserved_as_immutable_audit_evidence": True,
                "may_authorize_retry_without_successor_chain": False,
            },
            required_correction={
                "replace_raw_substring_with_lexical_component_boundary_check": True,
                "normalize_only_absolute_path_tokens_without_filesystem_resolution": True,
                "reject_exact_denied_paths_and_descendants": True,
                "allow_path_distinct_sibling_prefixes": True,
                "reject_traversal_and_embedded_absolute_option_paths": True,
                "retain_component_aware_mount_validation_and_denied_canary_probe": True,
                "bind_diagnostic_and_correction_in_runtime_governance": True,
                "new_test_receipt": runtime.CANARY_BOUNDARY_ACTIVE_FILENAMES[
                    "implementation_test_receipt"
                ],
                "new_source_snapshot": runtime.CANARY_BOUNDARY_ACTIVE_FILENAMES[
                    "source_snapshot"
                ],
                "new_pretrain_authorization": runtime.CANARY_BOUNDARY_ACTIVE_FILENAMES[
                    "active_authorization"
                ],
                "full_reauthorization_before_gpu_retry": True,
            },
            claim_boundary={
                "adaptive_retrospective_only": True,
                "outer_test_features_or_targets_opened": False,
                "accuracy_metric_computed": False,
                "gpu_accessed": False,
                "scientific_configuration_change_authorized": False,
                "commercial_claim_authorized": False,
            },
        ),
    )
    canary_boundary_authority = _write_exact_governance_json(
        campaign
        / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_CANARY_BOUNDARY.json",
        "canary_boundary_correction_authorization",
        _json_document(
            schema_version=1,
            classification="pretrain_adaptive_v3r1_v8r4a_denied_canary_component_boundary_correction_addendum",
            campaign_id=runtime.CAMPAIGN_ID,
            scientific_campaign_revision="V8R4",
            infrastructure_revision="V8R4A",
            authority_basis={
                "diagnostic": _authority_legacy_row(
                    project, canary_boundary_diagnostic
                ),
                "parent_fd_closure_authority": _authority_legacy_row(
                    project, fd_closure_authority
                ),
                "parent_implementation_test_receipt": dict(
                    runtime.CANARY_BOUNDARY_PARENT_BINDINGS[
                        "parent_implementation_test_receipt"
                    ]
                ),
                "parent_source_snapshot": dict(
                    runtime.CANARY_BOUNDARY_PARENT_BINDINGS[
                        "parent_source_snapshot"
                    ]
                ),
                "parent_pretrain_authorization": dict(
                    runtime.CANARY_BOUNDARY_PARENT_BINDINGS[
                        "parent_pretrain_authorization"
                    ]
                ),
                "user_goal_scope": "synthetic CPU-only canary-boundary fixture",
            },
            authorized_modifications=[
                {
                    "path": path,
                    "before_sha256": digest,
                    "allowed_change": "synthetic exact authorized change",
                }
                for path, digest in runtime.CANARY_BOUNDARY_AUTHORIZED_BEFORE.items()
            ],
            mandatory_invariants={
                "scientific_campaign_revision_unchanged": "V8R4",
                "infrastructure_revision_unchanged": "V8R4A",
                "variants_unchanged": ["H0_no_factor", "H1_factor", "H2_full"],
                "seeds_unchanged": list(runtime.SEEDS),
                "discovery_outer_folds_unchanged": [3, 4],
                **{
                    field: True
                    for field in (
                        "target_and_outer_reference_sealing_unchanged",
                        "gpu_budget_and_append_only_ledgers_unchanged",
                        "single_gpu_owner_and_kill_safe_lifecycle_unchanged",
                        "denied_canary_paths_unchanged",
                        "exact_denied_paths_and_descendants_fail_closed",
                        "path_distinct_sibling_prefixes_are_not_capabilities",
                        "embedded_absolute_option_paths_fail_closed",
                        "path_traversal_fails_before_normalization",
                        "component_aware_mount_validation_unchanged",
                        "sandbox_denied_canary_probe_unchanged",
                        "parent_fd1_chain_preserved_immutable",
                        "empty_failed_attempt_directories_may_be_reused_only_under_existing_exact_resume_rules",
                    )
                },
            },
            forbidden_changes={
                field: True
                for field in (
                    "model_architecture_or_loss",
                    "hyperparameters_or_epoch_counts",
                    "data_rows_splits_or_pack_bytes",
                    "seed_variant_or_fold_matrix",
                    "metric_or_selection_thresholds",
                    "outer_target_or_reference_access",
                    "renaming_or_weakening_denied_canaries_to_avoid_the_collision",
                    "removing_component_aware_mount_validation",
                    "removing_sandbox_denied_canary_probes",
                    "allowing_exact_or_descendant_denied_paths",
                    "filesystem_resolution_or_existence_dependency_for_absent_canaries",
                    "ledger_reset_truncation_or_rewrite",
                    "mutation_or_replacement_of_parent_evidence",
                    "commercial_or_confirmatory_claim",
                )
            },
            required_reauthorization=dict(
                runtime.CANARY_BOUNDARY_REQUIRED_REAUTHORIZATION
            ),
            claim_boundary={
                "adaptive_retrospective_only": True,
                "correction_is_infrastructure_only": True,
                "outer_test_features_or_targets_opened": False,
                "accuracy_metric_used": False,
                "gpu_execution_authorized_by_this_document": False,
                "successor_pretrain_authorization_required": True,
                "commercial_claim_authorized": False,
            },
        ),
    )
    contract = _write_exact_governance_json(
        project / runtime.CAMPAIGN_CONTRACT_RELATIVE_PATH,
        "campaign_contract",
        _json_document(
            schema_version=1,
            classification="synthetic-campaign-contract",
            campaign_id=runtime.CAMPAIGN_ID,
        ),
    )
    frozen_contract_diagnostic = _write_exact_governance_json(
        campaign
        / "diagnostics/v3r1_v8r4a_frozen_contract_encoding_false_rejection_failure.json",
        "frozen_contract_encoding_failure_diagnostic",
        _json_document(
            schema_version=1,
            classification=(
                "pretrain_adaptive_v3r1_v8r4a_frozen_contract_encoding_"
                "false_rejection_failure_diagnostic"
            ),
            campaign_id=runtime.CAMPAIGN_ID,
            scientific_campaign_revision="V8R4",
            infrastructure_revision="V8R4A",
            status="diagnosed_not_authorized_by_diagnostic",
            failed_attempt={
                "first_phase": "efficiency_benchmark",
                "outer_fold": 3,
                "target_runtime_child_return_code": 1,
                "coordinator_return_code": 79,
                "target_sandbox_child_launched": True,
                "gpu_wrapper_reached": False,
                "gpu_admission_reached": False,
                "training_reached": False,
                "gpu_usage_ledger_mutated": False,
                "gpu_execution_ledger_mutated": False,
                "benchmark_output_files_created": False,
            },
            frozen_contract_evidence={
                "path": runtime.CAMPAIGN_CONTRACT_RELATIVE_PATH.as_posix(),
                "file_sha256": runtime.FROZEN_CONTRACT_PARENT_BINDINGS[
                    "frozen_campaign_contract"
                ]["file_sha256"],
                "bytes": 16179,
                "content_sha256": runtime.FROZEN_CONTRACT_PARENT_BINDINGS[
                    "frozen_campaign_contract"
                ]["content_sha256"],
                "mode": "0444",
                "valid_unique_key_finite_json": True,
                "semantic_content_hash_valid": True,
                "exact_file_binding_valid": True,
                "roundtrip_differs_from_frozen_bytes": True,
                "may_be_rewritten_or_reformatted": False,
            },
            root_cause={"synthetic": True},
            immutable_failure_receipts={
                "capability_receipt": {
                    **dict(
                        runtime.FROZEN_CONTRACT_PARENT_BINDINGS[
                            "failed_capability_receipt"
                        ]
                    ),
                    "mode": "0444",
                },
                "completion_receipt": {
                    **dict(
                        runtime.FROZEN_CONTRACT_PARENT_BINDINGS[
                            "failed_completion_receipt"
                        ]
                    ),
                    "mode": "0444",
                    "return_code": 1,
                    "closed_replay_validated": True,
                },
                "same_exact_lifecycle_replays_recorded_return_code": True,
                "mutation_replacement_or_deletion_allowed": False,
            },
            failed_namespace_inventory={
                "lifecycle_root": (
                    runtime.SUPERSEDED_V8R4A_LIFECYCLE_ROOT_RELATIVE.as_posix()
                ),
                "benchmark_output_root": (
                    runtime.SUPERSEDED_V8R4A_OUTPUT_ROOT_RELATIVE.as_posix()
                ),
                "completion_receipt_binds_live_output_inventory": True,
                "adding_success_artifacts_under_old_output_root_would_invalidate_failure_evidence": True,
                "old_lifecycle_and_output_roots_must_be_preserved_and_denied_to_successor": True,
            },
            ledger_evidence={"synthetic": True},
            required_correction={
                "successor_lifecycle_root": (
                    runtime.SUPERSEDED_V8R4A_CONTRACT1_LIFECYCLE_ROOT_RELATIVE.as_posix()
                ),
                "successor_benchmark_output_root": (
                    runtime.SUPERSEDED_V8R4A_CONTRACT1_OUTPUT_ROOT_RELATIVE.as_posix()
                ),
                "preserve_frozen_execution_closure_authority_historical_output_literal": (
                    runtime.LEGACY_EXECUTION_CLOSURE_BENCHMARK_OUTPUT_RELATIVE.as_posix()
                ),
                "new_test_receipt": runtime.FROZEN_CONTRACT_ACTIVE_FILENAMES[
                    "implementation_test_receipt"
                ],
                "new_source_snapshot": runtime.FROZEN_CONTRACT_ACTIVE_FILENAMES[
                    "source_snapshot"
                ],
                "new_pretrain_authorization": runtime.FROZEN_CONTRACT_ACTIVE_FILENAMES[
                    "active_authorization"
                ],
                "deny_and_unmount_superseded_lifecycle_root": True,
                "deny_and_unmount_superseded_output_root": True,
                "full_reauthorization_before_gpu_retry": True,
            },
            claim_boundary={
                "adaptive_retrospective_only": True,
                "outer_test_features_or_targets_opened": False,
                "accuracy_metric_computed": False,
                "gpu_accessed": False,
                "scientific_configuration_change_authorized": False,
                "commercial_claim_authorized": False,
            },
        ),
    )
    frozen_contract_authority = _write_exact_governance_json(
        campaign
        / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_FROZEN_CONTRACT_ENCODING.json",
        "frozen_contract_encoding_correction_authorization",
        _json_document(
            schema_version=1,
            classification=(
                "pretrain_adaptive_v3r1_v8r4a_frozen_contract_exact_byte_"
                "encoding_correction_addendum"
            ),
            campaign_id=runtime.CAMPAIGN_ID,
            scientific_campaign_revision="V8R4",
            infrastructure_revision="V8R4A",
            authority_basis={
                "diagnostic": _authority_legacy_row(
                    project, frozen_contract_diagnostic
                ),
                "parent_canary_boundary_authority": _authority_legacy_row(
                    project, canary_boundary_authority
                ),
                "parent_canary_boundary_diagnostic": _authority_legacy_row(
                    project, canary_boundary_diagnostic
                ),
                "parent_implementation_test_receipt": dict(
                    runtime.FROZEN_CONTRACT_PARENT_BINDINGS[
                        "parent_implementation_test_receipt"
                    ]
                ),
                "parent_source_snapshot": dict(
                    runtime.FROZEN_CONTRACT_PARENT_BINDINGS[
                        "parent_source_snapshot"
                    ]
                ),
                "parent_pretrain_authorization": dict(
                    runtime.FROZEN_CONTRACT_PARENT_BINDINGS[
                        "parent_pretrain_authorization"
                    ]
                ),
                "frozen_campaign_contract": _authority_legacy_row(
                    project, contract
                ),
                "failed_capability_receipt": dict(
                    runtime.FROZEN_CONTRACT_PARENT_BINDINGS[
                        "failed_capability_receipt"
                    ]
                ),
                "failed_completion_receipt": dict(
                    runtime.FROZEN_CONTRACT_PARENT_BINDINGS[
                        "failed_completion_receipt"
                    ]
                ),
                "user_goal_scope": "synthetic CPU-only CONTRACT1 fixture",
            },
            authorized_modifications=[
                {
                    "path": path,
                    "before_sha256": digest,
                    "allowed_change": "synthetic exact authorized change",
                }
                for path, digest in runtime.FROZEN_CONTRACT_AUTHORIZED_BEFORE.items()
            ],
            mandatory_invariants={
                "scientific_campaign_revision_unchanged": "V8R4",
                "infrastructure_revision_unchanged": "V8R4A",
                "variants_unchanged": ["H0_no_factor", "H1_factor", "H2_full"],
                "seeds_unchanged": list(runtime.SEEDS),
                "discovery_outer_folds_unchanged": [3, 4],
                "superseded_canary1_lifecycle_root_preserved_immutable": (
                    runtime.SUPERSEDED_V8R4A_LIFECYCLE_ROOT_RELATIVE.as_posix()
                ),
                "superseded_canary1_output_root_preserved_immutable": (
                    runtime.SUPERSEDED_V8R4A_OUTPUT_ROOT_RELATIVE.as_posix()
                ),
                "successor_contract1_lifecycle_root": (
                    runtime.SUPERSEDED_V8R4A_CONTRACT1_LIFECYCLE_ROOT_RELATIVE.as_posix()
                ),
                "successor_contract1_output_root": (
                    runtime.SUPERSEDED_V8R4A_CONTRACT1_OUTPUT_ROOT_RELATIVE.as_posix()
                ),
                "historical_execution_closure_authority_literal_unchanged": (
                    runtime.LEGACY_EXECUTION_CLOSURE_BENCHMARK_OUTPUT_RELATIVE.as_posix()
                ),
                **{
                    field: True
                    for field in (
                        "target_and_outer_reference_sealing_unchanged",
                        "gpu_budget_and_append_only_ledgers_unchanged",
                        "single_gpu_owner_and_kill_safe_lifecycle_unchanged",
                        "exact_frozen_contract_path_sha_bytes_and_content_unchanged",
                        "frozen_contract_mode_nlink_and_identity_validation_unchanged",
                        "global_noncanonical_governance_rejection_unchanged_except_exact_contract",
                        "both_superseded_roots_denied_unmounted_and_command_inaccessible",
                        "failed_capability_and_completion_receipts_preserved_immutable",
                        "failed_completion_closed_replay_remains_valid",
                        "all_preexisting_denied_canaries_mount_checks_command_checks_descriptor_checks_and_sandbox_probes_retained",
                        "parent_canary1_chain_preserved_immutable",
                    )
                },
            },
            forbidden_changes={
                field: True
                for field in (
                    "model_architecture_or_loss",
                    "hyperparameters_or_epoch_counts",
                    "data_rows_splits_or_pack_bytes",
                    "seed_variant_or_fold_matrix",
                    "metric_or_selection_thresholds",
                    "outer_target_or_reference_access",
                    "global_relaxation_of_json_encoding_validation",
                    "contract_reformat_rewrite_copy_or_replacement",
                    "acceptance_by_semantic_hash_without_exact_file_sha_and_bytes",
                    "reuse_of_superseded_lifecycle_root",
                    "reuse_or_mutation_of_superseded_output_root",
                    "deletion_or_replacement_of_failed_receipts",
                    "reinterpretation_of_frozen_execution_closure_authority",
                    "removing_or_weakening_any_denied_canary",
                    "removing_mount_command_descriptor_or_sandbox_checks",
                    "ledger_reset_truncation_or_rewrite",
                    "mutation_or_replacement_of_parent_evidence",
                    "commercial_or_confirmatory_claim",
                )
            },
            required_reauthorization=dict(
                runtime.FROZEN_CONTRACT_REQUIRED_REAUTHORIZATION
            ),
            claim_boundary={
                "adaptive_retrospective_only": True,
                "correction_is_infrastructure_only": True,
                "outer_test_features_or_targets_opened": False,
                "accuracy_metric_used": False,
                "gpu_execution_authorized_by_this_document": False,
                "successor_pretrain_authorization_required": True,
                "commercial_claim_authorized": False,
            },
        ),
    )
    state_root_status = state_root.stat(follow_symlinks=False)
    gpu_state_parent_bind_diagnostic = _write_exact_governance_json(
        campaign
        / "diagnostics/v3r1_v8r4a_gpu_state_parent_mount_identity_failure.json",
        "gpu_state_parent_bind_failure_diagnostic",
        _json_document(
            schema_version=1,
            classification=(
                "pretrain_adaptive_v3r1_v8r4a_gpu_state_parent_mount_"
                "identity_failure_diagnostic"
            ),
            campaign_id=runtime.CAMPAIGN_ID,
            scientific_campaign_revision="V8R4",
            infrastructure_revision="V8R4A",
            status="diagnosed_not_authorized_by_diagnostic",
            failed_attempt={
                "first_phase": "efficiency_benchmark",
                "outer_fold": 3,
                "target_runtime_child_return_code": 1,
                "coordinator_return_code": 79,
                "target_sandbox_child_launched": True,
                "target_scoped_pretrain_validation_reached": True,
                **{
                    field: False
                    for field in (
                        "gpu_wrapper_reached",
                        "gpu_admission_reached",
                        "training_reached",
                        "accuracy_metric_computed",
                        "gpu_usage_ledger_mutated",
                        "gpu_execution_ledger_mutated",
                        "benchmark_output_files_created",
                    )
                },
            },
            trusted_host_gpu_state_root={
                "path": runtime.GPU_STATE_ROOT_RELATIVE.as_posix(),
                "mode": "0700",
                "st_dev": state_root_status.st_dev,
                "st_ino": state_root_status.st_ino,
                "exact_entries": ["admission", "execution", "usage"],
            },
            failed_mount_topology={
                "parent_identity_mount_absent": True,
                "exactly_three_mutable_child_mounts_present": True,
            },
            required_correction={
                "migration_validator_relaxation_allowed": False,
                "gpu_state_root_writable_mount_allowed": False,
                "synthetic_parent_chmod_substitution_allowed": False,
                "exact_parent_readonly_fd_bind_required": {
                    "path": runtime.GPU_STATE_ROOT_RELATIVE.as_posix(),
                    "mode": "0700",
                    "st_dev": state_root_status.st_dev,
                    "st_ino": state_root_status.st_ino,
                    "kind": "ro_bind_fd",
                },
                "exactly_three_mutable_direct_child_overlays_required": [
                    "admission",
                    "execution",
                    "usage",
                ],
                "parent_mount_must_precede_child_overlays": True,
                "successor_lifecycle_root": (
                    runtime.SUPERSEDED_V8R4A_ROOTBIND1_LIFECYCLE_ROOT_RELATIVE.as_posix()
                ),
                "successor_benchmark_output_root": (
                    runtime.SUPERSEDED_V8R4A_ROOTBIND1_OUTPUT_ROOT_RELATIVE.as_posix()
                ),
                "new_test_receipt": runtime.GPU_STATE_PARENT_BIND_ACTIVE_FILENAMES[
                    "implementation_test_receipt"
                ],
                "new_source_snapshot": runtime.GPU_STATE_PARENT_BIND_ACTIVE_FILENAMES[
                    "source_snapshot"
                ],
                "new_pretrain_authorization": runtime.GPU_STATE_PARENT_BIND_ACTIVE_FILENAMES[
                    "active_authorization"
                ],
                "full_reauthorization_before_gpu_retry": True,
            },
            claim_boundary={
                "adaptive_retrospective_only": True,
                "outer_test_features_or_targets_opened": False,
                "accuracy_metric_computed": False,
                "gpu_accessed": False,
                "scientific_configuration_change_authorized": False,
                "commercial_claim_authorized": False,
            },
        ),
    )
    gpu_state_parent_bind_authority = _write_exact_governance_json(
        campaign
        / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_GPU_STATE_PARENT_BIND.json",
        "gpu_state_parent_bind_correction_authorization",
        _json_document(
            schema_version=1,
            classification=(
                "pretrain_adaptive_v3r1_v8r4a_gpu_state_parent_readonly_"
                "bind_correction_addendum"
            ),
            campaign_id=runtime.CAMPAIGN_ID,
            scientific_campaign_revision="V8R4",
            infrastructure_revision="V8R4A",
            authority_basis={
                "diagnostic": _authority_legacy_row(
                    project, gpu_state_parent_bind_diagnostic
                ),
                "parent_frozen_contract_authority": _authority_legacy_row(
                    project, frozen_contract_authority
                ),
                "parent_frozen_contract_diagnostic": _authority_legacy_row(
                    project, frozen_contract_diagnostic
                ),
                "parent_implementation_test_receipt": dict(
                    runtime.GPU_STATE_PARENT_BIND_PARENT_BINDINGS[
                        "parent_implementation_test_receipt"
                    ]
                ),
                "parent_source_snapshot": dict(
                    runtime.GPU_STATE_PARENT_BIND_PARENT_BINDINGS[
                        "parent_source_snapshot"
                    ]
                ),
                "parent_pretrain_authorization": dict(
                    runtime.GPU_STATE_PARENT_BIND_PARENT_BINDINGS[
                        "parent_pretrain_authorization"
                    ]
                ),
                "frozen_campaign_contract": _authority_legacy_row(project, contract),
                "gpu_state_migration_receipt": _authority_legacy_row(
                    project, migration_receipt
                ),
                "failed_contract1_capability_receipt": dict(
                    runtime.GPU_STATE_PARENT_BIND_PARENT_BINDINGS[
                        "failed_contract1_capability_receipt"
                    ]
                ),
                "failed_contract1_completion_receipt": dict(
                    runtime.GPU_STATE_PARENT_BIND_PARENT_BINDINGS[
                        "failed_contract1_completion_receipt"
                    ]
                ),
                "user_goal_scope": "synthetic CPU-only ROOTBIND1 fixture",
            },
            authorized_modifications=[
                {
                    "path": path,
                    "before_sha256": digest,
                    "allowed_change": "synthetic exact authorized ROOTBIND1 change",
                }
                for path, digest in runtime.GPU_STATE_PARENT_BIND_AUTHORIZED_BEFORE.items()
            ],
            mandatory_invariants={
                "scientific_campaign_revision_unchanged": "V8R4",
                "infrastructure_revision_unchanged": "V8R4A",
                "gpu_state_root_path": runtime.GPU_STATE_ROOT_RELATIVE.as_posix(),
                "gpu_state_root_exact_mode": "0700",
                "gpu_state_root_exact_st_dev": state_root_status.st_dev,
                "gpu_state_root_exact_st_ino": state_root_status.st_ino,
                "gpu_state_root_mount_kind": "ro_bind_fd",
                "gpu_state_mutable_direct_children": [
                    "admission",
                    "execution",
                    "usage",
                ],
                "superseded_v8r4a_lifecycle_root_preserved_immutable": (
                    runtime.SUPERSEDED_V8R4A_LIFECYCLE_ROOT_RELATIVE.as_posix()
                ),
                "superseded_v8r4a_output_root_preserved_immutable": (
                    runtime.SUPERSEDED_V8R4A_OUTPUT_ROOT_RELATIVE.as_posix()
                ),
                "superseded_v8r4a_contract1_lifecycle_root_preserved_immutable": (
                    runtime.SUPERSEDED_V8R4A_CONTRACT1_LIFECYCLE_ROOT_RELATIVE.as_posix()
                ),
                "superseded_v8r4a_contract1_output_root_preserved_immutable": (
                    runtime.SUPERSEDED_V8R4A_CONTRACT1_OUTPUT_ROOT_RELATIVE.as_posix()
                ),
                "successor_rootbind1_lifecycle_root": (
                    runtime.SUPERSEDED_V8R4A_ROOTBIND1_LIFECYCLE_ROOT_RELATIVE.as_posix()
                ),
                "successor_rootbind1_output_root": (
                    runtime.SUPERSEDED_V8R4A_ROOTBIND1_OUTPUT_ROOT_RELATIVE.as_posix()
                ),
                "historical_execution_closure_authority_literal_unchanged": (
                    runtime.LEGACY_EXECUTION_CLOSURE_BENCHMARK_OUTPUT_RELATIVE.as_posix()
                ),
                **{
                    field: True
                    for field in (
                        "variants_seeds_folds_hyperparameters_and_metrics_unchanged",
                        "target_and_outer_reference_sealing_unchanged",
                        "gpu_budget_and_append_only_ledgers_unchanged",
                        "single_gpu_owner_and_kill_safe_lifecycle_unchanged",
                        "migration_validator_and_migration_receipt_unchanged",
                        "gpu_state_root_mount_precedes_children",
                        "gpu_state_root_direct_mutation_denied",
                        "exactly_three_mutable_state_directory_mounts",
                        "child_atomic_replace_fsync_and_existing_recovery_unchanged",
                        "all_other_readonly_writable_overlap_rejected",
                        "internal_guard_child_descriptor_set_exactly_zero_one_two",
                        "all_four_superseded_roots_denied_unmounted_and_command_inaccessible",
                        "both_failed_capability_and_completion_generations_preserved_immutable",
                        "both_failed_completions_closed_replay_remain_valid",
                        "all_preexisting_mount_command_descriptor_sandbox_and_denied_canary_checks_retained",
                        "parent_contract1_chain_preserved_immutable",
                    )
                },
            },
            forbidden_changes={
                field: True
                for field in (
                    "model_architecture_or_loss",
                    "hyperparameters_or_epoch_counts",
                    "data_rows_splits_or_pack_bytes",
                    "seed_variant_or_fold_matrix",
                    "metric_or_selection_thresholds",
                    "outer_target_or_reference_access",
                    "migration_validator_relaxation",
                    "gpu_state_root_writable_bind",
                    "synthetic_parent_chmod_as_identity_substitute",
                    "adding_gpu_state_root_to_writable_roots",
                    "allowing_more_or_fewer_than_three_mutable_state_children",
                    "generic_readonly_writable_overlap_relaxation",
                    "broad_runs_or_campaign_root_mount",
                    "reuse_or_mutation_of_any_superseded_lifecycle_or_output_root",
                    "deletion_or_replacement_of_any_failed_receipt",
                    "reinterpretation_of_frozen_execution_closure_authority",
                    "removing_or_weakening_any_denied_canary",
                    "ledger_reset_truncation_or_rewrite",
                    "mutation_or_replacement_of_parent_evidence",
                    "commercial_or_confirmatory_claim",
                )
            },
            required_reauthorization=dict(
                runtime.GPU_STATE_PARENT_BIND_REQUIRED_REAUTHORIZATION
            ),
            claim_boundary={
                "adaptive_retrospective_only": True,
                "correction_is_infrastructure_only": True,
                "outer_test_features_or_targets_opened": False,
                "accuracy_metric_used": False,
                "gpu_execution_authorized_by_this_document": False,
                "successor_pretrain_authorization_required": True,
                "commercial_claim_authorized": False,
            },
        ),
    )
    admitted_context_diagnostic_document = json.loads(
        (
            REAL_PROJECT
            / "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
            "diagnostics/v3r1_v8r4a_benchmark_admitted_pretrain_context_bridge_failure.json"
        ).read_text()
    )
    admitted_context_diagnostic = _write_exact_governance_json(
        campaign
        / "diagnostics/v3r1_v8r4a_benchmark_admitted_pretrain_context_bridge_failure.json",
        "admitted_context_failure_diagnostic",
        admitted_context_diagnostic_document,
    )
    admitted_context_authority_document = json.loads(
        (
            REAL_PROJECT
            / "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
            "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_BENCHMARK_ADMITTED_CONTEXT.json"
        ).read_text()
    )
    admitted_context_authority_document["authority_basis"].update(
        {
            "diagnostic": _governance_row(
                project, admitted_context_diagnostic
            ),
            "parent_gpu_state_parent_bind_authority": _authority_sha_row(
                project, gpu_state_parent_bind_authority
            ),
            "parent_gpu_state_parent_bind_diagnostic": _authority_sha_row(
                project, gpu_state_parent_bind_diagnostic
            ),
            "frozen_campaign_contract": _authority_sha_row(
                project, contract, content_hash=False
            ),
            "gpu_state_migration_receipt": _authority_sha_row(
                project, migration_receipt
            ),
            "user_goal_scope": "synthetic CPU-only CONTEXT1 fixture",
        }
    )
    admitted_context_authority_document.pop("content_sha256", None)
    admitted_context_authority = _write_exact_governance_json(
        campaign
        / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_BENCHMARK_ADMITTED_CONTEXT.json",
        "admitted_context_correction_authorization",
        admitted_context_authority_document,
    )

    implementation = _write_exact_governance_json(
        campaign / runtime.BENCHMARK_ADMITTED_CONTEXT_ACTIVE_FILENAMES[
            "implementation_test_receipt"
        ],
        "implementation_test_receipt",
        _json_document(
            schema_version=1,
            classification="adaptive_v3r1_v8r4a_implementation_test_receipt",
            campaign_id=runtime.CAMPAIGN_ID,
            authorization_generation="CONTEXT1",
            scientific_campaign_revision="V8R4",
            infrastructure_revision="V8R4A",
            all_tests_passed=True,
            gpu_accessed=False,
            target_or_outer_reference_accessed=False,
            correction_authorization=_governance_row(project, parent_authority),
            infrastructure_correction_authorization=_governance_row(
                project, infrastructure_authority
            ),
            source_closure_correction_authorization=_governance_row(
                project, source_closure_authority
            ),
            source_closure_dependency_authorization=_governance_row(
                project, source_dependency_authority
            ),
            kill_safe_correction_authorization=_governance_row(
                project, kill_safe_authority
            ),
            open_lifecycle_recovery_correction_authorization=_governance_row(
                project, open_lifecycle_authority
            ),
            execution_closure_correction_authorization=_governance_row(
                project, execution_closure_authority
            ),
            source_closure_failure_diagnostic=_governance_row(
                project, source_closure_diagnostic
            ),
            kill_safe_failure_diagnostic=_governance_row(
                project, kill_safe_diagnostic
            ),
            open_lifecycle_recovery_failure_diagnostic=_governance_row(
                project, open_lifecycle_diagnostic
            ),
            execution_closure_failure_diagnostic=_governance_row(
                project, execution_closure_diagnostic
            ),
            migration_source_succession_correction_authorization=_governance_row(
                project, migration_source_succession_authority
            ),
            migration_source_succession_failure_diagnostic=_governance_row(
                project, migration_source_succession_diagnostic
            ),
            fd_closure_correction_authorization=_governance_row(
                project, fd_closure_authority
            ),
            fd_closure_failure_diagnostic=_governance_row(
                project, fd_closure_diagnostic
            ),
            canary_boundary_correction_authorization=_governance_row(
                project, canary_boundary_authority
            ),
            canary_boundary_failure_diagnostic=_governance_row(
                project, canary_boundary_diagnostic
            ),
            frozen_contract_encoding_correction_authorization=_governance_row(
                project, frozen_contract_authority
            ),
            frozen_contract_encoding_failure_diagnostic=_governance_row(
                project, frozen_contract_diagnostic
            ),
            gpu_state_parent_bind_correction_authorization=_governance_row(
                project, gpu_state_parent_bind_authority
            ),
            gpu_state_parent_bind_failure_diagnostic=_governance_row(
                project, gpu_state_parent_bind_diagnostic
            ),
            admitted_context_correction_authorization=_governance_row(
                project, admitted_context_authority
            ),
            admitted_context_failure_diagnostic=_governance_row(
                project, admitted_context_diagnostic
            ),
            gpu_state_migration_receipt=_governance_row(
                project, migration_receipt
            ),
        ),
    )
    snapshot_path = campaign / runtime.BENCHMARK_ADMITTED_CONTEXT_ACTIVE_FILENAMES[
        "source_snapshot"
    ]
    snapshot = _source_snapshot(
        project,
        entry=entry,
        launcher=launcher,
        migration_module=migration_module,
        gpu_admission_wrapper=gpu_admission_wrapper,
        gpu_budget_module=gpu_budget_module,
        package_initializer=package_initializer,
        implementation_receipt=implementation,
        campaign_contract=contract,
    )
    snapshot.update(
        {
            "correction_authorization": _governance_row(project, parent_authority),
            "infrastructure_correction_authorization": _governance_row(
                project, infrastructure_authority
            ),
            "source_closure_correction_authorization": _governance_row(
                project, source_closure_authority
            ),
            "source_closure_dependency_authorization": _governance_row(
                project, source_dependency_authority
            ),
            "kill_safe_correction_authorization": _governance_row(
                project, kill_safe_authority
            ),
            "open_lifecycle_recovery_correction_authorization": _governance_row(
                project, open_lifecycle_authority
            ),
            "execution_closure_correction_authorization": _governance_row(
                project, execution_closure_authority
            ),
            "source_closure_failure_diagnostic": _governance_row(
                project, source_closure_diagnostic
            ),
            "kill_safe_failure_diagnostic": _governance_row(
                project, kill_safe_diagnostic
            ),
            "open_lifecycle_recovery_failure_diagnostic": _governance_row(
                project, open_lifecycle_diagnostic
            ),
            "execution_closure_failure_diagnostic": _governance_row(
                project, execution_closure_diagnostic
            ),
            "migration_source_succession_correction_authorization": _governance_row(
                project, migration_source_succession_authority
            ),
            "migration_source_succession_failure_diagnostic": _governance_row(
                project, migration_source_succession_diagnostic
            ),
            "fd_closure_correction_authorization": _governance_row(
                project, fd_closure_authority
            ),
            "fd_closure_failure_diagnostic": _governance_row(
                project, fd_closure_diagnostic
            ),
            "canary_boundary_correction_authorization": _governance_row(
                project, canary_boundary_authority
            ),
            "canary_boundary_failure_diagnostic": _governance_row(
                project, canary_boundary_diagnostic
            ),
            "frozen_contract_encoding_correction_authorization": _governance_row(
                project, frozen_contract_authority
            ),
            "frozen_contract_encoding_failure_diagnostic": _governance_row(
                project, frozen_contract_diagnostic
            ),
            "gpu_state_parent_bind_correction_authorization": _governance_row(
                project, gpu_state_parent_bind_authority
            ),
            "gpu_state_parent_bind_failure_diagnostic": _governance_row(
                project, gpu_state_parent_bind_diagnostic
            ),
            "admitted_context_correction_authorization": _governance_row(
                project, admitted_context_authority
            ),
            "admitted_context_failure_diagnostic": _governance_row(
                project, admitted_context_diagnostic
            ),
            "gpu_state_migration_receipt": _governance_row(
                project, migration_receipt
            ),
        }
    )
    snapshot.pop("content_sha256")
    snapshot["content_sha256"] = runtime.semantic_sha256(snapshot)
    _write_exact_governance_json(snapshot_path, "source_snapshot", snapshot)
    active = _write_exact_governance_json(
        campaign / runtime.BENCHMARK_ADMITTED_CONTEXT_ACTIVE_FILENAMES[
            "active_authorization"
        ],
        "active_authorization",
        _json_document(
            schema_version=1,
            classification="pretrain_adaptive_v3r1_v8r4a_authorization",
            campaign_id=runtime.CAMPAIGN_ID,
            authorization_generation="CONTEXT1",
            scientific_campaign_revision="V8R4",
            infrastructure_revision="V8R4A",
            status="authorized",
            adaptive_retrospective_only=True,
            training_authorized=True,
            production_target_sealed_runtime_authorized=True,
            promotion_authorized=False,
            commercial_claim_authorized=False,
            canonical_gpu_state_paths=runtime._canonical_gpu_state_path_document(),
            runtime_ledger_prefixes={
                role: dict(row)
                for role, row in runtime.BENCHMARK_ADMITTED_CONTEXT_LEDGER_PREFIXES.items()
            },
            efficiency_benchmark_scope=dict(
                runtime.BENCHMARK_ADMITTED_CONTEXT_EFFICIENCY_SCOPE
            ),
            source_snapshot=_governance_row(project, snapshot_path),
            implementation_test_receipt=_governance_row(project, implementation),
            correction_authorization=_governance_row(project, parent_authority),
            infrastructure_correction_authorization=_governance_row(
                project, infrastructure_authority
            ),
            source_closure_correction_authorization=_governance_row(
                project, source_closure_authority
            ),
            source_closure_dependency_authorization=_governance_row(
                project, source_dependency_authority
            ),
            kill_safe_correction_authorization=_governance_row(
                project, kill_safe_authority
            ),
            open_lifecycle_recovery_correction_authorization=_governance_row(
                project, open_lifecycle_authority
            ),
            execution_closure_correction_authorization=_governance_row(
                project, execution_closure_authority
            ),
            source_closure_failure_diagnostic=_governance_row(
                project, source_closure_diagnostic
            ),
            kill_safe_failure_diagnostic=_governance_row(
                project, kill_safe_diagnostic
            ),
            open_lifecycle_recovery_failure_diagnostic=_governance_row(
                project, open_lifecycle_diagnostic
            ),
            execution_closure_failure_diagnostic=_governance_row(
                project, execution_closure_diagnostic
            ),
            migration_source_succession_correction_authorization=_governance_row(
                project, migration_source_succession_authority
            ),
            migration_source_succession_failure_diagnostic=_governance_row(
                project, migration_source_succession_diagnostic
            ),
            fd_closure_correction_authorization=_governance_row(
                project, fd_closure_authority
            ),
            fd_closure_failure_diagnostic=_governance_row(
                project, fd_closure_diagnostic
            ),
            canary_boundary_correction_authorization=_governance_row(
                project, canary_boundary_authority
            ),
            canary_boundary_failure_diagnostic=_governance_row(
                project, canary_boundary_diagnostic
            ),
            frozen_contract_encoding_correction_authorization=_governance_row(
                project, frozen_contract_authority
            ),
            frozen_contract_encoding_failure_diagnostic=_governance_row(
                project, frozen_contract_diagnostic
            ),
            gpu_state_parent_bind_correction_authorization=_governance_row(
                project, gpu_state_parent_bind_authority
            ),
            gpu_state_parent_bind_failure_diagnostic=_governance_row(
                project, gpu_state_parent_bind_diagnostic
            ),
            admitted_context_correction_authorization=_governance_row(
                project, admitted_context_authority
            ),
            admitted_context_failure_diagnostic=_governance_row(
                project, admitted_context_diagnostic
            ),
            gpu_state_migration_receipt=_governance_row(
                project, migration_receipt
            ),
        ),
    )
    common: dict[str, Path] = {
        "correction_authorization": parent_authority,
        "infrastructure_correction_authorization": infrastructure_authority,
        "source_closure_correction_authorization": source_closure_authority,
        "source_closure_dependency_authorization": source_dependency_authority,
        "kill_safe_correction_authorization": kill_safe_authority,
        "open_lifecycle_recovery_correction_authorization": open_lifecycle_authority,
        "execution_closure_correction_authorization": execution_closure_authority,
        "failure_diagnostic": historical_diagnostic,
        "infrastructure_failure_diagnostic": infrastructure_diagnostic,
        "source_closure_failure_diagnostic": source_closure_diagnostic,
        "kill_safe_failure_diagnostic": kill_safe_diagnostic,
        "open_lifecycle_recovery_failure_diagnostic": open_lifecycle_diagnostic,
        "execution_closure_failure_diagnostic": execution_closure_diagnostic,
        "migration_source_succession_correction_authorization": (
            migration_source_succession_authority
        ),
        "migration_source_succession_failure_diagnostic": (
            migration_source_succession_diagnostic
        ),
        "fd_closure_correction_authorization": fd_closure_authority,
        "fd_closure_failure_diagnostic": fd_closure_diagnostic,
        "canary_boundary_correction_authorization": canary_boundary_authority,
        "canary_boundary_failure_diagnostic": canary_boundary_diagnostic,
        "frozen_contract_encoding_correction_authorization": (
            frozen_contract_authority
        ),
        "frozen_contract_encoding_failure_diagnostic": frozen_contract_diagnostic,
        "gpu_state_parent_bind_correction_authorization": (
            gpu_state_parent_bind_authority
        ),
        "gpu_state_parent_bind_failure_diagnostic": (
            gpu_state_parent_bind_diagnostic
        ),
        "admitted_context_correction_authorization": (
            admitted_context_authority
        ),
        "admitted_context_failure_diagnostic": admitted_context_diagnostic,
        "gpu_state_migration_receipt": migration_receipt,
        "active_authorization": active,
        "source_snapshot": snapshot_path,
        "implementation_test_receipt": implementation,
        "campaign_contract": contract,
    }

    pack_root: Path | None = None
    pack_index: Path | None = None
    if phase not in {"discovery_aggregation", "promotion_aggregation"}:
        promotion_authorization: Path | None = None
        if phase in {"promotion_training", "promotion_prediction"}:
            promotion_authorization = _write_json(
                campaign / "PROMOTION_AUTHORIZATION_V8R4.json",
                _json_document(
                    schema_version=1,
                    classification="adaptive_v3r1_v8r4_promotion_authorization",
                    campaign_id=runtime.CAMPAIGN_ID,
                    campaign_revision="V8R4",
                    authorized_now=True,
                    authorized_scopes=[
                        "promotion_training_pack",
                        "outer_prediction_pack",
                    ],
                ),
            )
            common["discovery_completion_seal"] = _write_json(
                campaign / "DISCOVERY_COMPLETION_SEAL_V8R4.json",
                _json_document(
                    schema_version=1,
                    classification="synthetic-discovery-completion",
                    campaign_id=runtime.CAMPAIGN_ID,
                ),
            )
            common["selection_lock"] = _write_json(
                campaign / "SELECTION_LOCK_V8R4.json",
                _json_document(
                    schema_version=1,
                    classification="synthetic-selection-lock",
                    campaign_id=runtime.CAMPAIGN_ID,
                ),
            )
            common["promotion_authorization"] = promotion_authorization
        pack_root, pack_index = _make_pack(
            project,
            phase=phase,
            outer_fold=outer_fold,
            promotion_authorization=promotion_authorization,
            selection_lock=common.get("selection_lock"),
        )
        common["sealed_pack_index"] = pack_index
    if phase == "discovery":
        owner = _write_json(
            campaign / "DISCOVERY_V8R3_ATTEMPT_000_QUARANTINE_OWNER_RECEIPT_V8R4.json",
            _json_document(schema_version=1, campaign_id=runtime.CAMPAIGN_ID, classification="owner"),
        )
        diagnostic = common["failure_diagnostic"]
        materials: list[Path] = []
        for number in range(11):
            suffix = ".pt" if number in {5, 6} else ".npz" if number == 8 else ".json"
            materials.append(
                _write_frozen(
                    project / "legacy_quarantine" / f"material_{number:02d}{suffix}",
                    f"material-{number}".encode(),
                )
            )
        seal_rows = [
            {
                "path": path.relative_to(project).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                "mode": 0o444,
            }
            for path in materials
        ]
        seal = _write_json(
            project / "governance/DISCOVERY_V8R3_ATTEMPT_000_QUARANTINED_OUTPUT_SEAL_V8R4.json",
            _json_document(
                schema_version=1,
                campaign_id=runtime.CAMPAIGN_ID,
                classification="quarantined-output",
                files=seal_rows,
                quarantine_owner_receipt={
                    "path": owner.relative_to(project).as_posix(),
                    "sha256": hashlib.sha256(owner.read_bytes()).hexdigest(),
                    "bytes": owner.stat().st_size,
                },
                diagnostic={
                    "path": diagnostic.relative_to(project).as_posix(),
                    "sha256": hashlib.sha256(diagnostic.read_bytes()).hexdigest(),
                    "bytes": diagnostic.stat().st_size,
                },
            ),
        )
        common["benchmark_receipt"] = _write_json(
                    campaign / "BENCHMARK_COMPLETION_RECEIPT.json",
            _json_document(schema_version=1, campaign_id=runtime.CAMPAIGN_ID, classification="benchmark"),
        )
        common["quarantine_owner_receipt"] = owner
        common["quarantined_output_seal"] = seal
        for number, path in enumerate(materials):
            common[f"quarantined_material_{number:02d}"] = path
    if phase == "discovery_aggregation":
        for outer in (3, 4):
            portable = _artifact_binding(project, contract)
            authorization_portable = _artifact_binding(project, active)
            common[f"discovery_shard_seal_outer{outer}"] = _write_json(
                campaign / f"DISCOVERY_SHARD_OUTER_{outer}_SEAL.json",
                _json_document(
                    schema_version=1,
                    classification="adaptive_v3r1_v8r4_discovery_capability_shard_seal",
                    campaign_id=runtime.CAMPAIGN_ID,
                    campaign_revision="V8R4",
                    infrastructure_revision="V8R4A",
                    outer_fold_shard=outer,
                    contract=portable,
                    pretrain_authorization=authorization_portable,
                    training_index=portable,
                    completed_units=9,
                    peer_outer_shard_pack_mounted_or_opened=False,
                    combined_target_bearing_cache_opened=False,
                    outer_prediction_pack_absent=True,
                    physical_boundary=dict(runtime.DISCOVERY_PHYSICAL_BOUNDARY),
                    gpu_usage_ledger_prefix={
                        "path": str(state_root / "usage/campaign_gpu_usage_chain_v6.jsonl"),
                        "sha256": "1" * 64,
                        "bytes": 0,
                        "records": 1,
                        "terminal_record_sha256": "2" * 64,
                        "settled_usage_ns": 1,
                        "elapsed_seconds": 1e-9,
                        "open_reservations": 0,
                    },
                    pre_discovery_efficiency_benchmark=portable,
                    v8r3_quarantine_owner=portable,
                    units=[
                        {
                            "outer_fold": outer,
                            "seed": seed,
                            "variant": variant,
                            "receipt": portable,
                        }
                        for seed in runtime.SEEDS
                        for variant in ("H0_no_factor", "H1_factor", "H2_full")
                    ],
                    cross_outer_validation_reuse_present=True,
                    fully_nested_confirmatory_oof=False,
                    prospective_confirmation_required=True,
                    ready_for_pack_free_shard_aggregation=True,
                    commercial_claim_authorized=False,
                ),
            )

    writable: dict[str, Path] = {}
    shard_output.mkdir(parents=True)
    lifecycle = project / runtime._canonical_lifecycle_relative(
        phase=phase, outer_fold=outer_fold, entry_name=entry_name
    )
    lifecycle.mkdir(parents=True)
    writable["output"] = shard_output
    writable["lifecycle"] = lifecycle
    for role in runtime.GPU_STATE_DIRECTORY_ROLES:
        writable[role] = state_root / role
    canaries = {
        role: tmp_path / "denied" / role
        for role in sorted(runtime.MANDATORY_DENIED_CANARY_ROLES)
    }
    canaries["other_output_root"] = project / runtime.OTHER_OUTPUT_ROOT_RELATIVE
    canaries["superseded_v8r4a_lifecycle_root"] = (
        project / runtime.SUPERSEDED_V8R4A_LIFECYCLE_ROOT_RELATIVE
    )
    canaries["superseded_v8r4a_output_root"] = (
        project / runtime.SUPERSEDED_V8R4A_OUTPUT_ROOT_RELATIVE
    )
    canaries["superseded_v8r4a_contract1_lifecycle_root"] = (
        project / runtime.SUPERSEDED_V8R4A_CONTRACT1_LIFECYCLE_ROOT_RELATIVE
    )
    canaries["superseded_v8r4a_contract1_output_root"] = (
        project / runtime.SUPERSEDED_V8R4A_CONTRACT1_OUTPUT_ROOT_RELATIVE
    )
    canaries["superseded_v8r4a_rootbind1_lifecycle_root"] = (
        project / runtime.SUPERSEDED_V8R4A_ROOTBIND1_LIFECYCLE_ROOT_RELATIVE
    )
    canaries["superseded_v8r4a_rootbind1_output_root"] = (
        project / runtime.SUPERSEDED_V8R4A_ROOTBIND1_OUTPUT_ROOT_RELATIVE
    )
    receipt = writable["lifecycle"] / runtime.CAPABILITY_RECEIPT_FILENAME
    request = runtime.RuntimeRequest(
        project_root=project,
        phase=phase,
        outer_fold=outer_fold,
        pack_root=pack_root,
        pack_index=pack_index,
        governance_files=common,
        writable_roots=writable,
        denied_canaries=canaries,
        capability_receipt=receipt,
        interpreter=REAL_INTERPRETER,
        venv_root=REAL_VENV,
        python_runtime_root=REAL_RUNTIME,
        command=(str(REAL_INTERPRETER), str(entry)),
        production=False,
    )
    return {
        "project": project,
        "entry": entry,
        "launcher": launcher,
        "marker": marker,
        "request": request,
        "writable": writable,
        "canaries": canaries,
        "unbound_project_files": unbound_project_files,
    }


def _prepare(bundle: Mapping[str, Any]) -> Any:
    return runtime.prepare_runtime(
        bundle["request"], launcher_path=bundle["launcher"]
    )


def test_first_context1_prelaunch_requires_exact_postfailure_ledger_floor(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path)
    with _prepare(bundle) as prepared:
        assert prepared.prelaunch_state.usage_state["record_count"] >= 77
        assert prepared.prelaunch_state.usage_state["settled_usage_ns"] >= (
            1_411_550_918_574
        )
        assert prepared.prelaunch_state.execution_state["record_count"] >= 10
        assert prepared.receipt["security_boundary"][
            "active_pretrain_postfailure_ledger_prefix_enforced"
        ] is True


@pytest.mark.parametrize(
    "defect",
    (
        "empty_usage",
        "empty_execution",
        "usage_rootbind1_rollback",
        "execution_rootbind1_rollback",
        "usage_prefix_tamper",
    ),
)
def test_context1_prelaunch_rejects_empty_rollback_or_tampered_ledgers_before_process(
    tmp_path: Path, defect: str
) -> None:
    bundle = _make_project(tmp_path)
    paths = runtime._expected_gpu_state_paths(bundle["project"])
    if defect == "empty_usage":
        paths["usage_ledger"].write_bytes(b"")
    elif defect == "empty_execution":
        paths["execution_ledger"].write_bytes(b"")
    elif defect == "usage_rootbind1_rollback":
        paths["usage_ledger"].write_bytes(
            b"".join(
                FROZEN_CONTEXT1_LEDGER_RAW["usage_ledger"].splitlines(
                    keepends=True
                )[:75]
            )
        )
    elif defect == "execution_rootbind1_rollback":
        paths["execution_ledger"].write_bytes(
            b"".join(
                FROZEN_CONTEXT1_LEDGER_RAW["execution_ledger"].splitlines(
                    keepends=True
                )[:8]
            )
        )
    else:
        raw = FROZEN_CONTEXT1_LEDGER_RAW["usage_ledger"]
        paths["usage_ledger"].write_bytes(b"[" + raw[1:])
    before = {
        role: paths[role].read_bytes()
        for role in ("usage_ledger", "execution_ledger")
    }
    with pytest.raises(
        runtime.TargetSealedError,
        match="postfailure prefix|lower-bound",
    ):
        _prepare(bundle)
    assert {
        role: paths[role].read_bytes()
        for role in ("usage_ledger", "execution_ledger")
    } == before
    assert not bundle["request"].capability_receipt.exists()


def test_context1_prefix_projection_rejects_sibling_path_without_opening_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefixes = {
        role: dict(row)
        for role, row in runtime.BENCHMARK_ADMITTED_CONTEXT_LEDGER_PREFIXES.items()
    }
    prefixes["usage_ledger"]["path"] += "-sibling"
    opened = False

    def forbidden_open(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal opened
        opened = True
        raise AssertionError("a rejected sibling prefix path was opened")

    monkeypatch.setattr(runtime, "_read_file_binding", forbidden_open)
    with pytest.raises(runtime.TargetSealedError, match="projection drifted"):
        runtime._validate_active_pretrain_live_ledger_prefixes(
            project_root=tmp_path,
            runtime_ledger_prefixes=prefixes,
            live_state=None,
            require_closed=False,
        )
    assert opened is False


def _recovery_source_bindings(
    bundle: Mapping[str, Any],
) -> tuple[Any, dict[Path, Any]]:
    request = bundle["request"]
    snapshot = json.loads(request.governance_files["source_snapshot"].read_bytes())
    bindings: dict[Path, Any] = {}
    for row in runtime._snapshot_file_rows(snapshot):
        path, expected_sha256, expected_bytes, expected_mode = (
            runtime._validate_snapshot_row(row, project_root=bundle["project"])
        )
        binding, _ = runtime._read_file_binding(
            path, label="test recovery source", require_immutable=True
        )
        assert (binding.sha256, binding.bytes, binding.mode) == (
            expected_sha256,
            expected_bytes,
            expected_mode,
        )
        bindings[path] = binding
    receipt_binding, _ = runtime._read_file_binding(
        request.governance_files["gpu_state_migration_receipt"],
        label="test migration receipt",
        require_immutable=True,
    )
    return receipt_binding, bindings


def _seed_open_gpu_lifecycle(
    bundle: Mapping[str, Any],
    *,
    owner: str = "dead",
    append_execution_start: bool = True,
    admitted_child_authorization: Mapping[str, Any] | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    receipt_binding, source_bindings = _recovery_source_bindings(bundle)
    wrapper, budget = runtime._load_exact_gpu_recovery_modules(
        bundle["project"], source_snapshot_bindings=source_bindings
    )
    paths = runtime._expected_gpu_state_paths(bundle["project"])
    command = [str(REAL_INTERPRETER), str(bundle["entry"]), "--synthetic"]
    if owner == "live":
        wrapper_pid = os.getpid()
        wrapper_ticks = budget.process_start_ticks(wrapper_pid)
        assert wrapper_ticks is not None
    else:
        wrapper_pid = 99_999_999
        assert budget.process_start_ticks(wrapper_pid) is None
        wrapper_ticks = 1
    context = {"outer_fold": 3, "seed": 20260828, "test_only": True}
    reservation_template: dict[str, Any] = {
            "lifecycle_id": f"test-{owner}-{time.time_ns()}",
            "campaign_id": runtime.CAMPAIGN_ID,
            "phase": "efficiency_benchmark",
            "context": context,
            "invocation_sha256": "a" * 64,
            "command_sha256": budget.command_sha256(command),
            "result_path": str(
                (bundle["writable"]["output"] / "terminal-result.json").resolve()
            ),
            "gpu_execution_ledger_path": str(paths["execution_ledger"]),
            "boot_id": budget.boot_id(),
            "wrapper_pid": wrapper_pid,
            "wrapper_start_ticks": wrapper_ticks,
            "wrapper_parent_pid": os.getppid(),
            "hostname": socket.gethostname(),
            "cwd": str(bundle["project"]),
            "gpu_lock_file": str(paths["admission_lock"]),
            "cuda_visible_devices": None,
            "realtime_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
    }
    if admitted_child_authorization is not None:
        reservation_template["admitted_child_authorization"] = dict(
            admitted_child_authorization
        )
    reservation, _, _ = budget.reconcile_and_reserve(
        paths["usage_ledger"],
        reservation_template,
    )
    if append_execution_start:
        wrapper.append_ledger(
            paths["execution_ledger"],
            {
            "schema_version": 1,
            "job_id": reservation["lifecycle_id"],
            "lifecycle_id": reservation["lifecycle_id"],
            "reservation_record_sha256": reservation["record_sha256"],
            "wrapper_pid": reservation["wrapper_pid"],
            "hostname": reservation["hostname"],
            "cwd": reservation["cwd"],
            "lock_file": reservation["gpu_lock_file"],
            "usage_ledger": str(paths["usage_ledger"]),
            "result_file": reservation["result_path"],
            "campaign_id": reservation["campaign_id"],
            "phase": reservation["phase"],
            "context": reservation["context"],
            "invocation_sha256": reservation["invocation_sha256"],
            "command": command,
            "command_sha256": reservation["command_sha256"],
            "cuda_visible_devices": reservation["cuda_visible_devices"],
            "event": "start",
            "utc": wrapper.utc_now(),
            },
        )
    # Return these too so tests can replay the same exact source generation.
    reservation["_test_receipt_binding"] = receipt_binding
    reservation["_test_source_bindings"] = source_bindings
    return wrapper, budget, reservation


def test_preflight_recovers_dead_post_reservation_start_before_strict_replay(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path)
    wrapper, budget, reservation = _seed_open_gpu_lifecycle(bundle, owner="dead")
    paths = runtime._expected_gpu_state_paths(bundle["project"])
    assert budget.verify_ledger(paths["usage_ledger"]).open_reservations
    assert wrapper._open_execution_starts(
        wrapper._decode_execution_ledger(paths["execution_ledger"].read_bytes())
    )

    with _prepare(bundle) as prepared:
        assert prepared.prelaunch_state.usage_state["open_reservation_count"] == 0
        assert prepared.prelaunch_state.execution_state["open_start_count"] == 0

    usage_state = budget.verify_ledger(paths["usage_ledger"])
    assert not usage_state.open_reservations
    terminals = [
        row
        for row in usage_state.records
        if row.get("lifecycle_id") == reservation["lifecycle_id"]
        and row.get("event") == "reconciled_terminal"
    ]
    assert len(terminals) == 1
    assert terminals[0]["process_identity_proven_dead"] is True
    assert terminals[0]["reuse_eligible"] is False
    execution_rows = wrapper._decode_execution_ledger(
        paths["execution_ledger"].read_bytes()
    )
    assert not wrapper._open_execution_starts(execution_rows)
    ends = [
        row
        for row in execution_rows
        if row.get("lifecycle_id") == reservation["lifecycle_id"]
        and row.get("event") == "end"
    ]
    assert len(ends) == 1
    assert ends[0]["recovered_from_durable_usage_terminal"] is True
    assert ends[0]["usage_terminal_event"] == "reconciled_terminal"


def test_recovery_module_loader_restores_host_import_namespace(tmp_path: Path) -> None:
    bundle = _make_project(tmp_path)
    _receipt, source_bindings = _recovery_source_bindings(bundle)
    before_path = list(sys.path)
    before_modules = {
        name: sys.modules.get(name)
        for name in ("snn_rr", "snn_rr.gpu_budget_ledger")
    }
    wrapper, budget = runtime._load_exact_gpu_recovery_modules(
        bundle["project"], source_snapshot_bindings=source_bindings
    )
    assert wrapper.budget is budget
    assert Path(wrapper.__file__).resolve() == (
        bundle["project"] / runtime.GPU_ADMISSION_WRAPPER_RELATIVE_PATH
    )
    assert sys.path == before_path
    assert {
        name: sys.modules.get(name)
        for name in ("snn_rr", "snn_rr.gpu_budget_ledger")
    } == before_modules


def test_preflight_recovers_dead_post_reservation_before_execution_start(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path)
    wrapper, budget, reservation = _seed_open_gpu_lifecycle(
        bundle,
        owner="dead",
        append_execution_start=False,
    )
    paths = runtime._expected_gpu_state_paths(bundle["project"])
    assert budget.verify_ledger(paths["usage_ledger"]).open_reservations
    assert (
        paths["execution_ledger"].read_bytes()
        == FROZEN_CONTEXT1_LEDGER_RAW["execution_ledger"]
    )

    with _prepare(bundle) as prepared:
        assert prepared.prelaunch_state.usage_state["open_reservation_count"] == 0
        assert prepared.prelaunch_state.execution_state["open_start_count"] == 0

    state = budget.verify_ledger(paths["usage_ledger"])
    assert not state.open_reservations
    assert sum(
        row.get("lifecycle_id") == reservation["lifecycle_id"]
        and row.get("event") == "reconciled_terminal"
        for row in state.records
    ) == 1
    assert wrapper._decode_execution_ledger(
        paths["execution_ledger"].read_bytes()
    ) == wrapper._decode_execution_ledger(
        FROZEN_CONTEXT1_LEDGER_RAW["execution_ledger"]
    )


def test_preflight_recovery_is_byte_and_inode_idempotent(tmp_path: Path) -> None:
    bundle = _make_project(tmp_path)
    _wrapper, _budget, reservation = _seed_open_gpu_lifecycle(bundle, owner="dead")
    with _prepare(bundle):
        pass
    paths = runtime._expected_gpu_state_paths(bundle["project"])
    before = {
        role: (
            paths[role].read_bytes(),
            paths[role].stat(follow_symlinks=False).st_ino,
        )
        for role in ("usage_ledger", "execution_ledger")
    }
    assert runtime._recover_dead_gpu_lifecycle_before_closed_validation(
        project_root=bundle["project"],
        receipt_binding=reservation["_test_receipt_binding"],
        source_snapshot_bindings=reservation["_test_source_bindings"],
    ) == 0
    after = {
        role: (
            paths[role].read_bytes(),
            paths[role].stat(follow_symlinks=False).st_ino,
        )
        for role in ("usage_ledger", "execution_ledger")
    }
    assert after == before


def test_preflight_refuses_matching_live_owner_without_mutation(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path)
    _wrapper, _budget, _reservation = _seed_open_gpu_lifecycle(
        bundle, owner="live"
    )
    paths = runtime._expected_gpu_state_paths(bundle["project"])
    before = {
        role: paths[role].read_bytes()
        for role in ("usage_ledger", "execution_ledger")
    }
    with pytest.raises(runtime.TargetSealedError, match="matching live wrapper"):
        _prepare(bundle)
    assert {
        role: paths[role].read_bytes()
        for role in ("usage_ledger", "execution_ledger")
    } == before
    assert not bundle["request"].capability_receipt.exists()


def test_preflight_refuses_tampered_reservation_authorization_without_charge(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path)
    _wrapper, _budget, _reservation = _seed_open_gpu_lifecycle(
        bundle,
        owner="dead",
        admitted_child_authorization={
            "path": str(bundle["project"] / "governance/missing-authorization.json"),
            "sha256": "0" * 64,
        },
    )
    paths = runtime._expected_gpu_state_paths(bundle["project"])
    before_usage = paths["usage_ledger"].read_bytes()
    before_execution = paths["execution_ledger"].read_bytes()
    with pytest.raises(
        runtime.TargetSealedError,
        match="kill recovery failed",
    ):
        _prepare(bundle)
    assert paths["usage_ledger"].read_bytes() == before_usage
    assert paths["execution_ledger"].read_bytes() == before_execution


def test_preflight_refuses_usage_hash_prefix_drift_without_mutation(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path)
    _wrapper, _budget, _reservation = _seed_open_gpu_lifecycle(
        bundle, owner="dead"
    )
    paths = runtime._expected_gpu_state_paths(bundle["project"])
    usage_lines = paths["usage_ledger"].read_bytes().splitlines(keepends=True)
    assert len(usage_lines) == 78
    tampered = json.loads(usage_lines[-1])
    tampered["context"]["seed"] = 0
    paths["usage_ledger"].write_bytes(
        b"".join(usage_lines[:-1])
        + runtime.canonical_json_bytes(tampered)
        + b"\n"
    )
    paths["usage_ledger"].chmod(0o644)
    before_usage = paths["usage_ledger"].read_bytes()
    before_execution = paths["execution_ledger"].read_bytes()
    with pytest.raises(
        runtime.TargetSealedError,
        match="hash drifted|kill recovery failed",
    ):
        _prepare(bundle)
    assert paths["usage_ledger"].read_bytes() == before_usage
    assert paths["execution_ledger"].read_bytes() == before_execution


def test_preflight_refuses_multiple_open_reservations_without_mutation(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path)
    _wrapper, budget, _reservation = _seed_open_gpu_lifecycle(
        bundle,
        owner="dead",
        append_execution_start=False,
    )
    paths = runtime._expected_gpu_state_paths(bundle["project"])
    usage_prefix = paths["usage_ledger"].read_bytes().splitlines(keepends=True)
    assert len(usage_prefix) == 78
    first = json.loads(usage_prefix[-1])
    first.pop("record_sha256")
    reservation_ns = 60 * 60 * 1_000_000_000
    first["reservation_ns"] = reservation_ns
    first["workload_timeout_ns"] = (
        reservation_ns
        - budget.TERMINATION_GRACE_NS
        - budget.ACCOUNTING_MARGIN_NS
    )
    first["record_sha256"] = budget.semantic_sha256(first)
    second = json.loads(json.dumps(first))
    second.pop("record_sha256")
    second["previous_record_sha256"] = first["record_sha256"]
    second["lifecycle_id"] = "second-dead-reservation"
    second["wrapper_pid"] = 99_999_998
    second["realtime_ns"] = time.time_ns()
    second["monotonic_ns"] = time.monotonic_ns()
    second["record_sha256"] = budget.semantic_sha256(second)
    paths["usage_ledger"].write_bytes(
        b"".join(usage_prefix[:-1])
        + budget.canonical_json_bytes(first)
        + b"\n"
        + budget.canonical_json_bytes(second)
        + b"\n"
    )
    paths["usage_ledger"].chmod(0o644)
    assert len(budget.verify_ledger(paths["usage_ledger"]).open_reservations) == 2
    before = paths["usage_ledger"].read_bytes()
    with pytest.raises(runtime.TargetSealedError, match="multiple open reservations"):
        _prepare(bundle)
    assert paths["usage_ledger"].read_bytes() == before
    assert (
        paths["execution_ledger"].read_bytes()
        == FROZEN_CONTEXT1_LEDGER_RAW["execution_ledger"]
    )


@pytest.mark.parametrize("defect", ["phase", "context", "command", "path"])
def test_preflight_execution_tamper_fails_closed_before_usage_charge(
    tmp_path: Path, defect: str
) -> None:
    bundle = _make_project(tmp_path)
    wrapper, _budget, _reservation = _seed_open_gpu_lifecycle(bundle, owner="dead")
    paths = runtime._expected_gpu_state_paths(bundle["project"])
    execution_rows = wrapper._decode_execution_ledger(
        paths["execution_ledger"].read_bytes()
    )
    assert len(execution_rows) == 11
    if defect == "phase":
        execution_rows[-1]["phase"] = "discovery"
    elif defect == "context":
        execution_rows[-1]["context"] = {"outer_fold": 4}
    elif defect == "command":
        execution_rows[-1]["command"] = ["/bin/false"]
    else:
        execution_rows[-1]["usage_ledger"] = str(
            paths["usage_ledger"].with_name("wrong.jsonl")
        )
    paths["execution_ledger"].write_bytes(
        b"".join(
            wrapper.budget.canonical_json_bytes(row) + b"\n"
            for row in execution_rows
        )
    )
    paths["execution_ledger"].chmod(0o644)
    before_usage = paths["usage_ledger"].read_bytes()
    before_execution = paths["execution_ledger"].read_bytes()
    with pytest.raises(
        runtime.TargetSealedError,
        match="start phase differs|kill recovery failed",
    ):
        _prepare(bundle)
    assert paths["usage_ledger"].read_bytes() == before_usage
    assert paths["execution_ledger"].read_bytes() == before_execution
    assert not bundle["request"].capability_receipt.exists()


def test_exact_outer_shard_plan_and_no_nested_watchdog_boundary(tmp_path: Path) -> None:
    bundle = _make_project(tmp_path)
    with _prepare(bundle) as prepared:
        command = prepared.bwrap_command
        assert command[0] == "/usr/bin/bwrap"
        assert "--die-with-parent" in command
        assert "--unshare-net" in command
        assert "--clearenv" in command
        assert "--unshare-pid" not in command
        assert "--new-session" not in command
        assert runtime.ADMITTED_CHILD_FD_ENV not in prepared.child_environment
        receipt = prepared.receipt
        assert receipt["sealed_pack_index"]["unit_count"] == 3
        assert receipt["outer_fold"] == 3
        assert receipt["security_boundary"][
            "admitted_fd_direct_watchdog_to_trainer_contract_preserved"
        ] is True
        mounted = {row["destination"] for row in prepared.mount_entries}
        by_destination = {
            row["destination"]: row for row in prepared.mount_entries
        }
        assert str(bundle["project"] / "scripts") in mounted
        assert str(bundle["project"] / "src") in mounted
        assert str(bundle["project"] / "tests") in mounted
        assert str(bundle["project"] / "configs") in mounted
        for directory in ("scripts", "src", "tests", "configs"):
            assert by_destination[str(bundle["project"] / directory)]["kind"] == "directory"
        assert all(
            str(path) not in by_destination
            for path in bundle["unbound_project_files"]
        )
        assert not any(
            row["kind"] == "ro_bind_fd"
            and row["destination"]
            in {
                str(bundle["project"] / "scripts"),
                str(bundle["project"] / "src"),
                str(bundle["project"] / "tests"),
                str(bundle["project"] / "configs"),
            }
            for row in prepared.mount_entries
        )
        assert not any("combined" in value for value in mounted)
        state_mounts = []
        state_root = str(bundle["project"] / runtime.GPU_STATE_ROOT_RELATIVE)
        assert by_destination[state_root]["kind"] == "ro_bind_fd"
        assert by_destination[state_root]["source"] == {
            **prepared.gpu_state_root_binding.document(),
            "exact_entries": ["admission", "execution", "usage"],
        }
        for role in runtime.GPU_STATE_DIRECTORY_ROLES:
            path = str(bundle["writable"][role])
            assert by_destination[path]["kind"] == "rw_bind_fd"
            assert by_destination[path]["source"]["exact_entries"] == sorted(
                runtime.GPU_STATE_EXACT_ENTRIES[role]
            )
            state_mounts.append(by_destination[path])
        assert len(state_mounts) == 3
        state_paths = {
            state_root,
            *(str(bundle["writable"][role]) for role in runtime.GPU_STATE_DIRECTORY_ROLES),
        }
        assert [
            row["destination"]
            for row in prepared.mount_entries
            if row["destination"] in state_paths
        ] == [
            state_root,
            *(str(bundle["writable"][role]) for role in sorted(runtime.GPU_STATE_DIRECTORY_ROLES)),
        ]
        assert not any(
            row["kind"] == "rw_bind_file_fd" for row in prepared.mount_entries
        )
        assert receipt["campaign_revision"] == "V8R4"
        assert receipt["infrastructure_revision"] == "V8R4A"
        assert receipt["security_boundary"]["atomic_replace_compatible"] is True
        assert receipt["security_boundary"][
            "gpu_state_parent_identity_readonly_bind"
        ] is True
        assert by_destination[str(bundle["writable"]["output"])]["kind"] == "rw_bind_fd"
        assert by_destination[str(bundle["writable"]["lifecycle"])]["kind"] == "ro_bind_fd"
        assert bundle["writable"]["output"] != bundle["writable"]["lifecycle"]


def test_read_binding_closes_once_and_is_repeatable(tmp_path: Path) -> None:
    path = _write_frozen(tmp_path / "one.json", b"{}\n")
    first, raw = runtime._read_file_binding(path, label="one", require_immutable=True)
    second, raw_again = runtime._read_file_binding(path, label="one", require_immutable=True)
    assert first == second
    assert raw == raw_again == b"{}\n"
    path.chmod(0o644)
    with pytest.raises(runtime.TargetSealedError, match="0444"):
        runtime._read_file_binding(path, label="one", require_immutable=True)


@pytest.mark.parametrize("field,value", [
    ("unit_count", 6),
    ("completed_units", 2),
    ("cross_outer_shard_mounted", True),
    ("outer_test_opened", True),
    ("combined_target_bearing_cache_consumer_access_authorized", True),
])
def test_pack_index_capability_and_cover_tamper_fails(
    tmp_path: Path, field: str, value: Any
) -> None:
    bundle = _make_project(tmp_path)
    index = bundle["request"].pack_index
    assert index is not None
    document = json.loads(index.read_text())
    document[field] = value
    document.pop("content_sha256")
    document["content_sha256"] = runtime.semantic_sha256(document)
    index.chmod(0o644)
    _write_json(index, document)
    with pytest.raises(runtime.TargetSealedError):
        _prepare(bundle)


def test_cross_outer_pack_or_six_seed_cover_is_rejected(tmp_path: Path) -> None:
    bundle = _make_project(tmp_path)
    index = bundle["request"].pack_index
    assert index is not None
    document = json.loads(index.read_text())
    document["units"][0]["outer_fold"] = 4
    document.pop("content_sha256")
    document["content_sha256"] = runtime.semantic_sha256(document)
    index.chmod(0o644)
    _write_json(index, document)
    with pytest.raises(runtime.TargetSealedError, match="identity|cover"):
        _prepare(bundle)


@pytest.mark.parametrize("phase", ["promotion_training", "promotion_prediction"])
def test_promotion_pack_phase_aware_exact_schema_and_authorization(
    tmp_path: Path, phase: str
) -> None:
    bundle = _make_project(tmp_path, phase=phase)
    with _prepare(bundle) as prepared:
        index = prepared.receipt["sealed_pack_index"]
        assert index["unit_count"] == 3
        document = json.loads(bundle["request"].pack_index.read_text())
        authorization = bundle["request"].governance_files[
            "promotion_authorization"
        ]
        assert document["promotion_authorization"] == _authorization_binding(
            authorization
        )
        assert {row["seed"] for row in document["units"]} == set(runtime.SEEDS)
        if phase == "promotion_prediction":
            assert document["classification"] == runtime.PREDICTION_PACK_INDEX_CLASSIFICATION
            assert document["physical_target_free_input_and_model_packs"] is True
            assert document["source_paths_or_peer_outputs_authorized_in_child"] is False
            assert document["model_source_shard_seal"]["path"] == (
                runtime.MODEL_SOURCE_SHARD_SEAL_FILENAME
            )
            assert "physical_nonouter_training_packs" not in document
            assert all(
                set(row["artifacts"])
                == runtime.PREDICTION_PACK_INDEX_ARTIFACT_KEYS
                for row in document["units"]
            )
        else:
            assert document["promotion_scope"] == "promotion_training_pack"
            assert document["physical_nonouter_training_packs"] is True
            assert "physical_target_free_outer_prediction_packs" not in document


def test_prediction_training_field_reuse_or_authorization_tamper_fails(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path, phase="promotion_prediction")
    index = bundle["request"].pack_index
    assert index is not None
    document = json.loads(index.read_text())
    document["physical_nonouter_training_packs"] = True
    document.pop("content_sha256")
    document["content_sha256"] = runtime.semantic_sha256(document)
    index.chmod(0o644)
    _write_json(index, document)
    with pytest.raises(runtime.TargetSealedError, match="schema"):
        _prepare(bundle)

    bundle = _make_project(tmp_path / "binding", phase="promotion_training")
    index = bundle["request"].pack_index
    assert index is not None
    document = json.loads(index.read_text())
    document["promotion_authorization"]["sha256"] = "0" * 64
    document.pop("content_sha256")
    document["content_sha256"] = runtime.semantic_sha256(document)
    index.chmod(0o644)
    _write_json(index, document)
    with pytest.raises(runtime.TargetSealedError, match="authorization binding"):
        _prepare(bundle)


def test_legacy_v8r4_target_free_prediction_index_is_rejected(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path, phase="promotion_prediction")
    index_path = bundle["request"].pack_index
    assert index_path is not None
    document = json.loads(index_path.read_text())
    document["classification"] = (
        "adaptive_v3r1_v8r4_target_free_prediction_shard_index"
    )
    for field in (
        "infrastructure_revision",
        "selected_variant",
        "physical_target_free_input_and_model_packs",
        "source_paths_or_peer_outputs_authorized_in_child",
        "model_source_shard_seal",
    ):
        document.pop(field)
    document["physical_target_free_outer_prediction_packs"] = True
    document["outer_prediction_packs_absent"] = False
    for row in document["units"]:
        for field in (
            "scientific_signature_sha256",
            "row_count",
            "global_cache_index_sha256",
            "source_kind",
        ):
            row.pop(field)
        row["artifacts"] = {
            role: row["artifacts"][role]
            for role in ("prediction_pack_manifest", "outer_predict_input")
        }
    document.pop("content_sha256")
    document["content_sha256"] = runtime.semantic_sha256(document)
    index_path.chmod(0o644)
    _write_json(index_path, document)
    with pytest.raises(runtime.TargetSealedError, match="index schema"):
        _prepare(bundle)


def test_model_source_shard_boundary_tamper_is_rejected_after_rebinding(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path, phase="promotion_prediction")
    root = bundle["request"].pack_root
    index_path = bundle["request"].pack_index
    assert root is not None and index_path is not None
    seal_path = root / runtime.MODEL_SOURCE_SHARD_SEAL_FILENAME
    seal = json.loads(seal_path.read_text())
    seal["target_or_prediction_values_present"] = True
    seal.pop("content_sha256")
    seal["content_sha256"] = runtime.semantic_sha256(seal)
    seal_path.chmod(0o644)
    _write_json(seal_path, seal)
    index = json.loads(index_path.read_text())
    index["model_source_shard_seal"] = _artifact_binding(root, seal_path)
    index.pop("content_sha256")
    index["content_sha256"] = runtime.semantic_sha256(index)
    index_path.chmod(0o644)
    _write_json(index_path, index)
    with pytest.raises(runtime.TargetSealedError, match="model-source shard seal"):
        _prepare(bundle)


def test_prediction_npz_extra_field_fails_even_with_rehashed_bindings(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path, phase="promotion_prediction")
    root = bundle["request"].pack_root
    index_path = bundle["request"].pack_index
    assert root is not None and index_path is not None
    index = json.loads(index_path.read_text())
    row = index["units"][0]
    unit = root / row["relative_path"]
    npz = unit / "outer_predict_input.npz"
    manifest_path = unit / "OUTER_PREDICTION_PACK_MANIFEST.json"
    unit.chmod(0o755)
    npz.chmod(0o644)
    with zipfile.ZipFile(npz, mode="a", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("target.npy", b"forbidden")
    npz.chmod(0o444)
    manifest = json.loads(manifest_path.read_text())
    manifest["output"] = _artifact_binding(unit, npz)
    manifest.pop("content_sha256")
    manifest["content_sha256"] = runtime.semantic_sha256(manifest)
    manifest_path.chmod(0o644)
    _write_json(manifest_path, manifest)
    successor_path = unit / runtime.MODEL_BOUND_PREDICTION_MANIFEST_FILENAME
    successor = json.loads(successor_path.read_text())
    successor["base_target_free_manifest"] = _artifact_binding(
        unit, manifest_path
    )
    successor["artifacts"]["outer_predict_input"] = _artifact_binding(unit, npz)
    successor.pop("content_sha256")
    successor["content_sha256"] = runtime.semantic_sha256(successor)
    successor_path.chmod(0o644)
    _write_json(successor_path, successor)
    row["artifacts"]["outer_predict_input"] = _artifact_binding(root, npz)
    row["artifacts"]["prediction_pack_manifest"] = _artifact_binding(
        root, manifest_path
    )
    row["artifacts"]["model_bound_prediction_pack_manifest"] = (
        _artifact_binding(root, successor_path)
    )
    model_seal_path = root / runtime.MODEL_SOURCE_SHARD_SEAL_FILENAME
    model_seal = json.loads(model_seal_path.read_text())
    sealed_unit = next(
        item for item in model_seal["units"] if item["seed"] == row["seed"]
    )
    sealed_unit["model_bound_prediction_pack_manifest"] = _artifact_binding(
        root, successor_path
    )
    model_seal.pop("content_sha256")
    model_seal["content_sha256"] = runtime.semantic_sha256(model_seal)
    model_seal_path.chmod(0o644)
    _write_json(model_seal_path, model_seal)
    index["model_source_shard_seal"] = _artifact_binding(root, model_seal_path)
    index.pop("content_sha256")
    index["content_sha256"] = runtime.semantic_sha256(index)
    index_path.chmod(0o644)
    root.chmod(0o755)
    _write_json(index_path, index)
    unit.chmod(0o555)
    root.chmod(0o555)
    with pytest.raises(runtime.TargetSealedError, match="ten-field allowlist"):
        _prepare(bundle)


def test_authorized_pack_nested_auth_and_extra_member_fail_closed(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path / "training", phase="promotion_training")
    root = bundle["request"].pack_root
    index_path = bundle["request"].pack_index
    assert root is not None and index_path is not None
    index = json.loads(index_path.read_text())
    row = index["units"][0]
    unit = root / row["relative_path"]
    cache_path = unit / "discovery_cache/manifest.json"
    cache = json.loads(cache_path.read_text())
    cache["promotion_authorization"]["sha256"] = "0" * 64
    cache.pop("content_sha256")
    cache["content_sha256"] = runtime.semantic_sha256(cache)
    cache_path.chmod(0o644)
    _write_json(cache_path, cache)
    partition_path = unit / "PARTITION_MANIFEST.json"
    partition = json.loads(partition_path.read_text())
    partition["outputs"]["discovery_cache_manifest"] = _artifact_binding(
        unit, cache_path
    )
    partition.pop("content_sha256")
    partition["content_sha256"] = runtime.semantic_sha256(partition)
    partition_path.chmod(0o644)
    _write_json(partition_path, partition)
    row["artifacts"]["cache_manifest"] = _artifact_binding(root, cache_path)
    row["artifacts"]["partition_manifest"] = _artifact_binding(
        root, partition_path
    )
    index.pop("content_sha256")
    index["content_sha256"] = runtime.semantic_sha256(index)
    index_path.chmod(0o644)
    _write_json(index_path, index)
    with pytest.raises(runtime.TargetSealedError, match="cache manifest capability"):
        _prepare(bundle)

    bundle = _make_project(tmp_path / "prediction", phase="promotion_prediction")
    root = bundle["request"].pack_root
    assert root is not None
    unit = next((root / "units").iterdir())
    unit.chmod(0o755)
    _write_frozen(unit / "unindexed_target.npy", b"forbidden")
    unit.chmod(0o555)
    with pytest.raises(runtime.TargetSealedError, match="exact inventory"):
        _prepare(bundle)


def test_pack_symlink_hardlink_and_writable_file_fail(tmp_path: Path) -> None:
    bundle = _make_project(tmp_path / "symlink")
    pack = bundle["request"].pack_root
    assert pack is not None
    member = next(pack.rglob("node_features.npy"))
    member.chmod(0o644)
    with pytest.raises(runtime.TargetSealedError, match="0444"):
        _prepare(bundle)

    bundle = _make_project(tmp_path / "hardlink")
    pack = bundle["request"].pack_root
    assert pack is not None
    member = next(pack.rglob("node_features.npy"))
    member.parent.chmod(0o755)
    os.link(member, member.with_name("alias.npy"))
    member.parent.chmod(0o555)
    with pytest.raises(runtime.TargetSealedError, match="aliased"):
        _prepare(bundle)

    bundle = _make_project(tmp_path / "link")
    pack = bundle["request"].pack_root
    assert pack is not None
    cache = next(pack.rglob("discovery_cache"))
    cache.chmod(0o755)
    os.symlink("node_features.npy", cache / "escape.npy")
    cache.chmod(0o555)
    with pytest.raises(runtime.TargetSealedError, match="symlink"):
        _prepare(bundle)


def test_exact_governance_role_and_source_snapshot_materials(tmp_path: Path) -> None:
    bundle = _make_project(tmp_path)
    request = bundle["request"]
    missing = dict(request.governance_files)
    missing.pop("campaign_contract")
    with pytest.raises(runtime.TargetSealedError, match="role set"):
        _prepare({**bundle, "request": replace(request, governance_files=missing)})

    source = bundle["project"] / "src/snn_rr/example.py"
    source.chmod(0o644)
    source.write_text("VALUE = 2\n")
    source.chmod(0o444)
    with pytest.raises(runtime.TargetSealedError, match="snapshot"):
        _prepare(bundle)


def test_governance_json_requires_exact_schema_self_hash_and_campaign(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path)
    path = bundle["request"].governance_files["implementation_test_receipt"]
    document = json.loads(path.read_text())

    missing_hash = dict(document)
    missing_hash.pop("content_sha256")
    with pytest.raises(runtime.TargetSealedError, match="self-hash is absent"):
        runtime._validate_governance_json(
            runtime.canonical_json_bytes(missing_hash),
            role="unregistered_direct_role",
        )

    missing_campaign = dict(document)
    missing_campaign.pop("campaign_id")
    missing_campaign.pop("content_sha256")
    missing_campaign["content_sha256"] = runtime.semantic_sha256(missing_campaign)
    with pytest.raises(runtime.TargetSealedError, match="campaign drifted"):
        runtime._validate_governance_json(
            runtime.canonical_json_bytes(missing_campaign),
            role="unregistered_direct_role",
        )

    extra = dict(document)
    extra["unrecognized"] = False
    extra.pop("content_sha256")
    extra["content_sha256"] = runtime.semantic_sha256(extra)
    with pytest.raises(runtime.TargetSealedError, match="exact schema drifted"):
        runtime._validate_governance_json(
            runtime.canonical_json_bytes(extra),
            role="implementation_test_receipt",
        )


def test_open_lifecycle_governance_roles_schema_and_cross_bindings_fail_closed(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path)
    request = bundle["request"]
    authority_role = "open_lifecycle_recovery_correction_authorization"
    diagnostic_role = "open_lifecycle_recovery_failure_diagnostic"

    missing = dict(request.governance_files)
    missing.pop(diagnostic_role)
    with pytest.raises(runtime.TargetSealedError, match="role set"):
        _prepare({**bundle, "request": replace(request, governance_files=missing)})

    for role, deleted_key in (
        (authority_role, "required_reauthorization"),
        (diagnostic_role, "status"),
    ):
        document = json.loads(request.governance_files[role].read_text())
        for mutation in ("delete", "add"):
            changed = dict(document)
            if mutation == "delete":
                changed.pop(deleted_key)
            else:
                changed["unexpected"] = False
            changed.pop("content_sha256")
            changed["content_sha256"] = runtime.semantic_sha256(changed)
            with pytest.raises(runtime.TargetSealedError, match="exact schema drifted"):
                runtime._validate_governance_json(
                    runtime.canonical_json_bytes(changed), role=role
                )

    documents: dict[str, dict[str, Any]] = {}
    bindings: dict[str, Any] = {}
    for role, path in request.governance_files.items():
        binding, raw = runtime._read_file_binding(
            path, label=role, require_immutable=True
        )
        bindings[role] = binding
        documents[role] = runtime._validate_governance_json(raw, role=role)

    for owner_role, field, expected_label in (
        (
            "implementation_test_receipt",
            authority_role,
            "test open-lifecycle recovery authority",
        ),
        (
            "source_snapshot",
            diagnostic_role,
            "snapshot open-lifecycle recovery diagnostic",
        ),
        (
            "active_authorization",
            authority_role,
            "pretrain open-lifecycle recovery authority",
        ),
    ):
        changed = json.loads(json.dumps(documents))
        changed[owner_role][field]["sha256"] = "0" * 64
        with pytest.raises(runtime.TargetSealedError, match=expected_label):
            runtime._validate_v8r4a_governance_chain(
                project_root=bundle["project"],
                documents=changed,
                bindings=bindings,
                production=False,
            )

    changed = json.loads(json.dumps(documents))
    changed[authority_role]["authority_basis"]["diagnostic"][
        "file_sha256"
    ] = "0" * 64
    with pytest.raises(
        runtime.TargetSealedError,
        match="open-lifecycle recovery diagnostic exact binding",
    ):
        runtime._validate_v8r4a_governance_chain(
            project_root=bundle["project"],
            documents=changed,
            bindings=bindings,
            production=False,
        )

    changed = json.loads(json.dumps(documents))
    changed[authority_role]["authority_basis"]["parent_kill_safe_addendum"][
        "file_sha256"
    ] = "0" * 64
    with pytest.raises(
        runtime.TargetSealedError,
        match="parent kill-safe authority exact binding",
    ):
        runtime._validate_v8r4a_governance_chain(
            project_root=bundle["project"],
            documents=changed,
            bindings=bindings,
            production=False,
        )


def test_migration_source_succession_schema_and_cross_bindings_fail_closed(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path)
    request = bundle["request"]
    authority_role = "migration_source_succession_correction_authorization"
    diagnostic_role = "migration_source_succession_failure_diagnostic"
    assert {authority_role, diagnostic_role} <= set(request.governance_files)
    for role, deleted in (
        (authority_role, "required_reauthorization"),
        (diagnostic_role, "root_cause"),
    ):
        original = json.loads(request.governance_files[role].read_text())
        for mutation in ("deletion", "addition"):
            changed = dict(original)
            if mutation == "deletion":
                changed.pop(deleted)
            else:
                changed["unexpected"] = False
            changed.pop("content_sha256")
            changed["content_sha256"] = runtime.semantic_sha256(changed)
            with pytest.raises(runtime.TargetSealedError, match="exact schema"):
                runtime._validate_governance_json(
                    runtime.canonical_json_bytes(changed), role=role
                )

    documents: dict[str, dict[str, Any]] = {}
    bindings: dict[str, Any] = {}
    for role, path in request.governance_files.items():
        binding, raw = runtime._read_file_binding(
            path, label=role, require_immutable=True
        )
        bindings[role] = binding
        documents[role] = runtime._validate_governance_json(raw, role=role)
    for owner in (
        "implementation_test_receipt",
        "source_snapshot",
        "active_authorization",
    ):
        changed = json.loads(json.dumps(documents))
        changed[owner][authority_role]["sha256"] = "0" * 64
        with pytest.raises(runtime.TargetSealedError, match="source-succession"):
            runtime._validate_v8r4a_governance_chain(
                project_root=bundle["project"],
                documents=changed,
                bindings=bindings,
                production=False,
            )
    changed = json.loads(json.dumps(documents))
    changed[authority_role]["authority_basis"]["immutable_migration_receipt"][
        "file_sha256"
    ] = "0" * 64
    with pytest.raises(runtime.TargetSealedError, match="migration receipt"):
        runtime._validate_v8r4a_governance_chain(
            project_root=bundle["project"],
            documents=changed,
            bindings=bindings,
            production=False,
        )


def test_fd_closure_governance_projection_and_cross_bindings_fail_closed(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path)
    request = bundle["request"]
    authority_role = "fd_closure_correction_authorization"
    diagnostic_role = "fd_closure_failure_diagnostic"
    documents: dict[str, dict[str, Any]] = {}
    bindings: dict[str, Any] = {}
    for role, path in request.governance_files.items():
        binding, raw = runtime._read_file_binding(
            path, label=role, require_immutable=True
        )
        bindings[role] = binding
        documents[role] = runtime._validate_governance_json(raw, role=role)

    for role, removed_key in (
        (authority_role, "required_reauthorization"),
        (diagnostic_role, "reproduction"),
    ):
        original = json.loads(request.governance_files[role].read_text())
        for mutation in ("remove", "add"):
            changed_document = dict(original)
            if mutation == "remove":
                changed_document.pop(removed_key)
            else:
                changed_document["unexpected"] = False
            changed_document.pop("content_sha256")
            changed_document["content_sha256"] = runtime.semantic_sha256(
                changed_document
            )
            with pytest.raises(runtime.TargetSealedError, match="exact schema"):
                runtime._validate_governance_json(
                    runtime.canonical_json_bytes(changed_document), role=role
                )

    for owner in (
        "implementation_test_receipt",
        "source_snapshot",
        "active_authorization",
    ):
        for role, label in (
            (authority_role, "FD-closure authority"),
            (diagnostic_role, "FD-closure diagnostic"),
        ):
            changed = json.loads(json.dumps(documents))
            changed[owner][role]["sha256"] = "0" * 64
            with pytest.raises(runtime.TargetSealedError, match=label):
                runtime._validate_v8r4a_governance_chain(
                    project_root=bundle["project"],
                    documents=changed,
                    bindings=bindings,
                    production=False,
                )

    changed = json.loads(json.dumps(documents))
    changed[authority_role]["authority_basis"]["parent_source_snapshot"][
        "file_sha256"
    ] = "0" * 64
    with pytest.raises(runtime.TargetSealedError, match="FD-closure projection"):
        runtime._validate_v8r4a_governance_chain(
            project_root=bundle["project"],
            documents=changed,
            bindings=bindings,
            production=False,
        )

    changed = json.loads(json.dumps(documents))
    changed[authority_role]["authority_basis"]["diagnostic"][
        "file_sha256"
    ] = "0" * 64
    with pytest.raises(runtime.TargetSealedError, match="FD-closure diagnostic"):
        runtime._validate_v8r4a_governance_chain(
            project_root=bundle["project"],
            documents=changed,
            bindings=bindings,
            production=False,
        )

    changed = json.loads(json.dumps(documents))
    changed[authority_role]["authority_basis"][
        "parent_execution_closure_authority"
    ]["file_sha256"] = "0" * 64
    with pytest.raises(
        runtime.TargetSealedError,
        match="FD-closure execution-closure parent",
    ):
        runtime._validate_v8r4a_governance_chain(
            project_root=bundle["project"],
            documents=changed,
            bindings=bindings,
            production=False,
        )

    literal_authority = json.loads(json.dumps(documents[authority_role]))
    literal_authority["authority_basis"]["diagnostic"] = dict(
        runtime.FD_CLOSURE_DIAGNOSTIC_LEGACY_BINDING
    )
    literal_authority["authority_basis"][
        "parent_execution_closure_authority"
    ] = dict(
        runtime.FD_CLOSURE_PARENT_BINDINGS[
            "parent_execution_closure_authority"
        ]
    )
    runtime._validate_fd_closure_projection(
        literal_authority, documents[diagnostic_role]
    )
    for basis_field in ("diagnostic", "parent_execution_closure_authority"):
        changed = json.loads(json.dumps(literal_authority))
        changed["authority_basis"][basis_field]["file_sha256"] = "0" * 64
        with pytest.raises(runtime.TargetSealedError, match="FD-closure projection"):
            runtime._validate_fd_closure_projection(
                changed, documents[diagnostic_role]
            )

    with pytest.raises(runtime.TargetSealedError, match="immutable binding"):
        runtime._validate_v8r4a_governance_chain(
            project_root=bundle["project"],
            documents=documents,
            bindings=bindings,
            production=True,
        )


def test_fd_closure_projection_never_opens_or_mounts_superseded_issuance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _make_project(tmp_path)
    project = bundle["project"]
    superseded = {
        (project / binding["path"]).resolve()
        for role, binding in runtime.FD_CLOSURE_PARENT_BINDINGS.items()
        if role != "parent_execution_closure_authority"
    }
    for path in superseded:
        _write_frozen(path, b"superseded issuance must remain unmounted\n")

    opened: set[Path] = set()
    original_open = runtime._open_pinned_file

    def tracking_open(path: Path, **kwargs: Any) -> Any:
        opened.add(Path(path).resolve())
        return original_open(path, **kwargs)

    monkeypatch.setattr(runtime, "_open_pinned_file", tracking_open)
    with _prepare(bundle) as prepared:
        mounted = {
            Path(str(row["destination"])).resolve()
            for row in prepared.mount_entries
        }
    assert superseded.isdisjoint(opened)
    assert superseded.isdisjoint(mounted)


def test_canary_boundary_governance_projection_and_cross_bindings_fail_closed(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path)
    request = bundle["request"]
    authority_role = "canary_boundary_correction_authorization"
    diagnostic_role = "canary_boundary_failure_diagnostic"
    documents: dict[str, dict[str, Any]] = {}
    bindings: dict[str, Any] = {}
    for role, path in request.governance_files.items():
        binding, raw = runtime._read_file_binding(
            path, label=role, require_immutable=True
        )
        bindings[role] = binding
        documents[role] = runtime._validate_governance_json(raw, role=role)

    for role, removed_key in (
        (authority_role, "required_reauthorization"),
        (diagnostic_role, "reproduction"),
    ):
        original = json.loads(request.governance_files[role].read_text())
        for mutation in ("remove", "add"):
            changed_document = dict(original)
            if mutation == "remove":
                changed_document.pop(removed_key)
            else:
                changed_document["unexpected"] = False
            changed_document.pop("content_sha256")
            changed_document["content_sha256"] = runtime.semantic_sha256(
                changed_document
            )
            with pytest.raises(runtime.TargetSealedError, match="exact schema"):
                runtime._validate_governance_json(
                    runtime.canonical_json_bytes(changed_document), role=role
                )

    for owner in (
        "implementation_test_receipt",
        "source_snapshot",
        "active_authorization",
    ):
        for role, label in (
            (authority_role, "canary-boundary authority"),
            (diagnostic_role, "canary-boundary diagnostic"),
        ):
            changed = json.loads(json.dumps(documents))
            changed[owner][role]["sha256"] = "0" * 64
            with pytest.raises(runtime.TargetSealedError, match=label):
                runtime._validate_v8r4a_governance_chain(
                    project_root=bundle["project"],
                    documents=changed,
                    bindings=bindings,
                    production=False,
                )

    changed = json.loads(json.dumps(documents))
    changed[authority_role]["authority_basis"]["parent_source_snapshot"][
        "file_sha256"
    ] = "0" * 64
    with pytest.raises(runtime.TargetSealedError, match="canary-boundary projection"):
        runtime._validate_v8r4a_governance_chain(
            project_root=bundle["project"],
            documents=changed,
            bindings=bindings,
            production=False,
        )

    changed = json.loads(json.dumps(documents))
    changed[authority_role]["authority_basis"]["diagnostic"][
        "file_sha256"
    ] = "0" * 64
    with pytest.raises(runtime.TargetSealedError, match="canary-boundary diagnostic"):
        runtime._validate_v8r4a_governance_chain(
            project_root=bundle["project"],
            documents=changed,
            bindings=bindings,
            production=False,
        )

    changed = json.loads(json.dumps(documents))
    changed[authority_role]["authority_basis"]["parent_fd_closure_authority"][
        "file_sha256"
    ] = "0" * 64
    with pytest.raises(
        runtime.TargetSealedError, match="canary-boundary FD-closure parent"
    ):
        runtime._validate_v8r4a_governance_chain(
            project_root=bundle["project"],
            documents=changed,
            bindings=bindings,
            production=False,
        )

    literal_authority = json.loads(json.dumps(documents[authority_role]))
    literal_authority["authority_basis"]["diagnostic"] = dict(
        runtime.CANARY_BOUNDARY_DIAGNOSTIC_LEGACY_BINDING
    )
    literal_authority["authority_basis"]["parent_fd_closure_authority"] = dict(
        runtime.CANARY_BOUNDARY_PARENT_BINDINGS["parent_fd_closure_authority"]
    )
    runtime._validate_canary_boundary_projection(
        literal_authority, documents[diagnostic_role]
    )
    for basis_field in ("diagnostic", "parent_fd_closure_authority"):
        changed = json.loads(json.dumps(literal_authority))
        changed["authority_basis"][basis_field]["file_sha256"] = "0" * 64
        with pytest.raises(
            runtime.TargetSealedError, match="canary-boundary projection"
        ):
            runtime._validate_canary_boundary_projection(
                changed, documents[diagnostic_role]
            )

    with pytest.raises(runtime.TargetSealedError, match="immutable binding"):
        runtime._validate_v8r4a_governance_chain(
            project_root=bundle["project"],
            documents=documents,
            bindings=bindings,
            production=True,
        )


def test_canary_projection_never_opens_or_mounts_parent_fd1_issuance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _make_project(tmp_path)
    project = bundle["project"]
    superseded = {
        (project / binding["path"]).resolve()
        for role, binding in runtime.CANARY_BOUNDARY_PARENT_BINDINGS.items()
        if role != "parent_fd_closure_authority"
    }
    for path in superseded:
        _write_frozen(path, b"parent FD1 issuance must remain unmounted\n")

    opened: set[Path] = set()
    original_open = runtime._open_pinned_file

    def tracking_open(path: Path, **kwargs: Any) -> Any:
        opened.add(Path(path).resolve())
        return original_open(path, **kwargs)

    monkeypatch.setattr(runtime, "_open_pinned_file", tracking_open)
    with _prepare(bundle) as prepared:
        mounted = {
            Path(str(row["destination"])).resolve()
            for row in prepared.mount_entries
        }
    assert superseded.isdisjoint(opened)
    assert superseded.isdisjoint(mounted)


def test_canary_boundary_immutable_documents_match_production_pins() -> None:
    campaign = (
        REAL_PROJECT
        / "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1"
    )
    authority_path = (
        campaign
        / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_CANARY_BOUNDARY.json"
    )
    diagnostic_path = (
        campaign
        / "diagnostics/v3r1_v8r4a_denied_canary_prefix_collision_failure.json"
    )
    authority_raw = authority_path.read_bytes()
    diagnostic_raw = diagnostic_path.read_bytes()
    assert hashlib.sha256(authority_raw).hexdigest() == (
        runtime.CANARY_BOUNDARY_AUTHORITY_FILE_SHA256
    )
    assert len(authority_raw) == runtime.CANARY_BOUNDARY_AUTHORITY_BYTES
    assert hashlib.sha256(diagnostic_raw).hexdigest() == (
        runtime.CANARY_BOUNDARY_DIAGNOSTIC_FILE_SHA256
    )
    assert len(diagnostic_raw) == runtime.CANARY_BOUNDARY_DIAGNOSTIC_BYTES
    authority = runtime._validate_governance_json(
        authority_raw, role="canary_boundary_correction_authorization"
    )
    diagnostic = runtime._validate_governance_json(
        diagnostic_raw, role="canary_boundary_failure_diagnostic"
    )
    runtime._validate_canary_boundary_projection(authority, diagnostic)


def test_frozen_contract_governance_schema_projection_and_crossbindings_fail_closed(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path)
    request = bundle["request"]
    authority_role = "frozen_contract_encoding_correction_authorization"
    diagnostic_role = "frozen_contract_encoding_failure_diagnostic"
    assert {authority_role, diagnostic_role} <= runtime.COMMON_GOVERNANCE_ROLES
    assert {
        "superseded_v8r4a_lifecycle_root",
        "superseded_v8r4a_output_root",
    } <= runtime.MANDATORY_DENIED_CANARY_ROLES

    missing = dict(request.governance_files)
    missing.pop(diagnostic_role)
    with pytest.raises(runtime.TargetSealedError, match="governance role set"):
        _prepare({**bundle, "request": replace(request, governance_files=missing)})

    for role, removed_key in (
        (authority_role, "required_reauthorization"),
        (diagnostic_role, "immutable_failure_receipts"),
    ):
        original = json.loads(request.governance_files[role].read_text())
        for mutation in ("remove", "add"):
            changed = dict(original)
            if mutation == "remove":
                changed.pop(removed_key)
            else:
                changed["unexpected"] = False
            changed.pop("content_sha256")
            changed["content_sha256"] = runtime.semantic_sha256(changed)
            with pytest.raises(runtime.TargetSealedError, match="exact schema"):
                runtime._validate_governance_json(
                    runtime.canonical_json_bytes(changed), role=role
                )

    documents: dict[str, dict[str, Any]] = {}
    bindings: dict[str, Any] = {}
    for role, path in request.governance_files.items():
        binding, raw = runtime._read_file_binding(
            path, label=role, require_immutable=True
        )
        bindings[role] = binding
        documents[role] = runtime._validate_governance_json(raw, role=role)

    assert {
        role: bindings[role].path.name
            for role in runtime.BENCHMARK_ADMITTED_CONTEXT_ACTIVE_FILENAMES
        } == dict(runtime.BENCHMARK_ADMITTED_CONTEXT_ACTIVE_FILENAMES)
    for owner in (
        "implementation_test_receipt",
        "source_snapshot",
        "active_authorization",
    ):
        for role, label in (
            (authority_role, "frozen-contract authority"),
            (diagnostic_role, "frozen-contract diagnostic"),
        ):
            changed = json.loads(json.dumps(documents))
            changed[owner][role]["sha256"] = "0" * 64
            with pytest.raises(runtime.TargetSealedError, match=label):
                runtime._validate_v8r4a_governance_chain(
                    project_root=bundle["project"],
                    documents=changed,
                    bindings=bindings,
                    production=False,
                )

    for field, label in (
        ("diagnostic", "frozen-contract diagnostic"),
        (
            "parent_canary_boundary_authority",
            "canary-boundary parent authority",
        ),
        (
            "parent_canary_boundary_diagnostic",
            "canary-boundary parent diagnostic",
        ),
        ("frozen_campaign_contract", "campaign contract"),
    ):
        changed = json.loads(json.dumps(documents))
        changed[authority_role]["authority_basis"][field]["file_sha256"] = (
            "0" * 64
        )
        with pytest.raises(runtime.TargetSealedError, match=label):
            runtime._validate_v8r4a_governance_chain(
                project_root=bundle["project"],
                documents=changed,
                bindings=bindings,
                production=False,
            )

    changed_authority = json.loads(json.dumps(documents[authority_role]))
    changed_authority["forbidden_changes"]["commercial_or_confirmatory_claim"] = (
        False
    )
    with pytest.raises(runtime.TargetSealedError, match="frozen-contract projection"):
        runtime._validate_frozen_contract_encoding_projection(
            changed_authority,
            documents[diagnostic_role],
            enforce_production_literal_bindings=False,
        )

    changed_diagnostic = json.loads(json.dumps(documents[diagnostic_role]))
    changed_diagnostic["immutable_failure_receipts"]["completion_receipt"][
        "return_code"
    ] = 0
    with pytest.raises(runtime.TargetSealedError, match="failed replay"):
        runtime._validate_frozen_contract_encoding_projection(
            documents[authority_role],
            changed_diagnostic,
            enforce_production_literal_bindings=False,
        )


def test_frozen_contract_immutable_documents_and_failed_receipts_match_pins() -> None:
    campaign = (
        REAL_PROJECT
        / "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1"
    )
    authority_path = (
        campaign
        / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_FROZEN_CONTRACT_ENCODING.json"
    )
    diagnostic_path = (
        campaign
        / "diagnostics/v3r1_v8r4a_frozen_contract_encoding_false_rejection_failure.json"
    )
    authority_raw = authority_path.read_bytes()
    diagnostic_raw = diagnostic_path.read_bytes()
    assert hashlib.sha256(authority_raw).hexdigest() == (
        runtime.FROZEN_CONTRACT_AUTHORITY_FILE_SHA256
    )
    assert len(authority_raw) == runtime.FROZEN_CONTRACT_AUTHORITY_BYTES
    assert hashlib.sha256(diagnostic_raw).hexdigest() == (
        runtime.FROZEN_CONTRACT_DIAGNOSTIC_FILE_SHA256
    )
    assert len(diagnostic_raw) == runtime.FROZEN_CONTRACT_DIAGNOSTIC_BYTES
    authority = runtime._validate_governance_json(
        authority_raw, role="frozen_contract_encoding_correction_authorization"
    )
    diagnostic = runtime._validate_governance_json(
        diagnostic_raw, role="frozen_contract_encoding_failure_diagnostic"
    )
    runtime._validate_frozen_contract_encoding_projection(authority, diagnostic)

    for role in ("failed_capability_receipt", "failed_completion_receipt"):
        expected = runtime.FROZEN_CONTRACT_PARENT_BINDINGS[role]
        path = REAL_PROJECT / expected["path"]
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected["file_sha256"]
        assert len(raw) == expected["bytes"]
        assert stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) == 0o444
    completion = json.loads(
        (
            REAL_PROJECT
            / runtime.FROZEN_CONTRACT_PARENT_BINDINGS[
                "failed_completion_receipt"
            ]["path"]
        ).read_text()
    )
    assert completion["return_code"] == 1


def test_contract1_projection_never_opens_or_mounts_old_chain_receipts_or_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _make_project(tmp_path)
    project = bundle["project"]
    old_files = {
        (project / binding["path"]).resolve()
        for role, binding in runtime.FROZEN_CONTRACT_PARENT_BINDINGS.items()
        if role
        in {
            "parent_implementation_test_receipt",
            "parent_source_snapshot",
            "parent_pretrain_authorization",
            "failed_capability_receipt",
            "failed_completion_receipt",
        }
    }
    for path in old_files:
        _write_frozen(path, b"immutable superseded audit evidence\n")
    old_roots = {
        (project / runtime.SUPERSEDED_V8R4A_LIFECYCLE_ROOT_RELATIVE).resolve(),
        (project / runtime.SUPERSEDED_V8R4A_OUTPUT_ROOT_RELATIVE).resolve(),
    }
    for path in old_roots:
        path.mkdir(parents=True, exist_ok=True)

    opened_files: set[Path] = set()
    read_files: set[Path] = set()
    opened_directories: set[Path] = set()
    original_file_open = runtime._open_pinned_file
    original_file_read = runtime._read_file_binding
    original_directory_open = runtime._open_pinned_directory

    def tracking_file_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        opened_files.add(Path(path).resolve())
        return original_file_open(path, *args, **kwargs)

    def tracking_file_read(path: Path, *args: Any, **kwargs: Any) -> Any:
        read_files.add(Path(path).resolve())
        return original_file_read(path, *args, **kwargs)

    def tracking_directory_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        opened_directories.add(Path(path).resolve())
        return original_directory_open(path, *args, **kwargs)

    monkeypatch.setattr(runtime, "_open_pinned_file", tracking_file_open)
    monkeypatch.setattr(runtime, "_read_file_binding", tracking_file_read)
    monkeypatch.setattr(runtime, "_open_pinned_directory", tracking_directory_open)
    with _prepare(bundle) as prepared:
        destinations = {
            Path(str(row["destination"])).resolve()
            for row in prepared.mount_entries
            if str(row["destination"]).startswith("/")
        }
        assert prepared.receipt["denied_canaries"][
            "superseded_v8r4a_lifecycle_root"
        ] == str(next(path for path in old_roots if "lifecycle" in path.name))
        assert prepared.receipt["denied_canaries"][
            "superseded_v8r4a_output_root"
        ] == str(next(path for path in old_roots if "benchmark" in path.name))
    assert old_files.isdisjoint(opened_files)
    assert old_files.isdisjoint(read_files)
    assert old_files.isdisjoint(destinations)
    assert old_roots.isdisjoint(opened_directories)
    assert old_roots.isdisjoint(destinations)

    active_roots = (
        bundle["writable"]["output"],
        bundle["writable"]["lifecycle"],
    )
    denied = {
        role: str(path)
        for role, path in bundle["request"].denied_canaries.items()
    }
    runtime._validate_command_denied_canaries(
        tuple(str(path) for path in active_roots), denied_canaries=denied
    )
    for role in (
        "superseded_v8r4a_lifecycle_root",
        "superseded_v8r4a_output_root",
    ):
        denied_path = bundle["request"].denied_canaries[role]
        for path in (denied_path, denied_path / "descendant"):
            with pytest.raises(runtime.TargetSealedError, match="denied capability"):
                runtime._validate_command_denied_canaries(
                    (str(path),), denied_canaries=denied
                )


def test_context1_target_projection_never_opens_parent_chain_or_superseded_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _make_project(tmp_path)
    project = bundle["project"]
    prohibited_files = {
        project / runtime.GPU_STATE_PARENT_BIND_PARENT_BINDINGS[field]["path"]
        for field in (
            "parent_implementation_test_receipt",
            "parent_source_snapshot",
            "parent_pretrain_authorization",
            "failed_contract1_capability_receipt",
            "failed_contract1_completion_receipt",
        )
    }
    prohibited_files.update(
        project / row["path"]
        for row in runtime.BENCHMARK_ADMITTED_CONTEXT_PINNED_UNMOUNTED_BASIS.values()
    )
    for path in prohibited_files:
        _write_frozen(path, b"immutable parent evidence must remain unmounted\n")
    superseded_roots = {
        project / runtime.SUPERSEDED_V8R4A_LIFECYCLE_ROOT_RELATIVE,
        project / runtime.SUPERSEDED_V8R4A_OUTPUT_ROOT_RELATIVE,
        project / runtime.SUPERSEDED_V8R4A_CONTRACT1_LIFECYCLE_ROOT_RELATIVE,
        project / runtime.SUPERSEDED_V8R4A_CONTRACT1_OUTPUT_ROOT_RELATIVE,
        project / runtime.SUPERSEDED_V8R4A_ROOTBIND1_LIFECYCLE_ROOT_RELATIVE,
        project / runtime.SUPERSEDED_V8R4A_ROOTBIND1_OUTPUT_ROOT_RELATIVE,
    }
    opened_files: set[Path] = set()
    opened_directories: set[Path] = set()
    original_file = runtime._open_pinned_file
    original_directory = runtime._open_pinned_directory

    def track_file(path: Path, **kwargs: Any) -> Any:
        opened_files.add(Path(path).resolve())
        return original_file(path, **kwargs)

    def track_directory(path: Path, **kwargs: Any) -> Any:
        opened_directories.add(Path(path).resolve())
        return original_directory(path, **kwargs)

    monkeypatch.setattr(runtime, "_open_pinned_file", track_file)
    monkeypatch.setattr(runtime, "_open_pinned_directory", track_directory)
    with _prepare(bundle) as prepared:
        destinations = {
            Path(str(row["destination"])).resolve()
            for row in prepared.mount_entries
        }
        assert {
            "gpu_state_parent_bind_correction_authorization",
                "gpu_state_parent_bind_failure_diagnostic",
                "admitted_context_correction_authorization",
                "admitted_context_failure_diagnostic",
        } <= set(prepared.receipt["governance_files"])
        assert {
            role
            for role in prepared.receipt["denied_canaries"]
            if role.startswith("superseded_v8r4a")
        } == {
            "superseded_v8r4a_lifecycle_root",
            "superseded_v8r4a_output_root",
            "superseded_v8r4a_contract1_lifecycle_root",
                "superseded_v8r4a_contract1_output_root",
                "superseded_v8r4a_rootbind1_lifecycle_root",
                "superseded_v8r4a_rootbind1_output_root",
            }
    assert prohibited_files.isdisjoint(opened_files)
    assert superseded_roots.isdisjoint(opened_directories)
    assert superseded_roots.isdisjoint(destinations)


def test_context1_projection_schema_bindings_and_generation_fail_closed(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path)
    request = bundle["request"]
    authority_role = "admitted_context_correction_authorization"
    diagnostic_role = "admitted_context_failure_diagnostic"
    assert {authority_role, diagnostic_role} <= runtime.COMMON_GOVERNANCE_ROLES
    assert {
        "superseded_v8r4a_rootbind1_lifecycle_root",
        "superseded_v8r4a_rootbind1_output_root",
    } <= runtime.MANDATORY_DENIED_CANARY_ROLES

    authority = runtime._validate_governance_json(
        request.governance_files[authority_role].read_bytes(),
        role=authority_role,
    )
    diagnostic = runtime._validate_governance_json(
        request.governance_files[diagnostic_role].read_bytes(),
        role=diagnostic_role,
    )
    runtime._validate_benchmark_admitted_context_projection(
        authority,
        diagnostic,
        enforce_production_literal_bindings=False,
    )
    changed = json.loads(json.dumps(authority))
    changed["mandatory_invariants"]["active_benchmark_context"][
        "authorization_generation"
    ] = "ROOTBIND1"
    with pytest.raises(runtime.TargetSealedError, match="authorization boundary"):
        runtime._validate_benchmark_admitted_context_projection(
            changed,
            diagnostic,
            enforce_production_literal_bindings=False,
        )
    changed_diagnostic = json.loads(json.dumps(diagnostic))
    changed_diagnostic["ledger_evidence"]["execution_postlaunch"][
        "open_start_count"
    ] = 1
    with pytest.raises(runtime.TargetSealedError, match="failure projection"):
        runtime._validate_benchmark_admitted_context_projection(
            authority,
            changed_diagnostic,
            enforce_production_literal_bindings=False,
        )

    documents: dict[str, dict[str, Any]] = {}
    bindings: dict[str, Any] = {}
    for role, path in request.governance_files.items():
        binding, raw = runtime._read_file_binding(
            path, label=role, require_immutable=True
        )
        bindings[role] = binding
        documents[role] = runtime._validate_governance_json(raw, role=role)
    tampered = json.loads(json.dumps(documents))
    tampered["active_authorization"][authority_role]["sha256"] = "0" * 64
    with pytest.raises(runtime.TargetSealedError, match="admitted-context authority"):
        runtime._validate_v8r4a_governance_chain(
            project_root=bundle["project"],
            documents=tampered,
            bindings=bindings,
            production=False,
        )
    wrong_scope = json.loads(json.dumps(documents))
    wrong_scope["active_authorization"]["efficiency_benchmark_scope"][
        "authorization_generation"
    ] = "ROOTBIND1"
    with pytest.raises(runtime.TargetSealedError, match="pretrain production"):
        runtime._validate_v8r4a_governance_chain(
            project_root=bundle["project"],
            documents=wrong_scope,
            bindings=bindings,
            production=False,
        )

    trio = (
        "implementation_test_receipt",
        "source_snapshot",
        "active_authorization",
    )
    for role in trio:
        assert "authorization_generation" in runtime.GOVERNANCE_TOP_LEVEL_KEYS[
            role
        ]
        assert documents[role]["authorization_generation"] == "CONTEXT1"
        missing = json.loads(json.dumps(documents[role]))
        missing.pop("authorization_generation")
        missing.pop("content_sha256")
        missing["content_sha256"] = runtime.semantic_sha256(missing)
        with pytest.raises(runtime.TargetSealedError, match="exact schema drifted"):
            runtime._validate_governance_json(
                runtime.canonical_json_bytes(missing), role=role
            )
        for generation in ("ROOTBIND1", "CONTEXT2", 1):
            changed_document = json.loads(json.dumps(documents[role]))
            changed_document["authorization_generation"] = generation
            changed_document.pop("content_sha256")
            changed_document["content_sha256"] = runtime.semantic_sha256(
                changed_document
            )
            rehashed = runtime._validate_governance_json(
                runtime.canonical_json_bytes(changed_document), role=role
            )
            changed_documents = dict(documents)
            changed_documents[role] = rehashed
            with pytest.raises(
                runtime.TargetSealedError,
                match="authorization generation is not exact CONTEXT1",
            ):
                runtime._validate_v8r4a_governance_chain(
                    project_root=bundle["project"],
                    documents=changed_documents,
                    bindings=bindings,
                    production=False,
                )


def test_context1_immutable_authority_and_diagnostic_match_exact_pins() -> None:
    campaign = (
        REAL_PROJECT
        / "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1"
    )
    authority_path = (
        campaign
        / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_BENCHMARK_ADMITTED_CONTEXT.json"
    )
    diagnostic_path = (
        campaign
        / "diagnostics/v3r1_v8r4a_benchmark_admitted_pretrain_context_bridge_failure.json"
    )
    authority_raw = authority_path.read_bytes()
    diagnostic_raw = diagnostic_path.read_bytes()
    assert hashlib.sha256(authority_raw).hexdigest() == (
        runtime.BENCHMARK_ADMITTED_CONTEXT_AUTHORITY_FILE_SHA256
    )
    assert len(authority_raw) == runtime.BENCHMARK_ADMITTED_CONTEXT_AUTHORITY_BYTES
    assert hashlib.sha256(diagnostic_raw).hexdigest() == (
        runtime.BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_FILE_SHA256
    )
    assert len(diagnostic_raw) == runtime.BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_BYTES
    authority = runtime._validate_governance_json(
        authority_raw, role="admitted_context_correction_authorization"
    )
    diagnostic = runtime._validate_governance_json(
        diagnostic_raw, role="admitted_context_failure_diagnostic"
    )
    runtime._validate_benchmark_admitted_context_projection(authority, diagnostic)


def test_context1_rootbind1_sibling_is_allowed_but_exact_denied_root_fails(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path)
    request = bundle["request"]
    active = request.writable_roots["output"]
    denied = request.denied_canaries["superseded_v8r4a_rootbind1_output_root"]
    assert active.name == "efficiency_benchmark_v8r4a_context1"
    assert denied.name == "efficiency_benchmark_v8r4a_rootbind1"
    canaries = {role: str(path) for role, path in request.denied_canaries.items()}
    runtime._validate_command_denied_canaries(
        (str(active),), denied_canaries=canaries
    )
    for path in (denied, denied / "descendant"):
        with pytest.raises(runtime.TargetSealedError, match="denied capability"):
            runtime._validate_command_denied_canaries(
                (str(path),), denied_canaries=canaries
            )


def test_active_context1_sibling_prefixes_are_not_denied_capabilities(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path)
    request = bundle["request"]
    active_output = request.writable_roots["output"]
    assert active_output.name == "efficiency_benchmark_v8r4a_context1"
    denied_v8 = request.denied_canaries["other_output_root"]
    denied_v8r4a = request.denied_canaries["superseded_v8r4a_output_root"]
    denied_rootbind1 = request.denied_canaries[
        "superseded_v8r4a_rootbind1_output_root"
    ]
    assert str(denied_v8) in str(active_output)
    assert str(denied_v8r4a) in str(active_output)
    assert str(denied_rootbind1.parent) in str(active_output.parent)
    execution_authority = json.loads(
        request.governance_files[
            "execution_closure_correction_authorization"
        ].read_text()
    )
    historical = execution_authority["authority_basis"][
        "historical_benchmark_prefix"
    ]
    assert historical["active_output_root"] == (
        runtime.LEGACY_EXECUTION_CLOSURE_BENCHMARK_OUTPUT_RELATIVE.as_posix()
    )
    runtime._validate_execution_closure_historical_projection(
        execution_authority
    )
    changed = json.loads(json.dumps(execution_authority))
    changed["authority_basis"]["historical_benchmark_prefix"][
        "active_output_root"
    ] = runtime.BENCHMARK_OUTPUT_RELATIVE.as_posix()
    with pytest.raises(runtime.TargetSealedError, match="historical benchmark"):
        runtime._validate_execution_closure_historical_projection(changed)
    command = (
        *request.command,
        "--output-root",
        str(active_output),
    )
    with runtime.prepare_runtime(
        replace(request, command=command),
        launcher_path=bundle["launcher"],
    ):
        pass


def test_infrastructure_diagnostic_is_distinct_and_exact_bound(tmp_path: Path) -> None:
    bundle = _make_project(tmp_path, phase="discovery")
    request = bundle["request"]
    assert (
        request.governance_files["failure_diagnostic"]
        != request.governance_files["infrastructure_failure_diagnostic"]
    )
    assert (
        request.governance_files["source_closure_failure_diagnostic"]
        != request.governance_files["infrastructure_failure_diagnostic"]
    )
    documents: dict[str, dict[str, Any]] = {}
    bindings: dict[str, Any] = {}
    for role, path in request.governance_files.items():
        if role.startswith("quarantined_material_"):
            continue
        binding, raw = runtime._read_file_binding(
            path, label=role, require_immutable=True
        )
        bindings[role] = binding
        documents[role] = runtime._validate_governance_json(raw, role=role)
    historical_documents = json.loads(json.dumps(documents))
    historical_documents["correction_authorization"]["diagnostic"][
        "file_sha256"
    ] = "0" * 64
    with pytest.raises(runtime.TargetSealedError, match="V8R4 authority diagnostic"):
        runtime._validate_v8r4a_governance_chain(
            project_root=bundle["project"],
            documents=historical_documents,
            bindings=bindings,
            production=False,
        )

    contract_documents = json.loads(json.dumps(documents))
    contract_path = runtime.CAMPAIGN_CONTRACT_RELATIVE_PATH.as_posix()
    contract_documents["source_snapshot"]["implementation_files"] = [
        row
        for row in contract_documents["source_snapshot"]["implementation_files"]
        if row["path"] != contract_path
    ]
    with pytest.raises(runtime.TargetSealedError, match="campaign contract cover"):
        runtime._validate_v8r4a_governance_chain(
            project_root=bundle["project"],
            documents=contract_documents,
            bindings=bindings,
            production=False,
        )

    documents["infrastructure_correction_authorization"] = dict(
        documents["infrastructure_correction_authorization"]
    )
    diagnostic = dict(
        documents["infrastructure_correction_authorization"]["diagnostic"]
    )
    diagnostic["file_sha256"] = "0" * 64
    documents["infrastructure_correction_authorization"]["diagnostic"] = diagnostic
    with pytest.raises(runtime.TargetSealedError, match="diagnostic exact binding"):
        runtime._validate_v8r4a_governance_chain(
            project_root=bundle["project"],
            documents=documents,
            bindings=bindings,
            production=False,
        )

    dependency_documents = json.loads(json.dumps(documents))
    dependency_documents["infrastructure_correction_authorization"] = json.loads(
        request.governance_files[
            "infrastructure_correction_authorization"
        ].read_text()
    )
    dependency_basis = dependency_documents[
        "source_closure_dependency_authorization"
    ]["authority_basis"]
    dependency_basis["parent_source_closure_addendum"]["file_sha256"] = "0" * 64
    with pytest.raises(
        runtime.TargetSealedError,
        match="source dependency parent authority exact binding",
    ):
        runtime._validate_v8r4a_governance_chain(
            project_root=bundle["project"],
            documents=dependency_documents,
            bindings=bindings,
            production=False,
        )

    diagnostic_documents = json.loads(json.dumps(documents))
    diagnostic_documents["infrastructure_correction_authorization"] = json.loads(
        request.governance_files[
            "infrastructure_correction_authorization"
        ].read_text()
    )
    diagnostic_documents["implementation_test_receipt"][
        "kill_safe_failure_diagnostic"
    ]["sha256"] = "0" * 64
    with pytest.raises(runtime.TargetSealedError, match="test kill-safe diagnostic"):
        runtime._validate_v8r4a_governance_chain(
            project_root=bundle["project"],
            documents=diagnostic_documents,
            bindings=bindings,
            production=False,
        )


@pytest.mark.parametrize("defect", ["extra_entry", "wrong_mode", "root_extra"])
def test_migrated_state_inventory_and_mode_defects_fail_closed(
    tmp_path: Path, defect: str
) -> None:
    bundle = _make_project(tmp_path)
    if defect == "extra_entry":
        (bundle["writable"]["usage"] / "unexpected").write_bytes(b"")
    elif defect == "wrong_mode":
        bundle["writable"]["execution"].chmod(0o755)
    else:
        extra = bundle["writable"]["usage"].parent / "other"
        extra.mkdir()
        extra.chmod(0o700)
    with pytest.raises(runtime.TargetSealedError, match="live validation|directory"):
        _prepare(bundle)


def test_state_directories_must_be_canonical_and_may_not_be_aliased(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path)
    request = bundle["request"]
    writable = dict(request.writable_roots)
    writable["usage"] = writable["execution"]
    with pytest.raises(runtime.TargetSealedError, match="canonical V8R4A"):
        _prepare({**bundle, "request": replace(request, writable_roots=writable)})


def test_discovery_quarantine_exact_eleven_file_cover(tmp_path: Path) -> None:
    bundle = _make_project(tmp_path, phase="discovery")
    with _prepare(bundle) as prepared:
        assert all(
            f"quarantined_material_{number:02d}" in prepared.governance_bindings
            for number in range(11)
        )
    material = bundle["request"].governance_files["quarantined_material_08"]
    material.chmod(0o644)
    material.write_bytes(b"tamper")
    material.chmod(0o444)
    with pytest.raises(runtime.TargetSealedError, match="quarantined material"):
        _prepare(bundle)


def test_aggregation_is_pack_free_and_safe_partial_output_can_resume(tmp_path: Path) -> None:
    bundle = _make_project(tmp_path, phase="discovery_aggregation")
    assert bundle["request"].pack_root is None
    with _prepare(bundle) as prepared:
        assert prepared.pack_binding is None
        assert "discovery_shard_seal_outer3" in prepared.governance_bindings
        assert "discovery_shard_seal_outer4" in prepared.governance_bindings
    _write_frozen(bundle["writable"]["output"] / "partial.json", b"{}\n")
    with _prepare(bundle) as resumed:
        assert any(
            row["path"] == "partial.json"
            for row in resumed.prelaunch_output_inventory
        )


def test_phase_outer_entry_output_and_lifecycle_topology_is_exact() -> None:
    cases = (
            (
                "efficiency_benchmark", 3, "benchmark_hfr_v3r1_efficiency.py",
                "efficiency_benchmark_v8r4a_context1", "outer_3",
        ),
        (
            "discovery", 4, "run_hfr_v3r1_discovery_campaign.py",
            "discovery_v8r4/shards/outer_4", "outer_4",
        ),
        (
            "promotion_training", 5, "run_fixed_hfr_v3r1_oof_campaign.py",
            "fixed_oof_v8r4/promotion_training_shards/outer_5", "outer_5",
        ),
        (
            "promotion_prediction", 2, "run_fixed_hfr_v3r1_oof_campaign.py",
            "fixed_oof_v8r4/prediction_shards/outer_2", "outer_2",
        ),
        (
            "discovery_aggregation", None, "run_hfr_v3r1_discovery_campaign.py",
            "discovery_v8r4/aggregation_v8r4a", "global",
        ),
        (
            "promotion_aggregation", None, "run_fixed_hfr_v3r1_oof_campaign.py",
            "fixed_oof_v8r4/aggregation_v8r4a", "global",
        ),
    )
    for phase, outer, entry, output_suffix, scope in cases:
        output = runtime._canonical_output_relative(
            phase=phase, outer_fold=outer, entry_name=entry
        )
        lifecycle = runtime._canonical_lifecycle_relative(
            phase=phase, outer_fold=outer, entry_name=entry
        )
        assert output.as_posix().endswith(output_suffix)
        assert lifecycle == (
            runtime.TARGET_LIFECYCLE_ROOT_RELATIVE
            / phase
            / Path(entry).stem
            / scope
        )
    assert {
        "discovery_shard_seal_outer3",
        "discovery_shard_seal_outer4",
    } == runtime.DISCOVERY_AGGREGATION_GOVERNANCE_ROLES
    assert {
        *(f"model_source_seal_outer{outer}" for outer in range(6)),
        *(f"prediction_shard_seal_outer{outer}" for outer in range(6)),
    } == runtime.FIXED_AGGREGATION_GOVERNANCE_ROLES
    assert {
        "canary_boundary_correction_authorization",
        "canary_boundary_failure_diagnostic",
    } <= runtime.COMMON_GOVERNANCE_ROLES
    promotion_roles = runtime._governance_roles_for(
        phase="promotion_aggregation",
        entry_name="run_fixed_hfr_v3r1_oof_campaign.py",
    )
    assert {"selection_lock", "promotion_authorization"} <= promotion_roles
    assert promotion_roles == (
        runtime.COMMON_GOVERNANCE_ROLES
        | runtime.FIXED_AGGREGATION_GOVERNANCE_ROLES
        | {"selection_lock", "promotion_authorization"}
    )


def test_promotion_training_refuses_reused_discovery_folds(tmp_path: Path) -> None:
    bundle = _make_project(tmp_path, phase="promotion_training")
    request = bundle["request"]
    with pytest.raises(runtime.TargetSealedError, match="authorized set"):
        runtime.prepare_runtime(
            replace(request, outer_fold=3), launcher_path=bundle["launcher"]
        )


@pytest.mark.parametrize(
    "defect", ("extra_key", "duplicate_unit", "wrong_outer", "contract_binding")
)
def test_discovery_aggregation_rejects_nonproducer_shard_schema(
    tmp_path: Path, defect: str
) -> None:
    bundle = _make_project(tmp_path, phase="discovery_aggregation")
    path = bundle["request"].governance_files["discovery_shard_seal_outer3"]
    document = json.loads(path.read_text())
    if defect == "extra_key":
        document["unexpected"] = False
    elif defect == "duplicate_unit":
        document["units"][1] = dict(document["units"][0])
    elif defect == "contract_binding":
        document["contract"]["sha256"] = "0" * 64
    else:
        document["outer_fold_shard"] = 4
    document.pop("content_sha256")
    document["content_sha256"] = runtime.semantic_sha256(document)
    path.chmod(0o644)
    path.write_bytes(runtime.canonical_json_bytes(document) + b"\n")
    path.chmod(0o444)
    with pytest.raises(runtime.TargetSealedError, match="aggregation"):
        _prepare(bundle)


def test_anonymous_create_once_publication_has_no_named_temp_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "lifecycle"
    parent.mkdir(mode=0o700)
    target = parent / "receipt.json"
    document = _json_document(
        schema_version=1,
        classification="anonymous-publication-regression",
        campaign_id=runtime.CAMPAIGN_ID,
    )
    observed: list[tuple[int, int, tuple[str, ...]]] = []
    real_link = runtime._linkat_empty_path

    def audited_link(source_fd: int, directory_fd: int, name: str) -> None:
        status = os.fstat(source_fd)
        observed.append(
            (status.st_nlink, stat.S_IMODE(status.st_mode), tuple(os.listdir(parent)))
        )
        real_link(source_fd, directory_fd, name)

    monkeypatch.setattr(runtime, "_linkat_empty_path", audited_link)
    first = runtime._create_once_immutable_json(target, document)
    assert observed == [(0, 0o444, ())]
    assert first.mode == 0o444
    assert target.stat().st_nlink == 1
    assert set(os.listdir(parent)) == {target.name}
    second = runtime._create_once_immutable_json(target, document)
    assert second == first
    assert set(os.listdir(parent)) == {target.name}
    changed = dict(document)
    changed["classification"] = "different"
    with pytest.raises(runtime.TargetSealedError, match="resume bytes differ"):
        runtime._create_once_immutable_json(target, changed)


def test_anonymous_prelink_failure_leaves_no_cleanup_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "lifecycle"
    parent.mkdir(mode=0o700)

    def fail_before_link(_source: int, _directory: int, _name: str) -> None:
        raise RuntimeError("fault before anonymous link")

    monkeypatch.setattr(runtime, "_linkat_empty_path", fail_before_link)
    with pytest.raises(RuntimeError, match="fault before anonymous link"):
        runtime._create_once_immutable_json(
            parent / "receipt.json",
            _json_document(
                schema_version=1,
                classification="prelink-fault",
                campaign_id=runtime.CAMPAIGN_ID,
            ),
        )
    assert list(parent.iterdir()) == []


def _install_valid_state_replace_residues(bundle: Mapping[str, Any]) -> tuple[Path, Path]:
    usage_target = (
        bundle["writable"]["usage"] / "campaign_gpu_usage_chain_v6.jsonl"
    )
    execution_target = (
        bundle["writable"]["execution"] / "gpu_execution_ledger_v7.jsonl"
    )
    execution_current = execution_target.read_bytes()
    execution_record = {
        "schema_version": 1,
        "job_id": "killed-before-replace",
        "event": "start",
        "utc": "2026-08-29T00:00:00+00:00",
    }
    usage_residue = usage_target.with_name(
        f".{usage_target.name}.v8r4a-replace.tmp"
    )
    execution_residue = execution_target.with_name(
        f".{execution_target.name}.v8r4a-replace.tmp"
    )
    execution_residue.write_bytes(
        execution_current
        + runtime.canonical_json_bytes(execution_record)
        + b"\n"
    )
    execution_residue.chmod(0o644)
    return usage_residue, execution_residue


def test_exact_state_replace_residues_are_cleaned_before_strict_validation(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path)
    usage_residue, execution_residue = _install_valid_state_replace_residues(bundle)
    with _prepare(bundle) as prepared:
        assert prepared.prelaunch_state.usage_state["open_reservation_count"] == 0
        assert prepared.prelaunch_state.execution_state["open_start_count"] == 0
    assert not usage_residue.exists()
    assert not execution_residue.exists()
    with _prepare(bundle):
        pass


@pytest.mark.parametrize("defect", ("non_successor", "wrong_mode", "symlink"))
def test_state_replace_residue_tamper_is_refused_without_mutation(
    tmp_path: Path, defect: str
) -> None:
    bundle = _make_project(tmp_path)
    usage = bundle["writable"]["usage"] / "campaign_gpu_usage_chain_v6.jsonl"
    residue = usage.with_name(f".{usage.name}.v8r4a-replace.tmp")
    payload = b"not-a-successor\n"
    if defect == "symlink":
        outside = tmp_path / "outside"
        outside.write_bytes(payload)
        outside.chmod(0o644)
        residue.symlink_to(outside)
    else:
        residue.write_bytes(payload)
        residue.chmod(0o444 if defect == "wrong_mode" else 0o644)
    before = usage.read_bytes()
    with pytest.raises(runtime.TargetSealedError, match="residue|recovery"):
        _prepare(bundle)
    assert usage.read_bytes() == before
    assert residue.exists() or residue.is_symlink()


def _fixed_aggregation_documents(
    tmp_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    bindings: dict[str, Any] = {}
    portable = {"path": "opaque", "sha256": "a" * 64, "bytes": 1}
    selection_path = _write_json(
        tmp_path / "selection.json",
        _json_document(schema_version=1, classification="selection"),
    )
    authorization_path = _write_json(
        tmp_path / "promotion.json",
        _json_document(schema_version=1, classification="promotion"),
    )
    selection_binding, _ = runtime._read_file_binding(
        selection_path, label="selection_lock", require_immutable=True
    )
    authorization_binding, _ = runtime._read_file_binding(
        authorization_path,
        label="promotion_authorization",
        require_immutable=True,
    )
    bindings["selection_lock"] = selection_binding
    bindings["promotion_authorization"] = authorization_binding
    selection_portable = _authorization_binding(selection_path)
    authorization_portable = _authorization_binding(authorization_path)
    for outer in range(6):
        cache_sha256 = f"{outer:x}" * 64
        model_role = f"model_source_seal_outer{outer}"
        model = _json_document(
            schema_version=1,
            classification="adaptive_v3r1_v8r4a_model_source_shard_seal",
            campaign_id=runtime.CAMPAIGN_ID,
            campaign_revision="V8R4",
            infrastructure_revision="V8R4A",
            outer_fold=outer,
            seeds=list(runtime.SEEDS),
            selected_variant="H1_factor",
            unit_count=3,
            exact_three_seed_cover=True,
            selection_lock=selection_portable,
            promotion_authorization=authorization_portable,
            units=[
                {
                    "outer_fold": outer,
                    "seed": seed,
                    "source_kind": "local_training",
                    "scientific_signature_sha256": "b" * 64,
                    "row_count": 11 + outer,
                    "global_cache_index_sha256": cache_sha256,
                    "model_bound_prediction_pack_manifest": portable,
                    "model_checkpoint": portable,
                    "model_scaler": portable,
                    "model_source_capability": portable,
                }
                for seed in runtime.SEEDS
            ],
            target_or_prediction_values_present=False,
            source_paths_or_peer_outputs_authorized_in_child=False,
            commercial_or_confirmatory_claim_allowed=False,
        )
        model_path = _write_json(tmp_path / f"model_{outer}.json", model)
        model_binding, _ = runtime._read_file_binding(
            model_path, label=model_role, require_immutable=True
        )
        documents[model_role] = model
        bindings[model_role] = model_binding
        prediction_role = f"prediction_shard_seal_outer{outer}"
        prediction = _json_document(
            schema_version=1,
            classification="adaptive_v3r1_v8r4a_prediction_shard_completion_seal",
            campaign_id=runtime.CAMPAIGN_ID,
            campaign_revision="V8R4",
            infrastructure_revision="V8R4A",
            outer_fold=outer,
            seeds=list(runtime.SEEDS),
            selected_variant="H1_factor",
            selected_release_mode="bounded",
            unit_count=3,
            exact_three_seed_cover=True,
            row_count_per_seed=11 + outer,
            cache_index_sha256=cache_sha256,
            prediction_pack_index=portable,
            model_source_shard_seal={
                "path": str(model_binding.path),
                "sha256": model_binding.sha256,
                "bytes": model_binding.bytes,
            },
            selection_lock=selection_portable,
            promotion_authorization=authorization_portable,
            gpu_usage_ledger_prefix={
                "path": "/state/usage.jsonl",
                "sha256": "c" * 64,
                "bytes": 1,
                "records": 1,
                "terminal_record_sha256": "d" * 64,
                "settled_usage_ns": 1,
                "elapsed_seconds": 1e-9,
                "open_reservations": 0,
            },
            units=[
                {
                    "outer_fold": outer,
                    "seed": seed,
                    "completion_receipt": portable,
                    "promotion_model_source": portable,
                    "rows": 11 + outer,
                    "cache_index_sha256": cache_sha256,
                    "prediction": portable,
                    "prediction_manifest": portable,
                }
                for seed in runtime.SEEDS
            ],
            target_fields_accessed_or_emitted=False,
            ready_for_pack_free_promotion_aggregation=True,
            commercial_claim_authorized=False,
        )
        prediction_path = _write_json(
            tmp_path / f"prediction_{outer}.json", prediction
        )
        prediction_binding, _ = runtime._read_file_binding(
            prediction_path, label=prediction_role, require_immutable=True
        )
        documents[prediction_role] = prediction
        bindings[prediction_role] = prediction_binding
    return documents, bindings


@pytest.mark.parametrize(
    "defect",
    (
        None,
        "extra_key",
        "wrong_model_binding",
        "wrong_authorization_binding",
        "duplicate_seed",
        "target_flag",
    ),
)
def test_fixed_aggregation_validates_exact_real_producer_seals_without_peer_roots(
    tmp_path: Path, defect: str | None
) -> None:
    documents, bindings = _fixed_aggregation_documents(tmp_path)
    changed = json.loads(json.dumps(documents))
    if defect == "extra_key":
        changed["model_source_seal_outer0"]["unexpected"] = False
    elif defect == "wrong_model_binding":
        changed["prediction_shard_seal_outer0"]["model_source_shard_seal"][
            "sha256"
        ] = "0" * 64
    elif defect == "wrong_authorization_binding":
        changed["model_source_seal_outer0"]["promotion_authorization"][
            "sha256"
        ] = "0" * 64
    elif defect == "duplicate_seed":
        changed["prediction_shard_seal_outer0"]["units"][1]["seed"] = (
            runtime.SEEDS[0]
        )
    elif defect == "target_flag":
        changed["prediction_shard_seal_outer0"][
            "target_fields_accessed_or_emitted"
        ] = True
    if defect is None:
        runtime._validate_fixed_aggregation_shard_seals(
            changed, bindings, project_root=tmp_path
        )
    else:
        with pytest.raises(runtime.TargetSealedError, match="aggregation"):
            runtime._validate_fixed_aggregation_shard_seals(
                changed, bindings, project_root=tmp_path
            )


@pytest.mark.parametrize("defect", ["symlink", "hardlink", "special", "mutable"])
def test_partial_output_alias_special_or_mutable_entry_is_refused(
    tmp_path: Path, defect: str
) -> None:
    bundle = _make_project(tmp_path)
    output = bundle["writable"]["output"]
    source = _write_frozen(output / "one.json", b"{}\n")
    if defect == "symlink":
        os.symlink("one.json", output / "alias.json")
    elif defect == "hardlink":
        os.link(source, output / "alias.json")
    elif defect == "special":
        os.mkfifo(output / "fifo", mode=0o444)
    else:
        source.chmod(0o644)
    with pytest.raises(runtime.TargetSealedError, match="symlink|aliased|regular|0444"):
        _prepare(bundle)


def test_partial_output_retry_reuses_the_canonical_lifecycle_capability(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path)
    with _prepare(bundle) as first:
        first_prelaunch = first.prelaunch_state
    trainer_residue_names = (
        "run_manifest.json",
        "scaler.json",
        "history.json",
        "last.pt",
        "best.pt",
        ".last.pt.killresidue",
        ".history.json.killresidue",
    )
    for name in trainer_residue_names:
        _write_frozen(
            bundle["writable"]["output"] / name,
            f"partial:{name}\n".encode(),
        )
    execution = bundle["writable"]["execution"] / "gpu_execution_ledger_v7.jsonl"
    replacement = execution.with_name("settled-replacement.tmp")
    replacement.write_bytes(execution.read_bytes())
    replacement.chmod(0o644)
    os.replace(replacement, execution)

    request = bundle["request"]
    with runtime.prepare_runtime(
        request, launcher_path=bundle["launcher"]
    ) as prepared:
        assert prepared.prelaunch_state != first_prelaunch
        observed = {
            row["path"]
            for row in prepared.prelaunch_output_inventory
            if row["kind"] == "file"
        }
        assert set(trainer_residue_names) <= observed
        by_destination = {
            row["destination"]: row for row in prepared.mount_entries
        }
        assert by_destination[str(bundle["writable"]["lifecycle"])]["kind"] == "ro_bind_fd"

    noncanonical = bundle["writable"]["lifecycle"].with_name("runtime_attempt_001")
    noncanonical.mkdir(mode=0o700)
    writable = dict(request.writable_roots)
    writable["lifecycle"] = noncanonical
    with pytest.raises(runtime.TargetSealedError, match="canonical"):
        runtime.prepare_runtime(
            replace(
                request,
                writable_roots=writable,
                capability_receipt=(
                    noncanonical / runtime.CAPABILITY_RECEIPT_FILENAME
                ),
            ),
            launcher_path=bundle["launcher"],
        )


def test_forbidden_environment_command_and_canary_overlap_fail(tmp_path: Path) -> None:
    bundle = _make_project(tmp_path)
    request = bundle["request"]
    with pytest.raises(runtime.TargetSealedError, match="forbidden environment"):
        _prepare(
            {
                **bundle,
                "request": replace(
                    request, propagated_environment={"HAI_EXPERIMENT": "secret"}
                ),
            }
        )
    with pytest.raises(runtime.TargetSealedError, match="canonical for phase"):
        _prepare(
            {
                **bundle,
                "request": replace(
                    request,
                    command=(str(REAL_INTERPRETER), str(bundle["launcher"])),
                ),
            }
        )
    canaries = dict(request.denied_canaries)
    canaries["target_root"] = request.pack_root
    with pytest.raises(runtime.TargetSealedError, match="reachable"):
        _prepare(
            {**bundle, "request": replace(request, denied_canaries=canaries)}
        )


def test_create_once_resume_tamper_and_changed_command_fail(tmp_path: Path) -> None:
    bundle = _make_project(tmp_path)
    with _prepare(bundle) as first:
        first_bytes = bundle["request"].capability_receipt.read_bytes()
        first_hash = first.receipt["content_sha256"]
    with _prepare(bundle) as second:
        assert second.receipt["content_sha256"] == first_hash
        assert bundle["request"].capability_receipt.read_bytes() == first_bytes
    receipt = bundle["request"].capability_receipt
    receipt.chmod(0o644)
    receipt.write_bytes(b"{}\n")
    receipt.chmod(0o444)
    with pytest.raises(runtime.TargetSealedError, match="resume bytes differ"):
        _prepare(bundle)


def test_capability_receipt_structural_tamper_detected(tmp_path: Path) -> None:
    bundle = _make_project(tmp_path)
    with _prepare(bundle):
        pass
    receipt = bundle["request"].capability_receipt
    document = json.loads(receipt.read_text())
    document["security_boundary"]["network_namespace_unshared"] = False
    document.pop("content_sha256")
    document["content_sha256"] = runtime.semantic_sha256(document)
    receipt.chmod(0o644)
    receipt.write_bytes(runtime.canonical_json_bytes(document) + b"\n")
    receipt.chmod(0o444)
    with pytest.raises(runtime.TargetSealedError, match="positive boundary"):
        runtime.validate_capability_receipt(receipt)


def test_run_does_not_create_session_timeout_or_nested_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _make_project(tmp_path)
    observed: dict[str, Any] = {}

    def forbidden_fd_normalization(**_kwargs: Any) -> None:
        raise AssertionError("synthetic in-process runs must not normalize caller FDs")

    monkeypatch.setattr(
        runtime, "_close_guard_runtime_noise_fds", forbidden_fd_normalization
    )

    class FakeProcess:
        pid = 43210

        def poll(self) -> int | None:
            return 0

        def wait(self) -> int:
            return 0

    def fake_popen(command: list[str], **kwargs: Any) -> FakeProcess:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return FakeProcess()

    assert runtime.run_target_sealed(
        bundle["request"],
        launcher_path=bundle["launcher"],
        popen_factory=fake_popen,
    ) == 0
    assert observed["kwargs"]["start_new_session"] is False
    assert "preexec_fn" not in observed["kwargs"]
    assert "timeout" not in observed["kwargs"]
    assert observed["kwargs"]["close_fds"] is True
    assert runtime.ADMITTED_CHILD_FD_ENV not in observed["kwargs"]["env"]


def _run_fresh_runtime_probe(body: str) -> subprocess.CompletedProcess[str]:
    loader = (
        "import importlib.util, pathlib, sys\n"
        f"path = pathlib.Path({str(SCRIPT)!r})\n"
        "spec = importlib.util.spec_from_file_location('target_fd_probe', path)\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules[spec.name] = module\n"
        "spec.loader.exec_module(module)\n"
        "runtime = module\n"
    )
    return subprocess.run(
        [sys.executable, "-I", "-c", loader + body],
        check=False,
        capture_output=True,
        text=True,
    )


def test_production_entry_normalizes_fresh_interpreter_urandom_fd(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "fd-reuse-marker"
    result = _run_fresh_runtime_probe(
        "import fcntl, os, types\n"
        "def extras():\n"
        "    rows = []\n"
        "    for name in os.listdir('/proc/self/fd'):\n"
        "        if not name.isdigit() or int(name) <= 2:\n"
        "            continue\n"
        "        try:\n"
        "            rows.append((int(name), os.readlink('/proc/self/fd/' + name)))\n"
        "        except OSError:\n"
        "            pass\n"
        "    return rows\n"
        "before = extras()\n"
        "assert len(before) <= 1, before\n"
        "if before:\n"
        "    assert before[0][1] == '/dev/urandom', before\n"
        "    assert fcntl.fcntl(before[0][0], fcntl.F_GETFD) == fcntl.FD_CLOEXEC\n"
        "runtime.validate_atomic_replace_compatibility = lambda request: None\n"
        "def resume(request):\n"
        "    runtime.audit_process_fds(allowed=(0, 1, 2))\n"
        "    return 19\n"
        "runtime._try_resume_completed = resume\n"
        "request = types.SimpleNamespace(production=True)\n"
        "assert runtime.run_target_sealed(request) == 19\n"
        "runtime.audit_process_fds(allowed=(0, 1, 2))\n"
        f"marker = pathlib.Path({str(marker)!r})\n"
        "marker.write_bytes(b'marker-bytes')\n"
        "descriptor = os.open(marker, os.O_RDONLY | os.O_CLOEXEC)\n"
        "if before:\n"
        "    assert descriptor == before[0][0]\n"
        "identity = os.fstat(descriptor)\n"
        "offset = os.lseek(descriptor, 0, os.SEEK_CUR)\n"
        "assert len(os.urandom(32)) == 32\n"
        "current = os.fstat(descriptor)\n"
        "assert (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino)\n"
        "assert os.lseek(descriptor, 0, os.SEEK_CUR) == offset\n"
        "assert os.read(descriptor, 12) == b'marker-bytes'\n"
        "runtime._close_guard_runtime_noise_fds(allowed=(0, 1, 2, descriptor))\n"
        "runtime.audit_process_fds(allowed=(0, 1, 2, descriptor))\n"
        "os.close(descriptor)\n"
        "marker.unlink()\n"
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("defect", ("wrong_target", "non_cloexec", "duplicate"))
def test_runtime_fd_normalization_rejects_without_partial_close(
    defect: str,
) -> None:
    result = _run_fresh_runtime_probe(
        "import fcntl, os\n"
        "noise = []\n"
        "for name in os.listdir('/proc/self/fd'):\n"
        "    if not name.isdigit() or int(name) <= 2:\n"
        "        continue\n"
        "    try:\n"
        "        if os.readlink('/proc/self/fd/' + name) == '/dev/urandom':\n"
        "            noise.append(int(name))\n"
        "    except OSError:\n"
        "        pass\n"
        "assert len(noise) == 1, noise\n"
        "cache = noise[0]\n"
        f"defect = {defect!r}\n"
        "if defect == 'wrong_target':\n"
        "    owned = os.open('/dev/null', os.O_RDONLY | os.O_CLOEXEC)\n"
        "elif defect == 'non_cloexec':\n"
        "    owned = os.open('/dev/urandom', os.O_RDONLY | os.O_CLOEXEC)\n"
        "    os.set_inheritable(owned, True)\n"
        "else:\n"
        "    owned = os.open('/dev/urandom', os.O_RDONLY | os.O_CLOEXEC)\n"
        "cache_identity = os.fstat(cache)\n"
        "owned_identity = os.fstat(owned)\n"
        "try:\n"
        "    runtime._close_guard_runtime_noise_fds()\n"
        "except runtime.TargetSealedError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('malformed descriptor set was accepted')\n"
        "current_cache = os.fstat(cache)\n"
        "current_owned = os.fstat(owned)\n"
        "assert (current_cache.st_dev, current_cache.st_ino) == "
        "(cache_identity.st_dev, cache_identity.st_ino)\n"
        "assert (current_owned.st_dev, current_owned.st_ino) == "
        "(owned_identity.st_dev, owned_identity.st_ino)\n"
        "if defect == 'non_cloexec':\n"
        "    assert os.get_inheritable(owned)\n"
        "os.close(owned)\n"
        "os.close(cache)\n"
    )
    assert result.returncode == 0, result.stderr


def test_runtime_fd_normalization_refuses_multithreaded_close() -> None:
    result = _run_fresh_runtime_probe(
        "import os, threading\n"
        "noise = [int(name) for name in os.listdir('/proc/self/fd') "
        "if name.isdigit() and int(name) > 2 and "
        "os.path.exists('/proc/self/fd/' + name)]\n"
        "assert len(noise) == 1, noise\n"
        "cache = noise[0]\n"
        "identity = os.fstat(cache)\n"
        "started = threading.Event()\n"
        "release = threading.Event()\n"
        "worker = threading.Thread(target=lambda: (started.set(), release.wait()))\n"
        "worker.start()\n"
        "started.wait()\n"
        "try:\n"
        "    runtime._close_guard_runtime_noise_fds()\n"
        "except runtime.TargetSealedError as error:\n"
        "    assert 'single-thread' in str(error)\n"
        "else:\n"
        "    raise AssertionError('multithreaded normalization was accepted')\n"
        "current = os.fstat(cache)\n"
        "assert (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino)\n"
        "release.set()\n"
        "worker.join()\n"
        "os.close(cache)\n"
    )
    assert result.returncode == 0, result.stderr


def test_runtime_fd_normalization_handles_memfd_fallback_noise() -> None:
    result = _run_fresh_runtime_probe(
        "import errno, os\n"
        "runtime._close_guard_runtime_noise_fds()\n"
        "saved_memfd = getattr(os, 'memfd_create', None)\n"
        "if saved_memfd is not None:\n"
        "    delattr(os, 'memfd_create')\n"
        "saved_open = os.open\n"
        "def no_tmpfile(path, flags, *args, **kwargs):\n"
        "    if path == '/tmp' and flags & getattr(os, 'O_TMPFILE', 0):\n"
        "        raise OSError(errno.EOPNOTSUPP, 'forced test fallback')\n"
        "    return saved_open(path, flags, *args, **kwargs)\n"
        "os.open = no_tmpfile\n"
        "try:\n"
        "    descriptor = runtime._create_memfd(b'sealed-spec')\n"
        "finally:\n"
        "    os.open = saved_open\n"
        "    if saved_memfd is not None:\n"
        "        os.memfd_create = saved_memfd\n"
        "try:\n"
        "    runtime._close_guard_runtime_noise_fds("
        "allowed=(0, 1, 2, descriptor))\n"
        "    runtime.audit_process_fds(allowed=(0, 1, 2, descriptor))\n"
        "    assert os.read(descriptor, 11) == b'sealed-spec'\n"
        "finally:\n"
        "    os.close(descriptor)\n"
    )
    assert result.returncode == 0, result.stderr


def test_atomic_replace_after_prelaunch_publishes_closed_completion_and_resumes(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path)
    usage = bundle["writable"]["usage"] / "campaign_gpu_usage_chain_v6.jsonl"
    before_inode = usage.stat().st_ino
    popen_count = 0

    class FakeProcess:
        pid = 43211

        def poll(self) -> int | None:
            return 0

        def wait(self) -> int:
            return 0

    def replacing_popen(_command: list[str], **_kwargs: Any) -> FakeProcess:
        nonlocal popen_count
        popen_count += 1
        temporary = usage.with_name("usage-replacement.tmp")
        temporary.write_bytes(usage.read_bytes())
        temporary.chmod(0o644)
        os.replace(temporary, usage)
        return FakeProcess()

    assert runtime.run_target_sealed(
        bundle["request"],
        launcher_path=bundle["launcher"],
        popen_factory=replacing_popen,
    ) == 0
    assert usage.stat().st_ino != before_inode
    completion = bundle["writable"]["lifecycle"] / runtime.COMPLETION_RECEIPT_FILENAME
    validated = runtime.validate_completion_receipt(
        completion,
        expected_phase=bundle["request"].phase,
        expected_outer_fold=bundle["request"].outer_fold,
    )
    assert (
        validated["prelaunch_state"].current_file_bindings["usage_ledger"]["st_ino"]
        != validated["postlaunch_state"].current_file_bindings["usage_ledger"]["st_ino"]
    )
    for path in (bundle["request"].capability_receipt, completion):
        status = path.stat(follow_symlinks=False)
        assert stat.S_IMODE(status.st_mode) == 0o444
        assert status.st_nlink == 1

    def forbidden_popen(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("completed invocation must resume without Popen")

    assert runtime.run_target_sealed(
        bundle["request"],
        launcher_path=bundle["launcher"],
        popen_factory=forbidden_popen,
    ) == 0
    assert popen_count == 1
    _write_frozen(bundle["writable"]["output"] / "late-extra.json", b"{}\n")
    with pytest.raises(runtime.TargetSealedError, match="output inventory"):
        runtime.run_target_sealed(
            bundle["request"],
            launcher_path=bundle["launcher"],
            popen_factory=forbidden_popen,
        )


@pytest.mark.parametrize(
    "defect", ["rewrite", "truncate", "open", "lock", "directory"]
)
def test_completed_resume_accepts_closed_shared_ledger_suffix_only(
    tmp_path: Path, defect: str
) -> None:
    bundle = _make_project(tmp_path)
    usage = bundle["writable"]["usage"] / "campaign_gpu_usage_chain_v6.jsonl"
    execution = (
        bundle["writable"]["execution"] / "gpu_execution_ledger_v7.jsonl"
    )

    class FakeProcess:
        pid = 43214

        def poll(self) -> int | None:
            return 0

        def wait(self) -> int:
            return 0

    def replace_bytes(path: Path, raw: bytes) -> None:
        temporary = path.with_name(f".{path.name}.replacement")
        temporary.write_bytes(raw)
        temporary.chmod(0o644)
        os.replace(temporary, path)

    def completing_popen(_command: list[str], **_kwargs: Any) -> FakeProcess:
        replace_bytes(
            usage,
            usage.read_bytes()
            + runtime.canonical_json_bytes({"event": "closed-A-usage"})
            + b"\n",
        )
        replace_bytes(
            execution,
            execution.read_bytes()
            + runtime.canonical_json_bytes({"event": "closed-A-execution"})
            + b"\n",
        )
        return FakeProcess()

    assert runtime.run_target_sealed(
        bundle["request"],
        launcher_path=bundle["launcher"],
        popen_factory=completing_popen,
    ) == 0
    replace_bytes(
        usage,
        usage.read_bytes()
        + runtime.canonical_json_bytes({"event": "closed-B-usage"})
        + b"\n",
    )
    replace_bytes(
        execution,
        execution.read_bytes()
        + runtime.canonical_json_bytes({"event": "closed-B-execution"})
        + b"\n",
    )

    def forbidden_popen(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("closed append-only completion must resume without Popen")

    assert runtime.run_target_sealed(
        bundle["request"],
        launcher_path=bundle["launcher"],
        popen_factory=forbidden_popen,
    ) == 0
    if defect == "rewrite":
        live = usage.read_bytes()
        replace_bytes(usage, b"[" + live[1:])
        match = "exact live ledger prefix"
    elif defect == "truncate":
        replace_bytes(usage, b"")
        match = "exact live ledger prefix|regressed"
    else:
        if defect == "open":
            replace_bytes(usage, usage.read_bytes() + b"OPEN_USAGE\n")
            match = "live validation|open"
        elif defect == "lock":
            usage_lock = (
                bundle["writable"]["usage"]
                / "campaign_gpu_usage_chain_v6.jsonl.lock"
            )
            replace_bytes(usage_lock, b"")
            match = "usage_ledger_lock lineage"
        else:
            usage_directory = bundle["writable"]["usage"]
            replacement_directory = usage_directory.with_name("usage-replacement")
            replacement_directory.mkdir(mode=0o700)
            for source in usage_directory.iterdir():
                target = replacement_directory / source.name
                target.write_bytes(source.read_bytes())
                target.chmod(0o644)
            displaced = usage_directory.parent.parent / "usage-displaced"
            os.rename(usage_directory, displaced)
            os.rename(replacement_directory, usage_directory)
            match = "directory/receipt lineage"
    with pytest.raises(runtime.TargetSealedError, match=match):
        runtime.run_target_sealed(
            bundle["request"],
            launcher_path=bundle["launcher"],
            popen_factory=forbidden_popen,
        )


def test_pre_popen_atomic_replace_tamper_fails_before_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _make_project(tmp_path)
    execution = bundle["writable"]["execution"] / "gpu_execution_ledger_v7.jsonl"
    original = runtime._pre_popen_revalidate
    popen_called = False

    def tamper_then_validate(prepared: Any) -> None:
        replacement = execution.with_name("execution-replacement.tmp")
        replacement.write_bytes(b"unexpected\n")
        replacement.chmod(0o644)
        os.replace(replacement, execution)
        original(prepared)

    def forbidden_popen(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal popen_called
        popen_called = True
        raise AssertionError("tampered state must fail before Popen")

    monkeypatch.setattr(runtime, "_pre_popen_revalidate", tamper_then_validate)
    with pytest.raises(
        runtime.TargetSealedError,
        match="changed before Popen|frozen postfailure prefix",
    ):
        runtime.run_target_sealed(
            bundle["request"],
            launcher_path=bundle["launcher"],
            popen_factory=forbidden_popen,
        )
    assert popen_called is False
    assert not (
        bundle["writable"]["lifecycle"] / runtime.COMPLETION_RECEIPT_FILENAME
    ).exists()


def test_pre_popen_migration_module_replace_fails_before_import_or_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _make_project(tmp_path)
    module = bundle["project"] / runtime.MIGRATION_MODULE_RELATIVE_PATH
    marker = tmp_path / "malicious-import-marker"
    original = runtime._pre_popen_revalidate

    def tamper_then_validate(prepared: Any) -> None:
        replacement = module.with_name("migration-replacement.tmp")
        replacement.write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n"
        )
        replacement.chmod(0o444)
        os.replace(replacement, module)
        original(prepared)

    monkeypatch.setattr(runtime, "_pre_popen_revalidate", tamper_then_validate)
    with pytest.raises(runtime.TargetSealedError, match="source snapshot capability"):
        runtime.run_target_sealed(
            bundle["request"],
            launcher_path=bundle["launcher"],
            popen_factory=lambda *_args, **_kwargs: pytest.fail("Popen called"),
        )
    assert not marker.exists()


def test_mutable_or_aliased_child_output_cannot_be_completed(tmp_path: Path) -> None:
    bundle = _make_project(tmp_path)
    mutable = bundle["writable"]["output"] / "mutable.json"

    class FakeProcess:
        pid = 43212

        def poll(self) -> int | None:
            return 0

        def wait(self) -> int:
            return 0

    def writing_popen(*_args: Any, **_kwargs: Any) -> FakeProcess:
        mutable.write_text("{}\n")
        mutable.chmod(0o644)
        return FakeProcess()

    with pytest.raises(runtime.TargetSealedError, match="0444"):
        runtime.run_target_sealed(
            bundle["request"],
            launcher_path=bundle["launcher"],
            popen_factory=writing_popen,
        )
    assert not (
        bundle["writable"]["lifecycle"] / runtime.COMPLETION_RECEIPT_FILENAME
    ).exists()


def test_production_requires_active_v8r4a_chain_before_receipt_or_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _make_project(tmp_path)
    request = replace(bundle["request"], production=True)
    active = request.governance_files["active_authorization"]
    document = json.loads(active.read_text())
    document["production_target_sealed_runtime_authorized"] = False
    document.pop("content_sha256")
    document["content_sha256"] = runtime.semantic_sha256(document)
    active.chmod(0o644)
    _write_json(active, document)
    monkeypatch.setattr(
        runtime, "_close_guard_runtime_noise_fds", lambda **_kwargs: None
    )
    monkeypatch.setattr(runtime, "audit_process_fds", lambda **_kwargs: (0, 1, 2))
    fd_authority = request.governance_files[
        "fd_closure_correction_authorization"
    ]
    fd_diagnostic = request.governance_files["fd_closure_failure_diagnostic"]
    monkeypatch.setattr(
        runtime,
        "FD_CLOSURE_AUTHORITY_FILE_SHA256",
        hashlib.sha256(fd_authority.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        runtime, "FD_CLOSURE_AUTHORITY_BYTES", fd_authority.stat().st_size
    )
    monkeypatch.setattr(
        runtime,
        "FD_CLOSURE_DIAGNOSTIC_FILE_SHA256",
        hashlib.sha256(fd_diagnostic.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        runtime, "FD_CLOSURE_DIAGNOSTIC_BYTES", fd_diagnostic.stat().st_size
    )
    fd_basis = json.loads(fd_authority.read_text())["authority_basis"]
    monkeypatch.setattr(
        runtime,
        "FD_CLOSURE_DIAGNOSTIC_LEGACY_BINDING",
        fd_basis["diagnostic"],
    )
    monkeypatch.setattr(
        runtime,
        "FD_CLOSURE_PARENT_BINDINGS",
        {
            **runtime.FD_CLOSURE_PARENT_BINDINGS,
            "parent_execution_closure_authority": fd_basis[
                "parent_execution_closure_authority"
            ],
        },
    )
    canary_authority = request.governance_files[
        "canary_boundary_correction_authorization"
    ]
    canary_diagnostic = request.governance_files[
        "canary_boundary_failure_diagnostic"
    ]
    monkeypatch.setattr(
        runtime,
        "CANARY_BOUNDARY_AUTHORITY_FILE_SHA256",
        hashlib.sha256(canary_authority.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        runtime,
        "CANARY_BOUNDARY_AUTHORITY_BYTES",
        canary_authority.stat().st_size,
    )
    monkeypatch.setattr(
        runtime,
        "CANARY_BOUNDARY_DIAGNOSTIC_FILE_SHA256",
        hashlib.sha256(canary_diagnostic.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        runtime,
        "CANARY_BOUNDARY_DIAGNOSTIC_BYTES",
        canary_diagnostic.stat().st_size,
    )
    canary_basis = json.loads(canary_authority.read_text())["authority_basis"]
    monkeypatch.setattr(
        runtime,
        "CANARY_BOUNDARY_DIAGNOSTIC_LEGACY_BINDING",
        canary_basis["diagnostic"],
    )
    monkeypatch.setattr(
        runtime,
        "CANARY_BOUNDARY_PARENT_BINDINGS",
        {
            **runtime.CANARY_BOUNDARY_PARENT_BINDINGS,
            "parent_fd_closure_authority": canary_basis[
                "parent_fd_closure_authority"
            ],
        },
    )
    frozen_authority = request.governance_files[
        "frozen_contract_encoding_correction_authorization"
    ]
    frozen_diagnostic = request.governance_files[
        "frozen_contract_encoding_failure_diagnostic"
    ]
    monkeypatch.setattr(
        runtime,
        "FROZEN_CONTRACT_AUTHORITY_FILE_SHA256",
        hashlib.sha256(frozen_authority.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        runtime,
        "FROZEN_CONTRACT_AUTHORITY_BYTES",
        frozen_authority.stat().st_size,
    )
    monkeypatch.setattr(
        runtime,
        "FROZEN_CONTRACT_DIAGNOSTIC_FILE_SHA256",
        hashlib.sha256(frozen_diagnostic.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        runtime,
        "FROZEN_CONTRACT_DIAGNOSTIC_BYTES",
        frozen_diagnostic.stat().st_size,
    )
    frozen_basis = json.loads(frozen_authority.read_text())["authority_basis"]
    monkeypatch.setattr(
        runtime,
        "FROZEN_CONTRACT_DIAGNOSTIC_LEGACY_BINDING",
        frozen_basis["diagnostic"],
    )
    monkeypatch.setattr(
        runtime,
        "FROZEN_CONTRACT_PARENT_BINDINGS",
        {
            **runtime.FROZEN_CONTRACT_PARENT_BINDINGS,
            **{
                field: frozen_basis[field]
                for field in (
                    "parent_canary_boundary_authority",
                    "parent_canary_boundary_diagnostic",
                    "frozen_campaign_contract",
                )
            },
        },
    )
    parent_bind_authority = request.governance_files[
        "gpu_state_parent_bind_correction_authorization"
    ]
    parent_bind_diagnostic = request.governance_files[
        "gpu_state_parent_bind_failure_diagnostic"
    ]
    monkeypatch.setattr(
        runtime,
        "GPU_STATE_PARENT_BIND_AUTHORITY_FILE_SHA256",
        hashlib.sha256(parent_bind_authority.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        runtime,
        "GPU_STATE_PARENT_BIND_AUTHORITY_BYTES",
        parent_bind_authority.stat().st_size,
    )
    monkeypatch.setattr(
        runtime,
        "GPU_STATE_PARENT_BIND_DIAGNOSTIC_FILE_SHA256",
        hashlib.sha256(parent_bind_diagnostic.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        runtime,
        "GPU_STATE_PARENT_BIND_DIAGNOSTIC_BYTES",
        parent_bind_diagnostic.stat().st_size,
    )
    parent_bind_basis = json.loads(parent_bind_authority.read_text())["authority_basis"]
    monkeypatch.setattr(
        runtime,
        "GPU_STATE_PARENT_BIND_DIAGNOSTIC_LEGACY_BINDING",
        parent_bind_basis["diagnostic"],
    )
    monkeypatch.setattr(
        runtime,
        "GPU_STATE_PARENT_BIND_PARENT_BINDINGS",
        {
            **runtime.GPU_STATE_PARENT_BIND_PARENT_BINDINGS,
            **{
                field: parent_bind_basis[field]
                for field in (
                    "parent_frozen_contract_authority",
                    "parent_frozen_contract_diagnostic",
                    "frozen_campaign_contract",
                    "gpu_state_migration_receipt",
                )
            },
            },
        )
    admitted_context_authority = request.governance_files[
        "admitted_context_correction_authorization"
    ]
    admitted_context_diagnostic = request.governance_files[
        "admitted_context_failure_diagnostic"
    ]
    admitted_context_authority_document = json.loads(
        admitted_context_authority.read_text()
    )
    admitted_context_diagnostic_document = json.loads(
        admitted_context_diagnostic.read_text()
    )
    monkeypatch.setattr(
        runtime,
        "BENCHMARK_ADMITTED_CONTEXT_AUTHORITY_FILE_SHA256",
        hashlib.sha256(admitted_context_authority.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        runtime,
        "BENCHMARK_ADMITTED_CONTEXT_AUTHORITY_BYTES",
        admitted_context_authority.stat().st_size,
    )
    monkeypatch.setattr(
        runtime,
        "BENCHMARK_ADMITTED_CONTEXT_AUTHORITY_CONTENT_SHA256",
        admitted_context_authority_document["content_sha256"],
    )
    monkeypatch.setattr(
        runtime,
        "BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_FILE_SHA256",
        hashlib.sha256(admitted_context_diagnostic.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        runtime,
        "BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_BYTES",
        admitted_context_diagnostic.stat().st_size,
    )
    monkeypatch.setattr(
        runtime,
        "BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_CONTENT_SHA256",
        admitted_context_diagnostic_document["content_sha256"],
    )
    state_status = (bundle["project"] / runtime.GPU_STATE_ROOT_RELATIVE).stat()
    monkeypatch.setattr(
        runtime, "GPU_STATE_ROOT_AUTHORIZED_ST_DEV", state_status.st_dev
    )
    monkeypatch.setattr(
        runtime, "GPU_STATE_ROOT_AUTHORIZED_ST_INO", state_status.st_ino
    )

    def forbidden_popen(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("production bwrap must not launch")

    with pytest.raises(runtime.TargetSealedError, match="pretrain production"):
        runtime.run_target_sealed(
            request,
            launcher_path=bundle["launcher"],
            popen_factory=forbidden_popen,
        )
    assert not request.capability_receipt.exists()


def test_private_pycache_prefix_ignores_hostile_source_tree_pyc(tmp_path: Path) -> None:
    bundle = _make_project(tmp_path)
    module = bundle["project"] / "scripts/poisoned_module.py"
    module.write_text("VALUE='EVIL'\n")
    original = module.stat()
    cache = Path(importlib.util.cache_from_source(str(module)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    py_compile.compile(str(module), cfile=str(cache), doraise=True)
    module.write_text("VALUE='SAFE'\n")
    os.utime(module, ns=(original.st_atime_ns, original.st_mtime_ns))
    module.chmod(0o444)
    cache.chmod(0o444)
    with _prepare(bundle) as prepared:
        assert prepared.child_environment["PYTHONPYCACHEPREFIX"] == "/tmp/pycache"
        assert "/tmp/pycache" in {
            row["destination"] for row in prepared.mount_entries
        }
    env = dict(prepared.child_environment)
    env["PYTHONPATH"] = str(bundle["project"] / "scripts")
    env["PYTHONPYCACHEPREFIX"] = str(tmp_path / "private-pycache")
    result = subprocess.run(
        [sys.executable, "-c", "import poisoned_module;print(poisoned_module.VALUE)"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "SAFE"


def _minimal_guard_receipt(path: Path, spec_values: Mapping[str, Any]) -> dict[str, Any]:
    boundary = {key: False for key in runtime.SECURITY_BOUNDARY_KEYS}
    for key in (
        "outer_campaign_runtime",
        "network_namespace_unshared",
        "ipc_namespace_unshared",
        "uts_namespace_unshared",
        "tmp_is_private_tmpfs",
        "die_with_parent",
        "capabilities_dropped",
        "environment_cleared_before_allowlist",
        "admitted_fd_direct_watchdog_to_trainer_contract_preserved",
        "child_fd_audit_required",
        "denied_canary_probe_required",
        "atomic_replace_compatible",
        "synthetic_validation_only",
        "v8r4a_migration_live_replay_validated",
        "dedicated_gpu_state_directory_capabilities",
        "gpu_state_parent_identity_readonly_bind",
        "exactly_three_mutable_state_directory_mounts",
        "benchmark_admitted_context_generation_isolated",
        "active_pretrain_postfailure_ledger_prefix_enforced",
        "usage_and_execution_closed_prelaunch",
        "lifecycle_mounted_read_only",
        "source_snapshot_exact_file_mounts",
    ):
        boundary[key] = True
    state_paths = {"admission": "/proc", "execution": "/dev", "usage": "/sys"}
    directories = {
        "root": {
            "exact_entries": sorted(runtime.GPU_STATE_DIRECTORY_ROLES),
            "mode": "0700",
            "path": runtime.GPU_STATE_ROOT_RELATIVE.as_posix(),
            "st_dev": 1,
            "st_ino": 1,
        }
    }
    for number, role in enumerate(sorted(runtime.GPU_STATE_DIRECTORY_ROLES), 2):
        directories[role] = {
            "exact_entries": sorted(runtime.GPU_STATE_EXACT_ENTRIES[role]),
            "mode": "0700",
            "path": runtime.GPU_STATE_DIRECTORY_RELATIVE_PATHS[role].as_posix(),
            "st_dev": 1,
            "st_ino": number,
        }
    files = {}
    for number, (role, (directory, name)) in enumerate(
        sorted(runtime.GPU_STATE_FILE_ROLES.items()), 10
    ):
        files[role] = {
            "bytes": 0,
            "mode": "0644",
            "nlink": 1,
            "path": (
                runtime.GPU_STATE_DIRECTORY_RELATIVE_PATHS[directory] / name
            ).as_posix(),
            "sha256": "0" * 64,
            "st_dev": 1,
            "st_ino": number,
        }
    state = {
        "migration_receipt": {
            "bytes": 1,
            "content_sha256": "1" * 64,
            "mode": "0444",
            "nlink": 1,
            "path": "migration.json",
            "sha256": "2" * 64,
            "st_dev": 1,
            "st_ino": 20,
        },
        "directories": directories,
        "files": files,
        "usage_state": {"open_reservation_count": 0},
        "execution_state": {"open_start_count": 0},
    }
    mounts = [
        {
            "kind": "ro_bind_fd",
            "destination": "/",
            "source": {
                **directories["root"],
                "path": "/",
            },
        },
        *[
        {
            "kind": "rw_bind_fd",
            "destination": state_paths[role],
            "source": {
                **directories[role],
                "path": state_paths[role],
            },
        }
        for role in sorted(runtime.GPU_STATE_DIRECTORY_ROLES)
        ],
    ]
    mounts.extend(
        [
            {"kind": "rw_bind_fd", "destination": "/tmp", "source": {}},
            {"kind": "ro_bind_fd", "destination": "/run", "source": {}},
        ]
    )
    writable = {
        "output": {"path": "/tmp", "mode": "0700", "st_dev": 1, "st_ino": 30},
        "lifecycle": {"path": "/run", "mode": "0700", "st_dev": 1, "st_ino": 31},
        **{
            role: {
                "path": state_paths[role],
                "mode": "0700",
                "st_dev": 1,
                "st_ino": 40 + number,
            }
            for number, role in enumerate(sorted(runtime.GPU_STATE_DIRECTORY_ROLES))
        },
    }
    command = spec_values["command"]
    environment = spec_values["environment"]
    receipt = {
        "schema_version": 1,
        "classification": runtime.RECEIPT_CLASSIFICATION,
        "campaign_id": runtime.CAMPAIGN_ID,
        "campaign_revision": "V8R4",
        "infrastructure_revision": "V8R4A",
        "phase": "discovery_aggregation",
        "outer_fold": None,
        "bubblewrap": {},
        "launcher": {},
        "interpreter": {},
        "sealed_pack_root": None,
        "sealed_pack_index": None,
        "governance_files": {},
        "writable_roots": writable,
        "prelaunch_gpu_state": state,
        "denied_canaries": spec_values["denied_canaries"],
        "mount_specification": mounts,
        "mount_specification_sha256": runtime.semantic_sha256(mounts),
        "environment": environment,
        "environment_sha256": runtime.semantic_sha256(environment),
        "command": command,
        "command_sha256": runtime.semantic_sha256(command),
        "security_boundary": boundary,
    }
    receipt["content_sha256"] = runtime.semantic_sha256(receipt)
    _write_frozen(path, runtime.canonical_json_bytes(receipt) + b"\n")
    return receipt


def _guard_spec(tmp_path: Path, *, canary_exists: bool = False) -> tuple[Path, Path, Path, dict[str, str]]:
    marker = tmp_path / "guard-marker"
    command = [sys.executable, "-c", f"open({str(marker)!r},'w').write('ok')"]
    environment = {
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", "/usr/bin"),
        "PWD": str(tmp_path),
    }
    canaries = {
        role: str(tmp_path / "denied" / role)
        for role in runtime.MANDATORY_DENIED_CANARY_ROLES
    }
    if canary_exists:
        target = Path(canaries["target_root"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("forbidden")
    receipt_path = tmp_path / "receipt.json"
    values = {
        "command": command,
        "environment": environment,
        "denied_canaries": canaries,
    }
    receipt = _minimal_guard_receipt(receipt_path, values)
    binding, _ = runtime._read_file_binding(
        receipt_path, label="receipt", require_immutable=True
    )
    spec = {
        "schema_version": 1,
        "classification": "adaptive_v3r1_v8r4a_target_sealed_child_guard_spec",
        "campaign_id": runtime.CAMPAIGN_ID,
        "campaign_revision": "V8R4",
        "infrastructure_revision": "V8R4A",
        "phase": "discovery_aggregation",
        "outer_fold": None,
        "capability_receipt": binding.document(),
        "mount_specification_sha256": receipt["mount_specification_sha256"],
        "environment": environment,
        "environment_sha256": runtime.semantic_sha256(environment),
        "denied_canaries": canaries,
        "available_paths": [str(receipt_path), str(tmp_path)],
        "command": command,
        "command_sha256": runtime.semantic_sha256(command),
        "required_open_fds": [0, 1, 2],
        "forbidden_environment": sorted(
            runtime.FORBIDDEN_ENV_NAMES | {runtime.ADMITTED_CHILD_FD_ENV}
        ),
    }
    spec["content_sha256"] = runtime.semantic_sha256(spec)
    spec_path = _write_frozen(
        tmp_path / "spec.json", runtime.canonical_json_bytes(spec) + b"\n"
    )
    return spec_path, marker, receipt_path, environment


def test_internal_guard_environment_fd_and_canary_boundary(tmp_path: Path) -> None:
    spec, marker, _receipt, environment = _guard_spec(tmp_path)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--internal-guard", str(spec)],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert marker.read_text() == "ok"

    denied = tmp_path / "denied-case"
    denied.mkdir()
    spec, marker, _receipt, environment = _guard_spec(denied, canary_exists=True)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--internal-guard", str(spec)],
        cwd=denied,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 73
    assert "denied canary" in result.stderr
    assert not marker.exists()


def test_child_command_accepts_only_exact_project_skeleton_not_descendants(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    mounted = project / "scripts/entry.py"
    mounted.parent.mkdir(parents=True)
    mounted.write_text("# mounted\n", encoding="utf-8")
    runtime._validate_command_paths(
        (str(mounted), "--project-root", str(project)),
        project_root=project,
        mounted_roots=(mounted,),
    )
    with pytest.raises(runtime.TargetSealedError, match="not mounted"):
        runtime._validate_command_paths(
            (str(project / "unmounted/secret"),),
            project_root=project,
            mounted_roots=(mounted,),
        )


def test_command_denied_canary_audit_uses_lexical_component_boundaries(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    denied = root / "efficiency_benchmark_v8"
    sibling = root / "efficiency_benchmark_v8r4a"
    mounted = sibling / "unit"
    canaries = {"historical_v8": str(denied)}

    # The exact production regression: the raw string prefix collides, but the
    # two basenames are distinct path components.
    assert str(denied) in str(sibling)
    runtime._validate_command_paths(
        (str(mounted), "relative-token-containing-efficiency_benchmark_v8"),
        project_root=tmp_path,
        mounted_roots=(sibling,),
    )
    runtime._validate_command_denied_canaries(
        (str(mounted), "relative-token-containing-efficiency_benchmark_v8"),
        denied_canaries=canaries,
    )

    for argument in (str(denied), str(denied / "child"), str(denied) + "/./"):
        with pytest.raises(runtime.TargetSealedError, match="denied capability"):
            runtime._validate_command_denied_canaries(
                (argument,), denied_canaries=canaries
            )


@pytest.mark.parametrize(
    "argument, message",
    (
        ("relative/../escape", "traversal"),
        ("../escape", "traversal"),
        ("..", "traversal"),
        ("/mounted/../escape", "traversal"),
        ("--output=/mounted/output", "absolute option path"),
        ("-x=/mounted/output", "absolute option path"),
    ),
)
def test_command_path_audit_rejects_traversal_and_embedded_absolute_options(
    tmp_path: Path, argument: str, message: str,
) -> None:
    mounted = tmp_path / "mounted"
    with pytest.raises(runtime.TargetSealedError, match=message):
        runtime._validate_command_paths(
            (argument,), project_root=tmp_path, mounted_roots=(mounted,)
        )


def test_gpu_state_parent_overlap_exception_is_exact_and_command_inaccessible(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "gpu_state_v8r4a"
    children = tuple(parent / role for role in sorted(runtime.GPU_STATE_DIRECTORY_ROLES))
    runtime._validate_mount_boundaries(
        ro_roots=(parent,),
        rw_roots=children,
        governance_files=(),
        denied_canaries={},
        gpu_state_readonly_parent=parent,
        gpu_state_mutable_children=children,
    )
    with pytest.raises(runtime.TargetSealedError, match="not mounted"):
        runtime._validate_command_paths(
            (str(parent),), project_root=tmp_path, mounted_roots=children
        )
    with pytest.raises(runtime.TargetSealedError, match="not mounted"):
        runtime._validate_command_paths(
            (str(parent / "unrelated"),),
            project_root=tmp_path,
            mounted_roots=children,
        )
    unrelated_parent = tmp_path / "unrelated"
    with pytest.raises(runtime.TargetSealedError, match="overlap"):
        runtime._validate_mount_boundaries(
            ro_roots=(parent, unrelated_parent),
            rw_roots=(*children, unrelated_parent / "child"),
            governance_files=(),
            denied_canaries={},
            gpu_state_readonly_parent=parent,
            gpu_state_mutable_children=children,
        )
    with pytest.raises(runtime.TargetSealedError, match="not exact"):
        runtime._validate_mount_boundaries(
            ro_roots=(parent,),
            rw_roots=children,
            governance_files=(),
            denied_canaries={},
            gpu_state_readonly_parent=parent,
            gpu_state_mutable_children=children[:2],
        )


@pytest.mark.parametrize(
    "defect",
    ("missing_parent", "writable_parent", "late_parent", "wrong_parent_identity"),
)
def test_rootbind1_capability_mount_abi_tamper_is_rejected(
    tmp_path: Path, defect: str,
) -> None:
    bundle = _make_project(tmp_path)
    with _prepare(bundle) as prepared:
        document = json.loads(json.dumps(prepared.receipt))
    root = str(bundle["project"] / runtime.GPU_STATE_ROOT_RELATIVE)
    mounts = document["mount_specification"]
    root_index = next(
        index for index, row in enumerate(mounts) if row.get("destination") == root
    )
    root_row = mounts[root_index]
    if defect == "missing_parent":
        mounts.pop(root_index)
    elif defect == "writable_parent":
        root_row["kind"] = "rw_bind_fd"
    elif defect == "late_parent":
        mounts.append(mounts.pop(root_index))
    else:
        root_row["source"]["st_ino"] += 1
    document["mount_specification_sha256"] = runtime.semantic_sha256(mounts)
    document.pop("content_sha256")
    document["content_sha256"] = runtime.semantic_sha256(document)
    path = _write_frozen(
        tmp_path / f"tampered-{defect}.json",
        runtime.canonical_json_bytes(document) + b"\n",
    )
    with pytest.raises(runtime.TargetSealedError, match="state mount ABI"):
        runtime.validate_capability_receipt(path)


def test_rootbind1_descriptor_inventory_and_live_mount_flags_fail_closed(
    tmp_path: Path,
) -> None:
    bundle = _make_project(tmp_path)
    with _prepare(bundle) as prepared:
        root = str(bundle["project"] / runtime.GPU_STATE_ROOT_RELATIVE)
        children = {
            role: str(bundle["writable"][role])
            for role in sorted(runtime.GPU_STATE_DIRECTORY_ROLES)
        }
        table = {root: [frozenset({"ro"})]}
        table.update({path: [frozenset({"rw"})] for path in children.values()})
        table[str(bundle["writable"]["lifecycle"])] = [frozenset({"ro"})]
        table[str(bundle["writable"]["output"])] = [frozenset({"rw"})]
        runtime._validate_live_gpu_state_mounts(table, prepared.receipt)
        wrong = dict(table)
        wrong[root] = [frozenset({"rw"})]
        with pytest.raises(runtime.TargetSealedError, match="access mode"):
            runtime._validate_live_gpu_state_mounts(wrong, prepared.receipt)
        for role, wrong_access in (("lifecycle", "rw"), ("output", "ro")):
            wrong = dict(table)
            wrong[str(bundle["writable"][role])] = [frozenset({wrong_access})]
            with pytest.raises(runtime.TargetSealedError, match="access mode"):
                runtime._validate_live_gpu_state_mounts(wrong, prepared.receipt)
        unexpected = Path(root) / "unexpected"
        unexpected.write_text("forbidden", encoding="utf-8")
        with pytest.raises(runtime.TargetSealedError, match="inventory"):
            runtime._revalidate_state_directory_descriptor(
                prepared.gpu_state_root_binding,
                prepared.gpu_state_root_descriptor,
                prepared.prelaunch_state.directory_bindings["root"],
                label="gpu_state_parent",
            )


def _bwrap_available() -> tuple[bool, str]:
    probe = subprocess.run(
        [
            "/usr/bin/bwrap",
            "--die-with-parent",
            "--unshare-net",
            "--ro-bind",
            "/usr",
            "/usr",
            "--symlink",
            "usr/lib",
            "/lib",
            "--symlink",
            "usr/lib64",
            "/lib64",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--",
            "/usr/bin/true",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0, probe.stderr.strip()


def test_bwrap_outer_campaign_smoke_when_host_allows_namespaces(tmp_path: Path) -> None:
    available, reason = _bwrap_available()
    if not available:
        pytest.skip(f"host sandbox blocks bubblewrap namespace creation: {reason}")
    bundle = _make_project(tmp_path)
    request = replace(bundle["request"], cuda_devices=(Path("/dev/null"),))
    result = runtime.run_target_sealed(
        request, launcher_path=bundle["launcher"]
    )
    assert result == 0
    payload = json.loads(bundle["marker"].read_text())
    assert runtime.ADMITTED_CHILD_FD_ENV not in payload["env"]
    assert "HAI_EXPERIMENT" not in payload["env"]
    assert bundle["request"].capability_receipt.stat().st_mode & 0o777 == 0o444


def test_bwrap_directory_fd_mount_allows_atomic_replace_when_available(
    tmp_path: Path,
) -> None:
    available, reason = _bwrap_available()
    if not available:
        pytest.skip(f"host sandbox blocks bubblewrap namespace creation: {reason}")
    directory = tmp_path / "state"
    directory.mkdir(mode=0o700)
    ledger = directory / "ledger.jsonl"
    ledger.write_text("prefix\n")
    ledger.chmod(0o644)
    original_inode = ledger.stat().st_ino
    descriptor = os.open(
        directory,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        code = (
            "import errno,os,pathlib,sys;"
            "p=pathlib.Path('/state/ledger.jsonl');"
            "t=p.with_name('replacement.tmp');t.write_text('new\\n');"
            "\ntry: os.replace(t,p)\n"
            "except OSError: sys.exit(21)\n"
            "else: sys.exit(0)"
        )
        result = subprocess.run(
            [
                "/usr/bin/bwrap",
                "--die-with-parent",
                "--unshare-net",
                "--ro-bind",
                "/usr",
                "/usr",
                "--symlink",
                "usr/lib",
                "/lib",
                "--symlink",
                "usr/lib64",
                "/lib64",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                    "--bind-fd",
                    str(descriptor),
                    "/state",
                "--",
                "/usr/bin/python3",
                "-c",
                code,
            ],
            pass_fds=(descriptor,),
            close_fds=True,
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        os.close(descriptor)
    assert result.returncode == 0, result.stderr
    assert ledger.read_text() == "new\n"
    assert ledger.stat().st_ino != original_inode


def test_bwrap_readonly_parent_with_three_readwrite_children_when_available(
    tmp_path: Path,
) -> None:
    available, reason = _bwrap_available()
    if not available:
        pytest.skip(f"host sandbox blocks bubblewrap namespace creation: {reason}")
    root = tmp_path / "gpu_state"
    root.mkdir(mode=0o700)
    descriptors: list[int] = []
    children: list[Path] = []
    for role in ("admission", "execution", "usage"):
        child = root / role
        child.mkdir(mode=0o700)
        (child / "ledger").write_text("prefix\n", encoding="utf-8")
        children.append(child)
    expected = root.stat()
    try:
        for path in (root, *children):
            descriptors.append(
                os.open(
                    path,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                )
            )
        code = (
            "import errno,os,pathlib,sys;"
            f"expected=({expected.st_dev},{expected.st_ino});"
            "assert (os.stat('/state').st_dev,os.stat('/state').st_ino)==expected;"
            "rows={};"
            "\nfor line in pathlib.Path('/proc/self/mountinfo').read_text().splitlines():\n"
            " p=line.split(' '); rows.setdefault(p[4],[]).append(set(p[5].split(',')))\n"
            "assert any('ro' in x and 'rw' not in x for x in rows['/state'])\n"
            "assert all(any('rw' in x and 'ro' not in x for x in rows['/state/'+r]) for r in ('admission','execution','usage'))\n"
            "try: pathlib.Path('/state/forbidden').write_text('x')\n"
            "except OSError as e: assert e.errno in {errno.EROFS,errno.EACCES,errno.EPERM}\n"
            "else: raise AssertionError('read-only parent mutated')\n"
            "p=pathlib.Path('/state/execution/ledger');t=p.with_name('replacement');t.write_text('new\\n');os.replace(t,p);"
            "fds=[];"
            "\nfor name in os.listdir('/proc/self/fd'):\n"
            " try: os.fstat(int(name));fds.append(int(name))\n"
            " except OSError: pass\n"
            "assert sorted(fds)==[0,1,2]"
        )
        command = [
            "/usr/bin/bwrap",
            "--die-with-parent",
            "--unshare-net",
            "--ro-bind",
            "/usr",
            "/usr",
            "--symlink",
            "usr/lib",
            "/lib",
            "--symlink",
            "usr/lib64",
            "/lib64",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--ro-bind-fd",
            str(descriptors[0]),
            "/state",
        ]
        for descriptor, role in zip(descriptors[1:], ("admission", "execution", "usage")):
            command.extend(["--bind-fd", str(descriptor), f"/state/{role}"])
        command.extend(["--", "/usr/bin/python3", "-c", code])
        result = subprocess.run(
            command,
            pass_fds=tuple(descriptors),
            close_fds=True,
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    assert result.returncode == 0, result.stderr
    assert (root / "execution/ledger").read_text() == "new\n"
    assert not (root / "forbidden").exists()


def test_bwrap_readonly_lifecycle_directory_blocks_receipt_replace_when_available(
    tmp_path: Path,
) -> None:
    available, reason = _bwrap_available()
    if not available:
        pytest.skip(f"host sandbox blocks bubblewrap namespace creation: {reason}")
    lifecycle = tmp_path / "lifecycle"
    lifecycle.mkdir(mode=0o700)
    receipt = _write_frozen(lifecycle / "receipt.json", b"{}\n")
    descriptor = os.open(
        lifecycle,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        code = (
            "import errno,os,pathlib,sys;"
            "p=pathlib.Path('/lifecycle/receipt.json');"
            "t=p.with_name('replacement.tmp');"
            "\ntry: t.write_text('new\\n')\n"
            "except OSError as e: sys.exit(0 if e.errno in {errno.EROFS,errno.EACCES,errno.EPERM} else 21)\n"
            "else:\n"
            "  try: os.replace(t,p)\n"
            "  except OSError as e: sys.exit(0 if e.errno in {errno.EROFS,errno.EACCES,errno.EPERM,errno.EBUSY} else 22)\n"
            "  else: sys.exit(23)"
        )
        result = subprocess.run(
            [
                "/usr/bin/bwrap",
                "--die-with-parent",
                "--unshare-net",
                "--ro-bind",
                "/usr",
                "/usr",
                "--symlink",
                "usr/lib",
                "/lib",
                "--symlink",
                "usr/lib64",
                "/lib64",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--ro-bind-fd",
                str(descriptor),
                "/lifecycle",
                "--",
                "/usr/bin/python3",
                "-c",
                code,
            ],
            pass_fds=(descriptor,),
            close_fds=True,
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        os.close(descriptor)
    assert result.returncode == 0, result.stderr
    assert receipt.read_bytes() == b"{}\n"


def test_real_public_migration_validator_abi_and_closed_live_state() -> None:
    module = REAL_PROJECT / runtime.MIGRATION_MODULE_RELATIVE_PATH
    receipt = (
        REAL_PROJECT
        / "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
        "GPU_STATE_MIGRATION_RECEIPT_V8R4A.json"
    )
    if not module.is_file() or not receipt.is_file():
        pytest.skip("V8R4A migrated state has not been integrated")
    module_status = module.stat(follow_symlinks=False)
    if stat.S_IMODE(module_status.st_mode) != 0o444:
        pytest.skip("successor migration validator has not been frozen/reissued")
    assert module_status.st_nlink == 1
    succession = json.loads(
        (
            REAL_PROJECT
            / "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
            "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_MIGRATION_SOURCE_SUCCESSION.json"
        ).read_text()
    )
    assert succession["classification"] == (
        "pretrain_adaptive_v3r1_v8r4a_migrated_state_source_succession_correction_addendum"
    )
    assert succession["content_sha256"] == runtime.semantic_sha256(
        {key: value for key, value in succession.items() if key != "content_sha256"}
    )
    migrator_rows = [
        row
        for row in succession["authorized_modifications"]
        if row.get("path") == runtime.MIGRATION_MODULE_RELATIVE_PATH.as_posix()
    ]
    assert len(migrator_rows) == 1
    assert migrator_rows[0]["before_sha256"] == (
        "f06cfb18ca5cf58bbe7c5b80248a3b05ca73e947e8023204654db84298eefdca"
    )
    binding, _ = runtime._read_file_binding(
        receipt, label="real migration receipt", require_immutable=True
    )
    state = runtime._validate_migrated_state_live(
        project_root=REAL_PROJECT, receipt_binding=binding
    )
    assert set(state.directory_bindings) == {
        "root",
        "admission",
        "execution",
        "usage",
    }
    assert state.usage_state["open_reservation_count"] == 0
    assert state.execution_state["open_start_count"] == 0


@pytest.mark.parametrize(
    "outer_fold,expected_sha256",
    [
        (3, "db204cdba72e9f3023c58ef37c1761cfd4ec2f4310449f8eaeeef7003afadb9b"),
        (4, "1ce1b1a0154c609b1fa1693ff5702b3e81be178f8ddbdd7ec25f559c39029f0a"),
    ],
)
def test_real_frozen_three_seed_shard_index_and_tree(
    outer_fold: int, expected_sha256: str
) -> None:
    root = (
        REAL_PROJECT
        / "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
        "v8r4_split_inputs"
        / f"discovery_shard_outer_{outer_fold}"
    )
    index = root / "V8R4_NONOUTER_TRAINING_INDEX.json"
    if not index.is_file():
        pytest.skip("authorized CPU pack build is not present")
    tree = runtime._tree_binding(root, label="real sealed shard", frozen=True)
    binding, raw = runtime._read_file_binding(
        index, label="real shard index", require_immutable=True
    )
    assert binding.sha256 == expected_sha256
    assert binding.bytes == 3172
    document = runtime._validate_pack_index(
        raw=raw,
        binding=binding,
        pack_root=tree,
        phase="discovery",
        outer_fold=outer_fold,
    )
    assert document["unit_count"] == 3
    assert {row["outer_fold"] for row in document["units"]} == {outer_fold}
