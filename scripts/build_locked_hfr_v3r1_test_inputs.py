#!/usr/bin/env python3
"""Build one create-once, target-free DHFER-v3r1 V8R4 promotion input.

The source is a physically separate outer-inference pack whose exact metadata
schema cannot contain fold, reference, QC, identity, or protocol fields.  The
consumer verifies every pack byte before opening metadata or feature arrays;
it never opens, hashes, or binds the historical combined target-bearing cache.
The output NPZ has an exact compile-time allow-list and a create-once receipt.

This program requires the post-discovery promotion authorization.  It has no
target or evaluation argument, so it cannot be repurposed to open references.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import run_hfr_v3r1_discovery_campaign as discovery  # noqa: E402
import select_hfr_v3r1_common_variant as selection_authority  # noqa: E402


SELECTION_RELATIVE = discovery.CAMPAIGN_RELATIVE / "DISCOVERY_SELECTION_LOCK.json"
PROMOTION_AUTH_RELATIVE = discovery.CAMPAIGN_RELATIVE / "PROMOTION_AUTHORIZATION.json"

CAMPAIGN_REVISION = "V8R4"
INFRASTRUCTURE_REVISION = discovery.INFRASTRUCTURE_REVISION
FORWARD_METADATA_COLUMNS = (
    "cache_index",
    "window_number",
    "classical_rr_bpm",
)
OUTER_PACK_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "classification",
        "campaign_id",
        "campaign_revision",
        "outer_fold",
        "seed",
        "row_count",
        "fields",
        "exact_allowlist",
        "forbidden_fields_emitted",
        "reference_identity_protocol_quality_decoded",
        "legacy_index",
        "legacy_cache_manifest",
        "legacy_proposer_stack",
        "promotion_authorization",
        "output",
        "global_cache_index_sha256",
        "object_arrays",
        "pickle",
        "commercial_or_confirmatory_claim_allowed",
        "content_sha256",
    }
)
OUTER_PACK_CLASSIFICATION = "adaptive_v3r1_v8r4_authorized_outer_prediction_pack"
MODEL_BOUND_PACK_CLASSIFICATION = (
    "adaptive_v3r1_v8r4a_model_bound_target_free_prediction_pack"
)
MODEL_SOURCE_CAPABILITY_CLASSIFICATION = (
    "adaptive_v3r1_v8r4a_promotion_model_source_capability"
)
MODEL_BOUND_MANIFEST_FILENAME = "MODEL_BOUND_OUTER_PREDICTION_PACK_MANIFEST.json"
MODEL_SOURCE_CAPABILITY_FILENAME = "MODEL_SOURCE_CAPABILITY.json"
MODEL_BOUND_UNIT_FILES = frozenset(
    {
        "OUTER_PREDICTION_PACK_MANIFEST.json",
        MODEL_BOUND_MANIFEST_FILENAME,
        MODEL_SOURCE_CAPABILITY_FILENAME,
        "outer_predict_input.npz",
        "model_checkpoint.pt",
        "model_scaler.json",
    }
)
SAFE_OUTPUT_FIELDS = (
    "cache_index",
    "node_features",
    "candidate_rr_bpm",
    "candidate_mask",
    "joint_radar_mask",
    "proposer_anchor_bpm",
    "proposer_anchor_std_bpm",
    "proposer_anchor_available",
    "classical_rr_bpm",
    "session_reset",
)

REUSED_DISCOVERY_FOLDS = (3, 4)
NEW_PROMOTION_TRAINING_FOLDS = (0, 1, 2, 5)
REUSE_POINTER_FILENAME = "discovery_reuse_pointer_receipt.json"
REUSE_POINTER_ARTIFACTS = tuple(discovery.REQUIRED_TRAIN_OUTPUTS)
_SIGNATURE_FORBIDDEN_KEYS = frozenset(
    {
        "output_directory",
        "campaign_phase_label",
        "promotion_authorization_path",
        "release_mode",
        "resume_flag",
    }
)
_SCIENTIFIC_SIGNATURE_KEYS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "campaign_revision",
        "classification",
        "contract_file_sha256",
        "outer_fold",
        "validation_fold",
        "seed",
        "variant",
        "model",
        "optimization",
        "source_bindings",
        "input_bindings",
        "pretrain_authorization",
        "population",
        "batching_execution",
        "checkpoint_selection",
    }
)


@dataclass(frozen=True)
class PromotionModelSource:
    """One exact checkpoint/scaler source for a fixed promotion unit."""

    kind: str
    receipt_path: Path
    output_dir: Path
    checkpoint: Path
    scaler: Path
    scientific_signature_sha256: str
    artifacts: Mapping[str, Mapping[str, Any]]
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class MountedPredictionPack:
    """A target-scoped successor pack validated without opening provenance paths."""

    root: Path
    input_path: Path
    model_source: PromotionModelSource
    manifest: Mapping[str, Any]
    capability: Mapping[str, Any]


def _require_regular_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise discovery.CampaignError(f"{label} is missing, non-regular, or symlinked: {path}")


def _validate_signature_object(
    signature: Any,
    signature_sha256: Any,
    *,
    outer_fold: int,
    seed: int,
    variant: str,
) -> str:
    if not isinstance(signature, Mapping):
        raise discovery.CampaignError("training manifest lacks its full scientific signature")
    if not discovery._is_sha256(signature_sha256):
        raise discovery.CampaignError("training manifest scientific signature hash is invalid")
    observed = discovery.semantic_sha256(signature)
    if observed != signature_sha256:
        raise discovery.CampaignError("training manifest scientific signature hash drifted")

    # The producer's frozen V8 rule strips exactly these five top-level keys;
    # nested values remain scientific and therefore must not be normalized here.
    forbidden = _SIGNATURE_FORBIDDEN_KEYS & set(map(str, signature))
    if forbidden:
        raise discovery.CampaignError(
            "scientific signature contains orchestration-only fields: "
            f"{sorted(forbidden)}"
        )
    if set(map(str, signature)) != _SCIENTIFIC_SIGNATURE_KEYS:
        raise discovery.CampaignError("scientific signature retained-field set drifted")
    for name, expected in (
        ("schema_version", 1),
        ("campaign_id", discovery.CAMPAIGN_ID),
        ("campaign_revision", CAMPAIGN_REVISION),
        ("contract_file_sha256", discovery.CONTRACT_FILE_SHA256),
        ("outer_fold", outer_fold),
        ("validation_fold", (outer_fold + 1) % 6),
        ("seed", seed),
        ("variant", variant),
    ):
        if signature.get(name) != expected:
            raise discovery.CampaignError(
                f"scientific signature unit identity drifted: {name}"
            )
    if signature.get("classification") not in {
        "adaptive_retrospective_historical_cohort_engineering",
        "synthetic_implementation_smoke_test",
    }:
        raise discovery.CampaignError("scientific signature classification drifted")
    for name in (
        "model",
        "optimization",
        "source_bindings",
        "input_bindings",
        "pretrain_authorization",
        "population",
        "batching_execution",
    ):
        if not isinstance(signature.get(name), Mapping):
            raise discovery.CampaignError(
                f"scientific signature retained object is invalid: {name}"
            )
    if signature.get("batching_execution") != {
        "training_batch_unit": "physical_session_group",
        "temporal_schedule": "aligned_tbptt_chunk_rounds",
        "padding_inert": True,
        "per_session_cvar_before_group_reduction": True,
        "valid_length_spike_weighting": True,
        "prediction_batch_sessions": 4,
    }:
        raise discovery.CampaignError("scientific signature batching execution drifted")
    return str(signature_sha256)


def _validate_bound_training_source(
    *,
    project_root: Path,
    receipt_path: Path,
    receipt: Mapping[str, Any],
    cache_dir: Path,
    outer_fold: int,
    seed: int,
    variant: str,
    expected_artifacts: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Mapping[str, Any]], str]:
    validated = receipt.get("validated_output")
    if not isinstance(validated, Mapping):
        raise discovery.CampaignError("training receipt lacks validated output")
    if (
        validated.get("campaign_revision") != CAMPAIGN_REVISION
        or validated.get("physical_boundary")
        != discovery.DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION
        or not isinstance(validated.get("row_access_audit"), Mapping)
        or validated["row_access_audit"].get("outer_row_access_attempts") != 0
        or validated.get("outer_fold") != outer_fold
        or validated.get("seed") != seed
        or validated.get("variant") != variant
    ):
        raise discovery.CampaignError("training receipt validated-output identity drifted")
    artifacts = validated.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(REUSE_POINTER_ARTIFACTS):
        raise discovery.CampaignError("training receipt artifact set is not exact")
    if expected_artifacts is not None and artifacts != expected_artifacts:
        raise discovery.CampaignError("reuse pointer artifact bindings drifted")
    resolved: dict[str, Path] = {}
    normalized_bindings: dict[str, Mapping[str, Any]] = {}
    for name in REUSE_POINTER_ARTIFACTS:
        binding = artifacts.get(name)
        if not isinstance(binding, Mapping):
            raise discovery.CampaignError(f"training receipt artifact binding is invalid: {name}")
        path = discovery.verify_binding(
            binding,
            project_root=project_root,
            owner=receipt_path,
            label=f"training source artifact {name}",
        )
        _require_regular_file(path, f"training source artifact {name}")
        if path.name != name:
            raise discovery.CampaignError(f"training source artifact filename drifted: {name}")
        resolved[name] = path
        normalized_bindings[name] = dict(binding)
    output_dirs = {path.parent for path in resolved.values()}
    if len(output_dirs) != 1:
        raise discovery.CampaignError("training source artifacts are mixed across output roots")
    output_dir = next(iter(output_dirs))
    live = discovery.validate_training_output(
        output_dir,
        outer_fold=outer_fold,
        seed=seed,
        variant=variant,
        cache_dir=cache_dir,
    )
    if live != validated:
        raise discovery.CampaignError("training source live validation differs from receipt")
    manifest = discovery.load_json(output_dir / "run_manifest.json", "training source manifest")
    lock = discovery.load_json(
        output_dir / "checkpoint_selection_lock.json", "training source selection lock"
    )
    signature_sha = _validate_signature_object(
        manifest.get("scientific_signature"),
        manifest.get("scientific_signature_sha256"),
        outer_fold=outer_fold,
        seed=seed,
        variant=variant,
    )
    if lock.get("scientific_signature_sha256") != signature_sha:
        raise discovery.CampaignError("checkpoint lock scientific signature drifted")
    if validated.get("scientific_signature_sha256") != signature_sha:
        raise discovery.CampaignError("training receipt scientific signature drifted")
    return output_dir, normalized_bindings, signature_sha


def validate_selected_discovery_source(
    *,
    project_root: Path,
    discovery_completion_seal: Mapping[str, Any],
    cache_dir: Path,
    outer_fold: int,
    seed: int,
    variant: str,
) -> PromotionModelSource:
    """Resolve the selected immutable discovery unit without opening outer test data."""

    if outer_fold not in REUSED_DISCOVERY_FOLDS or seed not in discovery.SEEDS:
        raise discovery.CampaignError("discovery reuse is outside the exact six-unit matrix")
    owner = project_root / "selection_lock.json"
    seal_path = discovery.verify_binding(
        discovery_completion_seal,
        project_root=project_root,
        owner=owner,
        label="selection discovery completion seal",
    )
    _require_regular_file(seal_path, "discovery completion seal")
    seal = discovery.load_json(seal_path, "discovery completion seal")
    if discovery.canonical_content_sha256(seal) != seal.get("content_sha256"):
        raise discovery.CampaignError("discovery completion seal content drifted")
    final_keys = {
        "schema_version", "classification", "campaign_id", "campaign_revision",
        "infrastructure_revision", "contract", "pretrain_authorization",
        "training_shards", "outer_runs", "seeds", "variants",
        "completed_units", "physical_boundary", "validation_targets_only",
        "gpu_elapsed_seconds", "gpu_hours_hard", "gpu_usage_ledger",
        "gpu_usage_ledger_path", "pre_discovery_efficiency_benchmark",
        "v8r3_successful_terminal_quarantine", "units",
        "cross_outer_validation_reuse_present", "fully_nested_confirmatory_oof",
        "prospective_confirmation_required", "ready_for_global_discovery_selection",
        "commercial_claim_authorized", "content_sha256",
    }
    if not (
        set(seal) == final_keys
        and seal.get("schema_version") == 1
        and seal.get("classification")
        == "adaptive_v3r1_v8r4_target_sealed_discovery_completion"
        and seal.get("campaign_id") == discovery.CAMPAIGN_ID
        and seal.get("campaign_revision") == CAMPAIGN_REVISION
        and seal.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and seal.get("completed_units") == 18
        and seal.get("outer_runs") == list(discovery.OUTER_RUNS)
        and seal.get("seeds") == list(discovery.SEEDS)
        and seal.get("variants") == list(discovery.VARIANTS)
        and isinstance(seal.get("training_shards"), list)
        and len(seal["training_shards"]) == 2
        and seal.get("physical_boundary")
        == discovery.DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION
        and seal.get("validation_targets_only") is True
        and seal.get("cross_outer_validation_reuse_present") is True
        and seal.get("fully_nested_confirmatory_oof") is False
        and seal.get("prospective_confirmation_required") is True
        and seal.get("ready_for_global_discovery_selection") is True
        and seal.get("commercial_claim_authorized") is False
    ):
        raise discovery.CampaignError("discovery completion seal safety fields drifted")
    units = seal.get("units")
    if not isinstance(units, list) or len(units) != len(discovery.EXPECTED_DISCOVERY_UNITS):
        raise discovery.CampaignError("discovery completion seal unit count drifted")
    expected_keys = set(discovery.EXPECTED_DISCOVERY_UNITS)
    indexed: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    for unit in units:
        if not isinstance(unit, Mapping) or set(unit) != {
            "outer_fold", "seed", "variant", "receipt"
        }:
            raise discovery.CampaignError("discovery completion seal has an invalid unit")
        key = (int(unit["outer_fold"]), int(unit["seed"]), str(unit["variant"]))
        if key in indexed:
            raise discovery.CampaignError("discovery completion seal has duplicate units")
        indexed[key] = unit
    if set(indexed) != expected_keys:
        raise discovery.CampaignError("discovery completion seal matrix drifted")
    key = (outer_fold, seed, variant)
    unit = indexed.get(key)
    if unit is None:
        raise discovery.CampaignError("selected discovery reuse unit is absent")
    receipt_path = discovery.verify_binding(
        unit.get("receipt", {}),
        project_root=project_root,
        owner=seal_path,
        label=f"selected discovery receipt {key}",
    )
    _require_regular_file(receipt_path, "selected discovery receipt")
    receipt = discovery.load_json(receipt_path, "selected discovery receipt")
    if discovery.canonical_content_sha256(receipt) != receipt.get("content_sha256"):
        raise discovery.CampaignError("selected discovery receipt content drifted")
    receipt_keys = {
        "schema_version", "classification", "campaign_id", "campaign_revision",
        "infrastructure_revision", "outer_test_opened", "outer_fold",
        "validation_fold", "seed", "variant", "invocation",
        "usage_ledger_path", "usage_record_sha256", "usage_record_sha256s",
        "terminal_results", "lifecycle_invocations", "gpu_execution_ledger_path",
        "gpu_admission_lock_path", "validated_output",
        "commercial_claim_authorized", "content_sha256",
    }
    if not (
        set(receipt) == receipt_keys
        and receipt.get("schema_version") == 1
        and receipt.get("classification")
        == "adaptive_v3r1_v8r4_discovery_unit_completion"
        and receipt.get("campaign_id") == discovery.CAMPAIGN_ID
        and receipt.get("campaign_revision") == CAMPAIGN_REVISION
        and receipt.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and receipt.get("outer_fold") == outer_fold
        and receipt.get("validation_fold") == (outer_fold + 1) % 6
        and receipt.get("seed") == seed
        and receipt.get("variant") == variant
        and receipt.get("outer_test_opened") is False
        and receipt.get("commercial_claim_authorized") is False
    ):
        raise discovery.CampaignError("selected discovery receipt identity/leakage drifted")
    output_dir, artifacts, signature_sha = _validate_bound_training_source(
        project_root=project_root,
        receipt_path=receipt_path,
        receipt=receipt,
        cache_dir=cache_dir,
        outer_fold=outer_fold,
        seed=seed,
        variant=variant,
    )
    return PromotionModelSource(
        kind="discovery",
        receipt_path=receipt_path,
        output_dir=output_dir,
        checkpoint=output_dir / "best.pt",
        scaler=output_dir / "scaler.json",
        scientific_signature_sha256=signature_sha,
        artifacts=artifacts,
        receipt=receipt,
    )


def resolve_promotion_model_source(
    *,
    project_root: Path,
    run_root: Path,
    cache_dir: Path,
    outer_fold: int,
    seed: int,
    variant: str,
) -> PromotionModelSource:
    """Require local-training XOR immutable discovery-pointer ownership."""

    unit_root = run_root / "training" / f"outer_{outer_fold}_seed_{seed}"
    local_path = unit_root / "completion_receipt.json"
    pointer_path = unit_root / REUSE_POINTER_FILENAME
    local_exists = os.path.lexists(local_path)
    pointer_exists = os.path.lexists(pointer_path)
    if local_exists == pointer_exists:
        raise discovery.CampaignError(
            "promotion unit must have exactly one local receipt or discovery pointer"
        )
    if outer_fold in REUSED_DISCOVERY_FOLDS and local_exists:
        raise discovery.CampaignError("reused promotion fold unexpectedly owns local GPU training")
    if outer_fold in NEW_PROMOTION_TRAINING_FOLDS and pointer_exists:
        raise discovery.CampaignError("new promotion fold unexpectedly uses a discovery pointer")
    if outer_fold not in range(6) or seed not in discovery.SEEDS:
        raise discovery.CampaignError("promotion model source is outside the fixed matrix")

    if local_exists:
        _require_regular_file(local_path, "local promotion training receipt")
        receipt = discovery.load_json(local_path, "local promotion training receipt")
        if discovery.canonical_content_sha256(receipt) != receipt.get("content_sha256"):
            raise discovery.CampaignError("local promotion training receipt content drifted")
        if (
            receipt.get("schema_version") != 1
            or receipt.get("classification")
            != "adaptive_v3r1_fixed_promotion_training_completion"
            or receipt.get("campaign_id") != discovery.CAMPAIGN_ID
            or receipt.get("campaign_revision") != CAMPAIGN_REVISION
            or receipt.get("infrastructure_revision") != INFRASTRUCTURE_REVISION
            or receipt.get("outer_fold") != outer_fold
            or receipt.get("seed") != seed
            or receipt.get("variant") != variant
            or receipt.get("outer_test_opened") is not False
            or receipt.get("validation_scores_changed_execution") is not False
            or receipt.get("commercial_claim_authorized") is not False
        ):
            raise discovery.CampaignError("local promotion training identity/leakage drifted")
        output_dir, artifacts, signature_sha = _validate_bound_training_source(
            project_root=project_root,
            receipt_path=local_path,
            receipt=receipt,
            cache_dir=cache_dir,
            outer_fold=outer_fold,
            seed=seed,
            variant=variant,
        )
        expected_output = unit_root / "attempt_000" / "output"
        if output_dir.resolve() != expected_output.resolve():
            raise discovery.CampaignError("local promotion output root drifted")
        return PromotionModelSource(
            kind="local_training",
            receipt_path=local_path,
            output_dir=output_dir,
            checkpoint=output_dir / "best.pt",
            scaler=output_dir / "scaler.json",
            scientific_signature_sha256=signature_sha,
            artifacts=artifacts,
            receipt=receipt,
        )

    _require_regular_file(pointer_path, "discovery reuse pointer")
    pointer = discovery.load_json(pointer_path, "discovery reuse pointer")
    if discovery.canonical_content_sha256(pointer) != pointer.get("content_sha256"):
        raise discovery.CampaignError("discovery reuse pointer content drifted")
    expected_keys = {
        "schema_version",
        "classification",
        "campaign_id",
        "campaign_revision",
        "infrastructure_revision",
        "outer_fold",
        "seed",
        "variant",
        "source_phase",
        "destination_phase",
        "scientific_signature_sha256",
        "discovery_completion_seal",
        "source_training_receipt",
        "artifacts",
        "selection_lock",
        "promotion_authorization",
        "owns_new_gpu_usage",
        "usage_record_sha256s",
        "outer_test_opened",
        "adaptive_retrospective_only",
        "commercial_claim_authorized",
        "content_sha256",
    }
    if set(pointer) != expected_keys:
        raise discovery.CampaignError("discovery reuse pointer schema drifted")
    if not (
        pointer.get("schema_version") == 1
        and pointer.get("classification")
        == "adaptive_v3r1_phase_independent_discovery_reuse_pointer"
        and pointer.get("campaign_id") == discovery.CAMPAIGN_ID
        and pointer.get("campaign_revision") == CAMPAIGN_REVISION
        and pointer.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and pointer.get("outer_fold") == outer_fold
        and pointer.get("seed") == seed
        and pointer.get("variant") == variant
        and pointer.get("source_phase") == "discovery"
        and pointer.get("destination_phase") == "promotion"
        and pointer.get("owns_new_gpu_usage") is False
        and pointer.get("usage_record_sha256s") == []
        and pointer.get("outer_test_opened") is False
        and pointer.get("adaptive_retrospective_only") is True
        and pointer.get("commercial_claim_authorized") is False
    ):
        raise discovery.CampaignError("discovery reuse pointer identity/accounting drifted")
    for label, name in (
        ("pointer selection lock", "selection_lock"),
        ("pointer promotion authorization", "promotion_authorization"),
    ):
        governance_path = discovery.verify_binding(
            pointer.get(name, {}),
            project_root=project_root,
            owner=pointer_path,
            label=label,
        )
        _require_regular_file(governance_path, label)
    selected_source = validate_selected_discovery_source(
        project_root=project_root,
        discovery_completion_seal=pointer.get("discovery_completion_seal", {}),
        cache_dir=cache_dir,
        outer_fold=outer_fold,
        seed=seed,
        variant=variant,
    )
    source_receipt_path = discovery.verify_binding(
        pointer.get("source_training_receipt", {}),
        project_root=project_root,
        owner=pointer_path,
        label="pointer source training receipt",
    )
    _require_regular_file(source_receipt_path, "pointer source training receipt")
    source_receipt = discovery.load_json(source_receipt_path, "pointer source training receipt")
    if discovery.canonical_content_sha256(source_receipt) != source_receipt.get(
        "content_sha256"
    ):
        raise discovery.CampaignError("pointer source training receipt content drifted")
    if source_receipt_path != selected_source.receipt_path:
        raise discovery.CampaignError("pointer receipt is not the selected sealed discovery unit")
    output_dir, artifacts, signature_sha = _validate_bound_training_source(
        project_root=project_root,
        receipt_path=source_receipt_path,
        receipt=source_receipt,
        cache_dir=cache_dir,
        outer_fold=outer_fold,
        seed=seed,
        variant=variant,
        expected_artifacts=pointer.get("artifacts"),
    )
    if pointer.get("scientific_signature_sha256") != signature_sha:
        raise discovery.CampaignError("discovery reuse pointer scientific signature drifted")
    if (
        selected_source.scientific_signature_sha256 != signature_sha
        or selected_source.artifacts != artifacts
    ):
        raise discovery.CampaignError("pointer differs from selected sealed discovery source")
    return PromotionModelSource(
        kind="discovery_pointer",
        receipt_path=pointer_path,
        output_dir=output_dir,
        checkpoint=output_dir / "best.pt",
        scaler=output_dir / "scaler.json",
        scientific_signature_sha256=signature_sha,
        artifacts=artifacts,
        receipt=pointer,
    )
FORBIDDEN_FIELD_TOKENS = (
    "target",
    "reference",
    "label",
    "ground_truth",
    "identity",
    "protocol",
    "quality",
    "qc",
    "future",
)


def validate_promotion_authorization(
    project_root: Path,
    *,
    selection_lock_path: Path | None = None,
    authorization_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Replay the one canonical nine-candidate selection before opening inputs."""

    root = project_root.expanduser().resolve()
    selection_path = (
        selection_lock_path.expanduser().resolve()
        if selection_lock_path is not None
        else (root / SELECTION_RELATIVE).resolve()
    )
    auth_path = (
        authorization_path.expanduser().resolve()
        if authorization_path is not None
        else (root / PROMOTION_AUTH_RELATIVE).resolve()
    )
    canonical_selection = (root / SELECTION_RELATIVE).resolve()
    canonical_authorization = (root / PROMOTION_AUTH_RELATIVE).resolve()
    if selection_path != canonical_selection or auth_path != canonical_authorization:
        raise discovery.CampaignError(
            "promotion selection/authorization must use canonical paths"
        )
    return selection_authority.validate_locked_selection_authorization(
        root,
        selection_lock_path=selection_path,
        promotion_authorization_path=auth_path,
    )


def _validate_outer_pack_manifest(
    cache_dir: Path, outer_fold: int, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify a target-free outer pack before any metadata/NumPy open."""

    root = cache_dir.expanduser().resolve()
    manifest_path = root / "OUTER_PREDICTION_PACK_MANIFEST.json"
    _require_regular_file(manifest_path, "V8R4 outer-pack manifest")
    document = discovery.load_json(manifest_path, "V8R4 target-free outer pack")
    if set(document) != OUTER_PACK_MANIFEST_KEYS:
        raise discovery.CampaignError("V8R4 outer-pack manifest schema drifted")
    if not (
        document.get("schema_version") == 1
        and document.get("classification") == OUTER_PACK_CLASSIFICATION
        and document.get("campaign_id") == discovery.CAMPAIGN_ID
        and document.get("campaign_revision") == CAMPAIGN_REVISION
        and document.get("outer_fold") == int(outer_fold)
        and document.get("seed") == int(seed)
        and type(document.get("row_count")) is int
        and int(document["row_count"]) > 0
        and document.get("fields") == list(SAFE_OUTPUT_FIELDS)
        and document.get("exact_allowlist") is True
        and document.get("forbidden_fields_emitted") is False
        and document.get("reference_identity_protocol_quality_decoded") is False
        and document.get("object_arrays") is False
        and document.get("pickle") is False
        and document.get("commercial_or_confirmatory_claim_allowed") is False
        and isinstance(document.get("global_cache_index_sha256"), str)
        and len(document["global_cache_index_sha256"]) == 64
        and discovery.canonical_content_sha256(document)
        == document.get("content_sha256")
    ):
        raise discovery.CampaignError("V8R4 outer-pack manifest invariant drifted")
    for opaque_name in (
        "legacy_index",
        "legacy_cache_manifest",
        "legacy_proposer_stack",
        "promotion_authorization",
    ):
        binding = document.get(opaque_name)
        if not (
            isinstance(binding, Mapping)
            and set(binding) == {"path", "sha256", "bytes"}
            and isinstance(binding.get("path"), str)
            and isinstance(binding.get("sha256"), str)
            and len(str(binding["sha256"])) == 64
            and type(binding.get("bytes")) is int
            and int(binding["bytes"]) > 0
        ):
            raise discovery.CampaignError(f"outer-pack {opaque_name} binding drifted")
    output_path = discovery.verify_binding(
        document.get("output", {}),
        project_root=PROJECT_ROOT,
        owner=manifest_path,
        label="authorized target-free outer prediction pack",
    )
    if output_path != root / "outer_predict_input.npz":
        raise discovery.CampaignError("authorized outer-pack output path drifted")
    return document, {
        "manifest": discovery.bind_file(manifest_path),
        "output": discovery.bind_file(output_path),
        "promotion_authorization": dict(document["promotion_authorization"]),
    }


def _validate_local_pack_file(
    path: Path,
    *,
    binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Hash one mounted pack file through a no-follow pinned descriptor."""

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise discovery.CampaignError(f"cannot securely open mounted pack file: {path}") from error
    try:
        info = os.fstat(descriptor)
        if not (
            stat.S_ISREG(info.st_mode)
            and stat.S_IMODE(info.st_mode) == 0o444
            and info.st_nlink == 1
        ):
            raise discovery.CampaignError(
                f"mounted pack file is not regular 0444/nlink1: {path}"
            )
        digest = hashlib.sha256()
        offset = 0
        while offset < info.st_size:
            payload = os.pread(
                descriptor, min(1024 * 1024, info.st_size - offset), offset
            )
            if not payload:
                raise discovery.CampaignError(f"short mounted pack read: {path}")
            digest.update(payload)
            offset += len(payload)
        result = {
            "path": str(path.resolve()),
            "sha256": digest.hexdigest(),
            "bytes": int(info.st_size),
        }
        if binding is not None:
            if not (
                set(binding) == {"path", "sha256", "bytes"}
                and Path(str(binding.get("path"))).name == path.name
                and binding.get("sha256") == result["sha256"]
                and binding.get("bytes") == result["bytes"]
            ):
                raise discovery.CampaignError(
                    f"mounted pack artifact binding drifted: {path.name}"
                )
        current = os.fstat(descriptor)
        if (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        ) != (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        ):
            raise discovery.CampaignError(f"mounted pack inode changed: {path}")
        return result
    finally:
        os.close(descriptor)


def _opaque_binding_schema(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and set(value) == {"path", "sha256", "bytes"}
        and isinstance(value.get("path"), str)
        and isinstance(value.get("sha256"), str)
        and len(str(value.get("sha256"))) == 64
        and type(value.get("bytes")) is int
        and int(value.get("bytes")) >= 0
    )


def validate_model_bound_prediction_pack(
    pack_root: Path,
    *,
    outer_fold: int,
    seed: int,
    selected_variant: str,
    expected_promotion_authorization: Mapping[str, Any] | None = None,
    expected_selection_lock: Mapping[str, Any] | None = None,
) -> MountedPredictionPack:
    """Validate the six-file successor ABI without following source provenance.

    Source receipt/checkpoint paths in ``MODEL_SOURCE_CAPABILITY.json`` are
    intentionally opaque evidence.  Only the copied checkpoint, scaler, input,
    and two local manifests are opened by the target-sealed child.
    """

    root = pack_root.expanduser().resolve()
    if outer_fold not in range(6) or seed not in discovery.SEEDS:
        raise discovery.CampaignError("mounted prediction pack identity is invalid")
    if not root.is_dir() or root.is_symlink():
        raise discovery.CampaignError("mounted prediction pack root is unsafe")
    observed = {entry.name for entry in os.scandir(root)}
    if observed != set(MODEL_BOUND_UNIT_FILES):
        raise discovery.CampaignError(
            "mounted prediction pack file inventory is not exact"
        )
    for name in MODEL_BOUND_UNIT_FILES:
        _validate_local_pack_file(root / name)

    base_manifest, base_binding = _validate_outer_pack_manifest(
        root, outer_fold, seed
    )
    manifest_path = root / MODEL_BOUND_MANIFEST_FILENAME
    manifest = discovery.load_json(manifest_path, "model-bound prediction manifest")
    expected_manifest_keys = {
        "schema_version",
        "classification",
        "campaign_id",
        "campaign_revision",
        "infrastructure_revision",
        "outer_fold",
        "seed",
        "selected_variant",
        "row_count",
        "global_cache_index_sha256",
        "fields",
        "exact_target_free_allowlist",
        "selection_lock",
        "promotion_authorization",
        "base_target_free_manifest",
        "artifacts",
        "exact_unit_file_inventory",
        "prediction_child_reads_model_only_from_this_pack",
        "source_paths_or_peer_outputs_authorized_in_child",
        "target_reference_quality_identity_protocol_present",
        "model_bytes_changed",
        "commercial_or_confirmatory_claim_allowed",
        "content_sha256",
    }
    artifacts = manifest.get("artifacts")
    if not (
        set(manifest) == expected_manifest_keys
        and manifest.get("content_sha256")
        == discovery.canonical_content_sha256(manifest)
        and manifest.get("schema_version") == 1
        and manifest.get("classification") == MODEL_BOUND_PACK_CLASSIFICATION
        and manifest.get("campaign_id") == discovery.CAMPAIGN_ID
        and manifest.get("campaign_revision") == CAMPAIGN_REVISION
        and manifest.get("infrastructure_revision") == "V8R4A"
        and manifest.get("outer_fold") == outer_fold
        and manifest.get("seed") == seed
        and manifest.get("selected_variant") == selected_variant
        and manifest.get("row_count") == base_manifest.get("row_count")
        and manifest.get("global_cache_index_sha256")
        == base_manifest.get("global_cache_index_sha256")
        and manifest.get("fields") == list(SAFE_OUTPUT_FIELDS)
        and manifest.get("exact_target_free_allowlist") is True
        and manifest.get("exact_unit_file_inventory")
        == sorted(MODEL_BOUND_UNIT_FILES)
        and manifest.get("prediction_child_reads_model_only_from_this_pack") is True
        and manifest.get("source_paths_or_peer_outputs_authorized_in_child") is False
        and manifest.get("target_reference_quality_identity_protocol_present") is False
        and manifest.get("model_bytes_changed") is False
        and manifest.get("commercial_or_confirmatory_claim_allowed") is False
        and isinstance(artifacts, Mapping)
        and set(artifacts)
        == {
            "outer_predict_input",
            "model_checkpoint",
            "model_scaler",
            "model_source_capability",
        }
    ):
        raise discovery.CampaignError("model-bound prediction manifest drifted")
    if expected_promotion_authorization is not None and (
        manifest.get("promotion_authorization")
        != dict(expected_promotion_authorization)
    ):
        raise discovery.CampaignError("mounted pack promotion authority drifted")
    if expected_selection_lock is not None and (
        manifest.get("selection_lock") != dict(expected_selection_lock)
    ):
        raise discovery.CampaignError("mounted pack selection binding drifted")
    if manifest.get("promotion_authorization") != base_manifest.get(
        "promotion_authorization"
    ):
        raise discovery.CampaignError("base/successor promotion authority differs")
    _validate_local_pack_file(
        root / "OUTER_PREDICTION_PACK_MANIFEST.json",
        binding=manifest.get("base_target_free_manifest"),
    )
    local_paths = {
        "outer_predict_input": root / "outer_predict_input.npz",
        "model_checkpoint": root / "model_checkpoint.pt",
        "model_scaler": root / "model_scaler.json",
        "model_source_capability": root / MODEL_SOURCE_CAPABILITY_FILENAME,
    }
    local_bindings = {
        name: _validate_local_pack_file(path, binding=artifacts[name])
        for name, path in local_paths.items()
    }
    if base_binding["output"]["sha256"] != local_bindings[
        "outer_predict_input"
    ]["sha256"]:
        raise discovery.CampaignError("base/successor target-free input differs")

    capability_path = root / MODEL_SOURCE_CAPABILITY_FILENAME
    capability = discovery.load_json(
        capability_path, "mounted model-source capability"
    )
    expected_capability_keys = {
        "schema_version",
        "classification",
        "campaign_id",
        "campaign_revision",
        "infrastructure_revision",
        "outer_fold",
        "seed",
        "selected_variant",
        "source_kind",
        "scientific_signature_sha256",
        "source_receipt",
        "source_checkpoint",
        "source_scaler",
        "packed_checkpoint",
        "packed_scaler",
        "selection_lock",
        "promotion_authorization",
        "source_deep_validated_before_copy",
        "source_paths_or_peer_outputs_authorized_in_child",
        "target_reference_quality_identity_protocol_present",
        "model_bytes_changed",
        "commercial_or_confirmatory_claim_allowed",
        "content_sha256",
    }
    if not (
        set(capability) == expected_capability_keys
        and capability.get("content_sha256")
        == discovery.canonical_content_sha256(capability)
        and capability.get("schema_version") == 1
        and capability.get("classification")
        == MODEL_SOURCE_CAPABILITY_CLASSIFICATION
        and capability.get("campaign_id") == discovery.CAMPAIGN_ID
        and capability.get("campaign_revision") == CAMPAIGN_REVISION
        and capability.get("infrastructure_revision") == "V8R4A"
        and capability.get("outer_fold") == outer_fold
        and capability.get("seed") == seed
        and capability.get("selected_variant") == selected_variant
        and capability.get("source_kind")
        in {"local_training", "discovery", "discovery_pointer"}
        and discovery._is_sha256(capability.get("scientific_signature_sha256"))
        and all(
            _opaque_binding_schema(capability.get(name))
            for name in (
                "source_receipt",
                "source_checkpoint",
                "source_scaler",
                "packed_checkpoint",
                "packed_scaler",
                "selection_lock",
                "promotion_authorization",
            )
        )
        and capability.get("selection_lock") == manifest.get("selection_lock")
        and capability.get("promotion_authorization")
        == manifest.get("promotion_authorization")
        and capability.get("source_deep_validated_before_copy") is True
        and capability.get("source_paths_or_peer_outputs_authorized_in_child") is False
        and capability.get("target_reference_quality_identity_protocol_present") is False
        and capability.get("model_bytes_changed") is False
        and capability.get("commercial_or_confirmatory_claim_allowed") is False
    ):
        raise discovery.CampaignError("mounted model-source capability drifted")
    _validate_local_pack_file(
        root / "model_checkpoint.pt", binding=capability["packed_checkpoint"]
    )
    _validate_local_pack_file(
        root / "model_scaler.json", binding=capability["packed_scaler"]
    )
    if not (
        capability["source_checkpoint"]["sha256"]
        == local_bindings["model_checkpoint"]["sha256"]
        and capability["source_checkpoint"]["bytes"]
        == local_bindings["model_checkpoint"]["bytes"]
        and capability["source_scaler"]["sha256"]
        == local_bindings["model_scaler"]["sha256"]
        and capability["source_scaler"]["bytes"]
        == local_bindings["model_scaler"]["bytes"]
    ):
        raise discovery.CampaignError("packed model bytes differ from source evidence")
    validation = validate_sanitized_input(
        root / "outer_predict_input.npz"
    )
    if validation.get("rows") != manifest.get("row_count"):
        raise discovery.CampaignError("mounted prediction input row count drifted")
    model_source = PromotionModelSource(
        kind="mounted_successor_pack",
        receipt_path=capability_path,
        output_dir=root,
        checkpoint=root / "model_checkpoint.pt",
        scaler=root / "model_scaler.json",
        scientific_signature_sha256=str(
            capability["scientific_signature_sha256"]
        ),
        artifacts={
            "best.pt": local_bindings["model_checkpoint"],
            "scaler.json": local_bindings["model_scaler"],
        },
        receipt=capability,
    )
    return MountedPredictionPack(
        root=root,
        input_path=root / "outer_predict_input.npz",
        model_source=model_source,
        manifest=manifest,
        capability=capability,
    )


def _metadata_path(cache_dir: Path) -> Path:
    raise discovery.CampaignError("V8R4 outer prediction consumer has no metadata path")


def _read_forward_metadata(cache_dir: Path, outer_fold: int) -> tuple[pd.DataFrame, np.ndarray]:
    path = _metadata_path(cache_dir)
    header = pd.read_csv(path, nrows=0)
    if tuple(map(str, header.columns)) != FORWARD_METADATA_COLUMNS:
        raise discovery.CampaignError(
            "target-free outer metadata is not the exact three-column schema"
        )
    frame = pd.read_csv(path, usecols=list(FORWARD_METADATA_COLUMNS))
    if tuple(map(str, frame.columns)) != FORWARD_METADATA_COLUMNS or frame.empty:
        raise discovery.CampaignError("target-free outer metadata topology drifted")
    index = frame["cache_index"].to_numpy(np.int64)
    if np.any(index < 0) or np.any(np.diff(index) <= 0):
        raise discovery.CampaignError("outer-fold cache index is duplicated")
    return frame, np.arange(len(frame), dtype=np.int64)


def _load_array(cache_dir: Path, names: Sequence[str], positions: np.ndarray) -> tuple[np.ndarray, Path]:
    matches = [cache_dir / name for name in names if (cache_dir / name).is_file()]
    if len(matches) != 1:
        raise discovery.CampaignError(
            f"cache array must resolve exactly once from {list(names)}"
        )
    path = matches[0]
    try:
        source = np.load(path, mmap_mode="r", allow_pickle=False)
        result = np.asarray(source[positions]).copy()
    except (OSError, ValueError, IndexError) as error:
        raise discovery.CampaignError(f"cannot select cache array {path}: {error}") from error
    return result, path


def _scalar(archive: Mapping[str, Any], name: str) -> Any:
    value = np.asarray(archive[name])
    if value.ndim != 0:
        raise discovery.CampaignError(f"{name} must be scalar")
    return value.item()


def _load_anchor(
    path: Path, *, expected_index: np.ndarray, outer_fold: int, seed: int
) -> tuple[dict[str, np.ndarray], list[str]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            fields = list(archive.files)
            forbidden = sorted(
                name
                for name in fields
                if any(token in name.lower() for token in FORBIDDEN_FIELD_TOKENS)
            )
            if forbidden:
                raise discovery.CampaignError(
                    f"proposer anchor contains forbidden context fields: {forbidden}"
                )
            index_name = "cache_index"
            bpm_name = next(
                (
                    name
                    for name in (
                        "proposer_anchor_bpm",
                        "fallback_rr_bpm",
                        "prediction_bpm",
                        "prediction",
                    )
                    if name in archive.files
                ),
                None,
            )
            std_name = next(
                (
                    name
                    for name in (
                        "proposer_anchor_std_bpm",
                        "fallback_std_bpm",
                        "rr_std",
                        "source_scale_bpm",
                    )
                    if name in archive.files
                ),
                None,
            )
            available_name = next(
                (
                    name
                    for name in (
                        "proposer_anchor_available",
                        "fallback_available",
                        "prediction_available",
                    )
                    if name in archive.files
                ),
                None,
            )
            if index_name not in archive.files or bpm_name is None or std_name is None:
                raise discovery.CampaignError("proposer anchor lacks index/value/scale")
            if "outer_fold" in archive.files and int(_scalar(archive, "outer_fold")) != outer_fold:
                raise discovery.CampaignError("proposer anchor outer-fold mismatch")
            if "fold_id" in archive.files and int(_scalar(archive, "fold_id")) != outer_fold:
                raise discovery.CampaignError("proposer anchor fold-id mismatch")
            if "seed" in archive.files and int(_scalar(archive, "seed")) != seed:
                raise discovery.CampaignError("proposer anchor seed mismatch")
            if "target_fields_present" in archive.files and bool(
                _scalar(archive, "target_fields_present")
            ):
                raise discovery.CampaignError("proposer anchor declares target fields")
            index = np.asarray(archive[index_name], dtype=np.int64)
            bpm = np.asarray(archive[bpm_name], dtype=np.float32)
            std = np.asarray(archive[std_name], dtype=np.float32)
            available = (
                np.asarray(archive[available_name], dtype=bool)
                if available_name is not None
                else np.isfinite(bpm)
            )
    except (OSError, ValueError, KeyError) as error:
        raise discovery.CampaignError(f"invalid proposer anchor {path}: {error}") from error
    if not np.array_equal(index, expected_index):
        missing = sorted(set(map(int, expected_index)) - set(map(int, index)))[:10]
        extra = sorted(set(map(int, index)) - set(map(int, expected_index)))[:10]
        raise discovery.CampaignError(
            f"proposer anchor is not the exact cache-index cover; missing={missing}, extra={extra}"
        )
    if (
        bpm.shape != index.shape
        or std.shape != index.shape
        or available.shape != index.shape
        or np.any(available & ~np.isfinite(bpm))
        or np.any(available & (~np.isfinite(std) | (std <= 0)))
    ):
        raise discovery.CampaignError("proposer anchor value/scale topology is invalid")
    safe_bpm = np.where(available, bpm, 0.0).astype(np.float32)
    safe_std = np.where(available, std, 1.0).astype(np.float32)
    return {
        "proposer_anchor_bpm": safe_bpm,
        "proposer_anchor_std_bpm": safe_std,
        "proposer_anchor_available": available.astype(bool),
    }, fields


def _derive_session_reset(window_number: np.ndarray) -> np.ndarray:
    numbers = np.asarray(window_number, dtype=np.int64)
    reset = np.zeros(len(numbers), dtype=bool)
    reset[0] = True
    if len(numbers) > 1:
        reset[1:] = numbers[1:] <= numbers[:-1]
    return reset


def _atomic_create_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Publish a complete immutable NPZ from an anonymous inode.

    A crash before ``linkat`` leaves no pathname; a crash after it leaves the
    final, fsynced, mode-0444 single-link file.  The caller owns idempotent
    replay and therefore treats an existing destination as a collision here.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise discovery.CampaignError(f"create-once sanitized input already exists: {path}")
    if not hasattr(os, "O_TMPFILE"):
        raise discovery.CampaignError("O_TMPFILE is required for kill-safe publication")
    descriptor = os.open(path.parent, os.O_RDWR | os.O_TMPFILE, 0o600)
    try:
        with os.fdopen(os.dup(descriptor), "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        anonymous = os.fstat(descriptor)
        if not (
            stat.S_ISREG(anonymous.st_mode)
            and stat.S_IMODE(anonymous.st_mode) == 0o444
            and anonymous.st_nlink == 0
        ):
            raise discovery.CampaignError(
                "sanitized input anonymous inode is not 0444/nlink0"
            )
        parent_flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            parent_flags |= os.O_NOFOLLOW
        parent_fd = os.open(path.parent, parent_flags)
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            linkat = libc.linkat
            linkat.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
            )
            linkat.restype = ctypes.c_int
            if linkat(descriptor, b"", parent_fd, os.fsencode(path.name), 0x1000) != 0:
                number = ctypes.get_errno()
                if number == errno.EEXIST:
                    raise discovery.CampaignError(
                        f"create-once sanitized input collision: {path}"
                    )
                raise discovery.CampaignError(
                    f"anonymous sanitized input publication failed: {os.strerror(number)}"
                )
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        published_fd = os.open(path, flags)
        try:
            published = os.fstat(published_fd)
            if not (
                stat.S_ISREG(published.st_mode)
                and stat.S_IMODE(published.st_mode) == 0o444
                and published.st_nlink == 1
                and (published.st_dev, published.st_ino)
                == (anonymous.st_dev, anonymous.st_ino)
            ):
                raise discovery.CampaignError(
                    "sanitized input publication is not the exact 0444/nlink1 inode"
                )
        finally:
            os.close(published_fd)
    finally:
        os.close(descriptor)


def validate_sanitized_input(
    path: Path,
    *,
    expected_index: np.ndarray | None = None,
) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            fields = tuple(archive.files)
            if set(fields) != set(SAFE_OUTPUT_FIELDS) or len(fields) != len(SAFE_OUTPUT_FIELDS):
                raise discovery.CampaignError(
                    f"sanitized input field allow-list mismatch: {sorted(fields)}"
                )
            arrays = {name: np.asarray(archive[name]) for name in SAFE_OUTPUT_FIELDS}
    except (OSError, ValueError, KeyError) as error:
        raise discovery.CampaignError(f"invalid sanitized input {path}: {error}") from error
    index = arrays["cache_index"].astype(np.int64)
    rows = len(index)
    if index.ndim != 1 or rows == 0 or len(np.unique(index)) != rows:
        raise discovery.CampaignError("sanitized input cache index is invalid")
    if expected_index is not None and not np.array_equal(index, expected_index):
        raise discovery.CampaignError("sanitized input exact cache-index cover drifted")
    if arrays["node_features"].shape[:2] != (rows, 12):
        raise discovery.CampaignError("sanitized node feature topology is invalid")
    if arrays["candidate_rr_bpm"].shape != (rows, 12):
        raise discovery.CampaignError("sanitized candidate topology is invalid")
    if arrays["candidate_mask"].shape != (rows, 12):
        raise discovery.CampaignError("sanitized candidate mask topology is invalid")
    if arrays["joint_radar_mask"].shape != (rows, 3):
        raise discovery.CampaignError("sanitized radar mask topology is invalid")
    for name in (
        "proposer_anchor_bpm",
        "proposer_anchor_std_bpm",
        "proposer_anchor_available",
        "classical_rr_bpm",
        "session_reset",
    ):
        if arrays[name].shape != (rows,):
            raise discovery.CampaignError(f"sanitized row field topology is invalid: {name}")
    if not np.isfinite(arrays["node_features"]).all():
        raise discovery.CampaignError("sanitized node features are non-finite")
    if not np.isfinite(arrays["candidate_rr_bpm"]).all():
        raise discovery.CampaignError("sanitized candidate values are non-finite")
    if not np.isfinite(arrays["classical_rr_bpm"]).all():
        raise discovery.CampaignError("sanitized classical RR is non-finite")
    available = arrays["proposer_anchor_available"].astype(bool)
    if np.any(available & ~np.isfinite(arrays["proposer_anchor_bpm"])) or np.any(
        available
        & (
            ~np.isfinite(arrays["proposer_anchor_std_bpm"])
            | (arrays["proposer_anchor_std_bpm"] <= 0)
        )
    ):
        raise discovery.CampaignError("sanitized proposer anchor is invalid")
    if not bool(arrays["session_reset"].astype(bool)[0]):
        raise discovery.CampaignError("sanitized input does not reset its first session")
    return {
        "rows": rows,
        "cache_index_sha256": discovery.semantic_sha256(index.tolist()),
        "fields": list(SAFE_OUTPUT_FIELDS),
        "target_fields_present": False,
        "identity_fields_present": False,
        "protocol_fields_present": False,
        "qc_fields_present": False,
    }


def _validate_sanitized_array_derivation(
    path: Path, expected: Mapping[str, np.ndarray]
) -> None:
    """Prove a resumed sanitized archive still equals its live safe inputs."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected) or len(archive.files) != len(expected):
                raise discovery.CampaignError(
                    "resumed sanitized input field set differs from its derivation"
                )
            for name in SAFE_OUTPUT_FIELDS:
                observed = np.asarray(archive[name])
                value = np.asarray(expected[name])
                if (
                    observed.dtype != value.dtype
                    or observed.shape != value.shape
                    or not np.array_equal(observed, value, equal_nan=True)
                ):
                    raise discovery.CampaignError(
                        "resumed sanitized input differs from live source values: "
                        f"{name}"
                    )
    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, discovery.CampaignError):
            raise
        raise discovery.CampaignError(
            f"cannot rederive resumed sanitized input {path}: {error}"
        ) from error


def _sanitized_receipt_value(
    *,
    outer_fold: int,
    seed: int,
    selection: Mapping[str, Any],
    authorization: Mapping[str, Any],
    governance: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
    anchor_source_fields: Sequence[str],
    output: Path,
    validation: Mapping[str, Any],
    model_source_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Reconstruct every semantic and byte binding in the create-once receipt."""

    return {
        "schema_version": 1,
        "classification": "adaptive_v3r1_v8r4_sanitized_target_free_promotion_input",
        "campaign_id": discovery.CAMPAIGN_ID,
        "campaign_revision": CAMPAIGN_REVISION,
        "infrastructure_revision": INFRASTRUCTURE_REVISION,
        "outer_fold": outer_fold,
        "seed": seed,
        "selected_variant": selection["selected_variant"],
        "selected_release_mode": selection["selected_release_mode"],
        "governance": dict(governance),
        "source_bindings": dict(source_bindings),
        "proposer_source_field_names": list(anchor_source_fields),
        "metadata_read_usecols": [],
        "source_metadata_opened_or_bound": False,
        "authorized_outer_npz_schema_exact_and_target_free": True,
        "combined_target_bearing_cache_opened_or_hashed": False,
        "outer_inference_pack_physically_separate": True,
        "output": discovery.bind_file(output),
        "validation": dict(validation),
        "exact_cache_index_cover": True,
        "target_reference_qc_identity_protocol_columns_physically_present": False,
        "target_fields_present": False,
        "future_context_present": False,
        "promotion_authorized": authorization["promotion_authorized"],
        "promotion_model_source": (
            None if model_source_binding is None else dict(model_source_binding)
        ),
        "commercial_claim_authorized": False,
    }


def build_locked_input(
    *,
    project_root: Path,
    cache_dir: Path,
    proposer_anchor: Path,
    outer_fold: int,
    seed: int,
    output: Path,
    receipt_path: Path,
    selection_lock_path: Path | None = None,
    promotion_authorization_path: Path | None = None,
    model_source: PromotionModelSource | None = None,
) -> dict[str, Any]:
    if outer_fold not in range(6) or seed not in discovery.SEEDS:
        raise discovery.CampaignError("promotion unit is outside the fixed 6x3 matrix")
    selection, authorization, governance = validate_promotion_authorization(
        project_root,
        selection_lock_path=selection_lock_path,
        authorization_path=promotion_authorization_path,
    )
    model_source_binding: dict[str, Any] | None = None
    if model_source is not None:
        if not (
            model_source.kind in {"local_training", "discovery_pointer"}
            and model_source.receipt.get("outer_fold") == outer_fold
            and model_source.receipt.get("seed") == seed
            and model_source.receipt.get("variant") == selection["selected_variant"]
        ):
            raise discovery.CampaignError("sanitized input model source identity drifted")
        model_source_binding = {
            "kind": model_source.kind,
            "receipt": discovery.bind_file(model_source.receipt_path),
            "checkpoint": discovery.bind_file(model_source.checkpoint),
            "scaler": discovery.bind_file(model_source.scaler),
            "scientific_signature_sha256": model_source.scientific_signature_sha256,
        }
    pack_manifest, outer_pack_binding = _validate_outer_pack_manifest(
        cache_dir, outer_fold, seed
    )
    pack_output = Path(str(outer_pack_binding["output"]["path"]))
    if proposer_anchor.expanduser().resolve() != pack_output:
        raise discovery.CampaignError(
            "V8R4 sanitizer accepts only the manifest-bound authorized outer pack"
        )
    authorization_path = (
        promotion_authorization_path.expanduser().resolve()
        if promotion_authorization_path is not None
        else (project_root / PROMOTION_AUTH_RELATIVE).resolve()
    )
    if dict(pack_manifest["promotion_authorization"]) != discovery.bind_file(
        authorization_path
    ):
        raise discovery.CampaignError(
            "outer prediction pack belongs to a different promotion authorization"
        )
    try:
        with np.load(pack_output, allow_pickle=False) as archive:
            if set(archive.files) != set(SAFE_OUTPUT_FIELDS):
                raise discovery.CampaignError(
                    "authorized outer prediction pack allow-list drifted"
                )
            arrays = {
                name: np.asarray(archive[name]).copy() for name in SAFE_OUTPUT_FIELDS
            }
    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, discovery.CampaignError):
            raise
        raise discovery.CampaignError(
            "authorized outer prediction pack is not pickle-free replayable"
        ) from error
    expected_index = np.asarray(arrays["cache_index"], dtype=np.int64)
    if not (
        expected_index.ndim == 1
        and len(expected_index) == int(pack_manifest["row_count"])
        and len(expected_index) > 0
        and np.all(expected_index >= 0)
        and np.all(np.diff(expected_index) > 0)
        and hashlib.sha256(
            np.ascontiguousarray(expected_index, dtype=np.int64).view(np.uint8)
        ).hexdigest()
        == pack_manifest["global_cache_index_sha256"]
    ):
        raise discovery.CampaignError("authorized outer pack cache-index proof drifted")
    source_bindings: dict[str, Any] = {
        "target_free_outer_inference_pack": outer_pack_binding,
    }
    anchor_source_fields = list(SAFE_OUTPUT_FIELDS)
    if set(arrays) != set(SAFE_OUTPUT_FIELDS):
        raise discovery.CampaignError("internal sanitized output allow-list violation")
    if receipt_path.exists() or output.exists():
        if not (receipt_path.is_file() and output.is_file()):
            raise discovery.CampaignError("partial create-once sanitized input publication")
        _require_regular_file(receipt_path, "sanitized input receipt")
        _require_regular_file(output, "sanitized input output")
        receipt = discovery.load_json(receipt_path, "sanitized input receipt")
        if discovery.canonical_content_sha256(receipt) != receipt.get("content_sha256"):
            raise discovery.CampaignError("sanitized input receipt content hash drifted")
        _validate_sanitized_array_derivation(output, arrays)
        validation = validate_sanitized_input(output, expected_index=expected_index)
        expected_receipt = _sanitized_receipt_value(
            outer_fold=outer_fold,
            seed=seed,
            selection=selection,
            authorization=authorization,
            governance=governance,
            source_bindings=source_bindings,
            anchor_source_fields=anchor_source_fields,
            output=output,
            validation=validation,
            model_source_binding=model_source_binding,
        )
        expected_receipt["content_sha256"] = discovery.semantic_sha256(
            expected_receipt
        )
        if receipt != expected_receipt:
            raise discovery.CampaignError(
                "sanitized input receipt differs from current exact provenance"
            )
        return receipt
    _atomic_create_npz(output, arrays)
    validation = validate_sanitized_input(output, expected_index=expected_index)
    return discovery.create_once_json(
        receipt_path,
        _sanitized_receipt_value(
            outer_fold=outer_fold,
            seed=seed,
            selection=selection,
            authorization=authorization,
            governance=governance,
            source_bindings=source_bindings,
            anchor_source_fields=anchor_source_fields,
            output=output,
            validation=validation,
            model_source_binding=model_source_binding,
        ),
    )


def _under(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--proposer-anchor", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--selection-lock", type=Path, default=SELECTION_RELATIVE)
    parser.add_argument(
        "--promotion-authorization", type=Path, default=PROMOTION_AUTH_RELATIVE
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.project_root.expanduser().resolve()
    output = _under(root, args.output)
    receipt = (
        _under(root, args.receipt)
        if args.receipt is not None
        else output.with_suffix(output.suffix + ".receipt.json")
    )
    try:
        result = build_locked_input(
            project_root=root,
            cache_dir=_under(root, args.cache),
            proposer_anchor=_under(root, args.proposer_anchor),
            outer_fold=args.outer_fold,
            seed=args.seed,
            output=output,
            receipt_path=receipt,
            selection_lock_path=_under(root, args.selection_lock),
            promotion_authorization_path=_under(root, args.promotion_authorization),
        )
    except discovery.CampaignError as error:
        print(json.dumps({"status": "failed_closed", "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
