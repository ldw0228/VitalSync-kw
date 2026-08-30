#!/usr/bin/env python3
"""Create the one global DHFER-v3r1 variant/release-mode promotion lock.

All nine registered variant/mode combinations are evaluated over the same six
outer-validation units.  The selection key and stable tie order are taken
verbatim from the adaptive v3r1 contract.  A promotion authorization is issued
only when the winning seven-component key is strictly lexicographically better
than the bound v2-i3 key.  This selector has no outer-test or target-join input.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import run_hfr_v3r1_discovery_campaign as discovery  # noqa: E402


DEFAULT_DISCOVERY_ROOT = discovery.AGGREGATION_OUTPUT_RELATIVE
DEFAULT_SELECTION_LOCK = discovery.CAMPAIGN_RELATIVE / "DISCOVERY_SELECTION_LOCK.json"
DEFAULT_PROMOTION_AUTHORIZATION = (
    discovery.CAMPAIGN_RELATIVE / "PROMOTION_AUTHORIZATION.json"
)
PRETRAIN_AUTHORIZATION_RELATIVE = (
    discovery.CAMPAIGN_RELATIVE / "PRETRAIN_AUTHORIZATION_V8R4A_CONTEXT1.json"
)
BENCHMARK_SCRIPT = SCRIPT_ROOT / "benchmark_hfr_v3r1_efficiency.py"
CAMPAIGN_REVISION = "V8R4"
INFRASTRUCTURE_REVISION = discovery.INFRASTRUCTURE_REVISION
SHARD_INDEX_BINDINGS = {
    3: {
        "path": discovery.SHARD_TRAINING_INDEX[3].as_posix(),
        "sha256": "db204cdba72e9f3023c58ef37c1761cfd4ec2f4310449f8eaeeef7003afadb9b",
        "bytes": 3_172,
    },
    4: {
        "path": discovery.SHARD_TRAINING_INDEX[4].as_posix(),
        "sha256": "1ce1b1a0154c609b1fa1693ff5702b3e81be178f8ddbdd7ec25f559c39029f0a",
        "bytes": 3_172,
    },
}

GATE_THRESHOLDS: tuple[tuple[str, str, float], ...] = (
    ("overall_mae_bpm", "maximum", 1.0),
    ("identity_macro_mae_bpm", "maximum", 1.0),
    ("rmse_bpm", "maximum", 1.8),
    ("within_2_fraction", "minimum", 0.9),
    ("over_5_fraction", "maximum", 0.03),
    ("high_rr_25_35_mae_bpm", "maximum", 2.0),
)


def canonical_discovery_aggregation_root(
    project_root: Path, candidate: Path | None = None
) -> Path:
    """Return the sole discovery seal owner accepted by selection.

    Discovery shards write under ``discovery_v8r4/shards`` while their
    pack-free exact-cover seal is owned by the distinct
    ``discovery_v8r4/aggregation_v8r4a`` output.  Selection creation and
    locked replay must therefore agree on this same dedicated root; accepting
    the historical parent would either miss the seal or validate a different
    capability topology.
    """

    root = project_root.expanduser().resolve()
    expected = (
        DEFAULT_DISCOVERY_ROOT.resolve()
        if DEFAULT_DISCOVERY_ROOT.is_absolute()
        else (root / DEFAULT_DISCOVERY_ROOT).resolve()
    )
    if candidate is not None:
        observed_raw = candidate.expanduser()
        observed = (
            observed_raw.resolve()
            if observed_raw.is_absolute()
            else (root / observed_raw).resolve()
        )
        if observed != expected:
            raise discovery.CampaignError(
                "selection requires the canonical dedicated discovery aggregation root"
            )
    return expected


def _load_benchmark_module() -> Any:
    """Load the V8 benchmark validator without a circular top-level import."""

    spec = importlib.util.spec_from_file_location(
        "hfr_v3r1_efficiency_benchmark_for_selector", BENCHMARK_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise discovery.CampaignError("selector cannot load the V8 benchmark validator")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise discovery.CampaignError(
            f"selector cannot import the V8 benchmark validator: {error}"
        ) from error
    if not callable(getattr(module, "validate_benchmark_receipt", None)):
        raise discovery.CampaignError("V8 benchmark validator entry point is missing")
    return module


def normalized_violations(metrics: Mapping[str, Any]) -> list[float]:
    normalized = discovery.normalize_accuracy_metrics(metrics)
    result: list[float] = []
    for name, direction, threshold in GATE_THRESHOLDS:
        value = float(normalized[name])
        if not math.isfinite(value):
            raise discovery.CampaignError(f"non-finite selection metric: {name}")
        if direction == "maximum":
            violation = max(0.0, value / threshold - 1.0)
        else:
            violation = max(0.0, (threshold - value) / threshold)
        result.append(float(violation))
    return result


def global_selection_key(
    records: Sequence[Mapping[str, Any]], *, parameter_count: int
) -> tuple[float | int, ...]:
    if len(records) != 6:
        raise discovery.CampaignError("a discovery selection candidate needs six units")
    unit_keys = {
        (int(item["outer_fold"]), int(item["seed"])) for item in records
    }
    expected = {
        (fold, seed) for fold in discovery.OUTER_RUNS for seed in discovery.SEEDS
    }
    if unit_keys != expected:
        raise discovery.CampaignError("selection records are not the exact six-unit cover")
    violations = [
        value for record in records for value in normalized_violations(record["metrics"])
    ]
    macros = [float(record["metrics"]["identity_macro_mae_bpm"]) for record in records]
    maes = [float(record["metrics"]["overall_mae_bpm"]) for record in records]
    return (
        int(sum(value > 0.0 for value in violations)),
        float(max(violations, default=0.0)),
        float(math.fsum(violations)),
        float(max(macros)),
        float(round(math.fsum(macros) / len(macros), 15)),
        float(round(math.fsum(maes) / len(maes), 15)),
        int(parameter_count),
    )


def rank_candidates(rankings: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Apply the contract's stable variant-then-release tie order."""

    variant_order = {value: index for index, value in enumerate(discovery.VARIANTS)}
    mode_order = {value: index for index, value in enumerate(discovery.RELEASE_MODES)}
    seen: set[tuple[str, str]] = set()
    normalized: list[dict[str, Any]] = []
    for raw in rankings:
        item = dict(raw)
        variant = str(item.get("variant"))
        mode = str(item.get("release_mode"))
        key = (variant, mode)
        if variant not in variant_order or mode not in mode_order:
            raise discovery.CampaignError(f"unregistered selection candidate: {key}")
        if key in seen:
            raise discovery.CampaignError(f"duplicate selection candidate: {key}")
        seen.add(key)
        values = item.get("selection_key")
        if not isinstance(values, list) or len(values) != 7 or not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in values
        ):
            raise discovery.CampaignError(f"invalid selection key: {key}")
        normalized.append(item)
    return sorted(
        normalized,
        key=lambda item: (
            tuple(item["selection_key"]),
            variant_order[str(item["variant"])],
            mode_order[str(item["release_mode"])],
        ),
    )


def _opaque_index_binding_matches(
    binding: Any, *, outer_fold: int, project_root: Path
) -> bool:
    """Compare an index capability without stat/open/hash in this process."""

    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256", "bytes"}:
        return False
    expected = SHARD_INDEX_BINDINGS[outer_fold]
    raw = Path(str(binding.get("path", "")))
    observed = Path(os.path.abspath(raw if raw.is_absolute() else project_root / raw))
    canonical = Path(os.path.abspath(project_root / Path(expected["path"])))
    return bool(
        observed == canonical
        and binding.get("sha256") == expected["sha256"]
        and binding.get("bytes") == expected["bytes"]
    )


def _projected_binding_identity(
    binding: Any, *, project_root: Path
) -> tuple[Path, str, int] | None:
    if not isinstance(binding, Mapping):
        return None
    raw_path = Path(str(binding.get("path", "")))
    path = raw_path if raw_path.is_absolute() else project_root / raw_path
    digest = binding.get("sha256")
    size = binding.get("bytes")
    if not (
        isinstance(digest, str)
        and len(digest) == 64
        and type(size) is int
        and size >= 0
    ):
        return None
    return Path(os.path.abspath(path)), digest, size


def _validate_discovery_seal(
    root: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    usage_ledger: Path | None = None,
    usage_state: Any | None = None,
) -> tuple[dict[tuple[int, int, str], dict[str, Any]], dict[str, Any]]:
    """Pack-free V8R4 replay of both shard seals and the exact 18 receipts."""

    project_root = project_root.expanduser().resolve()
    seal_path = root.expanduser().resolve() / "DISCOVERY_COMPLETION_SEAL.json"
    seal, _ = _read_exact_frozen_json(seal_path, "V8R4 discovery completion seal")
    final_keys = {
        "schema_version", "classification", "campaign_id", "campaign_revision",
        "infrastructure_revision",
        "contract", "pretrain_authorization", "training_shards", "outer_runs",
        "seeds", "variants", "completed_units", "physical_boundary",
        "validation_targets_only", "gpu_elapsed_seconds", "gpu_hours_hard",
        "gpu_usage_ledger", "gpu_usage_ledger_path",
        "pre_discovery_efficiency_benchmark",
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
        and seal.get("contract", {}).get("sha256") == discovery.CONTRACT_FILE_SHA256
        and seal.get("outer_runs") == list(discovery.OUTER_RUNS)
        and seal.get("seeds") == list(discovery.SEEDS)
        and seal.get("variants") == list(discovery.VARIANTS)
        and seal.get("completed_units") == 18
        and seal.get("physical_boundary")
        == discovery.DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION
        and seal.get("validation_targets_only") is True
        and seal.get("cross_outer_validation_reuse_present") is True
        and seal.get("fully_nested_confirmatory_oof") is False
        and seal.get("prospective_confirmation_required") is True
        and seal.get("ready_for_global_discovery_selection") is True
        and seal.get("commercial_claim_authorized") is False
    ):
        raise discovery.CampaignError("V8R4 discovery completion seal is unsafe")
    shards = seal.get("training_shards")
    if not isinstance(shards, list) or len(shards) != 2:
        raise discovery.CampaignError("V8R4 selection requires exactly two shard seals")
    shard_units: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    shard_seals: dict[int, tuple[Path, Mapping[str, Any]]] = {}
    shard_benchmark_bindings: list[Mapping[str, Any]] = []
    shard_quarantine_bindings: list[Mapping[str, Any]] = []
    shard_keys = {
        "schema_version", "classification", "campaign_id", "campaign_revision",
        "infrastructure_revision",
        "outer_fold_shard", "contract", "pretrain_authorization", "training_index",
        "completed_units", "peer_outer_shard_pack_mounted_or_opened",
        "combined_target_bearing_cache_opened", "outer_prediction_pack_absent",
        "physical_boundary", "gpu_usage_ledger_prefix",
        "pre_discovery_efficiency_benchmark", "v8r3_quarantine_owner", "units",
        "cross_outer_validation_reuse_present", "fully_nested_confirmatory_oof",
        "prospective_confirmation_required", "ready_for_pack_free_shard_aggregation",
        "commercial_claim_authorized", "content_sha256",
    }
    for record in shards:
        if not isinstance(record, Mapping) or set(record) != {
            "outer_fold", "seal", "training_index"
        }:
            raise discovery.CampaignError("V8R4 final shard binding schema drifted")
        outer = int(record.get("outer_fold", -1))
        if outer not in discovery.OUTER_RUNS or outer in shard_seals:
            raise discovery.CampaignError("V8R4 shard cover is duplicated or foreign")
        if not _opaque_index_binding_matches(
            record.get("training_index"), outer_fold=outer, project_root=project_root
        ):
            raise discovery.CampaignError("V8R4 opaque shard index binding drifted")
        shard_path = discovery.verify_binding(
            record.get("seal", {}), project_root=project_root, owner=seal_path,
            label=f"V8R4 outer-{outer} shard seal",
        )
        shard, _ = _read_exact_frozen_json(
            shard_path, f"V8R4 outer-{outer} shard seal"
        )
        if not (
            set(shard) == shard_keys
            and shard.get("classification")
            == "adaptive_v3r1_v8r4_discovery_capability_shard_seal"
            and shard.get("campaign_id") == discovery.CAMPAIGN_ID
            and shard.get("campaign_revision") == CAMPAIGN_REVISION
            and shard.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
            and shard.get("outer_fold_shard") == outer
            and shard.get("completed_units") == 9
            and shard.get("peer_outer_shard_pack_mounted_or_opened") is False
            and shard.get("combined_target_bearing_cache_opened") is False
            and shard.get("outer_prediction_pack_absent") is True
            and shard.get("physical_boundary")
            == discovery.DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION
            and shard.get("cross_outer_validation_reuse_present") is True
            and shard.get("fully_nested_confirmatory_oof") is False
            and shard.get("prospective_confirmation_required") is True
            and shard.get("ready_for_pack_free_shard_aggregation") is True
            and shard.get("commercial_claim_authorized") is False
            and shard.get("training_index") == record.get("training_index")
            and shard.get("contract") == seal.get("contract")
            and shard.get("pretrain_authorization") == seal.get("pretrain_authorization")
        ):
            raise discovery.CampaignError("V8R4 shard seal invariant drifted")
        units = shard.get("units")
        if not isinstance(units, list) or len(units) != 9:
            raise discovery.CampaignError("V8R4 shard seal lacks exact nine units")
        for unit in units:
            if not isinstance(unit, Mapping) or set(unit) != {
                "outer_fold", "seed", "variant", "receipt"
            }:
                raise discovery.CampaignError("V8R4 shard unit schema drifted")
            key = (int(unit["outer_fold"]), int(unit["seed"]), str(unit["variant"]))
            if key[0] != outer or key in shard_units:
                raise discovery.CampaignError("V8R4 shard unit cover drifted")
            shard_units[key] = unit
        benchmark_binding = shard.get("pre_discovery_efficiency_benchmark")
        quarantine_binding = shard.get("v8r3_quarantine_owner")
        if not isinstance(benchmark_binding, Mapping) or not isinstance(
            quarantine_binding, Mapping
        ):
            raise discovery.CampaignError("V8R4 shard owner binding is absent")
        shard_benchmark_bindings.append(benchmark_binding)
        shard_quarantine_bindings.append(quarantine_binding)
        shard_seals[outer] = (shard_path, shard)
    if set(shard_seals) != set(discovery.OUTER_RUNS) or set(shard_units) != set(
        discovery.EXPECTED_DISCOVERY_UNITS
    ):
        raise discovery.CampaignError("V8R4 two-shard exact cover is incomplete")

    final_units = seal.get("units")
    if not isinstance(final_units, list) or len(final_units) != 18:
        raise discovery.CampaignError("V8R4 final unit cover is incomplete")
    final_by_key: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    for unit in final_units:
        if not isinstance(unit, Mapping) or set(unit) != {
            "outer_fold", "seed", "variant", "receipt"
        }:
            raise discovery.CampaignError("V8R4 final unit schema drifted")
        key = (int(unit["outer_fold"]), int(unit["seed"]), str(unit["variant"]))
        if key in final_by_key or unit != shard_units.get(key):
            raise discovery.CampaignError("V8R4 final/shard unit binding disagrees")
        final_by_key[key] = unit
    if set(final_by_key) != set(discovery.EXPECTED_DISCOVERY_UNITS):
        raise discovery.CampaignError("V8R4 final unit exact cover drifted")

    receipt_keys = {
        "schema_version", "classification", "campaign_id", "campaign_revision",
        "infrastructure_revision",
        "outer_test_opened", "outer_fold", "validation_fold", "seed", "variant",
        "invocation", "usage_ledger_path", "usage_record_sha256",
        "usage_record_sha256s", "terminal_results", "lifecycle_invocations",
        "gpu_execution_ledger_path", "gpu_admission_lock_path", "validated_output",
        "commercial_claim_authorized", "content_sha256",
    }
    receipts: dict[tuple[int, int, str], dict[str, Any]] = {}
    receipt_specs: list[tuple[Mapping[str, Any], str, Mapping[str, Any]]] = []
    execution_ledgers: set[Path] = set()
    gpu_locks: set[Path] = set()
    for key in sorted(final_by_key):
        item = final_by_key[key]
        receipt_path = discovery.verify_binding(
            item["receipt"], project_root=project_root, owner=seal_path,
            label=f"V8R4 discovery receipt {key}",
        )
        receipt, _ = _read_exact_frozen_json(
            receipt_path, f"V8R4 discovery receipt {key}"
        )
        if not (
            set(receipt) == receipt_keys
            and receipt.get("schema_version") == 1
            and receipt.get("classification")
            == "adaptive_v3r1_v8r4_discovery_unit_completion"
            and receipt.get("campaign_id") == discovery.CAMPAIGN_ID
            and receipt.get("campaign_revision") == CAMPAIGN_REVISION
            and receipt.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
            and receipt.get("outer_test_opened") is False
            and (receipt.get("outer_fold"), receipt.get("seed"), receipt.get("variant"))
            == key
            and receipt.get("validation_fold") == (key[0] + 1) % 6
            and receipt.get("commercial_claim_authorized") is False
        ):
            raise discovery.CampaignError("V8R4 discovery receipt invariant drifted")
        validated = receipt.get("validated_output")
        validated_keys = {
            "campaign_revision", "outer_fold", "validation_fold", "seed",
            "variant", "parameter_count", "validation_rows",
            "valid_reference_rows", "release_metrics",
            "scientific_signature_sha256", "physical_boundary",
            "row_access_audit", "artifacts",
        }
        if not isinstance(validated, Mapping) or not (
            set(validated) == validated_keys
            and
            validated.get("campaign_revision") == CAMPAIGN_REVISION
            and validated.get("physical_boundary")
            == discovery.DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION
            and validated.get("outer_fold") == key[0]
            and validated.get("validation_fold") == (key[0] + 1) % 6
            and validated.get("seed") == key[1]
            and validated.get("variant") == key[2]
            and type(validated.get("parameter_count")) is int
            and 0 < int(validated["parameter_count"]) <= 400_000
            and type(validated.get("validation_rows")) is int
            and int(validated["validation_rows"]) > 0
            and type(validated.get("valid_reference_rows")) is int
            and 0 < int(validated["valid_reference_rows"])
            <= int(validated["validation_rows"])
            and discovery._is_sha256(validated.get("scientific_signature_sha256"))
            and isinstance(validated.get("row_access_audit"), Mapping)
        ):
            raise discovery.CampaignError("V8R4 discovery validated output drifted")
        audit = validated["row_access_audit"]
        audit_keys = {
            "campaign_revision", "outer_fold", "physical_pack_rows",
            "outer_rows_in_physical_pack", "outer_row_access_attempts",
            "implicit_whole_array_conversions", "accesses_by_array",
            "selected_rows_by_array", "unique_accessed_cache_indexes",
            "accessed_cache_indexes_sha256",
        }
        if not (
            set(audit) == audit_keys
            and audit.get("campaign_revision") == CAMPAIGN_REVISION
            and audit.get("outer_fold") == key[0]
            and type(audit.get("physical_pack_rows")) is int
            and int(audit["physical_pack_rows"]) > 0
            and audit.get("outer_rows_in_physical_pack") == 0
            and audit.get("outer_row_access_attempts") == 0
            and audit.get("implicit_whole_array_conversions") == 0
            and isinstance(audit.get("accesses_by_array"), Mapping)
            and isinstance(audit.get("selected_rows_by_array"), Mapping)
            and type(audit.get("unique_accessed_cache_indexes")) is int
            and int(audit["unique_accessed_cache_indexes"]) > 0
            and discovery._is_sha256(audit.get("accessed_cache_indexes_sha256"))
        ):
            raise discovery.CampaignError("V8R4 row-access audit drifted")
        artifacts = validated.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != set(
            discovery.REQUIRED_TRAIN_OUTPUTS
        ):
            raise discovery.CampaignError("V8R4 discovery artifact cover is absent")
        metrics_path = discovery.verify_binding(
            artifacts.get("validation_metrics.json", {}),
            project_root=project_root, owner=receipt_path,
            label=f"V8R4 validation metrics {key}",
        )
        metrics_document = discovery.load_json(metrics_path, "V8R4 validation metrics")
        if not (
            metrics_document.get("classification")
            == "adaptive_v3r1_v8r4_discovery_validation_only"
            and metrics_document.get("campaign_revision") == CAMPAIGN_REVISION
            and metrics_document.get("outer_test_rows_present") is False
        ):
            raise discovery.CampaignError("V8R4 validation metrics schema drifted")
        observed = discovery.validation_metrics_by_release_mode(metrics_document)
        normalized = {
            mode: discovery.normalize_accuracy_metrics(observed[mode])
            for mode in discovery.RELEASE_MODES
        }
        if normalized != validated.get("release_metrics"):
            raise discovery.CampaignError("V8R4 release metrics replay drifted")
        execution_ledgers.add(Path(str(receipt["gpu_execution_ledger_path"])).resolve())
        gpu_locks.add(Path(str(receipt["gpu_admission_lock_path"])).resolve())
        receipts[key] = receipt
        receipt_specs.append(
            (receipt, "discovery", {
                "campaign_revision": CAMPAIGN_REVISION,
                "outer_fold": key[0], "seed": key[1], "variant": key[2],
            })
        )
    if len(execution_ledgers) != 1 or len(gpu_locks) != 1:
        raise discovery.CampaignError("V8R4 discovery lifecycle capabilities disagree")

    ledger_binding = seal.get("gpu_usage_ledger")
    if not isinstance(ledger_binding, Mapping):
        raise discovery.CampaignError("V8R4 discovery ledger prefix is absent")
    resolved_ledger = discovery.resolve_binding_path(
        ledger_binding.get("path"), project_root=project_root, owner=seal_path
    )
    if usage_ledger is not None and resolved_ledger != usage_ledger.expanduser().resolve():
        raise discovery.CampaignError("V8R4 selector ledger capability drifted")
    if not (
        seal.get("gpu_usage_ledger_path") == str(resolved_ledger)
        and float(seal.get("gpu_hours_hard", -1.0)) == discovery.GPU_HOURS_HARD
    ):
        raise discovery.CampaignError("V8R4 final ledger capability drifted")
    if usage_state is None:
        try:
            with discovery.gpu_budget_ledger.locked_closed_snapshot(
                resolved_ledger,
                budget_ns=discovery.gpu_budget_ledger.GPU_BUDGET_NS,
                expected_legacy_genesis_sha256=discovery._expected_legacy_genesis(
                    resolved_ledger
                ),
            ) as locked_state:
                return _validate_discovery_seal(
                    root, project_root=project_root,
                    usage_ledger=resolved_ledger, usage_state=locked_state,
                )
        except (OSError, ValueError, RuntimeError) as error:
            if isinstance(error, discovery.CampaignError):
                raise
            raise discovery.CampaignError(
                f"V8R4 selector cannot lock its ledger prefix: {error}"
            ) from error

    benchmark_owner = seal.get("pre_discovery_efficiency_benchmark")
    if not isinstance(benchmark_owner, Mapping) or set(benchmark_owner) != {
        "receipt", "included_in_gpu_exact_cover", "excluded_from_selection",
        "artifacts_quarantined",
    } or not (
        benchmark_owner.get("included_in_gpu_exact_cover") is True
        and benchmark_owner.get("excluded_from_selection") is True
        and benchmark_owner.get("artifacts_quarantined") is True
    ):
        raise discovery.CampaignError("V8R4 benchmark ownership schema drifted")
    benchmark_path = discovery.verify_binding(
        benchmark_owner["receipt"], project_root=project_root, owner=seal_path,
        label="V8R4 benchmark receipt",
    )
    if any(binding != benchmark_owner["receipt"] for binding in shard_benchmark_bindings):
        raise discovery.CampaignError("V8R4 shards bind different benchmark owners")
    benchmark_module = _load_benchmark_module()
    benchmark_receipt = benchmark_module.validate_benchmark_receipt_pack_free(
        project_root=project_root,
        receipt_path=benchmark_path,
        expected_pretrain_authorization=seal.get("pretrain_authorization"),
    )
    if _projected_binding_identity(
        benchmark_receipt.get("pretrain_authorization"), project_root=project_root
    ) != _projected_binding_identity(
        seal.get("pretrain_authorization"), project_root=project_root
    ):
        raise discovery.CampaignError(
            "benchmark and discovery used different pretrain authorizations"
        )
    receipt_specs.insert(
        0, (benchmark_receipt, benchmark_module.BENCHMARK_PHASE,
            dict(benchmark_module.BENCHMARK_USAGE_IDENTITY))
    )
    quarantine = discovery.validate_v8r3_quarantine_owner_receipt(
        project_root=project_root,
        usage_ledger=resolved_ledger,
        gpu_ledger=next(iter(execution_ledgers)),
        gpu_lock=next(iter(gpu_locks)),
        usage_state=usage_state,
    )
    quarantine_binding = seal.get("v8r3_successful_terminal_quarantine")
    canonical_quarantine = discovery.bind_file(
        (project_root / discovery.V8R3_QUARANTINE_RELATIVE).resolve(),
        relative_to=project_root,
    )
    if not (
        isinstance(quarantine_binding, Mapping)
        and dict(quarantine_binding) == canonical_quarantine
        and all(binding == canonical_quarantine for binding in shard_quarantine_bindings)
        and quarantine.get("content_sha256") is not None
    ):
        raise discovery.CampaignError("V8R3 quarantine binding drifted")
    receipt_specs.insert(1, (quarantine, "discovery_v8r3_quarantine", {}))
    positions = {
        str(record.get("record_sha256")): number
        for number, record in enumerate(usage_state.records)
    }
    benchmark_hashes = [
        str(value) for value in benchmark_receipt.get("usage_record_sha256s", [])
    ]
    discovery_hashes = [
        str(value)
        for receipt in receipts.values()
        for value in receipt.get("usage_record_sha256s", [])
    ]
    if not (
        benchmark_hashes
        and discovery_hashes
        and all(value in positions for value in benchmark_hashes + discovery_hashes)
        and max(positions[value] for value in benchmark_hashes)
        < min(positions[value] for value in discovery_hashes)
    ):
        raise discovery.CampaignError("benchmark owner did not precede discovery")
    for shard_path, shard in shard_seals.values():
        prefix = shard.get("gpu_usage_ledger_prefix")
        if not isinstance(prefix, Mapping):
            raise discovery.CampaignError("V8R4 shard ledger prefix is absent")
        discovery.verify_usage_ledger_prefix_binding(
            resolved_ledger,
            prefix,
            project_root=project_root,
            owner=shard_path,
            terminal_record_sha256=str(prefix.get("terminal_record_sha256", "")),
            usage_state=usage_state,
        )
    _, elapsed = discovery.reconcile_usage_ledger(
        resolved_ledger,
        receipt_specs,
        usage_state=usage_state,
        allow_exact_historical_benchmark_prefix=True,
    )
    if float(seal.get("gpu_elapsed_seconds", -1.0)) != elapsed:
        raise discovery.CampaignError("V8R4 discovery elapsed usage drifted")
    raw = bytes(usage_state.raw_bytes)
    terminal_hash = (
        str(usage_state.records[-1].get("record_sha256"))
        if usage_state.records else None
    )
    if not (
        ledger_binding.get("bytes") == len(raw)
        and ledger_binding.get("sha256") == hashlib.sha256(raw).hexdigest()
        and ledger_binding.get("terminal_record_sha256") == terminal_hash
        and ledger_binding.get("open_reservations") == 0
    ):
        raise discovery.CampaignError("V8R4 final ledger exact prefix drifted")
    return receipts, discovery.bind_file(seal_path, relative_to=project_root)


def _canonical_locked_document(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    document = dict(payload)
    document.pop("content_sha256", None)
    document["content_sha256"] = discovery.semantic_sha256(document)
    raw = json.dumps(
        document,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    return document, raw


def _derived_selection_documents(
    *,
    project_root: Path,
    contract: Mapping[str, Any],
    contract_binding: Mapping[str, Any],
    pretrain: Mapping[str, Any],
    receipts: Mapping[tuple[int, int, str], Mapping[str, Any]],
    seal_binding: Mapping[str, Any],
    selection_lock_path: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Purely derive the only authorized selection and promotion documents."""

    contract_discovery = contract["discovery"]
    baseline = tuple(contract_discovery["v2_baseline_key"])
    if len(baseline) != 7 or not all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in baseline
    ):
        raise discovery.CampaignError("bound v2 selection key is invalid")
    rankings: list[dict[str, Any]] = []
    for variant in discovery.VARIANTS:
        counts = {
            int(receipts[(fold, seed, variant)]["validated_output"]["parameter_count"])
            for fold in discovery.OUTER_RUNS
            for seed in discovery.SEEDS
        }
        if len(counts) != 1:
            raise discovery.CampaignError(
                f"parameter count changed across {variant} units"
            )
        parameter_count = counts.pop()
        for mode in discovery.RELEASE_MODES:
            records = []
            unit_bindings = []
            for fold in discovery.OUTER_RUNS:
                for seed in discovery.SEEDS:
                    receipt = receipts[(fold, seed, variant)]
                    metrics = receipt["validated_output"]["release_metrics"][mode]
                    records.append(
                        {"outer_fold": fold, "seed": seed, "metrics": metrics}
                    )
                    unit_bindings.append(
                        {
                            "outer_fold": fold,
                            "seed": seed,
                            "receipt_sha256": receipt["content_sha256"],
                            "validation_metrics": receipt["validated_output"][
                                "artifacts"
                            ]["validation_metrics.json"],
                        }
                    )
            key = global_selection_key(records, parameter_count=parameter_count)
            rankings.append(
                {
                    "variant": variant,
                    "release_mode": mode,
                    "selection_key": list(key),
                    "parameter_count": parameter_count,
                    "units": unit_bindings,
                }
            )
    ordered = rank_candidates(rankings)
    selected = ordered[0]
    strict_improvement = tuple(selected["selection_key"]) < baseline
    lock, lock_raw = _canonical_locked_document(
        {
            "schema_version": 1,
            "classification": "adaptive_v3r1_v8r4_global_discovery_selection_lock",
            "campaign_id": discovery.CAMPAIGN_ID,
            "campaign_revision": CAMPAIGN_REVISION,
            "infrastructure_revision": INFRASTRUCTURE_REVISION,
            "contract": dict(contract_binding),
            "pretrain_authorization": pretrain["authorization_binding"],
            "discovery_completion_seal": dict(seal_binding),
            "outer_test_opened_before_selection": False,
            "selection_scope": "one_global_variant_and_release_mode_across_all_six_units",
            "selection_key_names": list(contract_discovery["selection_key"]),
            "ordering": "lexicographic_ascending",
            "stable_tie_order": {
                "variants": list(discovery.VARIANTS),
                "release_modes": list(discovery.RELEASE_MODES),
            },
            "v2_baseline_key": list(baseline),
            "ranking": ordered,
            "selected_variant": selected["variant"],
            "selected_release_mode": selected["release_mode"],
            "selected_parameter_count": selected["parameter_count"],
            "selected_key": selected["selection_key"],
            "strict_lexicographic_improvement_over_v2": strict_improvement,
            "promotion_eligible": strict_improvement,
            "promotion_authorized": strict_improvement,
            "release_mode_or_threshold_change_after_lock_allowed": False,
            "fixed_confidence_switch_probability_min": 0.8,
            "per_fold_or_seed_selection_used": False,
            "outer_test_features_or_targets_used": False,
            "selection_process_pack_free": True,
            "cross_outer_validation_reuse_present": True,
            "fully_nested_confirmatory_oof": False,
            "prospective_confirmation_required": True,
            "adaptive_retrospective_only": True,
            "commercial_claim_authorized": False,
        }
    )
    if not strict_improvement:
        return lock, None
    resolved_lock = selection_lock_path.expanduser().resolve()
    rendered_lock = (
        resolved_lock.relative_to(project_root.resolve()).as_posix()
        if resolved_lock.is_relative_to(project_root.resolve())
        else str(resolved_lock)
    )
    selection_binding = {
        "path": rendered_lock,
        "sha256": hashlib.sha256(lock_raw).hexdigest(),
        "bytes": len(lock_raw),
    }
    authorization, _ = _canonical_locked_document(
        {
            "schema_version": 1,
            "classification": "adaptive_v3r1_v8r4_promotion_authorization",
            "campaign_id": discovery.CAMPAIGN_ID,
            "campaign_revision": CAMPAIGN_REVISION,
            "infrastructure_revision": INFRASTRUCTURE_REVISION,
            "contract_file_sha256": discovery.CONTRACT_FILE_SHA256,
            "pretrain_authorization": pretrain["authorization_binding"],
            "discovery_completion_seal": dict(seal_binding),
            "discovery_selection_lock": selection_binding,
            "selected_variant": selected["variant"],
            "selected_release_mode": selected["release_mode"],
            "fixed_confidence_switch_probability_min": 0.8,
            "authorized_now": True,
            "authorized_scopes": [
                "promotion_training_pack",
                "outer_prediction_pack",
            ],
            "training_authorized": True,
            "promotion_authorized": True,
            "target_free_sanitized_test_inputs_only": True,
            "outer_test_targets_authorized": False,
            "release_mode_or_threshold_change_allowed": False,
            "cross_outer_validation_reuse_present": True,
            "fully_nested_confirmatory_oof": False,
            "prospective_confirmation_required": True,
            "adaptive_retrospective_only": True,
            "commercial_claim_authorized": False,
        }
    )
    return lock, authorization


def _select_common_variant_locked(
    *,
    project_root: Path,
    discovery_root: Path,
    selection_lock_path: Path,
    promotion_authorization_path: Path,
    usage_ledger: Path,
    usage_state: Any,
) -> dict[str, Any]:
    contract, contract_binding = discovery.validate_contract(project_root)
    pretrain = discovery.validate_pretrain_authorization(project_root)
    receipts, seal_binding = _validate_discovery_seal(
        discovery_root,
        project_root=project_root,
        usage_ledger=usage_ledger,
        usage_state=usage_state,
    )
    seal_path = discovery_root / "DISCOVERY_COMPLETION_SEAL.json"
    seal, _seal_raw = _read_exact_frozen_json(
        seal_path, "v3r1 discovery completion seal"
    )
    seal_authorization = seal.get("pretrain_authorization")
    active_authorization = pretrain.get("authorization_binding")
    _require_active_pretrain_binding(
        project_root=project_root,
        seal_path=seal_path,
        sealed_authorization=seal_authorization,
        active_authorization=active_authorization,
    )
    expected_lock, expected_authorization = _derived_selection_documents(
        project_root=project_root,
        contract=contract,
        contract_binding=contract_binding,
        pretrain=pretrain,
        receipts=receipts,
        seal_binding=seal_binding,
        selection_lock_path=selection_lock_path,
    )
    lock = discovery.create_once_json(selection_lock_path, expected_lock)
    if expected_authorization is not None:
        discovery.create_once_json(
            promotion_authorization_path, expected_authorization
        )
    elif promotion_authorization_path.exists():
        raise discovery.CampaignError(
            "promotion authorization exists although strict improvement failed"
        )
    return lock


def _read_exact_frozen_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    source = Path(os.path.abspath(path.expanduser()))
    try:
        before_path = os.stat(source, follow_symlinks=False)
        descriptor = os.open(
            source, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
    except OSError as error:
        raise discovery.CampaignError(f"cannot open {label}: {source}") from error
    try:
        before_fd = os.fstat(descriptor)
        if not (
            stat.S_ISREG(before_fd.st_mode)
            and before_fd.st_nlink == 1
            and stat.S_IMODE(before_fd.st_mode) == 0o444
            and (before_fd.st_dev, before_fd.st_ino)
            == (before_path.st_dev, before_path.st_ino)
        ):
            raise discovery.CampaignError(
                f"{label} must be a single-link exact-0444 regular file"
            )
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            chunks.append(block)
        after_fd = os.fstat(descriptor)
        after_path = os.stat(source, follow_symlinks=False)
        stable = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(
            getattr(before_fd, name) != getattr(after_fd, name) for name in stable
        ) or any(
            getattr(after_fd, name) != getattr(after_path, name) for name in stable
        ):
            raise discovery.CampaignError(f"{label} changed while being read")
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise discovery.CampaignError(f"duplicate {label} key: {key}")
            result[key] = value
        return result

    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                discovery.CampaignError(f"non-finite {label} value: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise discovery.CampaignError(f"invalid {label}: {error}") from error
    if not isinstance(document, dict) or discovery.canonical_content_sha256(
        document
    ) != document.get("content_sha256"):
        raise discovery.CampaignError(f"{label} canonical content drifted")
    return document, raw


def _active_pretrain_for_locked_validation(
    project_root: Path, admitted_binding: Mapping[str, Any] | None
) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    return (
        discovery.validate_pretrain_authorization(
            project_root, admitted_binding=admitted_binding
        ),
        admitted_binding,
    )


def _require_active_pretrain_binding(
    *,
    project_root: Path,
    seal_path: Path,
    sealed_authorization: Any,
    active_authorization: Any,
) -> None:
    """Require two verified bindings to the one canonical active V8 authority."""

    if not isinstance(sealed_authorization, Mapping) or not isinstance(
        active_authorization, Mapping
    ):
        raise discovery.CampaignError("selection lacks an active pretrain binding")
    sealed_path = discovery.verify_binding(
        sealed_authorization,
        project_root=project_root,
        owner=seal_path,
        label="selection-bound pretrain authorization",
    )
    active_path = discovery.verify_binding(
        active_authorization,
        project_root=project_root,
        owner=project_root / "scripts/validate_hfr_v3r1_authorization.py",
        label="active selection pretrain authorization",
    )
    canonical_path = (project_root / PRETRAIN_AUTHORIZATION_RELATIVE).resolve()
    if not (
        sealed_path == active_path == canonical_path
        and sealed_authorization.get("sha256")
        == active_authorization.get("sha256")
        and sealed_authorization.get("bytes")
        == active_authorization.get("bytes")
    ):
        raise discovery.CampaignError(
            "discovery seal belongs to a different pretrain authorization"
        )


def _selection_discovery_prefix_state(
    *,
    project_root: Path,
    discovery_root: Path,
    usage_ledger: Path,
    admitted_binding: Mapping[str, Any] | None,
    current_state: Any | None = None,
) -> Any:
    seal_path = discovery_root / "DISCOVERY_COMPLETION_SEAL.json"
    seal, _seal_raw = _read_exact_frozen_json(
        seal_path, "v3r1 discovery completion seal"
    )
    ledger_binding = seal.get("gpu_usage_ledger")
    if not isinstance(ledger_binding, Mapping):
        raise discovery.CampaignError("discovery seal lacks its usage-ledger prefix")
    bound_ledger = discovery.resolve_binding_path(
        ledger_binding.get("path"), project_root=project_root, owner=seal_path
    )
    if bound_ledger != usage_ledger:
        raise discovery.CampaignError("discovery seal usage ledger is non-canonical")
    if current_state is not None:
        current_raw = bytes(current_state.raw_bytes)
    elif admitted_binding is not None:
        live_bytes = admitted_binding.get("usage_ledger_prefix_bytes")
        live_sha = admitted_binding.get("usage_ledger_prefix_sha256")
        lifecycle_id = admitted_binding.get("lifecycle_id")
        if not (
            type(live_bytes) is int
            and live_bytes >= 0
            and isinstance(live_sha, str)
            and isinstance(lifecycle_id, str)
        ):
            raise discovery.CampaignError("admitted usage-ledger prefix is malformed")
        raw = usage_ledger.read_bytes()
        if len(raw) < live_bytes or hashlib.sha256(raw[:live_bytes]).hexdigest() != live_sha:
            raise discovery.CampaignError("admitted usage-ledger live prefix drifted")
        current_raw = raw[:live_bytes]
        try:
            live_state = discovery.gpu_budget_ledger.verify_ledger_bytes(
                current_raw,
                budget_ns=discovery.gpu_budget_ledger.GPU_BUDGET_NS,
                expected_legacy_genesis_sha256=discovery._expected_legacy_genesis(
                    usage_ledger
                ),
            )
        except (OSError, ValueError, RuntimeError) as error:
            raise discovery.CampaignError(
                f"admitted live usage-ledger prefix is invalid: {error}"
            ) from error
        if set(live_state.open_reservations) != {lifecycle_id}:
            raise discovery.CampaignError(
                "admitted selection does not own the sole live reservation"
            )
    else:
        raise discovery.CampaignError("selection ledger state is unavailable")
    prefix_bytes = ledger_binding.get("bytes")
    prefix_sha = ledger_binding.get("sha256", ledger_binding.get("file_sha256"))
    if not (
        type(prefix_bytes) is int
        and 0 <= prefix_bytes <= len(current_raw)
        and isinstance(prefix_sha, str)
    ):
        raise discovery.CampaignError("discovery usage-ledger prefix binding is malformed")
    prefix = current_raw[:prefix_bytes]
    if hashlib.sha256(prefix).hexdigest() != prefix_sha:
        raise discovery.CampaignError("discovery usage-ledger prefix bytes drifted")
    try:
        state = discovery.gpu_budget_ledger.verify_ledger_bytes(
            prefix,
            budget_ns=discovery.gpu_budget_ledger.GPU_BUDGET_NS,
            expected_legacy_genesis_sha256=discovery._expected_legacy_genesis(
                usage_ledger
            ),
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise discovery.CampaignError(
            f"discovery usage-ledger prefix replay failed: {error}"
        ) from error
    if state.open_reservations:
        raise discovery.CampaignError("discovery selection prefix is not closed")
    return state


def validate_locked_selection_authorization(
    project_root: Path,
    *,
    selection_lock_path: Path,
    promotion_authorization_path: Path,
    admitted_binding: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Recompute and byte-compare the immutable V8 selection authorization."""

    root = project_root.expanduser().resolve()
    selection_candidate = selection_lock_path.expanduser()
    authorization_candidate = promotion_authorization_path.expanduser()
    if not selection_candidate.is_absolute():
        selection_candidate = root / selection_candidate
    if not authorization_candidate.is_absolute():
        authorization_candidate = root / authorization_candidate
    selection_path = Path(os.path.abspath(selection_candidate))
    authorization_path = Path(os.path.abspath(authorization_candidate))
    canonical_selection = Path(os.path.abspath(root / DEFAULT_SELECTION_LOCK))
    canonical_authorization = Path(
        os.path.abspath(root / DEFAULT_PROMOTION_AUTHORIZATION)
    )
    discovery_root = canonical_discovery_aggregation_root(root)
    usage_ledger = (root / discovery.DEFAULT_USAGE_LEDGER).resolve()
    if selection_path != canonical_selection or authorization_path != canonical_authorization:
        raise discovery.CampaignError("selection/authorization path is non-canonical")
    selection, selection_raw = _read_exact_frozen_json(
        selection_path, "v3r1 discovery selection lock"
    )
    authorization, authorization_raw = _read_exact_frozen_json(
        authorization_path, "v3r1 promotion authorization"
    )
    contract, contract_binding = discovery.validate_contract(root)
    pretrain, verified_admitted = _active_pretrain_for_locked_validation(
        root, admitted_binding
    )

    def validate_with_state(current_state: Any | None) -> tuple[
        dict[str, Any], dict[str, Any], dict[str, Any]
    ]:
        prefix_state = _selection_discovery_prefix_state(
            project_root=root,
            discovery_root=discovery_root,
            usage_ledger=usage_ledger,
            admitted_binding=verified_admitted,
            current_state=current_state,
        )
        receipts, seal_binding = _validate_discovery_seal(
            discovery_root,
            project_root=root,
            usage_ledger=usage_ledger,
            usage_state=prefix_state,
        )
        seal_path = discovery_root / "DISCOVERY_COMPLETION_SEAL.json"
        seal, _seal_raw = _read_exact_frozen_json(
            seal_path, "v3r1 discovery completion seal"
        )
        sealed_authorization = seal.get("pretrain_authorization")
        active_authorization = pretrain.get("authorization_binding")
        _require_active_pretrain_binding(
            project_root=root,
            seal_path=seal_path,
            sealed_authorization=sealed_authorization,
            active_authorization=active_authorization,
        )
        expected_selection, expected_authorization = _derived_selection_documents(
            project_root=root,
            contract=contract,
            contract_binding=contract_binding,
            pretrain=pretrain,
            receipts=receipts,
            seal_binding=seal_binding,
            selection_lock_path=selection_path,
        )
        if expected_authorization is None:
            raise discovery.CampaignError("locked selection does not authorize promotion")
        _, expected_selection_raw = _canonical_locked_document(expected_selection)
        _, expected_authorization_raw = _canonical_locked_document(
            expected_authorization
        )
        if selection_raw != expected_selection_raw:
            raise discovery.CampaignError(
                "selection lock differs from pure nine-candidate derivation"
            )
        if authorization_raw != expected_authorization_raw:
            raise discovery.CampaignError(
                "promotion authorization differs from pure selection derivation"
            )
        governance = {
            "contract": contract_binding,
            "pretrain_authorization": pretrain["authorization_binding"],
            "selection_lock": discovery.bind_file(
                selection_path, relative_to=root
            ),
            "promotion_authorization": discovery.bind_file(
                authorization_path, relative_to=root
            ),
        }
        return selection, authorization, governance

    if verified_admitted is not None:
        return validate_with_state(None)
    try:
        with discovery.gpu_budget_ledger.locked_closed_snapshot(
            usage_ledger,
            budget_ns=discovery.gpu_budget_ledger.GPU_BUDGET_NS,
            expected_legacy_genesis_sha256=discovery._expected_legacy_genesis(
                usage_ledger
            ),
        ) as current_state:
            return validate_with_state(current_state)
    except (OSError, ValueError, RuntimeError) as error:
        if isinstance(error, discovery.CampaignError):
            raise
        raise discovery.CampaignError(
            f"cannot validate locked selection on a stable ledger: {error}"
        ) from error


def select_common_variant(
    *,
    project_root: Path,
    discovery_root: Path,
    selection_lock_path: Path,
    promotion_authorization_path: Path,
) -> dict[str, Any]:
    """Validate, rank, and publish under one stable closed ledger snapshot."""

    discovery_root = canonical_discovery_aggregation_root(
        project_root, discovery_root
    )
    seal_path = discovery_root / "DISCOVERY_COMPLETION_SEAL.json"
    preview = discovery.load_json(seal_path, "v3r1 discovery completion seal")
    if discovery.canonical_content_sha256(preview) != preview.get("content_sha256"):
        raise discovery.CampaignError("discovery completion seal content hash drifted")
    ledger_binding = preview.get("gpu_usage_ledger")
    if not isinstance(ledger_binding, Mapping):
        raise discovery.CampaignError("discovery seal lacks its GPU usage prefix")
    usage_ledger = discovery.resolve_binding_path(
        ledger_binding.get("path"), project_root=project_root, owner=seal_path
    )
    try:
        with discovery.gpu_budget_ledger.locked_closed_snapshot(
            usage_ledger,
            budget_ns=discovery.gpu_budget_ledger.GPU_BUDGET_NS,
            expected_legacy_genesis_sha256=discovery._expected_legacy_genesis(
                usage_ledger
            ),
        ) as usage_state:
            return _select_common_variant_locked(
                project_root=project_root,
                discovery_root=discovery_root,
                selection_lock_path=selection_lock_path,
                promotion_authorization_path=promotion_authorization_path,
                usage_ledger=usage_ledger,
                usage_state=usage_state,
            )
    except (OSError, ValueError, RuntimeError) as error:
        if isinstance(error, discovery.CampaignError):
            raise
        raise discovery.CampaignError(
            f"selector cannot hold a stable closed GPU snapshot: {error}"
        ) from error


def _under(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--discovery-root", type=Path, default=DEFAULT_DISCOVERY_ROOT)
    parser.add_argument("--selection-lock", type=Path, default=DEFAULT_SELECTION_LOCK)
    parser.add_argument(
        "--promotion-authorization",
        type=Path,
        default=DEFAULT_PROMOTION_AUTHORIZATION,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    try:
        lock = select_common_variant(
            project_root=project_root,
            discovery_root=_under(project_root, args.discovery_root),
            selection_lock_path=_under(project_root, args.selection_lock),
            promotion_authorization_path=_under(
                project_root, args.promotion_authorization
            ),
        )
    except discovery.CampaignError as error:
        print(json.dumps({"status": "failed_closed", "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(lock, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
