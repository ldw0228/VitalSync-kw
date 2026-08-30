#!/usr/bin/env python3
"""Fail-closed validator for the predeclared retrospective DHFER-SNN v3 design.

The current contract authorizes declaration/audit only.  ``--phase pretrain`` is
intentionally stricter and cannot pass until every contracted implementation
file exists and a separately hash-bound amendment changes the authorization
status.  This keeps model/trainer creation outside the active v2 runtime seal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_RELATIVE = Path(
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3/"
    "RETROSPECTIVE_CAMPAIGN_CONTRACT.json"
)
CONFIG_RELATIVE = Path("configs/harmonic_factor_router_v3.yaml")
VALIDATOR_RELATIVE = Path("scripts/validate_hfr_v3_contract.py")
VALIDATOR_TEST_RELATIVE = Path("tests/test_validate_hfr_v3_contract.py")

EXPECTED_CONTRACT_FILE_SHA256 = (
    "fbad12762e535e34c9e15496db983e97eb1ad26ebd7833ca6b923e7f054e9538"
)
EXPECTED_PAYLOAD_SHA256 = (
    "fad17b1337af4f0e99af341c46cf4c83b10756a91a96d2de16b548b2164070d8"
)
EXPECTED_CONFIG_FILE_SHA256 = (
    "357804255e538eda6520938ab36fb9af0efeeb928fe65ac1d1384f34b5da1669"
)
EXPECTED_CONFIG_SEMANTIC_SHA256 = (
    "ae93ab38d9235c55187f4e0c14ff660e3b613460db0f2b521734ab6f0cb8d913"
)
EXPECTED_SECTION_SHA256 = {
    "schema_version": "6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b",
    "campaign_id": "9beb9b1b324c6653b0c844b16a13cc9527cd54db45e3bc8ac94f097d0e470133",
    "created_date": "5fef69c92fe2101683b1f8429852bf137726ba2164f9cb7df783ba7974881a43",
    "status": "b007f0c5cd4e7648a6c1025980b9fe5adbbfc707604104419df8321e819564c7",
    "classification": "ac643b7e65b14a04e24fd86ca82f7f87f524dd38e21a236565a4738c0d7564e9",
    "entry_condition": "2b7f211569e31bee32c664dee2061d0cc11d35c6a8d271648b4ab5a7878e920f",
    "claim_boundary": "f8dfbb6d5ee059c4acff6dd75f23760ad39262b5363059ea91904026d9290b6d",
    "objective": "c859f1931d155bee538ff6f6809263ab97024ec69b09800b572b95363482432b",
    "immutable_population": "e1d5de6cc17e2362d5798fe6d59ba365b9728302e59d7d657b74ef758e472739",
    "diagnostic_evidence_boundary": "cc9aa1cd9e0c7f4c309045c6b3e249864073fb50b7aff3a5acbedd8b1f5190bd",
    "input_contract": "d14722df273e5587231661b2c4f257e7a393492ce31b6b6e175fd044711df60c",
    "architecture": "7b7c88a381bb13bfb69b6cde5eebf052302398bd91a5b908cea6c0a8658a421d",
    "fixed_ablations": "8f808f826140851e5fb6ef94dd5da975e3740cf32c4e71440ee3bcb0b0959769",
    "fixed_training": "f03cf8dd45f5ec5f15d810fe7d2dc061c0e9604745752849f127105b921ccf7c",
    "discovery_and_promotion": "0c21a53fe1ed048dd225a0aac303bee52c3c80c8a9acf9985eaab6213aa40258",
    "leakage_and_lock_order": "3068a1f70d75a4b54629fffb2037a00dd4dba2fc3277106950458e613db8e97c",
    "release_policy": "415d524d1394309d84d4b672b73525825046cc08d48e38a4efe81ab83374ebd4",
    "resource_budget": "44eb8d7dddc2ecd0b3dcc0c03920e3f7e818577a907ccd570e743004e8c25d81",
    "file_layout": "e60baab1712d9d58d0e01e54d25a59093ebef1a97b2748f5771dc385dadd74ea",
}

EXPECTED_VARIANTS = ("H0_no_factor", "H1_factor", "H2_full")
EXPECTED_SEEDS = (20260828, 20260829, 20260830)
EXPECTED_SELECTION_KEY = (
    "total_failed_accuracy_gates",
    "worst_normalized_gate_violation",
    "summed_normalized_gate_violation",
    "worst_unit_identity_macro_mae_bpm",
    "mean_identity_macro_mae_bpm",
    "mean_overall_mae_bpm",
    "parameter_count",
)
EXPECTED_CHECKPOINT_KEY = (
    "failed_accuracy_gates",
    "worst_normalized_gate_violation",
    "summed_normalized_gate_violation",
    "identity_macro_mae_bpm",
    "overall_mae_bpm",
    "epoch",
)
EXPECTED_RELATIONS = (
    "near",
    "receiver_is_2x_sender",
    "sender_is_2x_receiver",
    "receiver_is_3x_sender",
    "sender_is_3x_receiver",
    "receiver_is_4x_sender",
    "sender_is_4x_receiver",
)
EXPECTED_PRETRAIN_FILES = (
    "src/snn_rr/harmonic_feature_layout.py",
    "src/snn_rr/harmonic_factor_router_models.py",
    "scripts/train_harmonic_factor_router_snn.py",
    "scripts/run_hfr_discovery_campaign.py",
    "scripts/select_hfr_common_variant.py",
    "scripts/run_fixed_hfr_oof_campaign.py",
    "scripts/build_locked_hfr_test_inputs.py",
    "tests/test_harmonic_feature_layout.py",
    "tests/test_harmonic_factor_router_models.py",
    "tests/test_train_harmonic_factor_router_snn.py",
    "tests/test_run_hfr_campaign.py",
    "tests/test_locked_hfr_oof.py",
)


class ContractValidationError(RuntimeError):
    """A fail-closed design, provenance, or authorization violation."""


def _fail(message: str) -> None:
    raise ContractValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        _fail(f"value is not canonical finite JSON: {error}")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _unique_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    _fail(f"non-finite JSON constant: {value}")


def load_json_strict(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        _fail(f"cannot read contract {path}: {error}")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as error:
        _fail(f"invalid strict JSON in {path}: {error}")
    if not isinstance(value, dict):
        _fail("contract root must be an object")
    return value


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            _fail(f"unhashable YAML mapping key: {error}")
        if duplicate:
            _fail(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def load_yaml_strict(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        _fail(f"cannot read config {path}: {error}")
    try:
        value = yaml.load(raw, Loader=_UniqueKeySafeLoader)
    except (yaml.YAMLError, UnicodeError) as error:
        _fail(f"invalid strict YAML in {path}: {error}")
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        _fail("config root must be a string-keyed mapping")
    _assert_finite_tree(value, "config")
    return value


def _assert_finite_tree(value: Any, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _fail(f"non-finite number at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                _fail(f"non-string mapping key at {path}: {key!r}")
            _assert_finite_tree(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_tree(child, f"{path}[{index}]")


def _at(document: Mapping[str, Any], path: str) -> Any:
    current: Any = document
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            _fail(f"missing required field {path}")
        current = current[part]
    return current


def _require_equal(actual: Any, expected: Any, path: str) -> None:
    if actual != expected or type(actual) is not type(expected):
        _fail(f"{path} drifted: expected {expected!r}, found {actual!r}")


def _validate_semantic_hashes(
    contract: Mapping[str, Any], contract_path: Path, config: Mapping[str, Any], config_path: Path
) -> None:
    actual_contract_file_hash = sha256_file(contract_path)
    if actual_contract_file_hash != EXPECTED_CONTRACT_FILE_SHA256:
        _fail("contract byte hash drifted from the predeclared validator anchor")

    recorded = _at(contract, "semantic_hashes")
    if not isinstance(recorded, Mapping):
        _fail("semantic_hashes must be an object")
    payload = {key: value for key, value in contract.items() if key != "semantic_hashes"}
    actual_payload_hash = semantic_sha256(payload)
    _require_equal(
        _at(recorded, "payload_sha256"), EXPECTED_PAYLOAD_SHA256,
        "semantic_hashes.payload_sha256",
    )
    if actual_payload_hash != EXPECTED_PAYLOAD_SHA256:
        _fail("contract payload semantic hash drifted")

    recorded_sections = _at(recorded, "section_sha256")
    if not isinstance(recorded_sections, Mapping):
        _fail("semantic_hashes.section_sha256 must be an object")
    if set(payload) != set(EXPECTED_SECTION_SHA256):
        _fail("contract top-level semantic section set drifted")
    if dict(recorded_sections) != EXPECTED_SECTION_SHA256:
        _fail("recorded section hash table drifted")
    for name, expected_hash in EXPECTED_SECTION_SHA256.items():
        if semantic_sha256(payload[name]) != expected_hash:
            _fail(f"semantic section drifted: {name}")

    bound = _at(recorded, "bound_files")
    _require_equal(
        _at(bound, "config_path"), CONFIG_RELATIVE.as_posix(),
        "semantic_hashes.bound_files.config_path",
    )
    _require_equal(
        _at(bound, "config_file_sha256"), EXPECTED_CONFIG_FILE_SHA256,
        "semantic_hashes.bound_files.config_file_sha256",
    )
    _require_equal(
        _at(bound, "config_semantic_sha256"), EXPECTED_CONFIG_SEMANTIC_SHA256,
        "semantic_hashes.bound_files.config_semantic_sha256",
    )
    if sha256_file(config_path) != EXPECTED_CONFIG_FILE_SHA256:
        _fail("config byte hash drifted")
    if semantic_sha256(config) != EXPECTED_CONFIG_SEMANTIC_SHA256:
        _fail("config semantic hash drifted")


def _validate_design_invariants(contract: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    _require_equal(contract["schema_version"], 1, "schema_version")
    _require_equal(
        contract["campaign_id"], "directed_harmonic_factor_expert_snn_v3",
        "campaign_id",
    )
    _require_equal(
        contract["status"], "predeclared_design_only_no_training_authorized", "status"
    )
    for path, expected in (
        ("entry_condition.new_v2_iteration_created", False),
        ("entry_condition.v2_files_may_be_modified", False),
        ("claim_boundary.same_historical_cohort_repeatedly_observed", True),
        ("claim_boundary.v3_scores_are_retrospective", True),
        ("claim_boundary.v3_scores_are_confirmatory", False),
        ("claim_boundary.commercial_claim_allowed", False),
        ("claim_boundary.production_release_allowed", False),
        ("input_contract.node_feature_layout.total_width", 571),
        ("input_contract.node_feature_layout.core.shape", [46]),
        ("input_contract.node_feature_layout.rf.shape", [3, 7, 2, 9]),
        ("input_contract.node_feature_layout.svd.shape", [3, 7, 7]),
        ("input_contract.node_feature_layout.masked_cells_after_outer_train_scaling", "exact_zero"),
        ("input_contract.sanitized_inference_context.exact_fields", ["cache_index", "classical_rr_bpm"]),
        ("architecture.hidden_channels", 64),
        ("architecture.directed_candidate_graph.relations", list(EXPECTED_RELATIONS)),
        ("architecture.factor_router.factor_classes", [1, 2, 3, 4]),
        ("architecture.factor_router.fixed_logit_boost", 2.0),
        ("architecture.factor_router.classical_rr_unavailable_policy", "factor affinities and route boost exact zero and factor supervision masked"),
        ("architecture.anchor_expert.residual_limit_bpm", 12.0),
        ("architecture.anchor_expert.future_lags_allowed", False),
        ("architecture.maximum_parameters", 400000),
        ("architecture.parameter_cap_is_hard", True),
        ("fixed_ablations.order", list(EXPECTED_VARIANTS)),
        ("fixed_ablations.other_variants_or_hyperparameter_sweeps_allowed", False),
        ("discovery_and_promotion.discovery.outer_folds", [3, 4]),
        ("discovery_and_promotion.discovery.seeds", list(EXPECTED_SEEDS)),
        ("discovery_and_promotion.discovery.variants", list(EXPECTED_VARIANTS)),
        ("discovery_and_promotion.discovery.job_count", 18),
        ("discovery_and_promotion.discovery.test_features_constructed", False),
        ("discovery_and_promotion.discovery.test_targets_opened", False),
        ("discovery_and_promotion.common_variant_selection.key", list(EXPECTED_SELECTION_KEY)),
        ("discovery_and_promotion.common_variant_selection.one_variant_and_release_mode_for_all_folds_and_seeds", True),
        ("discovery_and_promotion.common_variant_selection.per_fold_or_per_seed_variant_or_release_mode_selection_allowed", False),
        ("discovery_and_promotion.common_variant_selection.release_modes", ["raw_anchor", "hard_source_argmax", "fixed_confidence_switch"]),
        ("discovery_and_promotion.promotion.outer_folds", [0, 1, 2, 3, 4, 5]),
        ("discovery_and_promotion.promotion.seeds", list(EXPECTED_SEEDS)),
        ("discovery_and_promotion.promotion.job_count", 18),
        ("discovery_and_promotion.promotion.all_18_prediction_units_must_be_sealed_before_any_target_receipt", True),
        ("leakage_and_lock_order.target_vault_inaccessible_before_prediction_seal", True),
        ("leakage_and_lock_order.test_loader_may_not_accept_reference_columns", True),
        ("release_policy.commercial_release_allowed", False),
        ("release_policy.fixed_confidence_switch.hard_source_probability_min", 0.8),
        ("release_policy.fixed_confidence_switch.otherwise", "raw_anchor"),
        ("release_policy.mode_selection_uses_common_variant_selection_key", True),
        ("resource_budget.gpu_hours_hard", 10.0),
        ("resource_budget.maximum_parallel_gpu_training_jobs", 1),
        ("resource_budget.discovery_jobs_max", 18),
        ("resource_budget.promotion_jobs_max", 18),
        ("resource_budget.total_training_jobs_max", 36),
        ("resource_budget.proposer_is_frozen", True),
        ("file_layout.required_before_training", list(EXPECTED_PRETRAIN_FILES)),
        ("file_layout.model_or_trainer_source_present_at_contract_declaration", False),
        ("fixed_training.checkpoint_selection.key", list(EXPECTED_CHECKPOINT_KEY)),
        ("fixed_training.checkpoint_selection.ordering", "lexicographic_ascending"),
        ("fixed_training.checkpoint_selection.outer_test_targets_allowed", False),
    ):
        _require_equal(_at(contract, path), expected, path)

    # Width/offset arithmetic is independently recomputed instead of trusted.
    layout = _at(contract, "input_contract.node_feature_layout")
    if 46 + 3 * 7 * 2 * 9 + 3 * 7 * 7 != layout["total_width"]:
        _fail("feature layout width arithmetic is inconsistent")
    if layout["rf"]["offset"] != layout["core"]["offset"] + layout["core"]["width"]:
        _fail("RF feature offset is not contiguous")
    if layout["svd"]["offset"] != layout["rf"]["offset"] + layout["rf"]["width"]:
        _fail("SVD feature offset is not contiguous")
    if layout["svd"]["offset"] + layout["svd"]["width"] != layout["total_width"]:
        _fail("feature layout terminal offset is inconsistent")

    # The fixed losses and optimizer are exact, not a hyperparameter search space.
    expected_losses = {
        "listwise_kl": 1.0,
        "mixture_nll": 0.25,
        "component_smooth_l1": 0.3,
        "anchor_residual_smooth_l1": 0.5,
        "anchor_nll": 0.15,
        "confident_class_balanced_factor_focal": 0.35,
        "wrong_harmonic_margin": 0.25,
        "factor_candidate_js_consistency": 0.1,
        "quality_bce": 0.1,
        "spike_rate": 0.005,
        "cvar20": 0.15,
    }
    _require_equal(_at(contract, "fixed_training.losses"), expected_losses, "fixed_training.losses")
    for path, expected in (
        ("fixed_training.optimizer", "AdamW"),
        ("fixed_training.learning_rate", 0.0003),
        ("fixed_training.weight_decay", 0.0001),
        ("fixed_training.epochs_max", 120),
        ("fixed_training.minimum_epochs", 20),
        ("fixed_training.patience", 18),
        ("fixed_training.chunk_windows", 32),
        ("fixed_training.warmup_windows", 2),
        ("fixed_training.gradient_accumulation_sessions", 4),
        ("fixed_training.gradient_clip", 2.0),
        ("fixed_training.factor_focal_gamma", 2.0),
        ("fixed_training.wrong_harmonic_margin_bpm", 1.0),
        ("fixed_training.wrong_harmonic_hardest_negatives", 2),
        ("fixed_training.tail_weight", 2.0),
        ("fixed_training.cvar_quantile", 0.2),
    ):
        _require_equal(_at(contract, path), expected, path)

    # Essential cross-file equivalence catches an internally inconsistent pair.
    cross_checks = (
        ("campaign_id", "campaign_id"),
        ("architecture.hidden_channels", "model.hidden_channels"),
        ("architecture.maximum_parameters", "model.maximum_parameters"),
        ("input_contract.node_feature_layout.total_width", "feature_layout.total_width"),
        ("input_contract.node_feature_layout.core.shape", "feature_layout.core.shape"),
        ("input_contract.node_feature_layout.rf.shape", "feature_layout.rf.shape"),
        ("input_contract.node_feature_layout.svd.shape", "feature_layout.svd.shape"),
        ("immutable_population.node_feature_schema_reference.path", "feature_layout.schema_reference.path"),
        ("immutable_population.node_feature_schema_reference.sha256", "feature_layout.schema_reference.file_sha256"),
        ("immutable_population.node_feature_schema_reference.ordered_names_semantic_sha256", "feature_layout.schema_reference.ordered_names_semantic_sha256"),
        ("discovery_and_promotion.discovery.outer_folds", "selection.discovery_outer_folds"),
        ("discovery_and_promotion.discovery.seeds", "selection.seeds"),
        ("discovery_and_promotion.discovery.variants", "selection.discovery_variants"),
        ("discovery_and_promotion.common_variant_selection.release_modes", "selection.discovery_release_modes"),
        ("discovery_and_promotion.common_variant_selection.key", "selection.selection_key"),
        ("discovery_and_promotion.promotion.outer_folds", "selection.promotion_outer_folds"),
        ("fixed_training.checkpoint_selection.key", "training.checkpoint_selection.key"),
        ("fixed_training.checkpoint_selection.ordering", "training.checkpoint_selection.ordering"),
        ("fixed_training.checkpoint_selection.outer_test_targets_allowed", "training.checkpoint_selection.outer_test_targets_allowed"),
        ("resource_budget.gpu_hours_hard", "resource_budget.gpu_hours_hard"),
        ("resource_budget.total_training_jobs_max", "resource_budget.total_training_jobs_max"),
    )
    for contract_path, config_path in cross_checks:
        _require_equal(_at(contract, contract_path), _at(config, config_path), f"cross-file {contract_path}")


def _validate_files(project_root: Path, contract: Mapping[str, Any], phase: str) -> None:
    layout = _at(contract, "file_layout")
    exact_now = {
        "contract": CONTRACT_RELATIVE.as_posix(),
        "config": CONFIG_RELATIVE.as_posix(),
        "validator": VALIDATOR_RELATIVE.as_posix(),
        "validator_tests": VALIDATOR_TEST_RELATIVE.as_posix(),
    }
    for key, expected in exact_now.items():
        _require_equal(_at(layout, key), expected, f"file_layout.{key}")
        if not (project_root / expected).is_file():
            _fail(f"required declaration file is missing: {expected}")

    exact_runtime_artifacts = {
        "entry_lock": (
            "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3/"
            "V2_I3_FAILURE_ENTRY_LOCK.json"
        ),
        "discovery_selection_lock": (
            "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3/"
            "DISCOVERY_SELECTION_LOCK.json"
        ),
        "promotion_prediction_seal": (
            "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3/"
            "PROMOTION_PREDICTION_SEAL.json"
        ),
        "target_access_receipt": (
            "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3/"
            "TARGET_ACCESS_RECEIPT.json"
        ),
        "retrospective_report": (
            "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3/"
            "RETROSPECTIVE_REPORT.json"
        ),
    }
    for key, expected in exact_runtime_artifacts.items():
        _require_equal(_at(layout, key), expected, f"file_layout.{key}")

    if phase == "pretrain":
        missing = [path for path in EXPECTED_PRETRAIN_FILES if not (project_root / path).is_file()]
        if missing:
            _fail("pretrain implementation layout is incomplete: " + ", ".join(missing))
        entry_lock = project_root / exact_runtime_artifacts["entry_lock"]
        if not entry_lock.is_file():
            _fail("pretrain entry lock is missing")
        _fail(
            "pretrain remains unauthorized: contract status is design-only and must be replaced "
            "by a separately hash-bound post-v2 amendment"
        )


def _validate_immutable_bindings(project_root: Path, contract: Mapping[str, Any]) -> None:
    population = _at(contract, "immutable_population")
    for name in (
        "fold_assignments",
        "rf_cache_manifest",
        "svd_cache_manifest",
        "frozen_leader_oof",
        "node_feature_schema_reference",
    ):
        binding = _at(population, name)
        relative = _at(binding, "path")
        expected = _at(binding, "sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            _fail(f"immutable binding {name} must contain string path and sha256")
        target = project_root / relative
        if not target.is_file():
            _fail(f"immutable input is missing: {relative}")
        if sha256_file(target) != expected:
            _fail(f"immutable input hash drifted: {name}")


def validate_contract(
    *,
    project_root: Path = PROJECT_ROOT,
    contract_path: Path | None = None,
    config_path: Path | None = None,
    phase: str = "design",
) -> dict[str, Any]:
    """Validate and return a compact provenance receipt; raise on any drift."""

    root = project_root.resolve()
    contract_file = (contract_path or root / CONTRACT_RELATIVE).resolve()
    config_file = (config_path or root / CONFIG_RELATIVE).resolve()
    if phase not in {"design", "pretrain"}:
        _fail(f"unknown validation phase: {phase}")

    contract = load_json_strict(contract_file)
    config = load_yaml_strict(config_file)
    _validate_semantic_hashes(contract, contract_file, config, config_file)
    _validate_design_invariants(contract, config)
    _validate_files(root, contract, phase)
    _validate_immutable_bindings(root, contract)
    return {
        "valid": True,
        "phase": phase,
        "campaign_id": contract["campaign_id"],
        "authorization": "design_only_no_training",
        "contract_file_sha256": EXPECTED_CONTRACT_FILE_SHA256,
        "contract_payload_sha256": EXPECTED_PAYLOAD_SHA256,
        "config_file_sha256": EXPECTED_CONFIG_FILE_SHA256,
        "config_semantic_sha256": EXPECTED_CONFIG_SEMANTIC_SHA256,
        "commercial_claim_allowed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--phase", choices=("design", "pretrain"), default="design")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = validate_contract(
            project_root=args.project_root,
            contract_path=args.contract,
            config_path=args.config,
            phase=args.phase,
        )
    except ContractValidationError as error:
        print(json.dumps({"valid": False, "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
