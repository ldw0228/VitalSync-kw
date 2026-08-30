from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_hfr_v3_contract", ROOT / "scripts/validate_hfr_v3_contract.py"
)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


CONTRACT = ROOT / validator.CONTRACT_RELATIVE
CONFIG = ROOT / validator.CONFIG_RELATIVE


def _write_contract(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _rehash_embedded_contract(document: dict) -> None:
    payload = {key: value for key, value in document.items() if key != "semantic_hashes"}
    document["semantic_hashes"]["payload_sha256"] = validator.semantic_sha256(payload)
    document["semantic_hashes"]["section_sha256"] = {
        key: validator.semantic_sha256(value) for key, value in payload.items()
    }


def test_repository_design_contract_passes_and_returns_design_only_receipt() -> None:
    receipt = validator.validate_contract(project_root=ROOT)
    assert receipt == {
        "valid": True,
        "phase": "design",
        "campaign_id": "directed_harmonic_factor_expert_snn_v3",
        "authorization": "design_only_no_training",
        "contract_file_sha256": validator.EXPECTED_CONTRACT_FILE_SHA256,
        "contract_payload_sha256": validator.EXPECTED_PAYLOAD_SHA256,
        "config_file_sha256": validator.EXPECTED_CONFIG_FILE_SHA256,
        "config_semantic_sha256": validator.EXPECTED_CONFIG_SEMANTIC_SHA256,
        "commercial_claim_allowed": False,
    }


def test_cli_design_validation_passes() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / validator.VALIDATOR_RELATIVE), "--phase", "design"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    receipt = json.loads(completed.stdout)
    assert receipt["valid"] is True
    assert receipt["authorization"] == "design_only_no_training"


@pytest.mark.parametrize(
    ("mutator", "expected_fragment"),
    [
        (
            lambda value: value["architecture"].__setitem__("maximum_parameters", 400001),
            "contract byte hash drifted",
        ),
        (
            lambda value: value["fixed_ablations"]["order"].reverse(),
            "contract byte hash drifted",
        ),
        (
            lambda value: value["discovery_and_promotion"]["common_variant_selection"].__setitem__(
                "key", list(reversed(value["discovery_and_promotion"]["common_variant_selection"]["key"]))
            ),
            "contract byte hash drifted",
        ),
        (
            lambda value: value["leakage_and_lock_order"].__setitem__(
                "target_vault_inaccessible_before_prediction_seal", False
            ),
            "contract byte hash drifted",
        ),
        (
            lambda value: value["claim_boundary"].__setitem__("commercial_claim_allowed", True),
            "contract byte hash drifted",
        ),
    ],
)
def test_contract_drift_fails_even_if_embedded_semantic_hashes_are_recomputed(
    tmp_path: Path, mutator, expected_fragment: str
) -> None:
    document = json.loads(CONTRACT.read_text(encoding="utf-8"))
    mutator(document)
    _rehash_embedded_contract(document)
    mutated = tmp_path / "contract.json"
    _write_contract(mutated, document)

    with pytest.raises(validator.ContractValidationError, match=expected_fragment):
        validator.validate_contract(project_root=ROOT, contract_path=mutated)


def test_config_byte_and_semantic_drift_fail_closed(tmp_path: Path) -> None:
    comment_only = tmp_path / "comment_only.yaml"
    comment_only.write_bytes(CONFIG.read_bytes() + b"# drift\n")
    with pytest.raises(validator.ContractValidationError, match="config byte hash drifted"):
        validator.validate_contract(project_root=ROOT, config_path=comment_only)

    changed_config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    changed_config["resource_budget"]["gpu_hours_hard"] = 10.1
    semantic_drift = tmp_path / "semantic_drift.yaml"
    semantic_drift.write_text(yaml.safe_dump(changed_config, sort_keys=False), encoding="utf-8")

    changed_contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    changed_contract["semantic_hashes"]["bound_files"]["config_file_sha256"] = hashlib.sha256(
        semantic_drift.read_bytes()
    ).hexdigest()
    changed_contract["semantic_hashes"]["bound_files"][
        "config_semantic_sha256"
    ] = validator.semantic_sha256(changed_config)
    rebound = tmp_path / "rebound_contract.json"
    _write_contract(rebound, changed_contract)

    with pytest.raises(validator.ContractValidationError):
        validator.validate_contract(
            project_root=ROOT, contract_path=rebound, config_path=semantic_drift
        )


def test_duplicate_json_and_yaml_keys_are_rejected_before_hash_validation(tmp_path: Path) -> None:
    duplicate_json = tmp_path / "duplicate.json"
    raw_contract = CONTRACT.read_text(encoding="utf-8")
    duplicate_json.write_text(
        raw_contract.replace('  "schema_version": 1,', '  "schema_version": 1,\n  "schema_version": 1,', 1),
        encoding="utf-8",
    )
    with pytest.raises(validator.ContractValidationError, match="duplicate JSON key"):
        validator.validate_contract(project_root=ROOT, contract_path=duplicate_json)

    duplicate_yaml = tmp_path / "duplicate.yaml"
    duplicate_yaml.write_bytes(CONFIG.read_bytes() + b"campaign_id: duplicate\n")
    with pytest.raises(validator.ContractValidationError, match="duplicate YAML key"):
        validator.validate_contract(project_root=ROOT, config_path=duplicate_yaml)


def test_pretrain_phase_is_fail_closed_until_post_v2_implementation_amendment() -> None:
    with pytest.raises(validator.ContractValidationError, match="pretrain"):
        validator.validate_contract(project_root=ROOT, phase="pretrain")


def test_contract_encodes_exact_cohort_design_jobs_and_sanitized_context() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert contract["claim_boundary"]["v3_scores_are_retrospective"] is True
    assert contract["claim_boundary"]["commercial_claim_allowed"] is False
    assert contract["input_contract"]["sanitized_inference_context"]["exact_fields"] == [
        "cache_index",
        "classical_rr_bpm",
    ]
    assert contract["immutable_population"]["node_feature_schema_reference"][
        "ordered_names_semantic_sha256"
    ] == "d7553f8b11733903393575d02bc6acd4a8edefd5ce0e538491295ec84d938f05"
    assert contract["fixed_ablations"]["order"] == list(validator.EXPECTED_VARIANTS)
    assert contract["discovery_and_promotion"]["discovery"]["job_count"] == 2 * 3 * 3
    assert contract["discovery_and_promotion"]["promotion"]["job_count"] == 6 * 3
    assert contract["resource_budget"]["gpu_hours_hard"] == 10.0
    assert contract["release_policy"]["fixed_confidence_switch"] == {
        "hard_source_probability_min": 0.8,
        "otherwise": "raw_anchor",
    }
    assert config["feature_layout"]["total_width"] == 46 + 3 * 7 * 2 * 9 + 3 * 7 * 7
    assert contract["file_layout"]["required_before_training"] == list(
        validator.EXPECTED_PRETRAIN_FILES
    )


def test_validation_inputs_are_not_mutated() -> None:
    document = json.loads(CONTRACT.read_text(encoding="utf-8"))
    before = copy.deepcopy(document)
    validator._validate_design_invariants(
        document, yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    )
    assert document == before
