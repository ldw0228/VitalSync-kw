from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_hfr_v3r1_v8r4a_campaign.py"
)
SPEC = importlib.util.spec_from_file_location("v8r4a_campaign_coordinator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
campaign = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = campaign
SPEC.loader.exec_module(campaign)


def promotion_state() -> Any:
    selection = {
        "selected_variant": "H2_full",
        "promotion_eligible": True,
        "promotion_authorized": True,
        "commercial_claim_authorized": False,
    }
    authorization = {
        "selected_variant": "H2_full",
        "authorized_now": True,
        "promotion_authorized": True,
        "authorized_scopes": [
            "promotion_training_pack",
            "outer_prediction_pack",
        ],
        "commercial_claim_authorized": False,
    }
    return campaign.PromotionState(
        selection=selection,
        authorization=authorization,
        governance={
            "selection_lock": {"path": "selection", "sha256": "a" * 64, "bytes": 1},
            "promotion_authorization": {
                "path": "authorization",
                "sha256": "b" * 64,
                "bytes": 1,
            },
        },
        builder_authorization=object(),
    )


class FakeBackend:
    def __init__(
        self,
        root: Path,
        *,
        fail_action: str | None = None,
        reused: Sequence[str] = (),
    ) -> None:
        self.root = root
        self.fail_action = fail_action
        self.reused = set(reused)
        self.calls: list[str] = []
        self.prelaunch_preflight_count = 0
        self.preflight_count = 0
        self.active = 0
        self.maximum_active = 0

    def prelaunch_preflight(self) -> None:
        self.prelaunch_preflight_count += 1

    def preflight(self) -> None:
        self.preflight_count += 1

    def execute_phase(self, action: str, promotion: Any | None) -> Any:
        if action.startswith("promotion_"):
            assert promotion is not None
        self.calls.append(action)
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            if action == self.fail_action:
                return campaign.PhaseOutcome(return_code=42)
            is_reused = action in self.reused
            return campaign.PhaseOutcome(
                return_code=0,
                reused=is_reused,
                launched=not is_reused,
            )
        finally:
            self.active -= 1

    def select_and_authorize(self) -> Any:
        self.calls.append("selection_and_promotion_authorization")
        return promotion_state()

    def build_promotion_training_pack(
        self, outer_fold: int, promotion: Any
    ) -> Any:
        assert promotion.selection["selected_variant"] == "H2_full"
        self.calls.append(f"promotion_training_pack_outer_{outer_fold}")
        root = self.root / f"training_pack_outer_{outer_fold}"
        return campaign.PackCapability(root=root, index=root / "index.json")

    def build_model_bound_prediction_packs(
        self, promotion: Any
    ) -> Mapping[int, Any]:
        assert promotion.authorization["promotion_authorized"] is True
        self.calls.append("model_bound_prediction_packs")
        result = {}
        for outer_fold in range(6):
            root = self.root / f"prediction_pack_outer_{outer_fold}"
            result[outer_fold] = campaign.PackCapability(
                root=root, index=root / "index.json"
            )
        return result


def test_coordinator_has_one_deterministic_complete_order(tmp_path: Path) -> None:
    backend = FakeBackend(tmp_path)
    result = campaign.CampaignCoordinator(backend).run()
    assert result.actions == campaign.EXPECTED_ACTION_ORDER
    assert tuple(backend.calls) == campaign.EXPECTED_ACTION_ORDER
    assert backend.maximum_active == 1
    assert backend.prelaunch_preflight_count == len(result.launched_phases)
    assert result.reused_phases == ()
    assert result.launched_phases == tuple(
        action
        for action in campaign.EXPECTED_ACTION_ORDER
        if action
        not in {
            "selection_and_promotion_authorization",
            "model_bound_prediction_packs",
        }
        and not action.startswith("promotion_training_pack_")
    )
    # The initial state and every mutating/launching dependency are guarded.
    assert backend.preflight_count > len(result.launched_phases)


@pytest.mark.parametrize(
    "failed",
    [
        "efficiency_benchmark",
        "discovery_outer_4",
        "discovery_aggregation",
        "promotion_training_outer_2",
        "promotion_prediction_outer_3",
        "promotion_aggregation",
    ],
)
def test_nonzero_gate_stops_every_successor(tmp_path: Path, failed: str) -> None:
    backend = FakeBackend(tmp_path, fail_action=failed)
    with pytest.raises(campaign.PhaseFailed) as captured:
        campaign.CampaignCoordinator(backend).run()
    assert captured.value.action == failed
    assert backend.calls[-1] == failed
    expected_prefix = campaign.EXPECTED_ACTION_ORDER[
        : campaign.EXPECTED_ACTION_ORDER.index(failed) + 1
    ]
    assert tuple(backend.calls) == expected_prefix


def test_resume_revalidates_full_graph_but_does_not_relaunch_completions(
    tmp_path: Path,
) -> None:
    reused = {
        "efficiency_benchmark",
        "discovery_outer_3",
        "promotion_training_outer_0",
        "promotion_prediction_outer_2",
        "promotion_aggregation",
    }
    backend = FakeBackend(tmp_path, reused=sorted(reused))
    result = campaign.CampaignCoordinator(backend).run()
    assert set(result.reused_phases) == reused
    assert not (set(result.launched_phases) & reused)
    assert tuple(backend.calls) == campaign.EXPECTED_ACTION_ORDER
    assert backend.maximum_active == 1


@pytest.mark.parametrize(
    ("selection_exists", "authorization_exists", "expected"),
    [(False, False, True), (True, False, True), (True, True, False)],
)
def test_selector_publication_prefix_is_kill_resumable(
    selection_exists: bool,
    authorization_exists: bool,
    expected: bool,
) -> None:
    assert campaign._selection_needs_publication(
        selection_exists=selection_exists,
        authorization_exists=authorization_exists,
    ) is expected


def test_selector_refuses_impossible_authorization_only_prefix() -> None:
    with pytest.raises(campaign.CampaignCoordinatorError, match="without"):
        campaign._selection_needs_publication(
            selection_exists=False,
            authorization_exists=True,
        )


def test_first_target_phase_can_recover_dead_open_lifecycle_before_strict_preflight(
    tmp_path: Path,
) -> None:
    class RecoveryBackend(FakeBackend):
        open_dead_lifecycle = True
        saw_open_prefix_preflight = False

        def prelaunch_preflight(self) -> None:
            if self.open_dead_lifecycle:
                self.saw_open_prefix_preflight = True
            super().prelaunch_preflight()

        def preflight(self) -> None:
            if self.open_dead_lifecycle:
                raise AssertionError(
                    "strict host preflight ran before target-owned recovery"
                )
            super().preflight()

        def execute_phase(self, action: str, promotion: Any | None) -> Any:
            outcome = super().execute_phase(action, promotion)
            if action == "efficiency_benchmark":
                self.open_dead_lifecycle = False
            return outcome

    backend = RecoveryBackend(tmp_path)
    result = campaign.CampaignCoordinator(backend).run()
    assert result.actions == campaign.EXPECTED_ACTION_ORDER
    assert backend.open_dead_lifecycle is False
    assert backend.saw_open_prefix_preflight is True
    assert backend.prelaunch_preflight_count == len(result.launched_phases)
    assert backend.preflight_count > 0


def test_production_prelaunch_preflight_enforces_context1_prefix_but_allows_recovery_suffix(
    tmp_path: Path,
) -> None:
    source_root = SCRIPT.parents[1]
    runtime = campaign._load_module(
        "context1_prelaunch_prefix_test_runtime",
        source_root / campaign.RUNTIME_SCRIPT_RELATIVE,
    )
    project = tmp_path / "project"
    project.mkdir()
    for row in runtime.BENCHMARK_ADMITTED_CONTEXT_LEDGER_PREFIXES.values():
        relative = Path(str(row["path"]))
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source_root / relative).read_bytes())
        target.chmod(0o644)
    active = project / campaign.CAMPAIGN_RELATIVE / (
        "PRETRAIN_AUTHORIZATION_V8R4A_CONTEXT1.json"
    )
    active.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "authorization_generation": "CONTEXT1",
        "runtime_ledger_prefixes": {
            role: dict(row)
            for role, row in runtime.BENCHMARK_ADMITTED_CONTEXT_LEDGER_PREFIXES.items()
        }
    }
    document["content_sha256"] = campaign._content_sha256(document)
    active.write_bytes(campaign._canonical_json_bytes(document) + b"\n")
    active.chmod(0o444)
    backend = object.__new__(campaign.ProductionBackend)
    backend.paths = campaign.CanonicalPaths(project)
    backend.runtime = runtime

    backend.prelaunch_preflight()
    usage = project / runtime.BENCHMARK_ADMITTED_CONTEXT_LEDGER_PREFIXES[
        "usage_ledger"
    ]["path"]
    usage.write_bytes(usage.read_bytes() + b"OPEN_RECOVERABLE_SUFFIX\n")
    backend.prelaunch_preflight()
    for generation in (None, "ROOTBIND1", "CONTEXT2", 1):
        changed = dict(document)
        if generation is None:
            changed.pop("authorization_generation")
        else:
            changed["authorization_generation"] = generation
        changed.pop("content_sha256")
        changed["content_sha256"] = campaign._content_sha256(changed)
        active.chmod(0o644)
        active.write_bytes(campaign._canonical_json_bytes(changed) + b"\n")
        active.chmod(0o444)
        with pytest.raises(
            campaign.CampaignCoordinatorError,
            match="prelaunch ledger prefix evidence drifted",
        ):
            backend.prelaunch_preflight()
    active.chmod(0o644)
    active.write_bytes(campaign._canonical_json_bytes(document) + b"\n")
    active.chmod(0o444)
    usage.write_bytes(
        b"".join(usage.read_bytes().splitlines(keepends=True)[:75])
    )
    with pytest.raises(
        campaign.CampaignCoordinatorError,
        match="prelaunch ledger prefix failed",
    ):
        backend.prelaunch_preflight()


def test_canonical_roots_are_fixed_and_phase_isolated(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    paths = campaign.CanonicalPaths(root)
    base = root / campaign.RUNS_RELATIVE
    assert paths.benchmark_output == base / "efficiency_benchmark_v8r4a_context1"
    assert paths.superseded_benchmark_output == base / "efficiency_benchmark_v8r4a"
    assert paths.superseded_contract1_benchmark_output == (
        base / "efficiency_benchmark_v8r4a_contract1"
    )
    assert paths.superseded_rootbind1_benchmark_output == (
        base / "efficiency_benchmark_v8r4a_rootbind1"
    )
    assert paths.discovery_output(3) == base / "discovery_v8r4/shards/outer_3"
    assert paths.discovery_output(4) == base / "discovery_v8r4/shards/outer_4"
    assert paths.discovery_aggregation == base / "discovery_v8r4/aggregation_v8r4a"
    assert paths.promotion_training_output(5) == (
        base / "fixed_oof_v8r4/promotion_training_shards/outer_5"
    )
    assert paths.prediction_output(0) == (
        base / "fixed_oof_v8r4/prediction_shards/outer_0"
    )
    assert paths.promotion_aggregation == base / "fixed_oof_v8r4/aggregation_v8r4a"
    assert paths.lifecycle("discovery", 3) == (
        base
        / "target_sealed_lifecycle_v8r4a_context1/discovery/"
        "run_hfr_v3r1_discovery_campaign/outer_3"
    )
    assert paths.lifecycle("promotion_aggregation", None) == (
        base
        / "target_sealed_lifecycle_v8r4a_context1/promotion_aggregation/"
        "run_fixed_hfr_v3r1_oof_campaign/global"
    )
    assert paths.superseded_lifecycle_root == (
        base / "target_sealed_lifecycle_v8r4a"
    )
    assert paths.superseded_contract1_lifecycle_root == (
        base / "target_sealed_lifecycle_v8r4a_contract1"
    )
    assert paths.superseded_rootbind1_lifecycle_root == (
        base / "target_sealed_lifecycle_v8r4a_rootbind1"
    )
    all_outputs = {
        paths.benchmark_output,
        paths.discovery_output(3),
        paths.discovery_output(4),
        paths.discovery_aggregation,
        *(paths.promotion_training_output(fold) for fold in (0, 1, 2, 5)),
        *(paths.prediction_output(fold) for fold in range(6)),
        paths.promotion_aggregation,
    }
    assert len(all_outputs) == 15


def test_denied_canary_capabilities_have_unique_host_paths(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    backend = object.__new__(campaign.ProductionBackend)
    backend.paths = campaign.CanonicalPaths(root)
    backend.runtime = SimpleNamespace(
        MANDATORY_DENIED_CANARY_ROLES=(
            "legacy_combined_cache",
            "raw_input_root",
            "target_root",
            "hai_experiment",
            "unadmitted_pack_root",
            "other_output_root",
            "superseded_v8r4a_lifecycle_root",
            "superseded_v8r4a_output_root",
            "superseded_v8r4a_contract1_lifecycle_root",
            "superseded_v8r4a_contract1_output_root",
            "superseded_v8r4a_rootbind1_lifecycle_root",
            "superseded_v8r4a_rootbind1_output_root",
        )
    )
    canaries = backend._denied_canaries("promotion_prediction", 3)
    assert set(canaries) == set(backend.runtime.MANDATORY_DENIED_CANARY_ROLES)
    assert len(set(canaries.values())) == len(canaries)
    denied_benchmark = canaries["other_output_root"]
    allowed_benchmark = backend.paths.benchmark_output
    assert denied_benchmark == (
        root / campaign.RUNS_RELATIVE / "efficiency_benchmark_v8"
    )
    assert allowed_benchmark == (
        root
        / campaign.RUNS_RELATIVE
        / "efficiency_benchmark_v8r4a_context1"
    )
    assert denied_benchmark.parent == allowed_benchmark.parent
    assert str(allowed_benchmark).startswith(str(denied_benchmark))
    with pytest.raises(ValueError):
        allowed_benchmark.relative_to(denied_benchmark)
    assert canaries["superseded_v8r4a_output_root"] == (
        root / campaign.RUNS_RELATIVE / "efficiency_benchmark_v8r4a"
    )
    assert canaries["superseded_v8r4a_lifecycle_root"] == (
        root / campaign.RUNS_RELATIVE / "target_sealed_lifecycle_v8r4a"
    )
    assert canaries["superseded_v8r4a_contract1_output_root"] == (
        root / campaign.RUNS_RELATIVE / "efficiency_benchmark_v8r4a_contract1"
    )
    assert canaries["superseded_v8r4a_contract1_lifecycle_root"] == (
        root
        / campaign.RUNS_RELATIVE
        / "target_sealed_lifecycle_v8r4a_contract1"
    )
    assert canaries["superseded_v8r4a_rootbind1_output_root"] == (
        root / campaign.RUNS_RELATIVE / "efficiency_benchmark_v8r4a_rootbind1"
    )
    assert canaries["superseded_v8r4a_rootbind1_lifecycle_root"] == (
        root
        / campaign.RUNS_RELATIVE
        / "target_sealed_lifecycle_v8r4a_rootbind1"
    )
    assert allowed_benchmark not in set(canaries.values())
    assert backend.paths.lifecycle_root not in set(canaries.values())
    assert str(canaries["superseded_v8r4a_output_root"]) in str(
        allowed_benchmark
    )
    assert str(canaries["superseded_v8r4a_lifecycle_root"]) in str(
        backend.paths.lifecycle_root
    )


def test_context1_coordinator_and_runtime_path_role_abis_are_exact() -> None:
    root = SCRIPT.parents[1]
    runtime = campaign._load_module(
        "rootbind1_runtime_abi_test",
        root / campaign.RUNTIME_SCRIPT_RELATIVE,
    )
    assert runtime.TARGET_LIFECYCLE_ROOT_RELATIVE == campaign.LIFECYCLE_RELATIVE
    assert runtime.BENCHMARK_OUTPUT_RELATIVE == campaign.BENCHMARK_RELATIVE
    assert (
        runtime.SUPERSEDED_V8R4A_LIFECYCLE_ROOT_RELATIVE
        == campaign.SUPERSEDED_LIFECYCLE_RELATIVE
    )
    assert (
        runtime.SUPERSEDED_V8R4A_OUTPUT_ROOT_RELATIVE
        == campaign.SUPERSEDED_BENCHMARK_RELATIVE
    )
    assert (
        runtime.SUPERSEDED_V8R4A_CONTRACT1_LIFECYCLE_ROOT_RELATIVE
        == campaign.SUPERSEDED_CONTRACT1_LIFECYCLE_RELATIVE
    )
    assert (
        runtime.SUPERSEDED_V8R4A_CONTRACT1_OUTPUT_ROOT_RELATIVE
        == campaign.SUPERSEDED_CONTRACT1_BENCHMARK_RELATIVE
    )
    assert (
        runtime.SUPERSEDED_V8R4A_ROOTBIND1_LIFECYCLE_ROOT_RELATIVE
        == campaign.SUPERSEDED_ROOTBIND1_LIFECYCLE_RELATIVE
    )
    assert (
        runtime.SUPERSEDED_V8R4A_ROOTBIND1_OUTPUT_ROOT_RELATIVE
        == campaign.SUPERSEDED_ROOTBIND1_BENCHMARK_RELATIVE
    )
    assert {
        "frozen_contract_encoding_correction_authorization",
        "frozen_contract_encoding_failure_diagnostic",
        "gpu_state_parent_bind_correction_authorization",
        "gpu_state_parent_bind_failure_diagnostic",
        "admitted_context_correction_authorization",
        "admitted_context_failure_diagnostic",
    } <= set(runtime.COMMON_GOVERNANCE_ROLES)
    assert all(
        "authorization_generation" in runtime.GOVERNANCE_TOP_LEVEL_KEYS[role]
        for role in (
            "implementation_test_receipt",
            "source_snapshot",
            "active_authorization",
        )
    )
    assert {
        "superseded_v8r4a_lifecycle_root",
        "superseded_v8r4a_output_root",
        "superseded_v8r4a_contract1_lifecycle_root",
        "superseded_v8r4a_contract1_output_root",
        "superseded_v8r4a_rootbind1_lifecycle_root",
        "superseded_v8r4a_rootbind1_output_root",
    } <= set(runtime.MANDATORY_DENIED_CANARY_ROLES)


def test_governance_catalog_uses_context1_successor_and_context_addendum(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    backend = object.__new__(campaign.ProductionBackend)
    backend.paths = campaign.CanonicalPaths(root)
    catalog = backend._governance_catalog(
        phase="promotion_aggregation", pack=None
    )
    assert catalog["active_authorization"].name == (
        "PRETRAIN_AUTHORIZATION_V8R4A_CONTEXT1.json"
    )
    assert catalog["source_snapshot"].name == (
        "V3R1_SOURCE_SNAPSHOT_V8R4A_CONTEXT1.json"
    )
    assert catalog["implementation_test_receipt"].name == (
        "IMPLEMENTATION_TEST_RECEIPT_V8R4A_CONTEXT1.json"
    )
    assert catalog["fd_closure_correction_authorization"].name == (
        "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_FD_CLOSURE.json"
    )
    assert catalog["fd_closure_failure_diagnostic"].name == (
        "v3r1_v8r4a_outer_guard_urandom_descriptor_failure.json"
    )
    assert catalog["canary_boundary_correction_authorization"] == (
        root
        / campaign.CAMPAIGN_RELATIVE
        / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_CANARY_BOUNDARY.json"
    )
    assert catalog["canary_boundary_failure_diagnostic"] == (
        root
        / campaign.CAMPAIGN_RELATIVE
        / "diagnostics/v3r1_v8r4a_denied_canary_prefix_collision_failure.json"
    )
    assert catalog["frozen_contract_encoding_correction_authorization"] == (
        root / campaign.FROZEN_CONTRACT_CORRECTION_RELATIVE
    )
    assert catalog["frozen_contract_encoding_failure_diagnostic"] == (
        root / campaign.FROZEN_CONTRACT_DIAGNOSTIC_RELATIVE
    )
    assert catalog["gpu_state_parent_bind_correction_authorization"] == (
        root / campaign.GPU_STATE_PARENT_BIND_CORRECTION_RELATIVE
    )
    assert catalog["gpu_state_parent_bind_failure_diagnostic"] == (
        root / campaign.GPU_STATE_PARENT_BIND_DIAGNOSTIC_RELATIVE
    )
    assert catalog["admitted_context_correction_authorization"] == (
        root / campaign.BENCHMARK_ADMITTED_CONTEXT_CORRECTION_RELATIVE
    )
    assert catalog["admitted_context_failure_diagnostic"] == (
        root / campaign.BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_RELATIVE
    )


def test_frozen_contract_correction_evidence_is_exact_and_fail_closed() -> None:
    root = SCRIPT.parents[1]
    authority, authority_binding = campaign._read_immutable_json(
        root / campaign.FROZEN_CONTRACT_CORRECTION_RELATIVE,
        "test frozen-contract authority",
    )
    diagnostic, diagnostic_binding = campaign._read_immutable_json(
        root / campaign.FROZEN_CONTRACT_DIAGNOSTIC_RELATIVE,
        "test frozen-contract diagnostic",
    )
    campaign._validate_frozen_contract_correction_evidence(
        authority,
        authority_binding,
        diagnostic,
        diagnostic_binding,
    )
    changed = {
        **authority,
        "claim_boundary": {
            **authority["claim_boundary"],
            "gpu_execution_authorized_by_this_document": True,
        },
    }
    with pytest.raises(
        campaign.CampaignCoordinatorError,
        match="correction authority boundary drifted",
    ):
        campaign._validate_frozen_contract_correction_evidence(
            changed,
            authority_binding,
            diagnostic,
            diagnostic_binding,
        )


def test_gpu_state_parent_bind_evidence_is_exact_and_fail_closed() -> None:
    root = SCRIPT.parents[1]
    authority, authority_binding = campaign._read_immutable_json(
        root / campaign.GPU_STATE_PARENT_BIND_CORRECTION_RELATIVE,
        "test GPU-state parent-bind authority",
    )
    diagnostic, diagnostic_binding = campaign._read_immutable_json(
        root / campaign.GPU_STATE_PARENT_BIND_DIAGNOSTIC_RELATIVE,
        "test GPU-state parent-bind diagnostic",
    )
    campaign._validate_gpu_state_parent_bind_correction_evidence(
        authority,
        authority_binding,
        diagnostic,
        diagnostic_binding,
    )
    changed = {
        **authority,
        "mandatory_invariants": {
            **authority["mandatory_invariants"],
            "gpu_state_root_mount_precedes_children": False,
        },
    }
    with pytest.raises(
        campaign.CampaignCoordinatorError,
        match="parent-bind correction authority boundary drifted",
    ):
        campaign._validate_gpu_state_parent_bind_correction_evidence(
            changed,
            authority_binding,
            diagnostic,
            diagnostic_binding,
        )


def test_benchmark_admitted_context_evidence_is_exact_and_fail_closed() -> None:
    root = SCRIPT.parents[1]
    authority, authority_binding = campaign._read_immutable_json(
        root / campaign.BENCHMARK_ADMITTED_CONTEXT_CORRECTION_RELATIVE,
        "test admitted-context authority",
    )
    diagnostic, diagnostic_binding = campaign._read_immutable_json(
        root / campaign.BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_RELATIVE,
        "test admitted-context diagnostic",
    )
    campaign._validate_benchmark_admitted_context_correction_evidence(
        authority,
        authority_binding,
        diagnostic,
        diagnostic_binding,
    )
    changed = {
        **authority,
        "mandatory_invariants": {
            **authority["mandatory_invariants"],
            "active_benchmark_context": {
                **authority["mandatory_invariants"]["active_benchmark_context"],
                "authorization_generation": "ROOTBIND1",
            },
        },
    }
    with pytest.raises(
        campaign.CampaignCoordinatorError,
        match="admitted-context correction authority boundary drifted",
    ):
        campaign._validate_benchmark_admitted_context_correction_evidence(
            changed,
            authority_binding,
            diagnostic,
            diagnostic_binding,
        )
    changed_diagnostic = {
        **diagnostic,
        "ledger_evidence": {
            **diagnostic["ledger_evidence"],
            "usage_postlaunch": {
                **diagnostic["ledger_evidence"]["usage_postlaunch"],
                "record_count": 76,
            },
        },
    }
    with pytest.raises(
        campaign.CampaignCoordinatorError,
        match="failure diagnostic boundary drifted",
    ):
        campaign._validate_benchmark_admitted_context_correction_evidence(
            authority,
            authority_binding,
            changed_diagnostic,
            diagnostic_binding,
        )


def test_host_helpers_reject_symlinked_ancestor_components(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    immutable = real / "governance.json"
    immutable.write_text("{}\n", encoding="utf-8")
    immutable.chmod(0o444)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(campaign.CampaignCoordinatorError, match="symlinked"):
        campaign._require_immutable_file(
            alias / immutable.name, "symlinked governance"
        )
    with pytest.raises(campaign.CampaignCoordinatorError, match="symlinked"):
        campaign._ensure_private_directory(
            alias / "output", boundary=alias
        )


def test_outer_command_is_target_sealed_and_contains_no_science_choice(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    interpreter = root / ".venv/bin/python"
    runtime = root / campaign.RUNTIME_SCRIPT_RELATIVE
    pack_root = root / "pack"
    invocation = campaign.PhaseInvocation(
        action="promotion_prediction_outer_3",
        phase="promotion_prediction",
        outer_fold=3,
        output_root=root / "output",
        lifecycle_root=root / "lifecycle",
        capability_receipt=root / "lifecycle" / campaign.CAPABILITY_FILENAME,
        pack=campaign.PackCapability(pack_root, pack_root / "index.json"),
        governance={"sealed_pack_index": pack_root / "index.json"},
        denied_canaries={
            role: root / "denied" / role
            for role in (
                "legacy_combined_cache",
                "raw_input_root",
                "target_root",
                "hai_experiment",
                "unadmitted_pack_root",
                "other_output_root",
                "superseded_v8r4a_lifecycle_root",
                "superseded_v8r4a_output_root",
                "superseded_v8r4a_contract1_lifecycle_root",
                "superseded_v8r4a_contract1_output_root",
                "superseded_v8r4a_rootbind1_lifecycle_root",
                "superseded_v8r4a_rootbind1_output_root",
            )
        },
        child_command=(
            str(interpreter),
            str(root / campaign.FIXED_SCRIPT_RELATIVE),
            "--prediction-shard",
            "3",
            "--target-sealed-capability-receipt",
            str(root / "lifecycle" / campaign.CAPABILITY_FILENAME),
        ),
    )
    argv = campaign.build_outer_runtime_argv(
        invocation,
        project_root=root,
        interpreter=interpreter,
        venv_root=root / ".venv",
        python_runtime_root=tmp_path / "python-runtime",
        cuda_runtime_roots=(),
        cuda_devices=(),
        propagated_environment={},
    )
    assert argv[:2] == (str(interpreter), str(runtime))
    outer = argv[: argv.index("--")]
    denied_rows = {
        outer[index + 1]
        for index, token in enumerate(outer)
        if token == "--deny-canary"
    }
    assert {
        f"{role}={root / 'denied' / role}"
        for role in (
            "superseded_v8r4a_lifecycle_root",
            "superseded_v8r4a_output_root",
            "superseded_v8r4a_contract1_lifecycle_root",
            "superseded_v8r4a_contract1_output_root",
            "superseded_v8r4a_rootbind1_lifecycle_root",
            "superseded_v8r4a_rootbind1_output_root",
        )
    } <= denied_rows
    child = argv[argv.index("--") + 1 :]
    assert child[:2] == (
        str(interpreter),
        str(root / campaign.FIXED_SCRIPT_RELATIVE),
    )
    assert not {
        "--variant",
        "--seed",
        "--release-mode",
        "--threshold",
        "--raw-input-root",
        "--target-root",
        "--smoke-test",
    } & set(child)
    assert Path(child[1]).name != "run_gpu_admitted.py"


def test_executor_refuses_direct_gpu_child_and_is_non_reentrant(
    tmp_path: Path,
) -> None:
    interpreter = tmp_path / "python"
    runtime = tmp_path / "run_hfr_v3r1_target_sealed.py"
    fixed = tmp_path / "run_fixed_hfr_v3r1_oof_campaign.py"
    observed: list[tuple[str, ...]] = []

    def runner(argv: Sequence[str], **kwargs: Any) -> Any:
        assert kwargs == {"check": False, "shell": False}
        observed.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0)

    executor = campaign.SynchronousTargetSealedExecutor(
        interpreter=interpreter,
        runtime_script=runtime,
        runner=runner,
    )
    command = (
        str(interpreter),
        str(runtime),
        "--phase",
        "promotion_prediction",
        "--",
        str(interpreter),
        str(fixed),
        "--prediction-shard",
        "0",
    )
    assert executor(command) == 0
    assert observed == [command]
    with pytest.raises(campaign.CampaignCoordinatorError, match="target-sealed"):
        executor(
            (
                str(interpreter),
                str(tmp_path / "run_gpu_admitted.py"),
                "--",
                str(interpreter),
                str(fixed),
            )
        )
    executor._active = True
    try:
        with pytest.raises(campaign.CampaignCoordinatorError, match="parallel"):
            executor(command)
    finally:
        executor._active = False


def test_cli_exposes_no_science_output_raw_or_target_controls() -> None:
    parser = campaign.build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert option_strings == {"-h", "--help", "--project-root"}
    with pytest.raises(SystemExit):
        parser.parse_args(["--outer-fold", "3"])
