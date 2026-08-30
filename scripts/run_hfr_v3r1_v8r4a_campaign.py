#!/usr/bin/env python3
"""Run the fixed V8R4A target-sealed campaign dependency graph.

This coordinator is deliberately CPU-only.  It may validate immutable
governance, build target-free capability packs, and resolve already-completed
model sources on the host.  Every phase that can start a GPU workload is
launched synchronously through :mod:`run_hfr_v3r1_target_sealed`; the
coordinator never invokes the admission wrapper or a trainer itself.

There are no command-line controls for folds, seeds, variants, thresholds,
release modes, output roots, raw data, or targets.  The only executable graph
is the V8R4A graph authorized by the execution-closure addendum.  A non-zero
phase result or a failed selection gate stops all successors.  Re-running the
coordinator replays the same immutable builders and target-sealed lifecycle
receipts, so a completion is reused only after the owning component has
validated its exact bytes and current append-only ledger prefix.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from types import ModuleType
from typing import Any, Callable, Final, Mapping, Protocol, Sequence


CAMPAIGN_ID: Final[str] = (
    "directed_harmonic_factor_expert_snn_v3r1_adaptive_retrospective"
)
SCIENTIFIC_REVISION: Final[str] = "V8R4"
INFRASTRUCTURE_REVISION: Final[str] = "V8R4A"
SEEDS: Final[tuple[int, ...]] = (20260828, 20260829, 20260830)
DISCOVERY_FOLDS: Final[tuple[int, ...]] = (3, 4)
PROMOTION_TRAINING_FOLDS: Final[tuple[int, ...]] = (0, 1, 2, 5)
ALL_FOLDS: Final[tuple[int, ...]] = tuple(range(6))

RUNS_RELATIVE: Final[Path] = Path(
    "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1"
)
CAMPAIGN_RELATIVE: Final[Path] = Path(
    "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1"
)
STATE_RELATIVE: Final[Path] = RUNS_RELATIVE / "gpu_state_v8r4a"
DISCOVERY_PACKS_RELATIVE: Final[Path] = RUNS_RELATIVE / "v8r4_split_inputs"
PROMOTION_PACKS_RELATIVE: Final[Path] = (
    RUNS_RELATIVE / "v8r4_promotion_authorized_inputs"
)
SUPERSEDED_BENCHMARK_RELATIVE: Final[Path] = (
    RUNS_RELATIVE / "efficiency_benchmark_v8r4a"
)
SUPERSEDED_CONTRACT1_BENCHMARK_RELATIVE: Final[Path] = (
    RUNS_RELATIVE / "efficiency_benchmark_v8r4a_contract1"
)
SUPERSEDED_ROOTBIND1_BENCHMARK_RELATIVE: Final[Path] = (
    RUNS_RELATIVE / "efficiency_benchmark_v8r4a_rootbind1"
)
BENCHMARK_RELATIVE: Final[Path] = (
    RUNS_RELATIVE / "efficiency_benchmark_v8r4a_context1"
)
DISCOVERY_RELATIVE: Final[Path] = RUNS_RELATIVE / "discovery_v8r4"
FIXED_RELATIVE: Final[Path] = RUNS_RELATIVE / "fixed_oof_v8r4"
SUPERSEDED_LIFECYCLE_RELATIVE: Final[Path] = (
    RUNS_RELATIVE / "target_sealed_lifecycle_v8r4a"
)
SUPERSEDED_CONTRACT1_LIFECYCLE_RELATIVE: Final[Path] = (
    RUNS_RELATIVE / "target_sealed_lifecycle_v8r4a_contract1"
)
SUPERSEDED_ROOTBIND1_LIFECYCLE_RELATIVE: Final[Path] = (
    RUNS_RELATIVE / "target_sealed_lifecycle_v8r4a_rootbind1"
)
LIFECYCLE_RELATIVE: Final[Path] = (
    RUNS_RELATIVE / "target_sealed_lifecycle_v8r4a_context1"
)

FROZEN_CONTRACT_CORRECTION_RELATIVE: Final[Path] = CAMPAIGN_RELATIVE / (
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_FROZEN_CONTRACT_ENCODING.json"
)
FROZEN_CONTRACT_CORRECTION_FILE_SHA256: Final[str] = (
    "2a59c7de8b8aefcbb3d21691db5a8da1e4b8f1a5461024f0338f389ff8ddcda1"
)
FROZEN_CONTRACT_CORRECTION_CONTENT_SHA256: Final[str] = (
    "b0df4b5d34bb5f55c6254d83459f81ee297177909658d101866d9e32c6c48c6f"
)
FROZEN_CONTRACT_CORRECTION_BYTES: Final[int] = 13_460
FROZEN_CONTRACT_DIAGNOSTIC_RELATIVE: Final[Path] = CAMPAIGN_RELATIVE / (
    "diagnostics/v3r1_v8r4a_frozen_contract_encoding_false_rejection_failure.json"
)
FROZEN_CONTRACT_DIAGNOSTIC_FILE_SHA256: Final[str] = (
    "1d358bdecb1a5605b138b1916451f8c306f957a27257d328aa71de8d83cbf8ee"
)
FROZEN_CONTRACT_DIAGNOSTIC_CONTENT_SHA256: Final[str] = (
    "bec8b74c21c0b0882f9ab147f68c1aa947e259ec39bf36557f3fdcdeb86abcc7"
)
FROZEN_CONTRACT_DIAGNOSTIC_BYTES: Final[int] = 8_653

GPU_STATE_PARENT_BIND_CORRECTION_RELATIVE: Final[Path] = CAMPAIGN_RELATIVE / (
    "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_GPU_STATE_PARENT_BIND.json"
)
GPU_STATE_PARENT_BIND_CORRECTION_FILE_SHA256: Final[str] = (
    "b73d68199acad6fff780c76f05bd3daadc62b03c160af6efc407792efa87a4cd"
)
GPU_STATE_PARENT_BIND_CORRECTION_CONTENT_SHA256: Final[str] = (
    "7917a0b003241181b6dd6fca6c127301538717050cff0bd63dd087fbcdaa07bf"
)
GPU_STATE_PARENT_BIND_CORRECTION_BYTES: Final[int] = 14_858
GPU_STATE_PARENT_BIND_DIAGNOSTIC_RELATIVE: Final[Path] = CAMPAIGN_RELATIVE / (
    "diagnostics/v3r1_v8r4a_gpu_state_parent_mount_identity_failure.json"
)
GPU_STATE_PARENT_BIND_DIAGNOSTIC_FILE_SHA256: Final[str] = (
    "fd2aa011020289d94ab0548c679888ecc924d3ec179ca5908560a6e82074d628"
)
GPU_STATE_PARENT_BIND_DIAGNOSTIC_CONTENT_SHA256: Final[str] = (
    "e4a315bc83e333d31920baef4c3db0f8cb2adc3b5f7e59b73a39795986073b67"
)
GPU_STATE_PARENT_BIND_DIAGNOSTIC_BYTES: Final[int] = 10_709

BENCHMARK_ADMITTED_CONTEXT_CORRECTION_RELATIVE: Final[Path] = (
    CAMPAIGN_RELATIVE
    / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_BENCHMARK_ADMITTED_CONTEXT.json"
)
BENCHMARK_ADMITTED_CONTEXT_CORRECTION_FILE_SHA256: Final[str] = (
    "c0646a3fb0e5b673850e570f7d0a1e91676e5116890d1a8e758e6603bbfa31e2"
)
BENCHMARK_ADMITTED_CONTEXT_CORRECTION_CONTENT_SHA256: Final[str] = (
    "d48ff6cb78fcf94e6d994cca96b144daca9da19f873bc8f5ef7e15246e6a1f5c"
)
BENCHMARK_ADMITTED_CONTEXT_CORRECTION_BYTES: Final[int] = 16_684
BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_RELATIVE: Final[Path] = (
    CAMPAIGN_RELATIVE
    / "diagnostics/v3r1_v8r4a_benchmark_admitted_pretrain_context_bridge_failure.json"
)
BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_FILE_SHA256: Final[str] = (
    "b7a360902a68c4a7cb72d320c2042bccaf965a6ea9df64b0d203a40dc64dd088"
)
BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_CONTENT_SHA256: Final[str] = (
    "51ffa6135eec896c385878b42ecd3d6bb440fad5965532d04341cec4cb4eb83e"
)
BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_BYTES: Final[int] = 9_019

RUNTIME_SCRIPT_RELATIVE: Final[Path] = Path(
    "scripts/run_hfr_v3r1_target_sealed.py"
)
BENCHMARK_SCRIPT_RELATIVE: Final[Path] = Path(
    "scripts/benchmark_hfr_v3r1_efficiency.py"
)
DISCOVERY_SCRIPT_RELATIVE: Final[Path] = Path(
    "scripts/run_hfr_v3r1_discovery_campaign.py"
)
FIXED_SCRIPT_RELATIVE: Final[Path] = Path(
    "scripts/run_fixed_hfr_v3r1_oof_campaign.py"
)
PACK_BUILDER_RELATIVE: Final[Path] = Path(
    "scripts/build_hfr_v3r1_sealed_input_pack_v8r4.py"
)
LOCKED_INPUTS_RELATIVE: Final[Path] = Path(
    "scripts/build_locked_hfr_v3r1_test_inputs.py"
)
SELECTOR_RELATIVE: Final[Path] = Path(
    "scripts/select_hfr_v3r1_common_variant.py"
)
VALIDATOR_RELATIVE: Final[Path] = Path(
    "scripts/validate_hfr_v3r1_authorization.py"
)
MIGRATOR_RELATIVE: Final[Path] = Path(
    "scripts/migrate_hfr_v3r1_gpu_state_v8r4a.py"
)

CAPABILITY_FILENAME: Final[str] = "TARGET_SEALED_CAPABILITY_RECEIPT_V8R4A.json"
COMPLETION_FILENAME: Final[str] = "TARGET_SEALED_COMPLETION_RECEIPT_V8R4A.json"
TRAINING_INDEX_FILENAME: Final[str] = "V8R4_NONOUTER_TRAINING_INDEX.json"
MODEL_BOUND_INDEX_FILENAME: Final[str] = (
    "V8R4A_MODEL_BOUND_TARGET_FREE_PREDICTION_INDEX.json"
)
MODEL_SOURCE_SEAL_FILENAME: Final[str] = "MODEL_SOURCE_SHARD_SEAL.json"
DISCOVERY_SHARD_SEAL_FILENAME: Final[str] = (
    "DISCOVERY_SHARD_COMPLETION_SEAL.json"
)
DISCOVERY_COMPLETION_FILENAME: Final[str] = "DISCOVERY_COMPLETION_SEAL.json"
PREDICTION_SHARD_SEAL_FILENAME: Final[str] = (
    "PREDICTION_SHARD_COMPLETION_SEAL.json"
)

EXPECTED_ACTION_ORDER: Final[tuple[str, ...]] = (
    "efficiency_benchmark",
    "discovery_outer_3",
    "discovery_outer_4",
    "discovery_aggregation",
    "selection_and_promotion_authorization",
    "promotion_training_pack_outer_0",
    "promotion_training_outer_0",
    "promotion_training_pack_outer_1",
    "promotion_training_outer_1",
    "promotion_training_pack_outer_2",
    "promotion_training_outer_2",
    "promotion_training_pack_outer_5",
    "promotion_training_outer_5",
    "model_bound_prediction_packs",
    "promotion_prediction_outer_0",
    "promotion_prediction_outer_1",
    "promotion_prediction_outer_2",
    "promotion_prediction_outer_3",
    "promotion_prediction_outer_4",
    "promotion_prediction_outer_5",
    "promotion_aggregation",
)


class CampaignCoordinatorError(RuntimeError):
    """The fixed campaign graph cannot continue without violating a gate."""


class PhaseFailed(CampaignCoordinatorError):
    """A target-sealed phase returned a terminal non-zero status."""

    def __init__(self, action: str, return_code: int) -> None:
        super().__init__(f"{action} failed closed with return code {return_code}")
        self.action = action
        self.return_code = return_code


@dataclass(frozen=True, slots=True)
class CanonicalPaths:
    """Every mutable or capability-bearing path in the fixed campaign."""

    project_root: Path

    def __post_init__(self) -> None:
        root = self.project_root.expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        root = Path(os.path.abspath(root))
        try:
            resolved = root.resolve(strict=True)
        except OSError as error:
            raise CampaignCoordinatorError(
                f"project root is unavailable: {root}: {error}"
            ) from error
        if resolved != root or not root.is_dir():
            raise CampaignCoordinatorError(
                "project root must be an existing non-symlink directory"
            )
        object.__setattr__(self, "project_root", root)

    @property
    def runs(self) -> Path:
        return self.project_root / RUNS_RELATIVE

    @property
    def campaign(self) -> Path:
        return self.project_root / CAMPAIGN_RELATIVE

    @property
    def state(self) -> Path:
        return self.project_root / STATE_RELATIVE

    @property
    def benchmark_output(self) -> Path:
        return self.project_root / BENCHMARK_RELATIVE

    @property
    def superseded_benchmark_output(self) -> Path:
        return self.project_root / SUPERSEDED_BENCHMARK_RELATIVE

    @property
    def superseded_contract1_benchmark_output(self) -> Path:
        return self.project_root / SUPERSEDED_CONTRACT1_BENCHMARK_RELATIVE

    @property
    def superseded_rootbind1_benchmark_output(self) -> Path:
        return self.project_root / SUPERSEDED_ROOTBIND1_BENCHMARK_RELATIVE

    @property
    def discovery_root(self) -> Path:
        return self.project_root / DISCOVERY_RELATIVE

    def discovery_output(self, outer_fold: int) -> Path:
        _require_member(outer_fold, DISCOVERY_FOLDS, "discovery outer fold")
        return self.discovery_root / "shards" / f"outer_{outer_fold}"

    @property
    def discovery_aggregation(self) -> Path:
        return self.discovery_root / "aggregation_v8r4a"

    @property
    def fixed_root(self) -> Path:
        return self.project_root / FIXED_RELATIVE

    def promotion_training_output(self, outer_fold: int) -> Path:
        _require_member(
            outer_fold, PROMOTION_TRAINING_FOLDS, "promotion training outer fold"
        )
        return self.fixed_root / "promotion_training_shards" / f"outer_{outer_fold}"

    def prediction_output(self, outer_fold: int) -> Path:
        _require_member(outer_fold, ALL_FOLDS, "prediction outer fold")
        return self.fixed_root / "prediction_shards" / f"outer_{outer_fold}"

    @property
    def promotion_aggregation(self) -> Path:
        return self.fixed_root / "aggregation_v8r4a"

    @property
    def lifecycle_root(self) -> Path:
        return self.project_root / LIFECYCLE_RELATIVE

    @property
    def superseded_lifecycle_root(self) -> Path:
        return self.project_root / SUPERSEDED_LIFECYCLE_RELATIVE

    @property
    def superseded_contract1_lifecycle_root(self) -> Path:
        return self.project_root / SUPERSEDED_CONTRACT1_LIFECYCLE_RELATIVE

    @property
    def superseded_rootbind1_lifecycle_root(self) -> Path:
        return self.project_root / SUPERSEDED_ROOTBIND1_LIFECYCLE_RELATIVE

    def lifecycle(self, phase: str, outer_fold: int | None) -> Path:
        entry_stems = {
            "efficiency_benchmark": "benchmark_hfr_v3r1_efficiency",
            "discovery": "run_hfr_v3r1_discovery_campaign",
            "discovery_aggregation": "run_hfr_v3r1_discovery_campaign",
            "promotion_training": "run_fixed_hfr_v3r1_oof_campaign",
            "promotion_prediction": "run_fixed_hfr_v3r1_oof_campaign",
            "promotion_aggregation": "run_fixed_hfr_v3r1_oof_campaign",
        }
        if phase not in entry_stems:
            raise CampaignCoordinatorError(f"unknown canonical phase: {phase}")
        scope = "global" if outer_fold is None else f"outer_{outer_fold}"
        return self.lifecycle_root / phase / entry_stems[phase] / scope

    def discovery_pack_root(self, outer_fold: int) -> Path:
        _require_member(outer_fold, DISCOVERY_FOLDS, "discovery pack outer fold")
        return (
            self.project_root
            / DISCOVERY_PACKS_RELATIVE
            / f"discovery_shard_outer_{outer_fold}"
        )

    def discovery_pack_index(self, outer_fold: int) -> Path:
        return self.discovery_pack_root(outer_fold) / TRAINING_INDEX_FILENAME

    @property
    def promotion_pack_root(self) -> Path:
        return self.project_root / PROMOTION_PACKS_RELATIVE

    def promotion_training_pack_root(self, outer_fold: int) -> Path:
        _require_member(
            outer_fold, PROMOTION_TRAINING_FOLDS, "promotion pack outer fold"
        )
        return (
            self.promotion_pack_root
            / "training_shards"
            / f"promotion_training_shard_outer_{outer_fold}"
        )

    def promotion_training_pack_index(self, outer_fold: int) -> Path:
        return self.promotion_training_pack_root(outer_fold) / TRAINING_INDEX_FILENAME

    def model_bound_pack_root(self, outer_fold: int) -> Path:
        _require_member(outer_fold, ALL_FOLDS, "model-bound pack outer fold")
        return (
            self.promotion_pack_root
            / "model_bound_prediction_shards"
            / f"prediction_shard_outer_{outer_fold}"
        )

    def model_bound_pack_index(self, outer_fold: int) -> Path:
        return self.model_bound_pack_root(outer_fold) / MODEL_BOUND_INDEX_FILENAME

    @property
    def selection_lock(self) -> Path:
        return self.campaign / "DISCOVERY_SELECTION_LOCK.json"

    @property
    def promotion_authorization(self) -> Path:
        return self.campaign / "PROMOTION_AUTHORIZATION.json"


@dataclass(frozen=True, slots=True)
class PackCapability:
    root: Path
    index: Path


@dataclass(frozen=True, slots=True)
class PromotionState:
    selection: Mapping[str, Any]
    authorization: Mapping[str, Any]
    governance: Mapping[str, Mapping[str, Any]]
    builder_authorization: Any = field(repr=False)


@dataclass(frozen=True, slots=True)
class PhaseInvocation:
    action: str
    phase: str
    outer_fold: int | None
    output_root: Path
    lifecycle_root: Path
    capability_receipt: Path
    pack: PackCapability | None
    governance: Mapping[str, Path]
    denied_canaries: Mapping[str, Path]
    child_command: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PhaseOutcome:
    return_code: int
    reused: bool = False
    launched: bool = True


@dataclass(frozen=True, slots=True)
class CampaignResult:
    actions: tuple[str, ...]
    reused_phases: tuple[str, ...]
    launched_phases: tuple[str, ...]


class CoordinatorBackend(Protocol):
    def prelaunch_preflight(self) -> None: ...

    def preflight(self) -> None: ...

    def execute_phase(self, action: str, promotion: PromotionState | None) -> PhaseOutcome: ...

    def select_and_authorize(self) -> PromotionState: ...

    def build_promotion_training_pack(
        self, outer_fold: int, promotion: PromotionState
    ) -> PackCapability: ...

    def build_model_bound_prediction_packs(
        self, promotion: PromotionState
    ) -> Mapping[int, PackCapability]: ...


class CampaignCoordinator:
    """Dependency-injected deterministic campaign state machine."""

    def __init__(self, backend: CoordinatorBackend) -> None:
        self.backend = backend

    def _preflight(self) -> None:
        self.backend.preflight()

    def _phase(
        self,
        action: str,
        promotion: PromotionState | None,
        actions: list[str],
        reused: list[str],
        launched: list[str],
    ) -> None:
        # The host first proves the immutable CONTEXT1 postfailure byte prefix.
        # This deliberately permits an open suffix so the target runtime can
        # still recover one demonstrably dead prior GPU lifecycle.  The target
        # then requires a closed strict replay before GPU admission, and every
        # successful phase is followed by the full host preflight.
        self.backend.prelaunch_preflight()
        outcome = self.backend.execute_phase(action, promotion)
        if type(outcome.return_code) is not int:
            raise CampaignCoordinatorError("phase executor returned a non-integer status")
        if outcome.return_code != 0:
            raise PhaseFailed(action, outcome.return_code)
        self._preflight()
        if outcome.reused:
            reused.append(action)
        if outcome.launched:
            launched.append(action)
        actions.append(action)

    def run(self) -> CampaignResult:
        actions: list[str] = []
        reused: list[str] = []
        launched: list[str] = []
        self._phase("efficiency_benchmark", None, actions, reused, launched)
        for outer_fold in DISCOVERY_FOLDS:
            self._phase(
                f"discovery_outer_{outer_fold}", None, actions, reused, launched
            )
        self._phase("discovery_aggregation", None, actions, reused, launched)

        self._preflight()
        promotion = self.backend.select_and_authorize()
        _validate_promotion_state_shape(promotion)
        actions.append("selection_and_promotion_authorization")

        for outer_fold in PROMOTION_TRAINING_FOLDS:
            self._preflight()
            capability = self.backend.build_promotion_training_pack(
                outer_fold, promotion
            )
            _validate_pack_capability(capability)
            actions.append(f"promotion_training_pack_outer_{outer_fold}")
            self._phase(
                f"promotion_training_outer_{outer_fold}",
                promotion,
                actions,
                reused,
                launched,
            )

        self._preflight()
        prediction_packs = dict(
            self.backend.build_model_bound_prediction_packs(promotion)
        )
        if set(prediction_packs) != set(ALL_FOLDS):
            raise CampaignCoordinatorError(
                "model-bound pack builder did not return all six folds"
            )
        for outer_fold in ALL_FOLDS:
            _validate_pack_capability(prediction_packs[outer_fold])
        actions.append("model_bound_prediction_packs")

        for outer_fold in ALL_FOLDS:
            self._phase(
                f"promotion_prediction_outer_{outer_fold}",
                promotion,
                actions,
                reused,
                launched,
            )
        self._phase(
            "promotion_aggregation", promotion, actions, reused, launched
        )

        if tuple(actions) != EXPECTED_ACTION_ORDER:
            raise CampaignCoordinatorError("coordinator action graph drifted")
        return CampaignResult(
            actions=tuple(actions),
            reused_phases=tuple(reused),
            launched_phases=tuple(launched),
        )


def _require_member(value: int, allowed: Sequence[int], label: str) -> None:
    if type(value) is not int or value not in allowed:
        raise CampaignCoordinatorError(f"invalid {label}: {value!r}")


def _validate_pack_capability(value: PackCapability) -> None:
    if not isinstance(value, PackCapability):
        raise CampaignCoordinatorError("backend returned an invalid pack capability")
    if value.index.parent != value.root:
        raise CampaignCoordinatorError("pack index is not owned by its shard root")


def _validate_promotion_state_shape(value: PromotionState) -> None:
    if not isinstance(value, PromotionState):
        raise CampaignCoordinatorError("selector returned an invalid promotion state")
    selection = value.selection
    authorization = value.authorization
    scopes = authorization.get("authorized_scopes")
    if not (
        selection.get("promotion_eligible") is True
        and selection.get("promotion_authorized") is True
        and selection.get("commercial_claim_authorized") is False
        and authorization.get("authorized_now") is True
        and authorization.get("promotion_authorized") is True
        and scopes == ["promotion_training_pack", "outer_prediction_pack"]
        and authorization.get("selected_variant")
        == selection.get("selected_variant")
        and authorization.get("commercial_claim_authorized") is False
    ):
        raise CampaignCoordinatorError("selection/promotion gate is not satisfied")


def _selection_needs_publication(
    *, selection_exists: bool, authorization_exists: bool
) -> bool:
    """Recognize only prefixes of the selector's fixed publication order."""

    if authorization_exists and not selection_exists:
        raise CampaignCoordinatorError(
            "promotion authorization exists without its earlier selection lock"
        )
    # A lock-only prefix is the sole valid SIGKILL boundary: replay validates
    # that lock byte-for-byte and publishes the still-absent authorization.
    return not authorization_exists


def _require_no_symlink_components(path: Path, *, label: str) -> Path:
    """Return one existing lexical path after lstat of every component."""

    absolute = Path(os.path.abspath(path.expanduser()))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            status = os.lstat(current)
        except OSError as error:
            raise CampaignCoordinatorError(
                f"{label} component is unavailable: {current}: {error}"
            ) from error
        if stat.S_ISLNK(status.st_mode):
            raise CampaignCoordinatorError(
                f"{label} has a symlinked component: {current}"
            )
    return absolute


def _load_module(name: str, path: Path) -> ModuleType:
    path = _require_no_symlink_components(path, label=f"module:{name}")
    try:
        status = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise CampaignCoordinatorError(f"module is unavailable: {path}: {error}") from error
    if not stat.S_ISREG(status.st_mode) or path.is_symlink():
        raise CampaignCoordinatorError(f"module path is not a regular file: {path}")
    unique = f"_snn_v8r4a_coordinator_{name}_{hashlib.sha256(os.fspath(path).encode()).hexdigest()[:12]}"
    specification = importlib.util.spec_from_file_location(unique, path)
    if specification is None or specification.loader is None:
        raise CampaignCoordinatorError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[unique] = module
    prior = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(unique, None)
        raise
    finally:
        sys.dont_write_bytecode = prior
    return module


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_sha256(document: Mapping[str, Any]) -> str:
    value = dict(document)
    value.pop("content_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _read_immutable_json(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _require_no_symlink_components(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CampaignCoordinatorError(f"cannot open {label}: {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not (
            stat.S_ISREG(before.st_mode)
            and stat.S_IMODE(before.st_mode) == 0o444
            and before.st_nlink == 1
        ):
            raise CampaignCoordinatorError(f"{label} is not immutable 0444/nlink1")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        stable = ("st_dev", "st_ino", "st_size", "st_mode", "st_nlink", "st_mtime_ns")
        if any(getattr(before, key) != getattr(after, key) for key in stable) or any(
            getattr(after, key) != getattr(named, key) for key in stable
        ):
            raise CampaignCoordinatorError(f"{label} changed while pinned")
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    try:
        document = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignCoordinatorError(f"invalid {label}: {error}") from error
    if not isinstance(document, dict):
        raise CampaignCoordinatorError(f"{label} is not a JSON object")
    if document.get("content_sha256") != _content_sha256(document):
        raise CampaignCoordinatorError(f"{label} self hash drifted")
    return document, {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _validate_frozen_contract_correction_evidence(
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    diagnostic_binding: Mapping[str, Any],
) -> None:
    """Pin the additive CONTRACT1 authority before resolving active issuance."""

    expected_diagnostic = {
        "path": FROZEN_CONTRACT_DIAGNOSTIC_RELATIVE.as_posix(),
        "file_sha256": FROZEN_CONTRACT_DIAGNOSTIC_FILE_SHA256,
        "bytes": FROZEN_CONTRACT_DIAGNOSTIC_BYTES,
        "content_sha256": FROZEN_CONTRACT_DIAGNOSTIC_CONTENT_SHA256,
    }
    expected_claim_boundary = {
        "adaptive_retrospective_only": True,
        "correction_is_infrastructure_only": True,
        "outer_test_features_or_targets_opened": False,
        "accuracy_metric_used": False,
        "gpu_execution_authorized_by_this_document": False,
        "successor_pretrain_authorization_required": True,
        "commercial_claim_authorized": False,
    }
    expected_diagnostic_claim_boundary = {
        "adaptive_retrospective_only": True,
        "outer_test_features_or_targets_opened": False,
        "accuracy_metric_computed": False,
        "gpu_accessed": False,
        "scientific_configuration_change_authorized": False,
        "commercial_claim_authorized": False,
    }
    authority_basis = authority.get("authority_basis")
    invariants = authority.get("mandatory_invariants")
    reauthorization = authority.get("required_reauthorization")
    correction = diagnostic.get("required_correction")
    if not (
        authority_binding.get("sha256")
        == FROZEN_CONTRACT_CORRECTION_FILE_SHA256
        and authority_binding.get("bytes") == FROZEN_CONTRACT_CORRECTION_BYTES
        and authority.get("content_sha256")
        == FROZEN_CONTRACT_CORRECTION_CONTENT_SHA256
        and authority.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_frozen_contract_exact_byte_encoding_correction_addendum"
        and authority.get("campaign_id") == CAMPAIGN_ID
        and authority.get("scientific_campaign_revision") == SCIENTIFIC_REVISION
        and authority.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and authority.get("claim_boundary") == expected_claim_boundary
        and isinstance(authority_basis, Mapping)
        and authority_basis.get("diagnostic") == expected_diagnostic
        and isinstance(invariants, Mapping)
        and invariants.get("successor_contract1_lifecycle_root")
        == SUPERSEDED_CONTRACT1_LIFECYCLE_RELATIVE.as_posix()
        and invariants.get("successor_contract1_output_root")
        == SUPERSEDED_CONTRACT1_BENCHMARK_RELATIVE.as_posix()
        and invariants.get("superseded_canary1_lifecycle_root_preserved_immutable")
        == SUPERSEDED_LIFECYCLE_RELATIVE.as_posix()
        and invariants.get("superseded_canary1_output_root_preserved_immutable")
        == SUPERSEDED_BENCHMARK_RELATIVE.as_posix()
        and invariants.get(
            "both_superseded_roots_denied_unmounted_and_command_inaccessible"
        )
        is True
        and isinstance(reauthorization, Mapping)
        and reauthorization.get("new_test_receipt")
        == "IMPLEMENTATION_TEST_RECEIPT_V8R4A_CONTRACT1.json"
        and reauthorization.get("new_source_snapshot")
        == "V3R1_SOURCE_SNAPSHOT_V8R4A_CONTRACT1.json"
        and reauthorization.get("new_pretrain_authorization")
        == "PRETRAIN_AUTHORIZATION_V8R4A_CONTRACT1.json"
        and reauthorization.get("new_governance_roles")
        == [
            "frozen_contract_encoding_correction_authorization",
            "frozen_contract_encoding_failure_diagnostic",
        ]
        and reauthorization.get("new_denied_canary_roles")
        == [
            "superseded_v8r4a_lifecycle_root",
            "superseded_v8r4a_output_root",
        ]
    ):
        raise CampaignCoordinatorError(
            "frozen-contract correction authority boundary drifted"
        )
    if not (
        diagnostic_binding.get("sha256")
        == FROZEN_CONTRACT_DIAGNOSTIC_FILE_SHA256
        and diagnostic_binding.get("bytes") == FROZEN_CONTRACT_DIAGNOSTIC_BYTES
        and diagnostic.get("content_sha256")
        == FROZEN_CONTRACT_DIAGNOSTIC_CONTENT_SHA256
        and diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_frozen_contract_encoding_false_rejection_failure_diagnostic"
        and diagnostic.get("status") == "diagnosed_not_authorized_by_diagnostic"
        and diagnostic.get("campaign_id") == CAMPAIGN_ID
        and diagnostic.get("scientific_campaign_revision") == SCIENTIFIC_REVISION
        and diagnostic.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and diagnostic.get("claim_boundary") == expected_diagnostic_claim_boundary
        and isinstance(correction, Mapping)
        and correction.get("successor_lifecycle_root")
        == SUPERSEDED_CONTRACT1_LIFECYCLE_RELATIVE.as_posix()
        and correction.get("successor_benchmark_output_root")
        == SUPERSEDED_CONTRACT1_BENCHMARK_RELATIVE.as_posix()
        and correction.get("deny_and_unmount_superseded_lifecycle_root") is True
        and correction.get("deny_and_unmount_superseded_output_root") is True
        and correction.get("new_test_receipt")
        == "IMPLEMENTATION_TEST_RECEIPT_V8R4A_CONTRACT1.json"
        and correction.get("new_source_snapshot")
        == "V3R1_SOURCE_SNAPSHOT_V8R4A_CONTRACT1.json"
        and correction.get("new_pretrain_authorization")
        == "PRETRAIN_AUTHORIZATION_V8R4A_CONTRACT1.json"
    ):
        raise CampaignCoordinatorError(
            "frozen-contract failure diagnostic boundary drifted"
        )


def _validate_gpu_state_parent_bind_correction_evidence(
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    diagnostic_binding: Mapping[str, Any],
) -> None:
    """Pin the additive ROOTBIND1 authority before resolving active issuance."""

    expected_diagnostic = {
        "path": GPU_STATE_PARENT_BIND_DIAGNOSTIC_RELATIVE.as_posix(),
        "file_sha256": GPU_STATE_PARENT_BIND_DIAGNOSTIC_FILE_SHA256,
        "bytes": GPU_STATE_PARENT_BIND_DIAGNOSTIC_BYTES,
        "content_sha256": GPU_STATE_PARENT_BIND_DIAGNOSTIC_CONTENT_SHA256,
    }
    expected_claim_boundary = {
        "adaptive_retrospective_only": True,
        "correction_is_infrastructure_only": True,
        "outer_test_features_or_targets_opened": False,
        "accuracy_metric_used": False,
        "gpu_execution_authorized_by_this_document": False,
        "successor_pretrain_authorization_required": True,
        "commercial_claim_authorized": False,
    }
    expected_diagnostic_claim_boundary = {
        "adaptive_retrospective_only": True,
        "outer_test_features_or_targets_opened": False,
        "accuracy_metric_computed": False,
        "gpu_accessed": False,
        "scientific_configuration_change_authorized": False,
        "commercial_claim_authorized": False,
    }
    authority_basis = authority.get("authority_basis")
    invariants = authority.get("mandatory_invariants")
    reauthorization = authority.get("required_reauthorization")
    correction = diagnostic.get("required_correction")
    if not (
        authority_binding.get("sha256")
        == GPU_STATE_PARENT_BIND_CORRECTION_FILE_SHA256
        and authority_binding.get("bytes")
        == GPU_STATE_PARENT_BIND_CORRECTION_BYTES
        and authority.get("content_sha256")
        == GPU_STATE_PARENT_BIND_CORRECTION_CONTENT_SHA256
        and authority.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_gpu_state_parent_readonly_bind_correction_addendum"
        and authority.get("campaign_id") == CAMPAIGN_ID
        and authority.get("scientific_campaign_revision") == SCIENTIFIC_REVISION
        and authority.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and authority.get("claim_boundary") == expected_claim_boundary
        and isinstance(authority_basis, Mapping)
        and authority_basis.get("diagnostic") == expected_diagnostic
        and isinstance(invariants, Mapping)
        and invariants.get("gpu_state_root_path") == STATE_RELATIVE.as_posix()
        and invariants.get("gpu_state_root_exact_mode") == "0700"
        and invariants.get("gpu_state_root_exact_st_dev") == 66_306
        and invariants.get("gpu_state_root_exact_st_ino") == 6_970_105
        and invariants.get("gpu_state_root_mount_kind") == "ro_bind_fd"
        and invariants.get("gpu_state_root_mount_precedes_children") is True
        and invariants.get("exactly_three_mutable_state_directory_mounts") is True
        and invariants.get("superseded_v8r4a_lifecycle_root_preserved_immutable")
        == SUPERSEDED_LIFECYCLE_RELATIVE.as_posix()
        and invariants.get("superseded_v8r4a_output_root_preserved_immutable")
        == SUPERSEDED_BENCHMARK_RELATIVE.as_posix()
        and invariants.get(
            "superseded_v8r4a_contract1_lifecycle_root_preserved_immutable"
        )
        == SUPERSEDED_CONTRACT1_LIFECYCLE_RELATIVE.as_posix()
        and invariants.get(
            "superseded_v8r4a_contract1_output_root_preserved_immutable"
        )
        == SUPERSEDED_CONTRACT1_BENCHMARK_RELATIVE.as_posix()
        and invariants.get("successor_rootbind1_lifecycle_root")
        == SUPERSEDED_ROOTBIND1_LIFECYCLE_RELATIVE.as_posix()
        and invariants.get("successor_rootbind1_output_root")
        == SUPERSEDED_ROOTBIND1_BENCHMARK_RELATIVE.as_posix()
        and invariants.get(
            "all_four_superseded_roots_denied_unmounted_and_command_inaccessible"
        )
        is True
        and isinstance(reauthorization, Mapping)
        and reauthorization.get("new_test_receipt")
        == "IMPLEMENTATION_TEST_RECEIPT_V8R4A_ROOTBIND1.json"
        and reauthorization.get("new_source_snapshot")
        == "V3R1_SOURCE_SNAPSHOT_V8R4A_ROOTBIND1.json"
        and reauthorization.get("new_pretrain_authorization")
        == "PRETRAIN_AUTHORIZATION_V8R4A_ROOTBIND1.json"
        and reauthorization.get("new_governance_roles")
        == [
            "gpu_state_parent_bind_correction_authorization",
            "gpu_state_parent_bind_failure_diagnostic",
        ]
        and reauthorization.get("new_denied_canary_roles")
        == [
            "superseded_v8r4a_contract1_lifecycle_root",
            "superseded_v8r4a_contract1_output_root",
        ]
    ):
        raise CampaignCoordinatorError(
            "GPU-state parent-bind correction authority boundary drifted"
        )
    if not (
        diagnostic_binding.get("sha256")
        == GPU_STATE_PARENT_BIND_DIAGNOSTIC_FILE_SHA256
        and diagnostic_binding.get("bytes")
        == GPU_STATE_PARENT_BIND_DIAGNOSTIC_BYTES
        and diagnostic.get("content_sha256")
        == GPU_STATE_PARENT_BIND_DIAGNOSTIC_CONTENT_SHA256
        and diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_gpu_state_parent_mount_identity_failure_diagnostic"
        and diagnostic.get("status") == "diagnosed_not_authorized_by_diagnostic"
        and diagnostic.get("campaign_id") == CAMPAIGN_ID
        and diagnostic.get("scientific_campaign_revision") == SCIENTIFIC_REVISION
        and diagnostic.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and diagnostic.get("claim_boundary") == expected_diagnostic_claim_boundary
        and isinstance(correction, Mapping)
        and correction.get("exact_parent_readonly_fd_bind_required")
        == {
            "path": STATE_RELATIVE.as_posix(),
            "mode": "0700",
            "st_dev": 66_306,
            "st_ino": 6_970_105,
            "kind": "ro_bind_fd",
        }
        and correction.get("exactly_three_mutable_direct_child_overlays_required")
        == ["admission", "execution", "usage"]
        and correction.get("successor_lifecycle_root")
        == SUPERSEDED_ROOTBIND1_LIFECYCLE_RELATIVE.as_posix()
        and correction.get("successor_benchmark_output_root")
        == SUPERSEDED_ROOTBIND1_BENCHMARK_RELATIVE.as_posix()
        and correction.get("deny_and_unmount_all_four_superseded_roots") is True
        and correction.get("new_test_receipt")
        == "IMPLEMENTATION_TEST_RECEIPT_V8R4A_ROOTBIND1.json"
        and correction.get("new_source_snapshot")
        == "V3R1_SOURCE_SNAPSHOT_V8R4A_ROOTBIND1.json"
        and correction.get("new_pretrain_authorization")
        == "PRETRAIN_AUTHORIZATION_V8R4A_ROOTBIND1.json"
    ):
        raise CampaignCoordinatorError(
            "GPU-state parent-bind failure diagnostic boundary drifted"
        )


def _validate_benchmark_admitted_context_correction_evidence(
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    diagnostic_binding: Mapping[str, Any],
) -> None:
    """Pin the exact CONTEXT1 succession without opening failed ROOTBIND1 roots."""

    expected_context = {
        "campaign_revision": "V8R4",
        "infrastructure_revision": "V8R4A",
        "authorization_generation": "CONTEXT1",
        "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
        "outer_fold": 3,
        "seed": 20260828,
        "variant": "H0_no_factor",
    }
    expected_superseded_context = dict(expected_context)
    expected_superseded_context.pop("authorization_generation")
    expected_claim = {
        "adaptive_retrospective_only": True,
        "correction_is_infrastructure_only": True,
        "outer_test_features_or_targets_opened": False,
        "accuracy_metric_used": False,
        "prior_gpu_admission_and_cuda_availability_probe_recorded": True,
        "prior_model_or_training_kernel_executed": False,
        "gpu_execution_authorized_by_this_document": False,
        "successor_pretrain_authorization_required": True,
        "commercial_claim_authorized": False,
    }
    expected_diagnostic_claim = {
        "adaptive_retrospective_only": True,
        "outer_test_features_or_targets_opened": False,
        "accuracy_metric_computed": False,
        "gpu_admission_reached": True,
        "cuda_availability_probe_occurred": True,
        "model_or_training_kernel_executed": False,
        "scientific_configuration_change_authorized": False,
        "commercial_claim_authorized": False,
    }
    basis = authority.get("authority_basis")
    invariants = authority.get("mandatory_invariants")
    required = authority.get("required_reauthorization")
    failed = diagnostic.get("failed_attempt")
    ledger = diagnostic.get("ledger_evidence")
    root_cause = diagnostic.get("root_cause")
    correction = diagnostic.get("required_correction")
    receipts = diagnostic.get("immutable_failure_receipts")
    basis_diagnostic = (
        basis.get("diagnostic") if isinstance(basis, Mapping) else None
    )
    basis_diagnostic = (
        basis_diagnostic if isinstance(basis_diagnostic, Mapping) else {}
    )
    basis_parent_authority = (
        basis.get("parent_gpu_state_parent_bind_authority")
        if isinstance(basis, Mapping)
        else None
    )
    basis_parent_authority = (
        basis_parent_authority
        if isinstance(basis_parent_authority, Mapping)
        else {}
    )
    basis_parent_diagnostic = (
        basis.get("parent_gpu_state_parent_bind_diagnostic")
        if isinstance(basis, Mapping)
        else None
    )
    basis_parent_diagnostic = (
        basis_parent_diagnostic
        if isinstance(basis_parent_diagnostic, Mapping)
        else {}
    )
    basis_test = basis.get("parent_implementation_test_receipt") if isinstance(basis, Mapping) else None
    basis_test = basis_test if isinstance(basis_test, Mapping) else {}
    basis_snapshot = basis.get("parent_source_snapshot") if isinstance(basis, Mapping) else None
    basis_snapshot = basis_snapshot if isinstance(basis_snapshot, Mapping) else {}
    basis_pretrain = basis.get("parent_pretrain_authorization") if isinstance(basis, Mapping) else None
    basis_pretrain = basis_pretrain if isinstance(basis_pretrain, Mapping) else {}
    basis_capability = basis.get("failed_rootbind1_target_capability_receipt") if isinstance(basis, Mapping) else None
    basis_capability = basis_capability if isinstance(basis_capability, Mapping) else {}
    basis_completion = basis.get("failed_rootbind1_target_completion_receipt") if isinstance(basis, Mapping) else None
    basis_completion = basis_completion if isinstance(basis_completion, Mapping) else {}
    basis_terminal = basis.get("failed_rootbind1_gpu_terminal_result") if isinstance(basis, Mapping) else None
    basis_terminal = basis_terminal if isinstance(basis_terminal, Mapping) else {}
    terminal_receipt = receipts.get("gpu_terminal_result") if isinstance(receipts, Mapping) else None
    terminal_receipt = terminal_receipt if isinstance(terminal_receipt, Mapping) else {}
    usage_postlaunch = ledger.get("usage_postlaunch") if isinstance(ledger, Mapping) else None
    usage_postlaunch = usage_postlaunch if isinstance(usage_postlaunch, Mapping) else {}
    execution_postlaunch = ledger.get("execution_postlaunch") if isinstance(ledger, Mapping) else None
    execution_postlaunch = execution_postlaunch if isinstance(execution_postlaunch, Mapping) else {}
    if not (
        authority_binding.get("sha256")
        == BENCHMARK_ADMITTED_CONTEXT_CORRECTION_FILE_SHA256
        and authority_binding.get("bytes")
        == BENCHMARK_ADMITTED_CONTEXT_CORRECTION_BYTES
        and authority.get("content_sha256")
        == BENCHMARK_ADMITTED_CONTEXT_CORRECTION_CONTENT_SHA256
        and authority.get("schema_version") == 1
        and authority.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_benchmark_admitted_pretrain_context_bridge_correction_addendum"
        and authority.get("campaign_id") == CAMPAIGN_ID
        and authority.get("scientific_campaign_revision") == SCIENTIFIC_REVISION
        and authority.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and authority.get("claim_boundary") == expected_claim
        and isinstance(basis, Mapping)
        and basis_diagnostic.get("path")
        == BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_RELATIVE.as_posix()
        and basis_diagnostic.get("sha256")
        == BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_FILE_SHA256
        and basis_diagnostic.get("bytes")
        == BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_BYTES
        and basis_diagnostic.get("content_sha256")
        == BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_CONTENT_SHA256
        and basis_parent_authority.get("sha256")
        == GPU_STATE_PARENT_BIND_CORRECTION_FILE_SHA256
        and basis_parent_diagnostic.get("sha256")
        == GPU_STATE_PARENT_BIND_DIAGNOSTIC_FILE_SHA256
        and basis_test.get("sha256")
        == "5654eb89eab4ccb97f20633dbe1832e8600694312d3b52162c0d8f1711f57ec5"
        and basis_snapshot.get("sha256")
        == "8ea5873fd2ebf43d975db123f4551e7d3aa849ff4aa404dfb5c862c23b735cae"
        and basis_pretrain.get("sha256")
        == "49ba2637b9c957d382f83c8847198f129eda2f08c099c184f717003a1129fba6"
        and basis_capability.get("sha256")
        == "b6ceccf7b4d3f0738de1cbead9038fe937a209295294a4af772a902dccbd20d8"
        and basis_completion.get("sha256")
        == "e08456bcdecddb10e38e7378837f785314dd8494125d2363b1063df7b4723747"
        and basis_terminal.get("sha256")
        == "b575caa298db286cad2a3ad3231aa84dcdb2af76ab04d15ea38eec7b1a50fbda"
        and isinstance(invariants, Mapping)
        and invariants.get("active_benchmark_context") == expected_context
        and invariants.get("superseded_rootbind1_context")
        == expected_superseded_context
        and invariants.get("superseded_rootbind1_terminal_record_sha256")
        == "9da404e120c87940617fed8ad8438d9470b785b304fe48c4ec8727d50fcafafd"
        and invariants.get("usage_postfailure_sha256")
        == "c59c9234c25c09a4cb7e7cc753942e0517acccdced5ba527336295db32dac029"
        and invariants.get("usage_postfailure_bytes") == 113_257
        and invariants.get("usage_postfailure_record_count") == 77
        and invariants.get("usage_postfailure_open_reservation_count") == 0
        and invariants.get("execution_postfailure_sha256")
        == "8acffd84488a1f9b4f9a0a95804108127135023d0e58c46afe7ffa7b553609b5"
        and invariants.get("execution_postfailure_bytes") == 29_961
        and invariants.get("execution_postfailure_record_count") == 10
        and invariants.get("execution_postfailure_open_start_count") == 0
        and invariants.get("successor_context1_lifecycle_root")
        == LIFECYCLE_RELATIVE.as_posix()
        and invariants.get("successor_context1_output_root")
        == BENCHMARK_RELATIVE.as_posix()
        and invariants.get(
            "all_six_superseded_roots_denied_unmounted_and_command_inaccessible"
        )
        is True
        and invariants.get("rootbind_parent_readonly_and_three_child_readwrite_topology_unchanged")
        is True
        and isinstance(required, Mapping)
        and required.get("new_test_receipt")
        == "IMPLEMENTATION_TEST_RECEIPT_V8R4A_CONTEXT1.json"
        and required.get("new_source_snapshot")
        == "V3R1_SOURCE_SNAPSHOT_V8R4A_CONTEXT1.json"
        and required.get("new_pretrain_authorization")
        == "PRETRAIN_AUTHORIZATION_V8R4A_CONTEXT1.json"
        and required.get("new_governance_roles")
        == [
            "admitted_context_correction_authorization",
            "admitted_context_failure_diagnostic",
        ]
        and required.get("new_denied_canary_roles")
        == [
            "superseded_v8r4a_rootbind1_lifecycle_root",
            "superseded_v8r4a_rootbind1_output_root",
        ]
        and required.get("required_true_security_boundary")
        == "benchmark_admitted_context_generation_isolated"
    ):
        raise CampaignCoordinatorError(
            "benchmark admitted-context correction authority boundary drifted"
        )
    if not (
        diagnostic_binding.get("sha256")
        == BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_FILE_SHA256
        and diagnostic_binding.get("bytes")
        == BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_BYTES
        and diagnostic.get("content_sha256")
        == BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_CONTENT_SHA256
        and diagnostic.get("schema_version") == 1
        and diagnostic.get("classification")
        == "pretrain_adaptive_v3r1_v8r4a_benchmark_admitted_pretrain_context_bridge_failure_diagnostic"
        and diagnostic.get("status") == "diagnosed_not_authorized_by_diagnostic"
        and diagnostic.get("campaign_id") == CAMPAIGN_ID
        and diagnostic.get("scientific_campaign_revision") == SCIENTIFIC_REVISION
        and diagnostic.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
        and diagnostic.get("claim_boundary") == expected_diagnostic_claim
        and isinstance(failed, Mapping)
        and failed.get("first_phase") == "efficiency_benchmark"
        and failed.get("outer_fold") == 3
        and failed.get("target_runtime_return_code") == 87
        and failed.get("gpu_wrapper_return_code") == 1
        and failed.get("gpu_admission_reached") is True
        and failed.get("admitted_child_binding_consumed_once") is True
        and failed.get("cuda_availability_probe_occurred") is True
        and failed.get("model_constructed") is False
        and failed.get("accuracy_metric_computed") is False
        and isinstance(receipts, Mapping)
        and terminal_receipt.get("terminal_record_sha256")
        == "9da404e120c87940617fed8ad8438d9470b785b304fe48c4ec8727d50fcafafd"
        and isinstance(ledger, Mapping)
        and usage_postlaunch.get("sha256")
        == "c59c9234c25c09a4cb7e7cc753942e0517acccdced5ba527336295db32dac029"
        and usage_postlaunch.get("record_count") == 77
        and usage_postlaunch.get("open_reservation_count") == 0
        and execution_postlaunch.get("sha256")
        == "8acffd84488a1f9b4f9a0a95804108127135023d0e58c46afe7ffa7b553609b5"
        and execution_postlaunch.get("record_count") == 10
        and execution_postlaunch.get("open_start_count") == 0
        and ledger.get("append_only_prefix_preserved") is True
        and ledger.get("both_ledgers_closed_after_failure") is True
        and isinstance(root_cause, Mapping)
        and root_cause.get("benchmark_internal_worker_called_trainer_primitive_without_prevalidated_pretrain")
        is True
        and root_cause.get("trainer_fail_closed_default_correctly_rejected_context_free_admitted_validation")
        is True
        and isinstance(correction, Mapping)
        and correction.get("independent_expected_context") == expected_context
        and correction.get("independent_expected_phase") == "efficiency_benchmark"
        and correction.get("independent_expected_outer_fold") == 3
        and correction.get("validated_pretrain_passed_to_primitive") is True
        and correction.get("successor_lifecycle_root")
        == LIFECYCLE_RELATIVE.as_posix()
        and correction.get("successor_benchmark_output_root")
        == BENCHMARK_RELATIVE.as_posix()
        and correction.get("deny_and_unmount_all_six_superseded_roots") is True
        and correction.get("new_test_receipt")
        == "IMPLEMENTATION_TEST_RECEIPT_V8R4A_CONTEXT1.json"
        and correction.get("new_source_snapshot")
        == "V3R1_SOURCE_SNAPSHOT_V8R4A_CONTEXT1.json"
        and correction.get("new_pretrain_authorization")
        == "PRETRAIN_AUTHORIZATION_V8R4A_CONTEXT1.json"
    ):
        raise CampaignCoordinatorError(
            "benchmark admitted-context failure diagnostic boundary drifted"
        )


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_immutable_file(path: Path, label: str) -> Path:
    absolute = _require_no_symlink_components(path, label=label)
    try:
        status = os.stat(absolute, follow_symlinks=False)
    except OSError as error:
        raise CampaignCoordinatorError(f"{label} is unavailable: {error}") from error
    if not (
        stat.S_ISREG(status.st_mode)
        and stat.S_IMODE(status.st_mode) == 0o444
        and status.st_nlink == 1
        and not absolute.is_symlink()
    ):
        raise CampaignCoordinatorError(f"{label} is not immutable 0444/nlink1")
    return absolute


def _require_frozen_tree(root: Path, label: str) -> None:
    root = _require_no_symlink_components(root, label=label)
    try:
        root_status = os.stat(root, follow_symlinks=False)
    except OSError as error:
        raise CampaignCoordinatorError(f"{label} is unavailable: {error}") from error
    if not stat.S_ISDIR(root_status.st_mode) or stat.S_IMODE(root_status.st_mode) != 0o555:
        raise CampaignCoordinatorError(f"{label} root is not immutable mode 0555")
    seen: set[tuple[int, int]] = set()
    file_count = 0
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        directories.sort()
        files.sort()
        current_path = Path(current)
        current_status = os.stat(current_path, follow_symlinks=False)
        if not stat.S_ISDIR(current_status.st_mode) or stat.S_IMODE(current_status.st_mode) != 0o555:
            raise CampaignCoordinatorError(f"{label} has a mutable directory: {current_path}")
        for name in directories:
            if stat.S_ISLNK(os.lstat(current_path / name).st_mode):
                raise CampaignCoordinatorError(f"{label} contains a symlink")
        for name in files:
            path = current_path / name
            status = os.lstat(path)
            identity = (status.st_dev, status.st_ino)
            if not (
                stat.S_ISREG(status.st_mode)
                and stat.S_IMODE(status.st_mode) == 0o444
                and status.st_nlink == 1
                and identity not in seen
            ):
                raise CampaignCoordinatorError(f"{label} contains a mutable/aliased file: {path}")
            seen.add(identity)
            file_count += 1
    if file_count == 0:
        raise CampaignCoordinatorError(f"{label} is empty")


def _ensure_private_directory(path: Path, *, boundary: Path) -> Path:
    path = Path(os.path.abspath(path.expanduser()))
    boundary = _require_no_symlink_components(boundary, label="mutable boundary")
    try:
        relative = path.relative_to(boundary)
    except ValueError as error:
        raise CampaignCoordinatorError("mutable directory escaped its canonical boundary") from error
    current = boundary
    if current.is_symlink() or not current.is_dir():
        raise CampaignCoordinatorError("canonical mutable boundary is unavailable")
    for part in relative.parts:
        if part in {"", ".", ".."}:
            raise CampaignCoordinatorError("mutable directory contains traversal")
        current /= part
        try:
            status = os.lstat(current)
        except FileNotFoundError:
            os.mkdir(current, 0o700)
            status = os.lstat(current)
        if not stat.S_ISDIR(status.st_mode) or stat.S_ISLNK(status.st_mode):
            raise CampaignCoordinatorError(f"mutable path is not a directory: {current}")
    status = os.stat(path, follow_symlinks=False)
    if stat.S_IMODE(status.st_mode) != 0o700:
        raise CampaignCoordinatorError(f"canonical output/lifecycle mode is not 0700: {path}")
    return path


class SynchronousTargetSealedExecutor:
    """A non-reentrant, shell-free, one-process target-sealed executor."""

    def __init__(
        self,
        *,
        interpreter: Path,
        runtime_script: Path,
        runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        self.interpreter = Path(os.path.abspath(interpreter))
        self.runtime_script = Path(os.path.abspath(runtime_script))
        self.runner = runner
        self._active = False

    def __call__(self, argv: Sequence[str]) -> int:
        command = tuple(argv)
        if len(command) < 5 or command[:2] != (
            str(self.interpreter),
            str(self.runtime_script),
        ):
            raise CampaignCoordinatorError(
                "GPU-capable execution must enter through target-sealed runtime"
            )
        try:
            separator = command.index("--")
        except ValueError as error:
            raise CampaignCoordinatorError("target-sealed command lacks child boundary") from error
        child = command[separator + 1 :]
        if len(child) < 2 or child[0] != str(self.interpreter):
            raise CampaignCoordinatorError("target-sealed child interpreter drifted")
        if Path(child[1]).name not in {
            "benchmark_hfr_v3r1_efficiency.py",
            "run_hfr_v3r1_discovery_campaign.py",
            "run_fixed_hfr_v3r1_oof_campaign.py",
        }:
            raise CampaignCoordinatorError("unapproved target-sealed child entry")
        forbidden_exact = {
            "--variant",
            "--seed",
            "--release-mode",
            "--threshold",
            "--raw-input-root",
            "--target-root",
            "--smoke-test",
            "--no-amp",
        }
        if any(token in forbidden_exact for token in child):
            raise CampaignCoordinatorError("child command attempts to alter frozen science")
        if "run_gpu_admitted.py" in (Path(part).name for part in command[:separator]):
            raise CampaignCoordinatorError("coordinator may not launch the GPU wrapper")
        if self._active:
            raise CampaignCoordinatorError("parallel/reentrant phase execution is forbidden")
        self._active = True
        try:
            result = self.runner(command, check=False, shell=False)
        finally:
            self._active = False
        return_code = result if type(result) is int else getattr(result, "returncode", None)
        if type(return_code) is not int:
            raise CampaignCoordinatorError("target-sealed runner returned no status")
        return return_code


def build_outer_runtime_argv(
    invocation: PhaseInvocation,
    *,
    project_root: Path,
    interpreter: Path,
    venv_root: Path,
    python_runtime_root: Path,
    cuda_runtime_roots: Sequence[Path],
    cuda_devices: Sequence[Path],
    propagated_environment: Mapping[str, str],
) -> tuple[str, ...]:
    runtime_script = project_root / RUNTIME_SCRIPT_RELATIVE
    argv: list[str] = [
        str(interpreter),
        str(runtime_script),
        "--project-root",
        str(project_root),
        "--phase",
        invocation.phase,
    ]
    if invocation.outer_fold is not None:
        argv.extend(["--outer-fold", str(invocation.outer_fold)])
    if invocation.pack is not None:
        argv.extend(
            [
                "--pack-root",
                str(invocation.pack.root),
                "--pack-index",
                str(invocation.pack.index),
            ]
        )
    for role, path in sorted(invocation.governance.items()):
        argv.extend(["--governance", f"{role}={path}"])
    writable = {
        "output": invocation.output_root,
        "lifecycle": invocation.lifecycle_root,
        "usage": project_root / STATE_RELATIVE / "usage",
        "execution": project_root / STATE_RELATIVE / "execution",
        "admission": project_root / STATE_RELATIVE / "admission",
    }
    for role, path in sorted(writable.items()):
        argv.extend(["--writable-root", f"{role}={path}"])
    for role, path in sorted(invocation.denied_canaries.items()):
        argv.extend(["--deny-canary", f"{role}={path}"])
    argv.extend(
        [
            "--capability-receipt",
            str(invocation.capability_receipt),
            "--interpreter",
            str(interpreter),
            "--venv-root",
            str(venv_root),
            "--python-runtime-root",
            str(python_runtime_root),
        ]
    )
    for root in sorted({Path(path) for path in cuda_runtime_roots}, key=os.fspath):
        argv.extend(["--cuda-runtime-root", str(root)])
    for device in sorted({Path(path) for path in cuda_devices}, key=os.fspath):
        argv.extend(["--cuda-device", str(device)])
    for name, value in sorted(propagated_environment.items()):
        argv.extend(["--env", f"{name}={value}"])
    argv.append("--")
    argv.extend(invocation.child_command)
    return tuple(argv)


class ProductionBackend:
    """Host-side implementation of the fixed coordinator graph."""

    def __init__(
        self,
        project_root: Path,
        *,
        phase_runner: Callable[[Sequence[str]], int] | None = None,
    ) -> None:
        self.paths = CanonicalPaths(project_root)
        self.interpreter = self.paths.project_root / ".venv/bin/python"
        if not os.path.lexists(self.interpreter):
            raise CampaignCoordinatorError("canonical project interpreter is absent")
        self.venv_root = self.paths.project_root / ".venv"
        try:
            resolved_interpreter = self.interpreter.resolve(strict=True)
        except OSError as error:
            raise CampaignCoordinatorError(f"cannot resolve project interpreter: {error}") from error
        self.python_runtime_root = resolved_interpreter.parent.parent
        self.runtime = _load_module(
            "runtime", self.paths.project_root / RUNTIME_SCRIPT_RELATIVE
        )
        required_phases = {
            "efficiency_benchmark",
            "discovery",
            "discovery_aggregation",
            "promotion_training",
            "promotion_prediction",
            "promotion_aggregation",
        }
        if not required_phases <= set(getattr(self.runtime, "PHASES", ())):
            raise CampaignCoordinatorError(
                "target runtime does not expose split discovery/promotion aggregation phases"
            )
        self.validator = _load_module(
            "validator", self.paths.project_root / VALIDATOR_RELATIVE
        )
        self.migrator = _load_module(
            "migrator", self.paths.project_root / MIGRATOR_RELATIVE
        )
        self.selector = _load_module(
            "selector", self.paths.project_root / SELECTOR_RELATIVE
        )
        self.pack_builder = _load_module(
            "pack_builder", self.paths.project_root / PACK_BUILDER_RELATIVE
        )
        self.fixed = _load_module(
            "fixed", self.paths.project_root / FIXED_SCRIPT_RELATIVE
        )
        self.locked_inputs = _load_module(
            "locked_inputs", self.paths.project_root / LOCKED_INPUTS_RELATIVE
        )
        self.discovery = _load_module(
            "discovery", self.paths.project_root / DISCOVERY_SCRIPT_RELATIVE
        )
        self._training_packs: dict[int, PackCapability] = {}
        self._prediction_packs: dict[int, PackCapability] = {}
        self._phase_runner = phase_runner or SynchronousTargetSealedExecutor(
            interpreter=self.interpreter,
            runtime_script=self.paths.project_root / RUNTIME_SCRIPT_RELATIVE,
        )

    def prelaunch_preflight(self) -> None:
        """Enforce CONTEXT1's immutable ledger floor without blocking recovery."""

        active_path = self.paths.campaign / (
            "PRETRAIN_AUTHORIZATION_V8R4A_CONTEXT1.json"
        )
        active, _binding = _read_immutable_json(
            active_path, "V8R4A CONTEXT1 active pretrain authorization"
        )
        expected = getattr(
            self.runtime, "BENCHMARK_ADMITTED_CONTEXT_LEDGER_PREFIXES", None
        )
        validator = getattr(
            self.runtime,
            "_validate_active_pretrain_live_ledger_prefixes",
            None,
        )
        prefixes = active.get("runtime_ledger_prefixes")
        if not (
            type(active.get("authorization_generation")) is str
            and active.get("authorization_generation") == "CONTEXT1"
            and isinstance(expected, Mapping)
            and callable(validator)
            and isinstance(prefixes, Mapping)
            and _canonical_json_bytes(prefixes) == _canonical_json_bytes(expected)
        ):
            raise CampaignCoordinatorError(
                "active CONTEXT1 prelaunch ledger prefix evidence drifted"
            )
        try:
            validator(
                project_root=self.paths.project_root,
                runtime_ledger_prefixes=prefixes,
                live_state=None,
                require_closed=False,
            )
        except BaseException as error:
            raise CampaignCoordinatorError(
                f"active CONTEXT1 prelaunch ledger prefix failed: {error}"
            ) from error

    def preflight(self) -> None:
        authority_path = self.paths.campaign / (
            "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_EXECUTION_CLOSURE.json"
        )
        authority, _binding = _read_immutable_json(
            authority_path, "V8R4A execution-closure authority"
        )
        if not (
            authority.get("classification")
            == "pretrain_adaptive_v3r1_v8r4a_kill_safe_capability_and_promotion_execution_closure_correction_addendum"
            and authority.get("campaign_id") == CAMPAIGN_ID
            and authority.get("scientific_campaign_revision") == SCIENTIFIC_REVISION
            and authority.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
            and authority.get("claim_boundary", {}).get(
                "gpu_execution_authorized_by_this_document"
            )
            is False
            and authority.get("claim_boundary", {}).get(
                "outer_fold_numeric_reference_authorized"
            )
            is False
        ):
            raise CampaignCoordinatorError("execution-closure authority boundary drifted")
        fd_authority_path = self.paths.campaign / (
            "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_FD_CLOSURE.json"
        )
        fd_authority, _fd_binding = _read_immutable_json(
            fd_authority_path, "V8R4A outer-guard FD-closure authority"
        )
        if not (
            fd_authority.get("classification")
            == "pretrain_adaptive_v3r1_v8r4a_outer_guard_urandom_descriptor_closure_correction_addendum"
            and fd_authority.get("campaign_id") == CAMPAIGN_ID
            and fd_authority.get("scientific_campaign_revision")
            == SCIENTIFIC_REVISION
            and fd_authority.get("infrastructure_revision")
            == INFRASTRUCTURE_REVISION
            and fd_authority.get("claim_boundary", {}).get(
                "gpu_execution_authorized_by_this_document"
            )
            is False
            and fd_authority.get("claim_boundary", {}).get(
                "successor_pretrain_authorization_required"
            )
            is True
            and fd_authority.get("claim_boundary", {}).get(
                "commercial_claim_authorized"
            )
            is False
        ):
            raise CampaignCoordinatorError("FD-closure authority boundary drifted")
        canary_authority_path = self.paths.campaign / (
            "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_CANARY_BOUNDARY.json"
        )
        canary_authority, _canary_binding = _read_immutable_json(
            canary_authority_path, "V8R4A denied-canary boundary authority"
        )
        if not (
            canary_authority.get("classification")
            == "pretrain_adaptive_v3r1_v8r4a_denied_canary_component_boundary_correction_addendum"
            and canary_authority.get("campaign_id") == CAMPAIGN_ID
            and canary_authority.get("scientific_campaign_revision")
            == SCIENTIFIC_REVISION
            and canary_authority.get("infrastructure_revision")
            == INFRASTRUCTURE_REVISION
            and canary_authority.get("claim_boundary")
            == {
                "adaptive_retrospective_only": True,
                "correction_is_infrastructure_only": True,
                "outer_test_features_or_targets_opened": False,
                "accuracy_metric_used": False,
                "gpu_execution_authorized_by_this_document": False,
                "successor_pretrain_authorization_required": True,
                "commercial_claim_authorized": False,
            }
        ):
            raise CampaignCoordinatorError(
                "denied-canary boundary authority drifted"
            )
        frozen_contract_authority, frozen_contract_authority_binding = (
            _read_immutable_json(
                self.paths.project_root / FROZEN_CONTRACT_CORRECTION_RELATIVE,
                "V8R4A frozen-contract encoding correction authority",
            )
        )
        frozen_contract_diagnostic, frozen_contract_diagnostic_binding = (
            _read_immutable_json(
                self.paths.project_root / FROZEN_CONTRACT_DIAGNOSTIC_RELATIVE,
                "V8R4A frozen-contract encoding failure diagnostic",
            )
        )
        _validate_frozen_contract_correction_evidence(
            frozen_contract_authority,
            frozen_contract_authority_binding,
            frozen_contract_diagnostic,
            frozen_contract_diagnostic_binding,
        )
        parent_bind_authority, parent_bind_authority_binding = (
            _read_immutable_json(
                self.paths.project_root / GPU_STATE_PARENT_BIND_CORRECTION_RELATIVE,
                "V8R4A GPU-state parent-bind correction authority",
            )
        )
        parent_bind_diagnostic, parent_bind_diagnostic_binding = (
            _read_immutable_json(
                self.paths.project_root / GPU_STATE_PARENT_BIND_DIAGNOSTIC_RELATIVE,
                "V8R4A GPU-state parent-bind failure diagnostic",
            )
        )
        _validate_gpu_state_parent_bind_correction_evidence(
            parent_bind_authority,
            parent_bind_authority_binding,
            parent_bind_diagnostic,
            parent_bind_diagnostic_binding,
        )
        admitted_context_authority, admitted_context_authority_binding = (
            _read_immutable_json(
                self.paths.project_root
                / BENCHMARK_ADMITTED_CONTEXT_CORRECTION_RELATIVE,
                "V8R4A benchmark admitted-context correction authority",
            )
        )
        admitted_context_diagnostic, admitted_context_diagnostic_binding = (
            _read_immutable_json(
                self.paths.project_root
                / BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_RELATIVE,
                "V8R4A benchmark admitted-context failure diagnostic",
            )
        )
        _validate_benchmark_admitted_context_correction_evidence(
            admitted_context_authority,
            admitted_context_authority_binding,
            admitted_context_diagnostic,
            admitted_context_diagnostic_binding,
        )
        active = self.validator.validate_pretrain(self.paths.project_root)
        if not (
            isinstance(active, Mapping)
            and active.get("valid") is True
            and active.get("scientific_campaign_revision") == SCIENTIFIC_REVISION
            and active.get("infrastructure_revision") == INFRASTRUCTURE_REVISION
            and active.get("authorization_generation") == "CONTEXT1"
            and active.get("training_authorized") is True
            and active.get("promotion_authorized") is False
            and active.get("commercial_claim_authorized") is False
        ):
            raise CampaignCoordinatorError("active V8R4A pretrain governance is invalid")
        receipt = self.paths.campaign / "GPU_STATE_MIGRATION_RECEIPT_V8R4A.json"
        state = self.migrator.validate_migrated_state(
            self.paths.project_root, receipt, require_closed=True
        )
        usage = getattr(state, "usage_state", None)
        execution = getattr(state, "execution_state", None)

        if isinstance(usage, Mapping):
            usage_closed = usage.get("open_reservation_count") == 0
        else:
            reservations = getattr(usage, "open_reservations", None)
            usage_closed = reservations is not None and len(reservations) == 0
        if not usage_closed:
            raise CampaignCoordinatorError("GPU usage ledger is not closed")
        if isinstance(execution, Mapping):
            execution_closed = execution.get("open_start_count") == 0
        else:
            starts = getattr(execution, "open_starts", None)
            execution_closed = starts is not None and len(starts) == 0
        if not execution_closed:
            raise CampaignCoordinatorError("GPU execution ledger is not closed")

    def select_and_authorize(self) -> PromotionState:
        selection_exists = os.path.lexists(self.paths.selection_lock)
        authorization_exists = os.path.lexists(self.paths.promotion_authorization)
        if _selection_needs_publication(
            selection_exists=selection_exists,
            authorization_exists=authorization_exists,
        ):
            self.selector.select_common_variant(
                project_root=self.paths.project_root,
                discovery_root=self.paths.discovery_aggregation,
                selection_lock_path=self.paths.selection_lock,
                promotion_authorization_path=self.paths.promotion_authorization,
            )
        # The selector's pure derivation check prevents stale or hand-authored
        # existing files from being treated as a resumable completion.
        self.selector.validate_locked_selection_authorization(
            self.paths.project_root,
            selection_lock_path=self.paths.selection_lock,
            promotion_authorization_path=self.paths.promotion_authorization,
        )
        selection, authorization, governance = (
            self.fixed.validate_v8r4_promotion_authority(
                project_root=self.paths.project_root,
                selection_path=self.paths.selection_lock,
                authorization_path=self.paths.promotion_authorization,
            )
        )
        authorization_path = _require_immutable_file(
            self.paths.promotion_authorization, "promotion authorization"
        )
        _authorization_document, authorization_binding = _read_immutable_json(
            authorization_path, "promotion authorization"
        )
        builder_authorization = self.pack_builder.validate_promotion_authorization(
            authorization_path,
            expected_sha256=authorization_binding["sha256"],
            expected_bytes=authorization_binding["bytes"],
            required_scope=self.pack_builder.PROMOTION_TRAINING_SCOPE,
        )
        state = PromotionState(
            selection=selection,
            authorization=authorization,
            governance=governance,
            builder_authorization=builder_authorization,
        )
        _validate_promotion_state_shape(state)
        return state

    def build_promotion_training_pack(
        self, outer_fold: int, promotion: PromotionState
    ) -> PackCapability:
        _require_member(
            outer_fold, PROMOTION_TRAINING_FOLDS, "promotion training pack outer fold"
        )
        result = self.pack_builder.build_pack_matrix(
            project_root=self.paths.project_root,
            training_index=self.pack_builder.DEFAULT_INDEX_RELATIVE,
            output_root=self.paths.promotion_pack_root / "training_shards",
            expected_index_sha256=self.pack_builder.DEFAULT_INDEX_SHA256,
            expected_index_bytes=self.pack_builder.DEFAULT_INDEX_BYTES,
            require_exact_matrix=True,
            selected_outer_fold=outer_fold,
            selected_seed=None,
            promotion_authorization=promotion.builder_authorization,
        )
        if not (
            isinstance(result, Mapping)
            and result.get("status") == "complete"
            and result.get("exact_three_seed_cover") is True
            and result.get("outer_fold") == outer_fold
        ):
            raise CampaignCoordinatorError(
                f"promotion training pack outer {outer_fold} is incomplete"
            )
        capability = PackCapability(
            self.paths.promotion_training_pack_root(outer_fold),
            self.paths.promotion_training_pack_index(outer_fold),
        )
        _require_immutable_file(capability.index, "promotion training pack index")
        _require_frozen_tree(capability.root, "promotion training pack")
        self._training_packs[outer_fold] = capability
        return capability

    def _deep_model_sources(
        self, promotion: PromotionState
    ) -> tuple[dict[tuple[int, int], Any], Sequence[Any], Any]:
        if set(self._training_packs) != set(PROMOTION_TRAINING_FOLDS):
            raise CampaignCoordinatorError("promotion model packs precede incomplete training")
        selected_variant = str(promotion.selection.get("selected_variant", ""))
        if not selected_variant:
            raise CampaignCoordinatorError("selected variant is absent")
        all_units, index_binding = self.pack_builder.load_training_index(
            self.paths.project_root,
            self.paths.project_root / self.pack_builder.DEFAULT_INDEX_RELATIVE,
            expected_sha256=self.pack_builder.DEFAULT_INDEX_SHA256,
            expected_bytes=self.pack_builder.DEFAULT_INDEX_BYTES,
            require_exact_matrix=True,
        )
        source_units = {(unit.outer_fold, unit.seed): unit for unit in all_units}
        if set(source_units) != {
            (outer_fold, seed) for outer_fold in ALL_FOLDS for seed in SEEDS
        }:
            raise CampaignCoordinatorError("immutable prediction index is not 18-unit exact cover")

        model_sources: dict[tuple[int, int], Any] = {}
        for outer_fold in PROMOTION_TRAINING_FOLDS:
            capability = self._training_packs[outer_fold]
            training, _training_binding = (
                self.fixed.load_target_scoped_promotion_training_pack(
                    project_root=self.paths.project_root,
                    index_path=capability.index,
                    outer_fold=outer_fold,
                    authorization_binding=promotion.governance[
                        "promotion_authorization"
                    ],
                )
            )
            for seed in SEEDS:
                source = self.locked_inputs.resolve_promotion_model_source(
                    project_root=self.paths.project_root,
                    run_root=self.paths.promotion_training_output(outer_fold),
                    cache_dir=training[(outer_fold, seed)].cache_dir,
                    outer_fold=outer_fold,
                    seed=seed,
                    variant=selected_variant,
                )
                if getattr(source, "kind", None) != "local_training":
                    raise CampaignCoordinatorError("local model source kind drifted")
                model_sources[(outer_fold, seed)] = source

        discovery_seal = self.paths.discovery_aggregation / DISCOVERY_COMPLETION_FILENAME
        discovery_binding = self.discovery.bind_file(discovery_seal)
        for outer_fold in DISCOVERY_FOLDS:
            training, _training_binding = self.discovery.load_training_index(
                self.paths.project_root,
                self.paths.discovery_pack_index(outer_fold),
                outer_fold_shard=outer_fold,
            )
            for seed in SEEDS:
                source = self.locked_inputs.validate_selected_discovery_source(
                    project_root=self.paths.project_root,
                    discovery_completion_seal=discovery_binding,
                    cache_dir=training[(outer_fold, seed)].cache_dir,
                    outer_fold=outer_fold,
                    seed=seed,
                    variant=selected_variant,
                )
                if getattr(source, "kind", None) != "discovery":
                    raise CampaignCoordinatorError("discovery model source kind drifted")
                model_sources[(outer_fold, seed)] = source
        if len(model_sources) != 18:
            raise CampaignCoordinatorError("host did not deep-resolve exactly 18 model sources")
        kinds = [getattr(source, "kind", None) for source in model_sources.values()]
        if kinds.count("local_training") != 12 or kinds.count("discovery") != 6:
            raise CampaignCoordinatorError("model-source ownership topology drifted")
        return model_sources, all_units, index_binding

    def build_model_bound_prediction_packs(
        self, promotion: PromotionState
    ) -> Mapping[int, PackCapability]:
        # Revalidate the selection gate immediately before model bytes become a
        # prediction capability.
        self.selector.validate_locked_selection_authorization(
            self.paths.project_root,
            selection_lock_path=self.paths.selection_lock,
            promotion_authorization_path=self.paths.promotion_authorization,
        )
        model_sources, all_units, index_binding = self._deep_model_sources(promotion)
        selected_variant = str(promotion.selection["selected_variant"])
        result: dict[int, PackCapability] = {}
        for outer_fold in ALL_FOLDS:
            units = [unit for unit in all_units if unit.outer_fold == outer_fold]
            built = self.pack_builder.build_model_bound_prediction_shard(
                units,
                model_sources={
                    seed: model_sources[(outer_fold, seed)] for seed in SEEDS
                },
                authorization=promotion.builder_authorization,
                index_binding=index_binding,
                selection_lock_path=self.paths.selection_lock,
                selected_variant=selected_variant,
                output_root=self.paths.model_bound_pack_root(outer_fold),
                outer_fold=outer_fold,
                selected_seed=None,
            )
            if not (
                isinstance(built, Mapping)
                and built.get("status") == "complete"
                and built.get("outer_fold") == outer_fold
                and built.get("completed_units") == 3
                and built.get("physical_target_free_input_and_model_packs") is True
                and built.get("source_paths_or_peer_outputs_authorized_in_child")
                is False
            ):
                raise CampaignCoordinatorError(
                    f"model-bound prediction pack outer {outer_fold} is incomplete"
                )
            capability = PackCapability(
                self.paths.model_bound_pack_root(outer_fold),
                self.paths.model_bound_pack_index(outer_fold),
            )
            _require_immutable_file(capability.index, "model-bound prediction index")
            _require_immutable_file(
                capability.root / MODEL_SOURCE_SEAL_FILENAME,
                "model-source shard seal",
            )
            _require_frozen_tree(capability.root, "model-bound prediction pack")
            result[outer_fold] = capability
        self._prediction_packs = result
        return dict(result)

    def _phase_shape(self, action: str) -> tuple[str, int | None]:
        if action == "efficiency_benchmark":
            return "efficiency_benchmark", 3
        match = re.fullmatch(r"discovery_outer_([34])", action)
        if match:
            return "discovery", int(match.group(1))
        if action == "discovery_aggregation":
            return "discovery_aggregation", None
        match = re.fullmatch(r"promotion_training_outer_([0125])", action)
        if match:
            return "promotion_training", int(match.group(1))
        match = re.fullmatch(r"promotion_prediction_outer_([0-5])", action)
        if match:
            return "promotion_prediction", int(match.group(1))
        if action == "promotion_aggregation":
            return "promotion_aggregation", None
        raise CampaignCoordinatorError(f"unknown phase action: {action}")

    def _output_for(self, phase: str, outer_fold: int | None) -> Path:
        if phase == "efficiency_benchmark":
            return self.paths.benchmark_output
        if phase == "discovery":
            assert outer_fold is not None
            return self.paths.discovery_output(outer_fold)
        if phase == "discovery_aggregation":
            return self.paths.discovery_aggregation
        if phase == "promotion_training":
            assert outer_fold is not None
            return self.paths.promotion_training_output(outer_fold)
        if phase == "promotion_prediction":
            assert outer_fold is not None
            return self.paths.prediction_output(outer_fold)
        if phase == "promotion_aggregation":
            return self.paths.promotion_aggregation
        raise CampaignCoordinatorError("output phase drifted")

    def _pack_for(self, phase: str, outer_fold: int | None) -> PackCapability | None:
        if phase in {"efficiency_benchmark", "discovery"}:
            assert outer_fold is not None
            capability = PackCapability(
                self.paths.discovery_pack_root(outer_fold),
                self.paths.discovery_pack_index(outer_fold),
            )
        elif phase == "promotion_training":
            assert outer_fold is not None
            capability = self._training_packs.get(outer_fold)
            if capability is None:
                raise CampaignCoordinatorError("promotion training pack was not built")
        elif phase == "promotion_prediction":
            assert outer_fold is not None
            capability = self._prediction_packs.get(outer_fold)
            if capability is None:
                raise CampaignCoordinatorError("model-bound prediction pack was not built")
        else:
            return None
        _require_immutable_file(capability.index, "sealed pack index")
        _require_frozen_tree(capability.root, "sealed pack")
        return capability

    def _governance_catalog(
        self,
        *,
        phase: str,
        pack: PackCapability | None,
    ) -> dict[str, Path]:
        campaign = self.paths.campaign
        catalog: dict[str, Path] = {
            "correction_authorization": campaign
            / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4.json",
            "infrastructure_correction_authorization": campaign
            / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A.json",
            "source_closure_correction_authorization": campaign
            / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_SOURCE_CLOSURE.json",
            "source_closure_dependency_authorization": campaign
            / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_SOURCE_CLOSURE_DEPENDENCIES.json",
            "kill_safe_correction_authorization": campaign
            / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_KILL_SAFE_RESUME.json",
            "open_lifecycle_recovery_correction_authorization": campaign
            / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_OPEN_LIFECYCLE_RECOVERY.json",
            "execution_closure_correction_authorization": campaign
            / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_EXECUTION_CLOSURE.json",
            "migration_source_succession_correction_authorization": campaign
            / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_MIGRATION_SOURCE_SUCCESSION.json",
            "fd_closure_correction_authorization": campaign
            / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_FD_CLOSURE.json",
            "canary_boundary_correction_authorization": campaign
            / "IMPLEMENTATION_CORRECTION_AUTHORIZATION_V8R4A_CANARY_BOUNDARY.json",
            "frozen_contract_encoding_correction_authorization": (
                self.paths.project_root / FROZEN_CONTRACT_CORRECTION_RELATIVE
            ),
            "gpu_state_parent_bind_correction_authorization": (
                self.paths.project_root / GPU_STATE_PARENT_BIND_CORRECTION_RELATIVE
            ),
            "admitted_context_correction_authorization": (
                self.paths.project_root
                / BENCHMARK_ADMITTED_CONTEXT_CORRECTION_RELATIVE
            ),
            "failure_diagnostic": campaign
            / "diagnostics/v3r1_v8r3_outer_capability_and_identity_dtype_failure_v8r4.json",
            "infrastructure_failure_diagnostic": campaign
            / "diagnostics/v3r1_v8r4_exact_file_atomic_replace_failure_v8r4a.json",
            "source_closure_failure_diagnostic": campaign
            / "diagnostics/v3r1_v8r4a_pretrain_validator_and_source_closure_failure.json",
            "kill_safe_failure_diagnostic": campaign
            / "diagnostics/v3r1_v8r4a_kill_safe_output_and_shared_ledger_resume_failure.json",
            "open_lifecycle_recovery_failure_diagnostic": campaign
            / "diagnostics/v3r1_v8r4a_open_gpu_lifecycle_kill_recovery_failure.json",
            "execution_closure_failure_diagnostic": campaign
            / "diagnostics/v3r1_v8r4a_terminal_execution_closure_failure.json",
            "migration_source_succession_failure_diagnostic": campaign
            / "diagnostics/v3r1_v8r4a_migrated_state_source_succession_failure.json",
            "fd_closure_failure_diagnostic": campaign
            / "diagnostics/v3r1_v8r4a_outer_guard_urandom_descriptor_failure.json",
            "canary_boundary_failure_diagnostic": campaign
            / "diagnostics/v3r1_v8r4a_denied_canary_prefix_collision_failure.json",
            "frozen_contract_encoding_failure_diagnostic": (
                self.paths.project_root / FROZEN_CONTRACT_DIAGNOSTIC_RELATIVE
            ),
            "gpu_state_parent_bind_failure_diagnostic": (
                self.paths.project_root / GPU_STATE_PARENT_BIND_DIAGNOSTIC_RELATIVE
            ),
            "admitted_context_failure_diagnostic": (
                self.paths.project_root
                / BENCHMARK_ADMITTED_CONTEXT_DIAGNOSTIC_RELATIVE
            ),
            "gpu_state_migration_receipt": campaign
            / "GPU_STATE_MIGRATION_RECEIPT_V8R4A.json",
            "active_authorization": campaign
            / "PRETRAIN_AUTHORIZATION_V8R4A_CONTEXT1.json",
            "source_snapshot": campaign
            / "V3R1_SOURCE_SNAPSHOT_V8R4A_CONTEXT1.json",
            "implementation_test_receipt": campaign
            / "IMPLEMENTATION_TEST_RECEIPT_V8R4A_CONTEXT1.json",
            "campaign_contract": campaign
            / "ADAPTIVE_RETROSPECTIVE_CAMPAIGN_CONTRACT.json",
            "benchmark_receipt": self.paths.benchmark_output
            / "BENCHMARK_COMPLETION_RECEIPT_V8R4.json",
            "quarantine_owner_receipt": campaign
            / "DISCOVERY_V8R3_ATTEMPT_000_QUARANTINE_OWNER_RECEIPT_V8R4.json",
            "quarantined_output_seal": campaign
            / "DISCOVERY_V8R3_ATTEMPT_000_QUARANTINED_OUTPUT_SEAL_V8R4.json",
            "discovery_completion_seal": self.paths.discovery_aggregation
            / DISCOVERY_COMPLETION_FILENAME,
            "selection_lock": self.paths.selection_lock,
            "promotion_authorization": self.paths.promotion_authorization,
        }
        if pack is not None:
            catalog["sealed_pack_index"] = pack.index
        quarantine_seal = catalog["quarantined_output_seal"]
        if phase == "discovery":
            document, _ = _read_immutable_json(
                quarantine_seal, "quarantined output seal"
            )
            rows = document.get("files")
            if not isinstance(rows, list) or len(rows) != 11:
                raise CampaignCoordinatorError("quarantined material cover drifted")
            for number, row in enumerate(rows):
                relative = row.get("path") if isinstance(row, Mapping) else None
                if not isinstance(relative, str):
                    raise CampaignCoordinatorError("quarantined material path is invalid")
                pure = PurePosixPath(relative)
                if pure.is_absolute() or ".." in pure.parts:
                    raise CampaignCoordinatorError("quarantined material path escapes root")
                catalog[f"quarantined_material_{number:02d}"] = (
                    self.paths.project_root / Path(relative)
                )
        for outer_fold in DISCOVERY_FOLDS:
            catalog[f"discovery_shard_seal_outer{outer_fold}"] = (
                self.paths.discovery_output(outer_fold)
                / DISCOVERY_SHARD_SEAL_FILENAME
            )
        for outer_fold in ALL_FOLDS:
            model_seal = self.paths.model_bound_pack_root(outer_fold) / MODEL_SOURCE_SEAL_FILENAME
            prediction_seal = self.paths.prediction_output(outer_fold) / PREDICTION_SHARD_SEAL_FILENAME
            catalog[f"model_source_seal_outer{outer_fold}"] = model_seal
            catalog[f"model_source_shard_seal_outer{outer_fold}"] = model_seal
            catalog[f"prediction_shard_seal_outer{outer_fold}"] = prediction_seal
        return catalog

    def _governance_for(
        self, phase: str, pack: PackCapability | None
    ) -> dict[str, Path]:
        roles_by_phase = getattr(self.runtime, "GOVERNANCE_ROLES_BY_PHASE", None)
        if not isinstance(roles_by_phase, Mapping) or phase not in roles_by_phase:
            raise CampaignCoordinatorError(f"runtime governance ABI lacks {phase}")
        entry_name = {
            "efficiency_benchmark": BENCHMARK_SCRIPT_RELATIVE.name,
            "discovery": DISCOVERY_SCRIPT_RELATIVE.name,
            "discovery_aggregation": DISCOVERY_SCRIPT_RELATIVE.name,
            "promotion_training": FIXED_SCRIPT_RELATIVE.name,
            "promotion_prediction": FIXED_SCRIPT_RELATIVE.name,
            "promotion_aggregation": FIXED_SCRIPT_RELATIVE.name,
        }[phase]
        role_resolver = getattr(self.runtime, "_governance_roles_for", None)
        required = set(
            role_resolver(phase=phase, entry_name=entry_name)
            if callable(role_resolver)
            else roles_by_phase[phase]
        )
        catalog = self._governance_catalog(phase=phase, pack=pack)
        unknown = required - set(catalog)
        if unknown:
            raise CampaignCoordinatorError(
                f"runtime governance ABI has unresolved roles: {sorted(unknown)}"
            )
        selected = {role: catalog[role] for role in sorted(required)}
        for role, path in selected.items():
            _require_immutable_file(path, f"governance:{role}")
        return selected

    def _denied_canaries(self, phase: str, outer_fold: int | None) -> dict[str, Path]:
        del phase
        other_fold = 4 if outer_fold == 3 else 3
        catalog = {
            "legacy_combined_cache": self.paths.project_root
            / "artifacts/cache/harmonic_set_v2_fixed_i3_pretest_v2",
            "raw_input_root": self.paths.project_root / "HAI_EXPERIMENT",
            "target_root": self.paths.project_root
            / "artifacts/campaigns/harmonic_candidate_set_snn_v2/nested_proposer/full_oof_test",
            "hai_experiment": self.paths.project_root / "HAI_EXPERIMENT/S01_CMS",
            "unadmitted_pack_root": self.paths.discovery_pack_root(other_fold),
            "other_output_root": self.paths.project_root
            / RUNS_RELATIVE
            / "efficiency_benchmark_v8",
            "superseded_v8r4a_lifecycle_root": (
                self.paths.superseded_lifecycle_root
            ),
            "superseded_v8r4a_output_root": (
                self.paths.superseded_benchmark_output
            ),
            "superseded_v8r4a_contract1_lifecycle_root": (
                self.paths.superseded_contract1_lifecycle_root
            ),
            "superseded_v8r4a_contract1_output_root": (
                self.paths.superseded_contract1_benchmark_output
            ),
            "superseded_v8r4a_rootbind1_lifecycle_root": (
                self.paths.superseded_rootbind1_lifecycle_root
            ),
            "superseded_v8r4a_rootbind1_output_root": (
                self.paths.superseded_rootbind1_benchmark_output
            ),
        }
        mandatory = set(
            getattr(self.runtime, "MANDATORY_DENIED_CANARY_ROLES", ())
        )
        unknown = mandatory - set(catalog)
        if unknown:
            raise CampaignCoordinatorError(
                f"runtime denied-canary ABI has unresolved roles: {sorted(unknown)}"
            )
        # Each host inode may be represented by exactly one canary capability.
        # The mandatory peer roles already cover the other pack/output roots;
        # aliases under reserved prefixes would make the runtime reject the
        # request before constructing the sandbox.
        return {role: catalog[role] for role in sorted(mandatory)}

    def _child_command(
        self,
        *,
        phase: str,
        outer_fold: int | None,
        output: Path,
        lifecycle: Path,
        pack: PackCapability | None,
    ) -> tuple[str, ...]:
        capability = lifecycle / CAPABILITY_FILENAME
        common = ["--project-root", str(self.paths.project_root)]
        if phase == "efficiency_benchmark":
            assert outer_fold == 3 and pack is not None
            return tuple(
                [
                    str(self.interpreter),
                    str(self.paths.project_root / BENCHMARK_SCRIPT_RELATIVE),
                    *common,
                    "--training-index",
                    str(pack.index),
                    "--run-root",
                    str(output),
                    "--python",
                    str(self.interpreter),
                    "--target-sealed-capability-receipt",
                    str(capability),
                ]
            )
        if phase in {"discovery", "discovery_aggregation"}:
            command = [
                str(self.interpreter),
                str(self.paths.project_root / DISCOVERY_SCRIPT_RELATIVE),
                *common,
            ]
            if phase == "discovery":
                assert outer_fold is not None and pack is not None
                command.extend(
                    [
                        "--outer-fold-shard",
                        str(outer_fold),
                        "--training-index",
                        str(pack.index),
                    ]
                )
            else:
                command.append("--aggregate-shards")
            command.extend(
                [
                    "--run-root",
                    str(output),
                    "--target-sealed-capability-receipt",
                    str(capability),
                    "--python",
                    str(self.interpreter),
                ]
            )
            return tuple(command)
        if phase in {
            "promotion_training",
            "promotion_prediction",
            "promotion_aggregation",
        }:
            command = [
                str(self.interpreter),
                str(self.paths.project_root / FIXED_SCRIPT_RELATIVE),
                *common,
            ]
            if phase == "promotion_training":
                assert outer_fold is not None and pack is not None
                command.extend(["--promotion-model-shard", str(outer_fold)])
            elif phase == "promotion_prediction":
                assert outer_fold is not None and pack is not None
                command.extend(["--prediction-shard", str(outer_fold)])
            else:
                command.append("--aggregate")
            if pack is not None:
                command.extend(["--sealed-pack-index", str(pack.index)])
            command.extend(
                [
                    "--target-sealed-capability-receipt",
                    str(capability),
                    "--run-root",
                    str(output),
                    "--python",
                    str(self.interpreter),
                ]
            )
            return tuple(command)
        raise CampaignCoordinatorError("child phase drifted")

    def _cuda_devices(self) -> tuple[Path, ...]:
        candidates = [Path("/dev") / name for name in getattr(self.runtime, "CUDA_DEVICE_BASENAMES", ())]
        try:
            candidates.extend(
                path
                for path in Path("/dev").iterdir()
                if re.fullmatch(r"nvidia[0-9]+", path.name)
            )
        except OSError:
            pass
        return tuple(
            sorted(
                {
                    path
                    for path in candidates
                    if path.exists() and stat.S_ISCHR(os.stat(path).st_mode)
                },
                key=os.fspath,
            )
        )

    def _environment(self) -> dict[str, str]:
        allowed = set(getattr(self.runtime, "SAFE_PROPAGATED_ENV", ()))
        return {
            name: os.environ[name]
            for name in sorted(allowed)
            if name in os.environ
        }

    def _phase_invocation(self, action: str) -> PhaseInvocation:
        phase, outer_fold = self._phase_shape(action)
        output = _ensure_private_directory(
            self._output_for(phase, outer_fold), boundary=self.paths.runs
        )
        lifecycle = _ensure_private_directory(
            self.paths.lifecycle(phase, outer_fold), boundary=self.paths.runs
        )
        pack = self._pack_for(phase, outer_fold)
        governance = self._governance_for(phase, pack)
        return PhaseInvocation(
            action=action,
            phase=phase,
            outer_fold=outer_fold,
            output_root=output,
            lifecycle_root=lifecycle,
            capability_receipt=lifecycle / CAPABILITY_FILENAME,
            pack=pack,
            governance=governance,
            denied_canaries=self._denied_canaries(phase, outer_fold),
            child_command=self._child_command(
                phase=phase,
                outer_fold=outer_fold,
                output=output,
                lifecycle=lifecycle,
                pack=pack,
            ),
        )

    def execute_phase(
        self, action: str, promotion: PromotionState | None
    ) -> PhaseOutcome:
        phase, _outer_fold = self._phase_shape(action)
        if phase.startswith("promotion_") and promotion is None:
            raise CampaignCoordinatorError("promotion phase lacks its selection gate")
        invocation = self._phase_invocation(action)
        completion_preexisted = os.path.lexists(
            invocation.lifecycle_root / COMPLETION_FILENAME
        )
        argv = build_outer_runtime_argv(
            invocation,
            project_root=self.paths.project_root,
            interpreter=self.interpreter,
            venv_root=self.venv_root,
            python_runtime_root=self.python_runtime_root,
            cuda_runtime_roots=(),
            cuda_devices=self._cuda_devices(),
            propagated_environment=self._environment(),
        )
        return_code = self._phase_runner(argv)
        return PhaseOutcome(
            return_code=return_code,
            reused=completion_preexisted and return_code == 0,
            launched=not completion_preexisted,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root only; campaign science and paths are not configurable",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        backend = ProductionBackend(args.project_root)
        result = CampaignCoordinator(backend).run()
    except (CampaignCoordinatorError, OSError, ValueError, RuntimeError) as error:
        print(
            json.dumps(
                {"status": "failed_closed", "error": str(error)}, sort_keys=True
            ),
            file=sys.stderr,
        )
        return 79
    print(
        json.dumps(
            {
                "status": "v8r4a_target_sealed_campaign_complete",
                "actions": list(result.actions),
                "reused_phases": list(result.reused_phases),
                "launched_phases": list(result.launched_phases),
                "outer_reference_opened": False,
                "commercial_claim_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
