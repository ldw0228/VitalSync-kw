"""Axis-preserving, differentiable-risk SNN proposal for the V8R5 campaign.

This is a separately versioned, non-authorizing successor architecture. It is
not source-independent: it deliberately reuses the frozen 571-wide V3R1 layout
contract and the hash-bound governed ``EpisodeSpikingCell`` implementation,
while replacing the V8R4 router/model topology to address two diagnosed
failure modes:

* evidence interacts non-linearly with radar/ratio/branch coordinates before
  any pooling; and
* training optimizes probability-weighted deployment cost while hard expert
  selection is isolated in deployment-named outputs and excluded from loss.

The forward interface is target-free.  Targets are accepted only by
``soft_risk_routing_loss`` on a permitted training split.  This source is an
unmeasured retrospective proposal and does not authorize protected training or
any commercial/medical claim.
"""

from __future__ import annotations

import math
import hashlib
import json
import os
import stat
import copy
from contextlib import ExitStack
from collections import OrderedDict
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Final, TypeAlias

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .harmonic_feature_layout_v3r1 import (
    EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256,
    FEATURE_LAYOUT_SEMANTIC_SHA256,
    RF_SVD_RATIOS,
    TOTAL_FEATURE_WIDTH,
    validate_ordered_feature_names,
)
from .svd_episode_models import EpisodeSpikingCell


DIRECTED_RELATIONS: Final[tuple[str, ...]] = (
    "near",
    "receiver_is_2x_sender",
    "sender_is_2x_receiver",
    "receiver_is_3x_sender",
    "sender_is_3x_receiver",
    "receiver_is_4x_sender",
    "sender_is_4x_receiver",
)
FACTOR_CLASSES: Final[tuple[float, ...]] = tuple(float(value) for value in RF_SVD_RATIOS)
NeuronState: TypeAlias = tuple[Tensor, Tensor]
RiskRouterState: TypeAlias = tuple[NeuronState, NeuronState]

_TRAINING_STAGE_MODULES: Final[Mapping[str, tuple[str, ...] | None]] = {
    "axis_and_expert_value_warmup": (
        "encoder",
        "graph",
        "pool",
        "episode_projection",
        "radar_context",
        "source_context",
        "temporal",
        "candidate_value_head",
        "anchor_value_head",
    ),
    "soft_risk_router_warmup": (
        "candidate_route_head",
        "anchor_route_head",
        "candidate_risk_head",
        "anchor_risk_head",
        "factor_head",
        "quality_head",
    ),
    "joint_finetune": None,
}
_ALL_LOSS_COMPONENTS: Final[tuple[str, ...]] = (
    "soft_expected_deployment_cost",
    "equivalence_set_cross_entropy",
    "equivalence_value_smooth_l1",
    "expected_abs_error_calibration",
    "tail2_bce",
    "tail5_bce",
    "scale_nll",
    "quality_bce",
)
_TRAINING_STAGE_ACTIVE_LOSSES: Final[Mapping[str, tuple[str, ...]]] = {
    "axis_and_expert_value_warmup": ("equivalence_value_smooth_l1",),
    "soft_risk_router_warmup": (
        "soft_expected_deployment_cost",
        "equivalence_set_cross_entropy",
        "expected_abs_error_calibration",
        "tail2_bce",
        "tail5_bce",
        "scale_nll",
        "quality_bce",
    ),
    "joint_finetune": _ALL_LOSS_COMPONENTS,
}

_MODEL_SOURCE_SHA256: Final[str] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _read_required_proposal_config_bytes(path: Path) -> bytes:
    """Read the mandatory proposal config or fail the module closed."""

    try:
        return path.read_bytes()
    except OSError as error:
        raise RuntimeError(
            "the mandatory V8R5 proposal config is unavailable; model import fails closed"
        ) from error


_PROPOSAL_CONFIG_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2] / "configs/axis_risk_router_snn_v8r5.yaml"
)
_PROPOSAL_CONFIG_BYTES: Final[bytes] = _read_required_proposal_config_bytes(
    _PROPOSAL_CONFIG_PATH
)
_PROPOSAL_CONFIG_SHA256: Final[str] = hashlib.sha256(
    _PROPOSAL_CONFIG_BYTES
).hexdigest()
_SPIKING_CELL_SOURCE_PATH: Final[Path] = Path(__file__).with_name(
    "svd_episode_models.py"
)
_SPIKING_CELL_SOURCE_SHA256: Final[str] = hashlib.sha256(
    _SPIKING_CELL_SOURCE_PATH.read_bytes()
).hexdigest()
_FEATURE_LAYOUT_SOURCE_PATH: Final[Path] = Path(__file__).with_name(
    "harmonic_feature_layout_v3r1.py"
)
_FEATURE_LAYOUT_SOURCE_SHA256: Final[str] = hashlib.sha256(
    _FEATURE_LAYOUT_SOURCE_PATH.read_bytes()
).hexdigest()
_BEHAVIOR_CONTRACT_FIELDS: Final[tuple[str, ...]] = (
    "hidden_channels",
    "graph_blocks",
    "simulation_steps",
    "dropout",
    "beta",
    "adaptation_decay",
    "adaptation_strength",
    "rr_min_bpm",
    "rr_max_bpm",
    "tail2_risk_weight",
    "tail5_risk_weight",
    "candidate_residual_limit_bpm",
    "anchor_residual_limit_bpm",
    "near_relation_tolerance_bpm",
    "ratio_relation_tolerance_bpm",
    "edge_log_ratio_bandwidth",
    "factor_affinity_bandwidth_bpm",
)
_BEHAVIOR_RUNTIME_ATTRIBUTES: Final[tuple[str, ...]] = (
    "hidden_channels",
    "graph_blocks",
    "simulation_steps",
    "dropout_probability",
    "beta",
    "adaptation_decay",
    "adaptation_strength",
    "rr_min_bpm",
    "rr_max_bpm",
    "tail2_risk_weight",
    "tail5_risk_weight",
    "candidate_residual_limit_bpm",
    "anchor_residual_limit_bpm",
    "near_relation_tolerance_bpm",
    "ratio_relation_tolerance_bpm",
    "edge_log_ratio_bandwidth",
    "factor_affinity_bandwidth_bpm",
)
if len(_BEHAVIOR_RUNTIME_ATTRIBUTES) != len(_BEHAVIOR_CONTRACT_FIELDS):
    raise RuntimeError("V8R5 behavior field/attribute contract drifted")
_FROZEN_MODEL_ATTRIBUTES: Final[frozenset[str]] = frozenset(
    {
        *_BEHAVIOR_RUNTIME_ATTRIBUTES,
        "MAX_CANDIDATES",
        "maximum_parameters",
        "ordered_feature_names_semantic_sha256",
        "structural_layout_semantic_sha256",
        "_checkpoint_source_receipt",
        "_runtime_structure_receipt",
        "_behavior_contract",
        "_route_temperature",
        "_v8r5_runtime_contract_frozen",
    }
)
_CHECKPOINT_RECEIPT_SCHEMA_VERSION: Final[int] = 1
_RUNTIME_STRUCTURE_RECEIPT_SCHEMA_VERSION: Final[int] = 1
_LOCKED_MAX_CANDIDATES: Final[int] = 12
_NONFINITE_SOURCE_POLICY: Final[str] = (
    "remove_nonfinite_expert_then_finite_classical_fallback_or_unavailable_quality_zero"
)
_SOURCE_EXECUTION_BINDING_SCOPE: Final[str] = (
    "initialization_time_disk_bytes_not_actual_loader_compiled_bytes"
)
_SOURCE_EXECUTION_TERMINAL_BLOCKER: Final[str] = (
    "external_launcher_executed_byte_closure_and_verifier_absent"
)


def _canonical_checkpoint_source_receipt(
    *,
    ordered_feature_names_semantic_sha256: str,
    structural_layout_semantic_sha256: str,
) -> bytes:
    """Return the weights-only-safe immutable source/dependency contract.

    The canonical JSON is persisted as a uint8 state-dict buffer.  A checkpoint
    therefore cannot silently transplant weights from a different source,
    config, feature layout, or governed spiking-cell generation merely because
    the parameter shapes happen to agree.
    """

    document = {
        "checkpoint_receipt_schema_version": _CHECKPOINT_RECEIPT_SCHEMA_VERSION,
        "model_family": "axis_risk_router_snn_v8r5_unmeasured_proposal",
        "model_source_sha256": _MODEL_SOURCE_SHA256,
        "model_source_binding_scope": _SOURCE_EXECUTION_BINDING_SCOPE,
        "binds_actual_loader_compiled_bytes": False,
        "training_authorization_terminal_blocker": (
            _SOURCE_EXECUTION_TERMINAL_BLOCKER
        ),
        "proposal_config_sha256": _PROPOSAL_CONFIG_SHA256,
        "proposal_config_absence_policy": "module_import_fails_closed",
        "feature_layout_source_sha256": _FEATURE_LAYOUT_SOURCE_SHA256,
        "spiking_cell_source_sha256": _SPIKING_CELL_SOURCE_SHA256,
        "ordered_feature_names_semantic_sha256": (
            ordered_feature_names_semantic_sha256
        ),
        "structural_layout_semantic_sha256": structural_layout_semantic_sha256,
        "total_feature_width": TOTAL_FEATURE_WIDTH,
        "nonfinite_source_policy": _NONFINITE_SOURCE_POLICY,
        "hard_selection_policy": "eval_deployment_only_absent_in_training",
        "runtime_structure_receipt_schema_version": (
            _RUNTIME_STRUCTURE_RECEIPT_SCHEMA_VERSION
        ),
    }
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


SnapshotSignature: TypeAlias = tuple[int, int, int, int, int, int, int]


def _snapshot_signature(value: os.stat_result) -> SnapshotSignature:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
        value.st_nlink,
    )


def _read_file_snapshot(path: Path) -> tuple[bytes, SnapshotSignature]:
    try:
        with path.open("rb") as handle:
            before = _snapshot_signature(os.fstat(handle.fileno()))
            payload = handle.read()
            after = _snapshot_signature(os.fstat(handle.fileno()))
    except OSError as error:
        raise ValueError(f"cannot read cache contract payload {path}") from error
    if before != after or len(payload) != before[2]:
        raise ValueError(f"cache contract payload changed while reading: {path}")
    return payload, before


def _load_strict_json_object(
    path: Path,
) -> tuple[dict[str, object], str, SnapshotSignature]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    payload, signature = _read_file_snapshot(path)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read strict JSON contract {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON contract {path} must contain an object")
    return value, hashlib.sha256(payload).hexdigest(), signature


def _canonical_mapping_sha256(
    value: Mapping[str, object], *, exclude: str | None = None
) -> str:
    payload = dict(value)
    if exclude is not None:
        payload.pop(exclude, None)
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("cache manifest cannot be canonically encoded") from error
    return hashlib.sha256(encoded).hexdigest()


def _validate_bound_v8r5_payloads(
    root: Path,
    outputs: Mapping[str, object],
    feature_shape: list[object],
    *,
    names_sha256: str,
    names_signature: SnapshotSignature,
) -> dict[str, str]:
    """Hash, mmap, and semantically inspect one stable inode per payload."""

    required_outputs = {
        "node_features": "node_features.npy",
        "node_feature_availability": "node_feature_availability.npy",
        "candidate_bpm": "candidate_bpm.npy",
        "candidate_mask": "candidate_mask.npy",
        "joint_radar_mask": "joint_radar_mask.npy",
        "feature_names": "feature_names.json",
    }
    verified_outputs: dict[str, str] = {}
    signatures: dict[str, SnapshotSignature] = {}
    paths: dict[str, Path] = {}
    with ExitStack() as stack:
        handles = {}
        for logical_name, canonical_filename in required_outputs.items():
            binding = outputs.get(logical_name)
            if not isinstance(binding, dict):
                raise ValueError(f"cache output binding {logical_name!r} is absent")
            filename = binding.get("filename")
            expected_sha256 = binding.get("sha256")
            expected_bytes = binding.get("bytes")
            if (
                filename != canonical_filename
                or not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
                or type(expected_bytes) is not int
                or expected_bytes < 1
            ):
                raise ValueError(f"cache output binding {logical_name!r} is malformed")
            path = root / canonical_filename
            try:
                handle = stack.enter_context(path.open("rb"))
            except OSError as error:
                raise ValueError(f"cannot open cache output {logical_name!r}") from error
            before = _snapshot_signature(os.fstat(handle.fileno()))
            if (
                not stat.S_ISREG(before[5])
                or before[6] < 1
                or before[2] != expected_bytes
            ):
                raise ValueError(f"cache output {logical_name!r} inode/size mismatch")
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            after = _snapshot_signature(os.fstat(handle.fileno()))
            if before != after:
                raise ValueError(f"cache output {logical_name!r} changed while hashing")
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError(f"cache output {logical_name!r} hash mismatch")
            if logical_name == "feature_names" and (
                actual_sha256 != names_sha256 or before != names_signature
            ):
                raise ValueError("parsed feature_names.json differs from its bound inode")
            handle.seek(0)
            handles[logical_name] = handle
            paths[logical_name] = path
            signatures[logical_name] = before
            verified_outputs[logical_name] = actual_sha256

        import numpy as np

        proc_root = Path("/proc/self/fd")
        if not proc_root.is_dir():
            raise ValueError("stable-fd NPY validation requires Linux /proc/self/fd")
        arrays = {
            name: np.load(
                str(proc_root / str(handles[name].fileno())),
                mmap_mode="r",
                allow_pickle=False,
            )
            for name in required_outputs
            if required_outputs[name].endswith(".npy")
        }
        feature_array = arrays["node_features"]
        availability_array = arrays["node_feature_availability"]
        candidate_bpm = arrays["candidate_bpm"]
        candidate_mask = arrays["candidate_mask"]
        joint_radar_mask = arrays["joint_radar_mask"]
        row_count = feature_shape[0]
        if (
            list(feature_array.shape) != feature_shape
            or feature_array.dtype != np.dtype("float32")
            or list(availability_array.shape) != feature_shape
            or availability_array.dtype != np.dtype("bool")
            or candidate_bpm.shape != (row_count, 12)
            or candidate_bpm.dtype != np.dtype("float32")
            or candidate_mask.shape != (row_count, 12)
            or candidate_mask.dtype != np.dtype("bool")
            or joint_radar_mask.shape != (row_count, 3)
            or joint_radar_mask.dtype != np.dtype("bool")
        ):
            raise ValueError("actual cache arrays disagree with their manifest contract")
        for begin in range(0, row_count, 256):
            end = min(begin + 256, row_count)
            feature_chunk = np.asarray(feature_array[begin:end])
            availability_chunk = np.asarray(availability_array[begin:end])
            bpm_chunk = np.asarray(candidate_bpm[begin:end])
            candidate_chunk = np.asarray(candidate_mask[begin:end])
            radar_chunk = np.asarray(joint_radar_mask[begin:end])
            if not np.isfinite(feature_chunk[availability_chunk]).all():
                raise ValueError("available cache features contain non-finite values")
            if np.count_nonzero(feature_chunk[~availability_chunk]):
                raise ValueError("unavailable cache features are not exact zero")
            if not np.array_equal(availability_chunk[..., 0], candidate_chunk):
                raise ValueError(
                    "candidate feature availability differs from candidate_mask"
                )
            if np.any(availability_chunk[~candidate_chunk]):
                raise ValueError("padded candidates expose available features")
            valid_bpm = (
                np.isfinite(bpm_chunk)
                & (bpm_chunk >= 6.0)
                & (bpm_chunk <= 45.0)
            )
            if np.any(candidate_chunk & ~valid_bpm):
                raise ValueError("available candidate BPM is non-finite or out of range")
            if np.count_nonzero(bpm_chunk[~candidate_chunk]):
                raise ValueError("padded candidate BPM must be exact zero")
            if not np.array_equal(feature_chunk[..., 0], bpm_chunk):
                raise ValueError("candidate_bpm differs from node feature zero")
            ceiling = build_structural_availability_mask(
                torch.from_numpy(bpm_chunk.copy()),
                torch.from_numpy(candidate_chunk.copy()),
                torch.from_numpy(radar_chunk.copy()),
            ).numpy()
            if np.any(availability_chunk & ~ceiling):
                raise ValueError(
                    "cache availability exceeds the canonical structural ceiling"
                )
        # fstat proves hashing, NPY parsing, and semantic scans all used the
        # exact same open inode. Path stat additionally rejects ABA replacement.
        for logical_name, handle in handles.items():
            if _snapshot_signature(os.fstat(handle.fileno())) != signatures[logical_name]:
                raise ValueError(f"cache output {logical_name!r} mutated during validation")
            if _snapshot_signature(paths[logical_name].stat()) != signatures[logical_name]:
                raise ValueError(f"cache output {logical_name!r} path was replaced")
    return verified_outputs


def validate_v8r5_cache_contract(cache_root: str | Path) -> dict[str, object]:
    """Validate a concrete on-disk cache, never caller-asserted digest strings.

    This closes the path where a caller could pass the repository constants to
    the model while loading a different feature order.  It proves only schema
    compatibility; it deliberately does not issue scientific training
    authorization.
    """

    root = Path(cache_root).resolve()
    manifest_path = root / "manifest.json"
    names_path = root / "feature_names.json"
    manifest, manifest_sha256, manifest_signature = _load_strict_json_object(
        manifest_path
    )
    names_document, names_sha256, names_signature = _load_strict_json_object(
        names_path
    )
    if (
        manifest.get("schema") != "snn_rr.harmonic_candidate_cache.v2"
        or type(manifest.get("format_version")) is not int
        or manifest.get("format_version") != 2
        or manifest.get("complete") is not True
    ):
        raise ValueError("V8R5 requires a complete harmonic cache format_version=2")
    if manifest.get("content_sha256") != _canonical_mapping_sha256(
        manifest, exclude="content_sha256"
    ):
        raise ValueError("cache manifest content_sha256 is invalid")
    expected_contract = {
        "ordered_feature_names_semantic_sha256": EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256,
        "feature_layout_semantic_sha256": FEATURE_LAYOUT_SEMANTIC_SHA256,
        "axis_risk_router_v8r5_compatible": True,
    }
    for field, expected in expected_contract.items():
        manifest_value = manifest.get(field)
        names_value = names_document.get(field)
        if field == "axis_risk_router_v8r5_compatible":
            matches = manifest_value is True and names_value is True
        else:
            matches = manifest_value == expected and names_value == expected
        if not matches:
            raise ValueError(f"cache {field} does not match the frozen V8R5 contract")
    ordered_names = names_document.get("node_feature_names")
    if not isinstance(ordered_names, list) or not all(
        isinstance(name, str) for name in ordered_names
    ):
        raise ValueError("cache node_feature_names must be an ordered string list")
    if len(ordered_names) != TOTAL_FEATURE_WIDTH:
        raise ValueError("cache node feature width is not the canonical 571")
    validate_ordered_feature_names(ordered_names)
    forward_arrays = names_document.get("forward_arrays")
    required_forward_arrays = {
        "node_features",
        "node_feature_availability",
        "candidate_bpm",
        "candidate_mask",
        "joint_radar_mask",
    }
    if (
        not isinstance(forward_arrays, list)
        or not all(isinstance(name, str) for name in forward_arrays)
        or not required_forward_arrays.issubset(forward_arrays)
    ):
        raise ValueError("cache forward-array inventory is incomplete for V8R5")
    settings = manifest.get("settings")
    if not isinstance(settings, dict) or (
        type(settings.get("format_version")) is not int
        or settings.get("format_version") != 2
        or type(settings.get("maximum_candidates")) is not int
        or settings.get("maximum_candidates") != 12
        or settings.get("proposer_features") is not True
        or type(settings.get("svd_components")) is not int
        or settings.get("svd_components") != 12
    ):
        raise ValueError("cache builder settings are not the canonical V8R5 settings")
    feature_shape = manifest.get("node_feature_shape")
    availability_shape = manifest.get("node_feature_availability_shape")
    row_count = manifest.get("row_count")
    if (
        not isinstance(feature_shape, list)
        or len(feature_shape) != 3
        or any(type(dimension) is not int for dimension in feature_shape)
        or type(row_count) is not int
        or row_count < 1
        or feature_shape[0] != row_count
        or feature_shape[-2:] != [12, TOTAL_FEATURE_WIDTH]
        or availability_shape != feature_shape
        or manifest.get("node_feature_dtype") != "float32"
        or manifest.get("node_feature_availability_dtype") != "bool"
    ):
        raise ValueError("cache feature/availability shape or dtype contract drifted")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("cache output bindings are absent")
    verified_outputs = _validate_bound_v8r5_payloads(
        root,
        outputs,
        feature_shape,
        names_sha256=names_sha256,
        names_signature=names_signature,
    )
    if _snapshot_signature(manifest_path.stat()) != manifest_signature:
        raise ValueError("cache manifest changed during validation")
    return {
        "schema_compatible": True,
        "training_authorized": False,
        "cache_format_version": 2,
        **expected_contract,
        "manifest_sha256": manifest_sha256,
        "verified_output_sha256": verified_outputs,
    }


def _finite_real(name: str, value: Real) -> float:
    """Return a finite scalar without accepting booleans as numeric settings."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    float32 = torch.finfo(torch.float32)
    magnitude = abs(result)
    if magnitude > float32.max or (0.0 < magnitude < float32.tiny):
        raise ValueError(f"{name} must be representable as a normal float32 scalar")
    return result


def _positive_real(name: str, value: Real) -> float:
    result = _finite_real(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_real(name: str, value: Real) -> float:
    result = _finite_real(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _inverse_softplus(value: float) -> float:
    value = _positive_real("inverse-softplus input", value)
    return math.log(math.expm1(value))


def _logit(probability: float) -> float:
    probability = _finite_real("logit probability", probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("logit probability must be in (0,1)")
    return math.log(probability / (1.0 - probability))


def _ordered_tail_outputs(
    tail2_logits: Tensor,
    tail5_conditional_logits: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``P(error>2)``, ``P(error>5)`` and a finite P>5 logit.

    P>5 is parameterized as P>2 times a conditional probability, making the
    required nested-event monotonicity structural rather than a soft penalty.
    """

    if tail2_logits.shape != tail5_conditional_logits.shape:
        raise ValueError("tail logits must share shape")
    logits2 = tail2_logits.float()
    conditional_logits5 = tail5_conditional_logits.float()
    if not torch.isfinite(logits2).all() or not torch.isfinite(
        conditional_logits5
    ).all():
        raise RuntimeError("tail-risk head produced non-finite logits")
    probability2 = logits2.sigmoid()
    probability5 = probability2 * conditional_logits5.sigmoid()
    epsilon = torch.finfo(probability5.dtype).eps
    safe_probability5 = probability5.clamp(epsilon, 1.0 - epsilon)
    logits5 = torch.log(safe_probability5) - torch.log1p(-safe_probability5)
    return probability2, probability5, logits5


def _masked_softmax(values: Tensor, mask: Tensor, *, dim: int) -> Tensor:
    if mask.dtype != torch.bool:
        raise ValueError("masked-softmax mask must be boolean")
    # Keep normalization in float32 under AMP.  Fully masked structural axes
    # are expected (missing radar/ratio/candidate), so retain exact zero rather
    # than producing 0/0 -> NaN in fp16.
    working = values.float()
    masked = working.masked_fill(~mask, -1.0e4)
    probability = masked.softmax(dim=dim) * mask.to(working.dtype)
    denominator = probability.sum(dim=dim, keepdim=True)
    normalized = probability / denominator.clamp_min(
        torch.finfo(probability.dtype).tiny
    )
    return normalized.to(values.dtype)


def build_directed_harmonic_relations(
    candidate_rr_bpm: Tensor,
    candidate_mask: Tensor,
    *,
    near_tolerance_bpm: float = 0.5,
    ratio_tolerance_bpm: float = 0.75,
) -> Tensor:
    """Build receiver-by-sender adjacency for seven directed relations."""

    if candidate_rr_bpm.ndim < 1 or candidate_rr_bpm.shape != candidate_mask.shape:
        raise ValueError("candidate RR and mask must share shape [...,K]")
    if candidate_mask.dtype != torch.bool or candidate_rr_bpm.shape[-1] < 1:
        raise ValueError("candidate mask must be boolean and K must be positive")
    if candidate_rr_bpm.device != candidate_mask.device:
        raise ValueError("candidate RR and mask must share a device")
    near_tolerance_bpm = _positive_real(
        "near_tolerance_bpm", near_tolerance_bpm
    )
    ratio_tolerance_bpm = _positive_real(
        "ratio_tolerance_bpm", ratio_tolerance_bpm
    )
    rr = candidate_rr_bpm.float() if not torch.is_floating_point(candidate_rr_bpm) else candidate_rr_bpm
    if (candidate_mask & (~torch.isfinite(rr) | (rr <= 0.0))).any():
        raise ValueError("available candidate RR must be finite and positive")
    valid = candidate_mask & torch.isfinite(rr) & (rr > 0.0)
    receiver = rr.unsqueeze(-1)
    sender = rr.unsqueeze(-2)
    pair = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    identity = torch.eye(rr.shape[-1], dtype=torch.bool, device=rr.device).reshape(
        (1,) * (rr.ndim - 1) + (rr.shape[-1], rr.shape[-1])
    )
    pair &= ~identity
    relations = [pair & ((receiver - sender).abs() <= near_tolerance_bpm)]
    for factor in (2.0, 3.0, 4.0):
        relations.append(
            pair & ((receiver - factor * sender).abs() <= ratio_tolerance_bpm)
        )
        relations.append(
            pair & ((sender - factor * receiver).abs() <= ratio_tolerance_bpm)
        )
    return torch.stack(relations, dim=-1)


def build_directed_harmonic_edge_weights(
    candidate_rr_bpm: Tensor,
    candidate_mask: Tensor,
    *,
    log_ratio_bandwidth: float = 0.08,
    near_tolerance_bpm: float = 0.5,
    ratio_tolerance_bpm: float = 0.75,
) -> Tensor:
    """Return continuous directed log-ratio proximity for seven relations.

    Boolean topology prevents unrelated candidates from communicating.  These
    weights additionally retain how closely each directed edge matches its
    physical ratio instead of treating every in-band pair as identical.
    """

    log_ratio_bandwidth = _positive_real(
        "log_ratio_bandwidth", log_ratio_bandwidth
    )
    rr = (
        candidate_rr_bpm
        if torch.is_floating_point(candidate_rr_bpm)
        else candidate_rr_bpm.float()
    )
    relations = build_directed_harmonic_relations(
        rr,
        candidate_mask,
        near_tolerance_bpm=near_tolerance_bpm,
        ratio_tolerance_bpm=ratio_tolerance_bpm,
    )
    valid_rr = candidate_mask & torch.isfinite(rr) & (rr > 0.0)
    safe_rr = torch.where(valid_rr, rr, torch.ones_like(rr)).clamp_min(
        torch.finfo(rr.dtype).tiny
    )
    observed = safe_rr.unsqueeze(-1).log() - safe_rr.unsqueeze(-2).log()
    expected = observed.new_tensor(
        (0.0, math.log(2.0), -math.log(2.0), math.log(3.0), -math.log(3.0), math.log(4.0), -math.log(4.0))
    )
    proximity = torch.exp(
        -(observed.unsqueeze(-1) - expected).abs() / float(log_ratio_bandwidth)
    )
    return torch.where(relations, proximity, torch.zeros_like(proximity))


def build_structural_availability_mask(
    candidate_rr_bpm: Tensor,
    candidate_mask: Tensor,
    joint_radar_mask: Tensor,
    *,
    rr_min_bpm: float = 6.0,
    rr_max_bpm: float = 45.0,
) -> Tensor:
    """Return the target-free structural availability mask ``[...,K,571]``."""

    if candidate_rr_bpm.shape != candidate_mask.shape or candidate_mask.dtype != torch.bool:
        raise ValueError("candidate RR/mask contract is invalid")
    if joint_radar_mask.dtype != torch.bool or joint_radar_mask.shape != (
        *candidate_rr_bpm.shape[:-1],
        3,
    ):
        raise ValueError("joint_radar_mask must be boolean [...,3]")
    if not (
        candidate_rr_bpm.device
        == candidate_mask.device
        == joint_radar_mask.device
    ):
        raise ValueError("candidate and radar mask tensors must share a device")
    rr_min_bpm = _positive_real("rr_min_bpm", rr_min_bpm)
    rr_max_bpm = _positive_real("rr_max_bpm", rr_max_bpm)
    if rr_min_bpm >= rr_max_bpm:
        raise ValueError("rr_min_bpm must be below rr_max_bpm")
    rr = candidate_rr_bpm.float() if not torch.is_floating_point(candidate_rr_bpm) else candidate_rr_bpm
    if (
        candidate_mask
        & (
            ~torch.isfinite(rr)
            | (rr < rr_min_bpm)
            | (rr > rr_max_bpm)
        )
    ).any():
        raise ValueError("available candidate RR must be finite and in range")
    # Match the frozen canonical 571-layout exactly: core cells follow
    # candidate availability, while only RF/SVD cells additionally depend on
    # per-radar availability.  The layout digest would be misleading if the
    # Torch successor silently imposed a stricter core mask.
    node = candidate_mask
    ratios = rr.new_tensor(RF_SVD_RATIOS)
    ratio_rr = rr.unsqueeze(-1) * ratios
    ratio = node.unsqueeze(-1) & (ratio_rr >= rr_min_bpm) & (ratio_rr <= rr_max_bpm)
    cell = node[..., None, None] & joint_radar_mask[..., None, :, None] & ratio[..., None, :]
    core = node[..., None].expand(*node.shape, 46)
    # The frozen cache has only the raw-power RF branch.  IQ remains absent.
    rf_branch = torch.stack((cell, torch.zeros_like(cell)), dim=-1)
    rf = rf_branch[..., None].expand(*rf_branch.shape, 9).reshape(*node.shape, 378)
    svd = cell[..., None].expand(*cell.shape, 7).reshape(*node.shape, 147)
    return torch.cat((core, rf, svd), dim=-1)


class _AxisPreservingEvidenceEncoder(nn.Module):
    """Jointly encode evidence and coordinates before axial attention."""

    def __init__(
        self,
        *,
        hidden_channels: int,
        rr_min_bpm: float,
        rr_max_bpm: float,
    ) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.rr_min_bpm = float(rr_min_bpm)
        self.rr_max_bpm = float(rr_max_bpm)
        cell_channels = 24
        coordinate_channels = 8
        self.cell_channels = cell_channels
        self.core = nn.Sequential(
            nn.Linear(92, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.LayerNorm(32)
        )
        self.rf_evidence = nn.Sequential(nn.Linear(18, 16), nn.SiLU(), nn.LayerNorm(16))
        self.svd_evidence = nn.Sequential(nn.Linear(14, 16), nn.SiLU(), nn.LayerNorm(16))
        self.radar_embedding = nn.Embedding(3, coordinate_channels)
        self.ratio_embedding = nn.Embedding(7, coordinate_channels)
        self.branch_embedding = nn.Embedding(2, coordinate_channels)
        self.modality_embedding = nn.Embedding(2, coordinate_channels)
        self.candidate_coordinate = nn.Sequential(
            nn.Linear(3, coordinate_channels), nn.SiLU(), nn.Linear(coordinate_channels, coordinate_channels)
        )
        # Concatenation makes coordinate/evidence interaction explicit before pooling.
        joint_width = 16 + 4 * coordinate_channels + 1
        self.rf_joint = nn.Sequential(
            nn.Linear(joint_width, 40), nn.SiLU(), nn.Linear(40, cell_channels), nn.LayerNorm(cell_channels)
        )
        self.svd_joint = nn.Sequential(
            nn.Linear(joint_width, 40), nn.SiLU(), nn.Linear(40, cell_channels), nn.LayerNorm(cell_channels)
        )
        self.query = nn.Sequential(
            nn.Linear(coordinate_channels, cell_channels), nn.SiLU(), nn.Linear(cell_channels, cell_channels)
        )
        self.ratio_score = nn.Linear(cell_channels, 1, bias=False)
        self.radar_score = nn.Linear(cell_channels, 1, bias=False)
        self.output = nn.Sequential(
            nn.Linear(32 + 4 * cell_channels, hidden_channels),
            nn.SiLU(),
            nn.LayerNorm(hidden_channels),
        )

    @staticmethod
    def _pool_axis(
        values: Tensor,
        mask: Tensor,
        query: Tensor,
        scorer: nn.Linear,
        *,
        dim: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        score = scorer(torch.tanh(values + query)).squeeze(-1)
        if score.shape != mask.shape:
            raise ValueError("axial attention score/mask shapes drifted")
        # ``dim`` names an axis of ``values``.  Removing the trailing channel
        # dimension from ``score`` changes the meaning of negative indices:
        # for example ratio is -2 in values but -1 in score.  Convert once to
        # a positive axis so normalization and reduction address the same
        # radar/ratio dimension and never accidentally normalize over the
        # candidate axis.
        value_axis = dim % values.ndim
        if value_axis >= values.ndim - 1:
            raise ValueError("axial attention cannot pool the channel axis")
        weights = _masked_softmax(score, mask, dim=value_axis)
        return (values * weights.unsqueeze(-1)).sum(dim=value_axis), weights

    def _candidate_coordinate_components(self, candidate_rr_bpm: Tensor) -> Tensor:
        safe = candidate_rr_bpm.clamp_min(self.rr_min_bpm)
        log_midpoint = 0.5 * (
            math.log(self.rr_min_bpm) + math.log(self.rr_max_bpm)
        )
        log_half_range = 0.5 * (
            math.log(self.rr_max_bpm) - math.log(self.rr_min_bpm)
        )
        normalized = ((safe.log() - log_midpoint) / log_half_range).clamp(-1.0, 1.0)
        return torch.stack(
            (normalized, torch.sin(math.pi * normalized), torch.cos(math.pi * normalized)), dim=-1
        )

    def _candidate_coordinate(self, candidate_rr_bpm: Tensor) -> Tensor:
        return self.candidate_coordinate(
            self._candidate_coordinate_components(candidate_rr_bpm)
        )

    def _axial_pool(
        self,
        values: Tensor,
        mask: Tensor,
        query: Tensor,
    ) -> tuple[Tensor, Mapping[str, Tensor]]:
        # values/mask: [..., radar, ratio, channels]/[..., radar, ratio]
        query_ratio = query[..., None, None, :]
        per_radar, ratio_weights = self._pool_axis(
            values, mask, query_ratio, self.ratio_score, dim=-2
        )
        radar_mask = mask.any(dim=-1)
        radar_summary, radar_weights = self._pool_axis(
            per_radar, radar_mask, query[..., None, :], self.radar_score, dim=-2
        )

        per_ratio, reverse_radar_weights = self._pool_axis(
            values, mask, query_ratio, self.radar_score, dim=-3
        )
        ratio_mask = mask.any(dim=-2)
        ratio_summary, reverse_ratio_weights = self._pool_axis(
            per_ratio, ratio_mask, query[..., None, :], self.ratio_score, dim=-2
        )
        return torch.cat((radar_summary, ratio_summary), dim=-1), {
            "ratio_then_radar_ratio_weights": ratio_weights,
            "ratio_then_radar_radar_weights": radar_weights,
            "radar_then_ratio_radar_weights": reverse_radar_weights,
            "radar_then_ratio_ratio_weights": reverse_ratio_weights,
        }

    def forward(
        self,
        features: Tensor,
        availability: Tensor,
        candidate_rr_bpm: Tensor,
    ) -> tuple[Tensor, Mapping[str, Tensor]]:
        if features.ndim != 4 or features.shape[-1] != TOTAL_FEATURE_WIDTH:
            raise ValueError("features must be [batch,time,K,571]")
        if availability.shape != features.shape or availability.dtype != torch.bool:
            raise ValueError("availability must be boolean and match features")
        if candidate_rr_bpm.shape != features.shape[:3]:
            raise ValueError("candidate_rr_bpm must be [batch,time,K]")

        node_mask = availability[..., 0]
        core = features[..., :46]
        core_mask = availability[..., :46]
        rf = features[..., 46:424].reshape(*features.shape[:3], 3, 7, 2, 9)
        svd = features[..., 424:].reshape(*features.shape[:3], 3, 7, 7)
        rf_feature_mask = availability[..., 46:424].reshape(
            *features.shape[:3], 3, 7, 2, 9
        )
        svd_feature_mask = availability[..., 424:].reshape(
            *features.shape[:3], 3, 7, 7
        )
        rf_mask = rf_feature_mask.any(dim=-1)
        svd_mask = svd_feature_mask.any(dim=-1)

        coordinate = self._candidate_coordinate(candidate_rr_bpm)
        query = self.query(coordinate)
        radar = self.radar_embedding(torch.arange(3, device=features.device))
        ratio = self.ratio_embedding(torch.arange(7, device=features.device))
        branch = self.branch_embedding(torch.arange(2, device=features.device))
        modality = self.modality_embedding(torch.arange(2, device=features.device))

        rf_shape = (*features.shape[:3], 3, 7, 2)
        rf_parts = (
            self.rf_evidence(
                torch.cat((rf, rf_feature_mask.to(features.dtype)), dim=-1)
            ),
            radar.reshape(1, 1, 1, 3, 1, 1, -1).expand(*rf_shape, -1),
            ratio.reshape(1, 1, 1, 1, 7, 1, -1).expand(*rf_shape, -1),
            branch.reshape(1, 1, 1, 1, 1, 2, -1).expand(*rf_shape, -1),
            coordinate[..., None, None, None, :].expand(*rf_shape, -1),
            rf_mask.unsqueeze(-1).to(features.dtype),
        )
        rf_joint = self.rf_joint(torch.cat(rf_parts, dim=-1))
        rf_joint = rf_joint * rf_mask.unsqueeze(-1).to(features.dtype)
        branch_count = rf_mask.to(features.dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
        rf_branch_pooled = rf_joint.sum(dim=-2) / branch_count
        rf_cell_mask = rf_mask.any(dim=-1)
        rf_summary, rf_attention = self._axial_pool(rf_branch_pooled, rf_cell_mask, query)

        svd_shape = (*features.shape[:3], 3, 7)
        svd_parts = (
            self.svd_evidence(
                torch.cat((svd, svd_feature_mask.to(features.dtype)), dim=-1)
            ),
            radar.reshape(1, 1, 1, 3, 1, -1).expand(*svd_shape, -1),
            ratio.reshape(1, 1, 1, 1, 7, -1).expand(*svd_shape, -1),
            modality[1].reshape(1, 1, 1, 1, 1, -1).expand(*svd_shape, -1),
            coordinate[..., None, None, :].expand(*svd_shape, -1),
            svd_mask.unsqueeze(-1).to(features.dtype),
        )
        svd_joint = self.svd_joint(torch.cat(svd_parts, dim=-1))
        svd_joint = svd_joint * svd_mask.unsqueeze(-1).to(features.dtype)
        svd_summary, svd_attention = self._axial_pool(svd_joint, svd_mask, query)

        core_encoded = self.core(
            torch.cat((core, core_mask.to(features.dtype)), dim=-1)
        ) * node_mask.unsqueeze(-1).to(features.dtype)
        encoded = self.output(torch.cat((core_encoded, rf_summary, svd_summary), dim=-1))
        encoded = encoded * node_mask.unsqueeze(-1).to(features.dtype)
        return encoded, {
            "structural_availability": availability,
            "core_feature_mask": core_mask,
            "rf_feature_mask": rf_feature_mask,
            "svd_feature_mask": svd_feature_mask,
            "rf_branch_mask": rf_mask,
            "rf_cell_mask": rf_cell_mask,
            "svd_cell_mask": svd_mask,
            "rf_axial_attention": rf_attention,
            "svd_axial_attention": svd_attention,
        }


class _DirectedGraphPLIFBlock(nn.Module):
    def __init__(self, channels: int, *, beta: float, dropout: float) -> None:
        super().__init__()
        self.relations = nn.ModuleList(
            nn.Linear(channels, channels, bias=False) for _ in DIRECTED_RELATIONS
        )
        self.current = nn.Linear((len(DIRECTED_RELATIONS) + 1) * channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.cell = EpisodeSpikingCell(channels, cell_type="plif", beta=beta)
        self.readout = nn.Linear(2 * channels, channels)
        self.output_norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        nodes: Tensor,
        edge_weights: Tensor,
        node_mask: Tensor,
        *,
        simulation_steps: int,
    ) -> tuple[Tensor, Tensor]:
        messages: list[Tensor] = []
        for index, projection in enumerate(self.relations):
            adjacency = edge_weights[..., index].to(nodes.dtype)
            degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
            messages.append(projection(torch.matmul(adjacency, nodes) / degree))
        current = self.norm(self.current(torch.cat((nodes, *messages), dim=-1)))
        current = current * node_mask.unsqueeze(-1).to(nodes.dtype)
        state = self.cell.initial_state(torch.zeros_like(current))
        spike_sum = torch.zeros_like(current)
        for _ in range(simulation_steps):
            spikes, state = self.cell.forward_step(current, state, node_mask)
            spike_sum += spikes
        state_finite = (
            torch.isfinite(state[0]).all(dim=(-2, -1))
            & torch.isfinite(state[1]).all(dim=(-2, -1))
            & torch.isfinite(spike_sum).all(dim=(-2, -1))
        )
        state_mask = state_finite[:, None, None]
        state = (
            torch.where(state_mask, state[0], torch.zeros_like(state[0])),
            torch.where(state_mask, state[1], torch.zeros_like(state[1])),
        )
        spike_sum = torch.where(
            state_mask, spike_sum, torch.zeros_like(spike_sum)
        )
        rates = spike_sum / float(simulation_steps)
        update = self.readout(torch.cat((rates, torch.tanh(state[0])), dim=-1))
        output = self.output_norm(nodes + self.dropout(update))
        return (
            output * node_mask.unsqueeze(-1).to(output.dtype),
            rates,
            state_finite,
        )


class _CandidatePool(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(channels))
        nn.init.normal_(self.query, std=channels**-0.5)

    def forward(self, nodes: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        score = (nodes * self.query).sum(dim=-1) / math.sqrt(nodes.shape[-1])
        weight = _masked_softmax(score, mask, dim=-1)
        return (nodes * weight.unsqueeze(-1)).sum(dim=-2), weight


class _CausalPLIFALIF(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        simulation_steps: int,
        beta: float,
        adaptation_decay: float,
        adaptation_strength: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.simulation_steps = int(simulation_steps)
        self.synapses = nn.ModuleList(nn.Linear(channels, channels) for _ in range(2))
        self.norms = nn.ModuleList(nn.LayerNorm(channels) for _ in range(2))
        self.cells = nn.ModuleList(
            (
                EpisodeSpikingCell(channels, cell_type="plif", beta=beta),
                EpisodeSpikingCell(
                    channels,
                    cell_type="alif",
                    beta=beta,
                    adaptation_decay=adaptation_decay,
                    adaptation_strength=adaptation_strength,
                ),
            )
        )
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(3 * channels, channels), nn.SiLU(), nn.LayerNorm(channels)
        )

    def initial_state(self, reference: Tensor) -> RiskRouterState:
        canonical = torch.zeros(
            reference.shape,
            device=reference.device,
            dtype=torch.float32,
        )
        return tuple(cell.initial_state(canonical) for cell in self.cells)  # type: ignore[return-value]

    def _validate_state(self, state: RiskRouterState, reference: Tensor) -> RiskRouterState:
        if not isinstance(state, (tuple, list)) or len(state) != 2:
            raise ValueError("state must contain PLIF and ALIF state")
        result: list[NeuronState] = []
        for layer in state:
            if not isinstance(layer, (tuple, list)) or len(layer) != 2:
                raise ValueError("each temporal state must be (membrane,adaptation)")
            membrane, adaptation = layer
            if not isinstance(membrane, Tensor) or not isinstance(adaptation, Tensor):
                raise ValueError("temporal state entries must be tensors")
            if membrane.shape != reference.shape or adaptation.shape != reference.shape:
                raise ValueError("temporal state shape drifted")
            if membrane.dtype != torch.float32 or adaptation.dtype != torch.float32:
                raise ValueError("temporal state dtype must remain canonical float32")
            converted = (
                membrane.to(device=reference.device),
                adaptation.to(device=reference.device),
            )
            if not all(torch.isfinite(value).all() for value in converted):
                raise ValueError("temporal state must be finite")
            result.append(converted)
        return tuple(result)  # type: ignore[return-value]

    def forward(
        self,
        values: Tensor,
        sequence_mask: Tensor,
        reset_mask: Tensor,
        state: RiskRouterState | None,
    ) -> tuple[Tensor, Tensor, RiskRouterState, Tensor]:
        batch, windows, channels = values.shape
        if batch < 1 or windows < 1:
            raise ValueError("causal temporal input cannot have an empty batch/time axis")
        if channels != self.channels or sequence_mask.shape != (batch, windows):
            raise ValueError("causal temporal input shape mismatch")
        reference = torch.zeros(
            (batch, channels), device=values.device, dtype=torch.float32
        )
        states = list(self.initial_state(reference) if state is None else self._validate_state(state, reference))
        tokens: list[Tensor] = []
        rates: list[Tensor] = []
        state_integrity: list[Tensor] = []
        for window in range(windows):
            states = [
                (
                    membrane
                    * (~reset_mask[:, window]).to(membrane.dtype).unsqueeze(-1),
                    adaptation
                    * (~reset_mask[:, window]).to(adaptation.dtype).unsqueeze(-1),
                )
                for membrane, adaptation in states
            ]
            analog = values[:, window]
            layer_rate = torch.zeros(
                (batch, 2), device=values.device, dtype=torch.float32
            )
            last_spikes = torch.zeros_like(analog)
            last_membrane = torch.zeros_like(analog)
            for _ in range(self.simulation_steps):
                current = analog
                for index, (synapse, norm, cell) in enumerate(
                    zip(self.synapses, self.norms, self.cells, strict=True)
                ):
                    current = norm(synapse(current))
                    spikes, states[index] = cell.forward_step(
                        current, states[index], sequence_mask[:, window]
                    )
                    layer_rate[:, index] += spikes.mean(dim=-1)
                    last_spikes = spikes
                    last_membrane = states[index][0]
                    current = self.dropout(spikes)
            window_state_finite = (
                torch.isfinite(layer_rate).all(dim=-1)
                & torch.isfinite(last_spikes).all(dim=-1)
                & torch.isfinite(last_membrane).all(dim=-1)
            )
            for membrane, adaptation in states:
                window_state_finite = (
                    window_state_finite
                    & torch.isfinite(membrane).all(dim=-1)
                    & torch.isfinite(adaptation).all(dim=-1)
                )
            finite_mask = window_state_finite.unsqueeze(-1)
            states = [
                (
                    torch.where(
                        finite_mask, membrane, torch.zeros_like(membrane)
                    ),
                    torch.where(
                        finite_mask, adaptation, torch.zeros_like(adaptation)
                    ),
                )
                for membrane, adaptation in states
            ]
            layer_rate = torch.where(
                finite_mask, layer_rate, torch.zeros_like(layer_rate)
            )
            last_spikes = torch.where(
                finite_mask, last_spikes, torch.zeros_like(last_spikes)
            )
            last_membrane = torch.where(
                finite_mask, last_membrane, torch.zeros_like(last_membrane)
            )
            token = self.readout(
                torch.cat((analog, last_spikes, torch.tanh(last_membrane)), dim=-1)
            ) * sequence_mask[:, window, None].to(values.dtype)
            window_execution_finite = (
                window_state_finite
                & torch.isfinite(token).all(dim=-1)
                & torch.isfinite(layer_rate).all(dim=-1)
            )
            execution_mask = window_execution_finite.unsqueeze(-1)
            token = torch.where(
                execution_mask, token, torch.zeros_like(token)
            )
            layer_rate = torch.where(
                execution_mask, layer_rate, torch.zeros_like(layer_rate)
            )
            states = [
                (
                    torch.where(
                        execution_mask, membrane, torch.zeros_like(membrane)
                    ),
                    torch.where(
                        execution_mask, adaptation, torch.zeros_like(adaptation)
                    ),
                )
                for membrane, adaptation in states
            ]
            tokens.append(token)
            rates.append(layer_rate / float(self.simulation_steps))
            state_integrity.append(window_execution_finite)
        return (  # type: ignore[return-value]
            torch.stack(tokens, dim=1),
            torch.stack(rates, dim=1),
            tuple(states),
            torch.stack(state_integrity, dim=1),
        )


class AxisRiskRouterSNNV8R5(nn.Module):
    """Coordinate-aware graph with soft training and isolated hard deployment."""

    MAX_CANDIDATES = _LOCKED_MAX_CANDIDATES

    def __setattr__(self, name: str, value: object) -> None:
        """Keep constructor-bound inference behavior immutable.

        The persistent behavior buffer is an attestation, not the runtime
        source of truth.  Letting callers mutate (for example)
        ``tail5_risk_weight`` after construction would otherwise change
        deployment while an old checkpoint and old behavior buffer still
        appeared compatible.  Route temperature is intentionally mutable only
        through its validated in-place setter.
        """

        if self.__dict__.get("_v8r5_runtime_contract_frozen", False) and (
            name in _FROZEN_MODEL_ATTRIBUTES
        ):
            raise AttributeError(
                f"V8R5 runtime contract attribute {name!r} is immutable"
            )
        super().__setattr__(name, value)

    def __init__(
        self,
        *,
        ordered_feature_names_semantic_sha256: str,
        structural_layout_semantic_sha256: str = FEATURE_LAYOUT_SEMANTIC_SHA256,
        hidden_channels: int = 64,
        graph_blocks: int = 2,
        simulation_steps: int = 8,
        dropout: float = 0.05,
        beta: float = 0.92,
        adaptation_decay: float = 0.97,
        adaptation_strength: float = 0.40,
        rr_min_bpm: float = 6.0,
        rr_max_bpm: float = 45.0,
        route_temperature: float = 1.0,
        tail2_risk_weight: float = 1.0,
        tail5_risk_weight: float = 3.0,
        candidate_residual_limit_bpm: float = 0.75,
        anchor_residual_limit_bpm: float = 12.0,
        near_relation_tolerance_bpm: float = 0.5,
        ratio_relation_tolerance_bpm: float = 0.75,
        edge_log_ratio_bandwidth: float = 0.08,
        factor_affinity_bandwidth_bpm: float = 0.75,
        maximum_parameters: int = 400_000,
    ) -> None:
        super().__init__()
        if ordered_feature_names_semantic_sha256 != EXPECTED_FEATURE_NAMES_SEMANTIC_SHA256:
            raise ValueError("ordered 571-feature schema digest drifted")
        if structural_layout_semantic_sha256 != FEATURE_LAYOUT_SEMANTIC_SHA256:
            raise ValueError("structural 571-feature layout digest drifted")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (hidden_channels, graph_blocks, simulation_steps)
        ):
            raise ValueError("hidden_channels, graph_blocks, and steps must be integers")
        if hidden_channels != 64 or graph_blocks != 2 or simulation_steps != 8:
            raise ValueError("V8R5 proposal locks hidden=64, graph_blocks=2, steps=8")
        dropout = _finite_real("dropout", dropout)
        beta = _finite_real("beta", beta)
        adaptation_decay = _finite_real("adaptation_decay", adaptation_decay)
        adaptation_strength = _positive_real(
            "adaptation_strength", adaptation_strength
        )
        rr_min_bpm = _positive_real("rr_min_bpm", rr_min_bpm)
        rr_max_bpm = _positive_real("rr_max_bpm", rr_max_bpm)
        route_temperature = _positive_real("route_temperature", route_temperature)
        tail2_risk_weight = _nonnegative_real(
            "tail2_risk_weight", tail2_risk_weight
        )
        tail5_risk_weight = _nonnegative_real(
            "tail5_risk_weight", tail5_risk_weight
        )
        candidate_residual_limit_bpm = _nonnegative_real(
            "candidate_residual_limit_bpm", candidate_residual_limit_bpm
        )
        anchor_residual_limit_bpm = _nonnegative_real(
            "anchor_residual_limit_bpm", anchor_residual_limit_bpm
        )
        near_relation_tolerance_bpm = _positive_real(
            "near_relation_tolerance_bpm", near_relation_tolerance_bpm
        )
        ratio_relation_tolerance_bpm = _positive_real(
            "ratio_relation_tolerance_bpm", ratio_relation_tolerance_bpm
        )
        edge_log_ratio_bandwidth = _positive_real(
            "edge_log_ratio_bandwidth", edge_log_ratio_bandwidth
        )
        factor_affinity_bandwidth_bpm = _positive_real(
            "factor_affinity_bandwidth_bpm", factor_affinity_bandwidth_bpm
        )
        if not 0.0 <= dropout < 1.0 or not 0.0 < beta < 1.0:
            raise ValueError("dropout/beta outside valid range")
        if not 0.0 < adaptation_decay < 1.0:
            raise ValueError("adaptation_decay must be in (0,1)")
        if not rr_min_bpm < rr_max_bpm:
            raise ValueError("RR bounds/route temperature are invalid")
        if (
            isinstance(maximum_parameters, bool)
            or not isinstance(maximum_parameters, int)
            or maximum_parameters < 1
        ):
            raise ValueError("maximum_parameters must be a positive integer")

        self.hidden_channels = int(hidden_channels)
        self.graph_blocks = int(graph_blocks)
        self.simulation_steps = int(simulation_steps)
        self.dropout_probability = dropout
        self.beta = beta
        self.adaptation_decay = adaptation_decay
        self.adaptation_strength = adaptation_strength
        self.rr_min_bpm = rr_min_bpm
        self.rr_max_bpm = rr_max_bpm
        self.tail2_risk_weight = tail2_risk_weight
        self.tail5_risk_weight = tail5_risk_weight
        self.candidate_residual_limit_bpm = candidate_residual_limit_bpm
        self.anchor_residual_limit_bpm = anchor_residual_limit_bpm
        self.near_relation_tolerance_bpm = near_relation_tolerance_bpm
        self.ratio_relation_tolerance_bpm = ratio_relation_tolerance_bpm
        self.edge_log_ratio_bandwidth = edge_log_ratio_bandwidth
        self.factor_affinity_bandwidth_bpm = factor_affinity_bandwidth_bpm
        self.maximum_parameters = int(maximum_parameters)
        self.ordered_feature_names_semantic_sha256 = ordered_feature_names_semantic_sha256
        self.structural_layout_semantic_sha256 = structural_layout_semantic_sha256
        checkpoint_receipt = _canonical_checkpoint_source_receipt(
            ordered_feature_names_semantic_sha256=(
                ordered_feature_names_semantic_sha256
            ),
            structural_layout_semantic_sha256=structural_layout_semantic_sha256,
        )
        self.register_buffer(
            "_checkpoint_source_receipt",
            torch.tensor(list(checkpoint_receipt), dtype=torch.uint8),
            persistent=True,
        )
        temperature_buffer = torch.tensor(route_temperature, dtype=torch.float32)
        if not torch.isfinite(temperature_buffer) or temperature_buffer.item() <= 0.0:
            raise ValueError("route_temperature is not representable as float32")
        self.register_buffer(
            "_route_temperature", temperature_buffer, persistent=True
        )
        behavior_values = (
            float(hidden_channels),
            float(graph_blocks),
            float(simulation_steps),
            dropout,
            beta,
            adaptation_decay,
            adaptation_strength,
            rr_min_bpm,
            rr_max_bpm,
            tail2_risk_weight,
            tail5_risk_weight,
            candidate_residual_limit_bpm,
            anchor_residual_limit_bpm,
            near_relation_tolerance_bpm,
            ratio_relation_tolerance_bpm,
            edge_log_ratio_bandwidth,
            factor_affinity_bandwidth_bpm,
        )
        self.register_buffer(
            "_behavior_contract",
            torch.tensor(behavior_values, dtype=torch.float64),
            persistent=True,
        )

        self.encoder = _AxisPreservingEvidenceEncoder(
            hidden_channels=hidden_channels,
            rr_min_bpm=rr_min_bpm,
            rr_max_bpm=rr_max_bpm,
        )
        self.graph = nn.ModuleList(
            _DirectedGraphPLIFBlock(hidden_channels, beta=beta, dropout=dropout)
            for _ in range(graph_blocks)
        )
        self.pool = _CandidatePool(hidden_channels)
        self.episode_projection = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels), nn.SiLU(), nn.LayerNorm(hidden_channels)
        )
        self.radar_context = nn.Linear(3, hidden_channels, bias=False)
        self.source_context = nn.Linear(5, hidden_channels, bias=False)
        self.temporal = _CausalPLIFALIF(
            hidden_channels,
            simulation_steps=simulation_steps,
            beta=beta,
            adaptation_decay=adaptation_decay,
            adaptation_strength=adaptation_strength,
            dropout=dropout,
        )
        # Keep value, routing-preference, and calibrated-risk parameters fully
        # disjoint. In particular, a routing loss must not be able to alter a
        # risk head through a shared hidden layer after the direct risk tensor is
        # detached below.
        self.candidate_value_head = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )
        self.anchor_value_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )
        self.candidate_route_head = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )
        self.anchor_route_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )
        # scale, expected absolute error, P(error>2), P(error>5 | error>2)
        self.candidate_risk_head = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 4),
        )
        self.anchor_risk_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 4),
        )
        self.factor_head = nn.Linear(hidden_channels, len(FACTOR_CLASSES))
        self.quality_head = nn.Linear(hidden_channels, 1)
        self._initialize_safe_heads()

        runtime_structure_receipt = self._canonical_runtime_structure_receipt()
        self.register_buffer(
            "_runtime_structure_receipt",
            torch.tensor(list(runtime_structure_receipt), dtype=torch.uint8),
            persistent=True,
        )

        if self.parameter_count() > int(maximum_parameters):
            raise ValueError(
                f"V8R5 parameter cap exceeded: {self.parameter_count()}>{maximum_parameters}"
            )
        self._v8r5_runtime_contract_frozen = True

    def _initialize_safe_heads(self) -> None:
        for head in (
            self.candidate_value_head,
            self.anchor_value_head,
            self.candidate_route_head,
            self.anchor_route_head,
            self.candidate_risk_head,
            self.anchor_risk_head,
        ):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        candidate_risk_bias = torch.tensor(
            [
                _inverse_softplus(1.0 - 0.25),
                _inverse_softplus(2.5 - 0.05),
                _logit(0.60),
                _logit(0.10 / 0.60),
            ],
            dtype=self.candidate_risk_head[-1].bias.dtype,
        )
        anchor_risk_bias = torch.tensor(
            [
                _inverse_softplus(1.5 - 0.25),
                _inverse_softplus(1.0 - 0.05),
                _logit(0.20),
                _logit(0.03 / 0.20),
            ],
            dtype=self.anchor_risk_head[-1].bias.dtype,
        )
        with torch.no_grad():
            self.anchor_route_head[-1].bias.fill_(4.0)
            self.candidate_risk_head[-1].bias.copy_(candidate_risk_bias)
            self.anchor_risk_head[-1].bias.copy_(anchor_risk_bias)
            nn.init.zeros_(self.factor_head.weight)
            nn.init.zeros_(self.factor_head.bias)
            nn.init.zeros_(self.quality_head.weight)
            nn.init.zeros_(self.quality_head.bias)

    def configure_training_stage(self, stage_name: str) -> tuple[str, ...]:
        """Apply the frozen V8R5 module-level stage contract."""

        if stage_name not in _TRAINING_STAGE_MODULES:
            raise ValueError(
                f"unknown V8R5 training stage {stage_name!r}; expected "
                f"{sorted(_TRAINING_STAGE_MODULES)}"
            )
        selected = _TRAINING_STAGE_MODULES[stage_name]
        if selected is None:
            for parameter in self.parameters():
                parameter.requires_grad_(True)
            return tuple(name for name, _ in self.named_children())
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for name in selected:
            module = getattr(self, name)
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        return selected

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

    def state_dict(self, *args, **kwargs):
        """Export state only when its non-tensor behavior receipts are current."""

        self._assert_checkpoint_source_receipt(self._checkpoint_source_receipt)
        self._assert_runtime_structure_receipt(self._runtime_structure_receipt)
        self._assert_runtime_behavior_contract()
        _ = self.route_temperature
        return super().state_dict(*args, **kwargs)

    @property
    def route_temperature(self) -> float:
        value = self._route_temperature.detach()
        if value.numel() != 1 or not torch.isfinite(value).all() or value.item() <= 0.0:
            raise RuntimeError("stored route temperature is invalid")
        return float(value.item())

    @torch.no_grad()
    def set_route_temperature(self, value: float) -> None:
        """Update the persisted routing temperature after strict validation."""

        validated = _positive_real("route_temperature", value)
        converted = self._route_temperature.new_tensor(validated)
        if not torch.isfinite(converted) or converted.item() <= 0.0:
            raise ValueError(
                "route_temperature is not representable in the buffer dtype"
            )
        self._route_temperature.copy_(converted)

    def _expected_checkpoint_source_receipt(self) -> Tensor:
        payload = _canonical_checkpoint_source_receipt(
            ordered_feature_names_semantic_sha256=(
                self.ordered_feature_names_semantic_sha256
            ),
            structural_layout_semantic_sha256=(
                self.structural_layout_semantic_sha256
            ),
        )
        return torch.tensor(list(payload), dtype=torch.uint8)

    def _canonical_runtime_structure_receipt(self) -> bytes:
        """Serialize non-tensor module semantics that ``state_dict`` omits."""

        modules: list[dict[str, object]] = []
        for path, module in self.named_modules():
            if type(module.training) is not bool or module.training is not self.training:
                raise RuntimeError(
                    "V8R5 child-module runtime training/eval mode diverged from the root"
                )
            module_type = type(module)
            entry: dict[str, object] = {
                "path": path,
                "type": f"{module_type.__module__}.{module_type.__qualname__}",
                "children": [
                    {"name": name, "present": child is not None}
                    for name, child in module._modules.items()
                ],
            }
            if module_type is AxisRiskRouterSNNV8R5:
                if (
                    type(module.MAX_CANDIDATES) is not int
                    or module.MAX_CANDIDATES != _LOCKED_MAX_CANDIDATES
                ):
                    raise RuntimeError("V8R5 maximum-candidate runtime drifted")
                entry.update(
                    {
                        "maximum_candidates": module.MAX_CANDIDATES,
                        "maximum_parameters": module.maximum_parameters,
                    }
                )
            elif module_type is _AxisPreservingEvidenceEncoder:
                entry.update(
                    {
                        "hidden_channels": module.hidden_channels,
                        "rr_min_bpm": module.rr_min_bpm,
                        "rr_max_bpm": module.rr_max_bpm,
                        "cell_channels": module.cell_channels,
                    }
                )
            elif module_type is _CausalPLIFALIF:
                entry.update(
                    {
                        "channels": module.channels,
                        "simulation_steps": module.simulation_steps,
                    }
                )
            elif module_type is EpisodeSpikingCell:
                spike = module.spike_function
                closure_values: list[object] = []
                for cell in getattr(spike, "__closure__", None) or ():
                    value = cell.cell_contents
                    if type(value) in (bool, int, str) or value is None:
                        closure_values.append(value)
                    elif isinstance(value, Real) and not isinstance(value, bool):
                        closure_values.append(_finite_real("spike closure", value))
                    else:
                        raise RuntimeError(
                            "V8R5 spiking surrogate closure is not canonical"
                        )
                entry.update(
                    {
                        "channels": module.channels,
                        "cell_type": module.cell_type,
                        "spike_function_module": getattr(spike, "__module__", None),
                        "spike_function_qualname": getattr(
                            spike, "__qualname__", None
                        ),
                        "spike_function_closure": closure_values,
                    }
                )
            elif module_type is nn.Linear:
                entry.update(
                    {
                        "in_features": module.in_features,
                        "out_features": module.out_features,
                        "bias": module.bias is not None,
                    }
                )
            elif module_type is nn.LayerNorm:
                entry.update(
                    {
                        "normalized_shape": list(module.normalized_shape),
                        "eps": _positive_real("LayerNorm eps", module.eps),
                        "elementwise_affine": module.elementwise_affine,
                        "bias": module.bias is not None,
                    }
                )
            elif module_type is nn.Dropout:
                probability = _finite_real("Dropout p", module.p)
                if not 0.0 <= probability < 1.0:
                    raise RuntimeError("V8R5 dropout probability drifted")
                entry.update({"p": probability, "inplace": module.inplace})
            elif module_type is nn.SiLU:
                entry["inplace"] = module.inplace
            elif module_type is nn.Embedding:
                max_norm = (
                    None
                    if module.max_norm is None
                    else _positive_real("Embedding max_norm", module.max_norm)
                )
                entry.update(
                    {
                        "num_embeddings": module.num_embeddings,
                        "embedding_dim": module.embedding_dim,
                        "padding_idx": module.padding_idx,
                        "max_norm": max_norm,
                        "norm_type": _positive_real(
                            "Embedding norm_type", module.norm_type
                        ),
                        "scale_grad_by_freq": module.scale_grad_by_freq,
                        "sparse": module.sparse,
                    }
                )
            elif module_type not in (
                _DirectedGraphPLIFBlock,
                _CandidatePool,
                nn.ModuleList,
                nn.Sequential,
            ):
                raise RuntimeError(
                    f"unsupported V8R5 runtime module type at {path!r}"
                )
            modules.append(entry)
        document = {
            "runtime_structure_receipt_schema_version": (
                _RUNTIME_STRUCTURE_RECEIPT_SCHEMA_VERSION
            ),
            "modules": modules,
        }
        try:
            return json.dumps(
                document,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "V8R5 runtime structure cannot be canonically serialized"
            ) from error

    def _assert_runtime_structure_receipt(
        self, stored: object | None = None
    ) -> None:
        expected = torch.tensor(
            list(self._canonical_runtime_structure_receipt()),
            dtype=torch.uint8,
        )
        current = self._runtime_structure_receipt.detach().to(device="cpu")
        if (
            current.dtype != torch.uint8
            or current.ndim != 1
            or not torch.equal(current, expected)
        ):
            raise RuntimeError(
                "constructed V8R5 runtime structure differs from its persistent receipt"
            )
        if stored is not None and (
            not isinstance(stored, Tensor)
            or stored.dtype != torch.uint8
            or stored.ndim != 1
            or not torch.equal(stored.detach().to(device="cpu"), expected)
        ):
            raise RuntimeError(
                "checkpoint runtime structure receipt differs from the constructed model"
            )

    def _assert_no_state_dict_load_hooks(self) -> None:
        for path, module in self.named_modules():
            if module._load_state_dict_pre_hooks or module._load_state_dict_post_hooks:
                raise RuntimeError(
                    "V8R5 strict checkpoint loading forbids state-dict load hooks; "
                    f"found one at {path or '<root>'!r}"
                )

    @staticmethod
    def _assert_finite_checkpoint_tensors(
        state_dict: Mapping[str, Tensor],
        *,
        subject: str = "checkpoint",
    ) -> None:
        """Reject every non-finite floating/complex tensor before any copy.

        ``nn.Module.load_state_dict`` can copy earlier entries before reporting a
        later malformed one.  Performing the complete finite scan first keeps a
        NaN/Inf checkpoint from partially mutating the live model.
        """

        for name, value in state_dict.items():
            if not isinstance(name, str) or not isinstance(value, Tensor):
                raise RuntimeError(
                    f"V8R5 {subject} state must map string keys to tensors"
                )
            if not (value.is_floating_point() or value.is_complex()):
                continue
            try:
                finite = bool(torch.isfinite(value.detach()).all().item())
            except (RuntimeError, TypeError) as error:
                raise RuntimeError(
                    f"{subject} tensor {name!r} cannot be finite-validated"
                ) from error
            if not finite:
                raise RuntimeError(
                    f"{subject} floating/complex tensor {name!r} is non-finite"
                )

    def _named_live_parameters_and_buffers(self) -> OrderedDict[str, Tensor]:
        """Return every live tensor state entry, including non-persistent buffers."""

        live: OrderedDict[str, Tensor] = OrderedDict()
        for name, value in self.named_parameters():
            live[f"parameter:{name}"] = value
        for name, value in self.named_buffers():
            live[f"buffer:{name}"] = value
        return live

    def _assert_all_live_parameters_and_buffers_finite(self) -> None:
        """Validate persistent and non-persistent live tensor state."""

        self._assert_finite_checkpoint_tensors(
            self._named_live_parameters_and_buffers(),
            subject="constructed model",
        )

    def _fresh_runtime_behavior_contract(self) -> Tensor:
        """Derive the behavior tuple from attributes actually used at runtime."""

        raw = tuple(
            getattr(self, attribute)
            for attribute in _BEHAVIOR_RUNTIME_ATTRIBUTES
        )
        integer_values = raw[:3]
        if any(type(value) is not int for value in integer_values) or tuple(
            integer_values
        ) != (64, 2, 8):
            raise RuntimeError("V8R5 integer runtime behavior drifted")
        try:
            dropout = _finite_real("dropout", raw[3])
            beta = _finite_real("beta", raw[4])
            adaptation_decay = _finite_real("adaptation_decay", raw[5])
            adaptation_strength = _positive_real(
                "adaptation_strength", raw[6]
            )
            rr_min_bpm = _positive_real("rr_min_bpm", raw[7])
            rr_max_bpm = _positive_real("rr_max_bpm", raw[8])
            tail2_risk_weight = _nonnegative_real(
                "tail2_risk_weight", raw[9]
            )
            tail5_risk_weight = _nonnegative_real(
                "tail5_risk_weight", raw[10]
            )
            candidate_residual_limit_bpm = _nonnegative_real(
                "candidate_residual_limit_bpm", raw[11]
            )
            anchor_residual_limit_bpm = _nonnegative_real(
                "anchor_residual_limit_bpm", raw[12]
            )
            near_relation_tolerance_bpm = _positive_real(
                "near_relation_tolerance_bpm", raw[13]
            )
            ratio_relation_tolerance_bpm = _positive_real(
                "ratio_relation_tolerance_bpm", raw[14]
            )
            edge_log_ratio_bandwidth = _positive_real(
                "edge_log_ratio_bandwidth", raw[15]
            )
            factor_affinity_bandwidth_bpm = _positive_real(
                "factor_affinity_bandwidth_bpm", raw[16]
            )
        except ValueError as error:
            raise RuntimeError("V8R5 runtime behavior is invalid") from error
        if not (
            0.0 <= dropout < 1.0
            and 0.0 < beta < 1.0
            and 0.0 < adaptation_decay < 1.0
            and rr_min_bpm < rr_max_bpm
        ):
            raise RuntimeError("V8R5 runtime behavior bounds drifted")
        # A few constructor settings are copied into child-module Python
        # attributes and therefore do not appear in ``state_dict``.  Bind those
        # actual consumers too; checking only the convenient top-level mirror
        # would let strict loading retain a mutated dropout or encoder bound.
        if (
            type(self.encoder) is not _AxisPreservingEvidenceEncoder
            or self.encoder.hidden_channels != integer_values[0]
            or self.encoder.rr_min_bpm != rr_min_bpm
            or self.encoder.rr_max_bpm != rr_max_bpm
            or type(self.graph) is not nn.ModuleList
            or len(self.graph) != integer_values[1]
            or type(self.temporal) is not _CausalPLIFALIF
            or self.temporal.channels != integer_values[0]
            or self.temporal.simulation_steps != integer_values[2]
        ):
            raise RuntimeError("V8R5 derived runtime structure drifted")
        dropout_modules: list[object] = [
            *(block.dropout for block in self.graph),
            self.temporal.dropout,
            self.candidate_value_head[2],
            self.anchor_value_head[2],
            self.candidate_route_head[2],
            self.anchor_route_head[2],
            self.candidate_risk_head[2],
            self.anchor_risk_head[2],
        ]
        if any(
            type(module) is not nn.Dropout or module.p != dropout
            for module in dropout_modules
        ):
            raise RuntimeError("V8R5 derived runtime dropout behavior drifted")
        if (
            any(
                type(block) is not _DirectedGraphPLIFBlock
                or type(block.cell) is not EpisodeSpikingCell
                or block.cell.channels != integer_values[0]
                or block.cell.cell_type != "plif"
                for block in self.graph
            )
            or type(self.temporal.cells) is not nn.ModuleList
            or len(self.temporal.cells) != 2
            or tuple(cell.cell_type for cell in self.temporal.cells)
            != ("plif", "alif")
            or any(
                type(cell) is not EpisodeSpikingCell
                or cell.channels != integer_values[0]
                for cell in self.temporal.cells
            )
        ):
            raise RuntimeError("V8R5 derived spiking runtime structure drifted")
        values = (
            *(float(value) for value in integer_values),
            dropout,
            beta,
            adaptation_decay,
            adaptation_strength,
            rr_min_bpm,
            rr_max_bpm,
            tail2_risk_weight,
            tail5_risk_weight,
            candidate_residual_limit_bpm,
            anchor_residual_limit_bpm,
            near_relation_tolerance_bpm,
            ratio_relation_tolerance_bpm,
            edge_log_ratio_bandwidth,
            factor_affinity_bandwidth_bpm,
        )
        contract = torch.tensor(values, dtype=torch.float64)
        if contract.shape != (len(_BEHAVIOR_CONTRACT_FIELDS),) or not bool(
            torch.isfinite(contract).all().item()
        ):
            raise RuntimeError("V8R5 fresh runtime behavior tuple is invalid")
        return contract

    def _assert_runtime_behavior_contract(self, stored: object | None = None) -> None:
        """Compare fresh runtime attributes, the live buffer, and checkpoint."""

        expected = self._fresh_runtime_behavior_contract()
        current = self._behavior_contract.detach().to(device="cpu")
        if (
            current.dtype != torch.float64
            or current.shape != expected.shape
            or not torch.equal(current, expected)
        ):
            raise RuntimeError(
                "constructed model runtime behavior differs from its persistent contract"
            )
        if stored is not None and (
            not isinstance(stored, Tensor)
            or stored.dtype != torch.float64
            or stored.shape != expected.shape
            or not torch.equal(stored.detach().to(device="cpu"), expected)
        ):
            raise RuntimeError(
                "checkpoint behavior contract differs from fresh constructed behavior"
            )

    def _assert_checkpoint_route_temperature(self, stored: object) -> None:
        current = self._route_temperature.detach().to(device="cpu")
        if (
            not isinstance(stored, Tensor)
            or stored.dtype != torch.float32
            or stored.shape != current.shape
            or stored.numel() != 1
            or not bool(torch.isfinite(stored.detach()).all().item())
            or stored.item() <= 0.0
            or not torch.equal(stored.detach().to(device="cpu"), current)
        ):
            raise RuntimeError(
                "checkpoint route temperature is absent, invalid, or differs from "
                "the constructed model"
            )

    def _private_strict_checkpoint_snapshot(
        self, state_dict: Mapping[str, Tensor]
    ) -> OrderedDict[str, Tensor]:
        """Validate and privately clone every state entry before model mutation."""

        if not isinstance(state_dict, Mapping):
            raise TypeError("V8R5 checkpoint state_dict must be a mapping")
        try:
            items = list(state_dict.items())
        except Exception as error:
            raise RuntimeError("V8R5 checkpoint mapping cannot be snapshotted") from error
        incoming: OrderedDict[str, Tensor] = OrderedDict()
        for name, value in items:
            if not isinstance(name, str) or name in incoming:
                raise RuntimeError(
                    "V8R5 checkpoint keys must be unique strings"
                )
            if not isinstance(value, Tensor):
                raise RuntimeError(
                    f"V8R5 checkpoint entry {name!r} is not a tensor"
                )
            incoming[name] = value

        expected = nn.Module.state_dict(self)
        self._assert_all_live_parameters_and_buffers_finite()
        expected_keys = set(expected)
        incoming_keys = set(incoming)
        missing = sorted(expected_keys - incoming_keys)
        unexpected = sorted(incoming_keys - expected_keys)
        if missing or unexpected:
            raise RuntimeError(
                "V8R5 strict checkpoint key mismatch; "
                f"Missing key(s): {missing}; Unexpected key(s): {unexpected}"
            )

        private: OrderedDict[str, Tensor] = OrderedDict()
        for name, expected_value in expected.items():
            value = incoming[name]
            if value.shape != expected_value.shape:
                raise RuntimeError(
                    f"checkpoint tensor {name!r} shape differs: "
                    f"{tuple(value.shape)} != {tuple(expected_value.shape)}"
                )
            if value.dtype != expected_value.dtype:
                raise RuntimeError(
                    f"checkpoint tensor {name!r} dtype differs: "
                    f"{value.dtype} != {expected_value.dtype}; implicit casts are forbidden"
                )
            if (
                value.layout != torch.strided
                or expected_value.layout != torch.strided
            ):
                raise RuntimeError(
                    f"checkpoint tensor {name!r} must use the exact dense strided layout"
                )
            if value.device.type == "meta" or expected_value.device.type == "meta":
                raise RuntimeError(
                    f"checkpoint tensor {name!r} cannot load from or into a meta tensor"
                )
            try:
                snapshot = value.detach().to(
                    device=expected_value.device,
                    dtype=expected_value.dtype,
                    copy=True,
                )
            except (RuntimeError, TypeError) as error:
                raise RuntimeError(
                    f"checkpoint tensor {name!r} cannot be privately materialized"
                ) from error
            if (
                snapshot.shape != expected_value.shape
                or snapshot.dtype != expected_value.dtype
                or snapshot.device != expected_value.device
                or snapshot.layout != expected_value.layout
            ):
                raise RuntimeError(
                    f"checkpoint tensor {name!r} changed during private materialization"
                )
            private[name] = snapshot
        self._assert_finite_checkpoint_tensors(private)
        metadata = getattr(expected, "_metadata", None)
        if metadata is not None:
            private._metadata = copy.deepcopy(metadata)  # type: ignore[attr-defined]
        return private

    def _assert_loaded_checkpoint_integrity(self) -> None:
        self._assert_checkpoint_source_receipt(self._checkpoint_source_receipt)
        self._assert_runtime_structure_receipt(self._runtime_structure_receipt)
        self._assert_runtime_behavior_contract()
        self._assert_all_live_parameters_and_buffers_finite()
        _ = self.route_temperature

    @torch.no_grad()
    def _restore_checkpoint_state_without_load_hooks(
        self, before: Mapping[str, Tensor]
    ) -> None:
        """Restore an exact live-state snapshot without re-entering load hooks.

        A post-load hook can raise after PyTorch has copied parameters.  Calling
        ``load_state_dict`` again for rollback would re-enter that same hook and
        can leave a second mutation behind.  Exact key/shape/dtype checks have
        already locked the state topology, so direct in-place restoration is
        the smaller transactional primitive.
        """

        live = self._named_live_parameters_and_buffers()
        if tuple(live) != tuple(before):
            raise RuntimeError("live V8R5 state topology changed during rollback")
        for name, destination in live.items():
            source = before[name]
            if (
                destination.shape != source.shape
                or destination.dtype != source.dtype
                or destination.device != source.device
                or destination.layout != torch.strided
                or source.layout != torch.strided
            ):
                raise RuntimeError(
                    f"live V8R5 tensor {name!r} changed contract during rollback"
                )
            destination.copy_(source)

    def _assert_checkpoint_source_receipt(self, stored: object) -> None:
        expected = self._expected_checkpoint_source_receipt()
        current = self._checkpoint_source_receipt.detach().to(device="cpu")
        if (
            current.dtype != torch.uint8
            or current.ndim != 1
            or not torch.equal(current, expected)
        ):
            raise RuntimeError(
                "constructed model source/layout/config/dependency receipt is invalid"
            )
        if (
            not isinstance(stored, Tensor)
            or stored.dtype != torch.uint8
            or stored.ndim != 1
            or not torch.equal(stored.detach().to(device="cpu"), expected)
        ):
            raise RuntimeError(
                "checkpoint source/layout/config/dependency receipt differs from "
                "the constructed model"
            )

    def load_state_dict(
        self,
        state_dict: Mapping[str, Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        if strict is not True:
            raise ValueError(
                "V8R5 checkpoints require strict=True; partial/default-filled loads are forbidden"
            )
        if assign is not False:
            raise ValueError(
                "V8R5 checkpoints require assign=False; parameter/buffer replacement is forbidden"
            )
        self._assert_no_state_dict_load_hooks()
        private_state = self._private_strict_checkpoint_snapshot(state_dict)
        self._assert_checkpoint_source_receipt(
            private_state.get("_checkpoint_source_receipt")
        )
        self._assert_runtime_structure_receipt(
            private_state.get("_runtime_structure_receipt")
        )
        self._assert_runtime_behavior_contract(
            private_state.get("_behavior_contract")
        )
        self._assert_checkpoint_route_temperature(
            private_state.get("_route_temperature")
        )

        # Exercise the complete recursive PyTorch loader on a private model
        # first.  Exact keys/shapes/dtypes make copying deterministic; the
        # shadow pass additionally catches module hooks or future submodule
        # loader changes without touching the live instance.
        try:
            shadow = copy.deepcopy(self)
            nn.Module.load_state_dict(
                shadow, private_state, strict=True, assign=False
            )
            shadow._assert_loaded_checkpoint_integrity()
        except Exception as error:
            raise RuntimeError(
                "V8R5 checkpoint failed private transactional preflight"
            ) from error

        before = OrderedDict(
            (name, value.detach().clone())
            for name, value in self._named_live_parameters_and_buffers().items()
        )
        try:
            self._assert_no_state_dict_load_hooks()
            result = super().load_state_dict(
                private_state, strict=True, assign=False
            )
            self._assert_loaded_checkpoint_integrity()
        except Exception as error:
            try:
                self._restore_checkpoint_state_without_load_hooks(before)
                self._assert_loaded_checkpoint_integrity()
            except Exception as rollback_error:  # pragma: no cover - catastrophic.
                raise RuntimeError(
                    "V8R5 checkpoint load failed and transactional rollback failed"
                ) from rollback_error
            raise RuntimeError(
                "V8R5 checkpoint load failed after preflight; live state was rolled back"
            ) from error
        return result

    def layout_receipt(self) -> dict[str, str | int | float | bool]:
        self._assert_checkpoint_source_receipt(self._checkpoint_source_receipt)
        self._assert_runtime_structure_receipt(self._runtime_structure_receipt)
        self._assert_runtime_behavior_contract()
        checkpoint_source_receipt = bytes(
            self._checkpoint_source_receipt.detach().cpu().tolist()
        )
        fresh_behavior = self._fresh_runtime_behavior_contract()
        behavior = {
            name: float(value)
            for name, value in zip(
                _BEHAVIOR_CONTRACT_FIELDS,
                fresh_behavior.tolist(),
                strict=True,
            )
        }
        encoded_behavior = json.dumps(
            behavior,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return {
            "receipt_schema_version": 2,
            "checkpoint_receipt_schema_version": (
                _CHECKPOINT_RECEIPT_SCHEMA_VERSION
            ),
            "checkpoint_source_receipt_sha256": hashlib.sha256(
                checkpoint_source_receipt
            ).hexdigest(),
            "runtime_structure_receipt_sha256": hashlib.sha256(
                bytes(self._runtime_structure_receipt.detach().cpu().tolist())
            ).hexdigest(),
            "runtime_structure_receipt_schema_version": (
                _RUNTIME_STRUCTURE_RECEIPT_SCHEMA_VERSION
            ),
            "model_family": "axis_risk_router_snn_v8r5_unmeasured_proposal",
            "model_source_sha256": _MODEL_SOURCE_SHA256,
            "model_source_binding_scope": _SOURCE_EXECUTION_BINDING_SCOPE,
            "binds_actual_loader_compiled_bytes": False,
            "training_authorization_terminal_blocker": (
                _SOURCE_EXECUTION_TERMINAL_BLOCKER
            ),
            "spiking_cell_source_sha256": _SPIKING_CELL_SOURCE_SHA256,
            "feature_layout_source_sha256": _FEATURE_LAYOUT_SOURCE_SHA256,
            "ancestry_policy": "successor_with_hash_bound_layout_and_spiking_cell_reuse",
            "allowed_internal_dependencies": (
                "harmonic_feature_layout_v3r1;svd_episode_models.EpisodeSpikingCell"
            ),
            "proposal_config_sha256": _PROPOSAL_CONFIG_SHA256,
            "proposal_config_absence_policy": "module_import_fails_closed",
            "total_width": TOTAL_FEATURE_WIDTH,
            "ordered_feature_names_semantic_sha256": self.ordered_feature_names_semantic_sha256,
            "structural_layout_semantic_sha256": self.structural_layout_semantic_sha256,
            "parameter_count": self.parameter_count(),
            "simulation_steps": self.simulation_steps,
            "rr_min_bpm": self.rr_min_bpm,
            "rr_max_bpm": self.rr_max_bpm,
            "route_temperature": self.route_temperature,
            "hidden_channels": self.hidden_channels,
            "graph_blocks": self.graph_blocks,
            "dropout": self.dropout_probability,
            "beta": self.beta,
            "adaptation_decay": self.adaptation_decay,
            "adaptation_strength": self.adaptation_strength,
            "tail2_risk_weight": self.tail2_risk_weight,
            "tail5_risk_weight": self.tail5_risk_weight,
            "candidate_residual_limit_bpm": self.candidate_residual_limit_bpm,
            "anchor_residual_limit_bpm": self.anchor_residual_limit_bpm,
            "near_relation_tolerance_bpm": self.near_relation_tolerance_bpm,
            "ratio_relation_tolerance_bpm": self.ratio_relation_tolerance_bpm,
            "edge_log_ratio_bandwidth": self.edge_log_ratio_bandwidth,
            "factor_affinity_bandwidth_bpm": self.factor_affinity_bandwidth_bpm,
            "maximum_parameters": self.maximum_parameters,
            "behavior_contract_sha256": hashlib.sha256(
                encoded_behavior
            ).hexdigest(),
            "continuous_edge_evidence": "directed_log_ratio_proximity",
            "hard_selection_policy": "eval_deployment_only_absent_in_training",
            "nonfinite_source_policy": _NONFINITE_SOURCE_POLICY,
            "head_parameterization": "disjoint_value_route_preference_calibrated_risk",
            "route_gradient_to_shared_representation": "stopped",
            "cache_contract_validator": "validate_v8r5_cache_contract",
            "temporal_state_dtype": "float32",
            "training_authorized": False,
            "commercial_claim_allowed": False,
        }

    def assert_safe_initialization(self) -> None:
        self._assert_checkpoint_source_receipt(self._checkpoint_source_receipt)
        self._assert_runtime_structure_receipt(self._runtime_structure_receipt)
        self._assert_runtime_behavior_contract()
        for name, parameter in self.named_parameters():
            if not torch.isfinite(parameter.detach()).all():
                raise RuntimeError(f"head/model parameter is non-finite: {name}")
        for name, head in (
            ("candidate_value", self.candidate_value_head),
            ("anchor_value", self.anchor_value_head),
            ("candidate_route", self.candidate_route_head),
            ("anchor_route", self.anchor_route_head),
            ("candidate_risk", self.candidate_risk_head),
            ("anchor_risk", self.anchor_risk_head),
        ):
            if torch.count_nonzero(head[-1].weight.detach()):
                raise RuntimeError(f"{name} output head no longer has safe zero weights")
        for name, head in (
            ("factor", self.factor_head),
            ("quality", self.quality_head),
        ):
            if torch.count_nonzero(head.weight.detach()) or torch.count_nonzero(
                head.bias.detach()
            ):
                raise RuntimeError(f"{name} head must initialize to exact zero")
        if self.anchor_route_head[-1].bias.detach()[0].item() != 4.0:
            raise RuntimeError("anchor preference must initialize to exactly 4")
        if self.candidate_route_head[-1].bias.detach()[0].item() != 0.0:
            raise RuntimeError("candidate preference must initialize to exactly 0")
        _ = self.route_temperature

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> RiskRouterState:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if dtype not in (None, torch.float32):
            raise ValueError("initial state dtype is locked to float32")
        parameter = next(self.parameters())
        reference = torch.zeros(
            (batch_size, self.hidden_channels),
            device=parameter.device if device is None else device,
            dtype=torch.float32,
        )
        return self.temporal.initial_state(reference)

    def _validate_and_sanitize(
        self,
        node_features: Tensor,
        node_feature_availability: Tensor,
        candidate_rr_bpm: Tensor,
        candidate_mask: Tensor,
        sequence_mask: Tensor,
        joint_radar_mask: Tensor,
        proposer_anchor_bpm: Tensor,
        proposer_anchor_std_bpm: Tensor,
        proposer_anchor_available: Tensor,
        classical_rr_bpm: Tensor,
        classical_rr_available: Tensor,
        reset_mask: Tensor | None,
    ) -> tuple[Tensor, ...]:
        if node_features.ndim != 4 or node_features.shape[-1] != TOTAL_FEATURE_WIDTH:
            raise ValueError("node_features must be [batch,time,K,571]")
        if (
            node_feature_availability.shape != node_features.shape
            or node_feature_availability.dtype != torch.bool
        ):
            raise ValueError(
                "node_feature_availability must be boolean [batch,time,K,571]"
            )
        batch, windows, candidates, _ = node_features.shape
        if batch < 1 or windows < 1:
            raise ValueError("node_features cannot have an empty batch/time axis")
        if not 1 <= candidates <= self.MAX_CANDIDATES:
            raise ValueError("candidate count must be in [1,12]")
        if candidate_rr_bpm.shape != (batch, windows, candidates):
            raise ValueError("candidate_rr_bpm shape drifted")
        if candidate_mask.shape != candidate_rr_bpm.shape or candidate_mask.dtype != torch.bool:
            raise ValueError("candidate_mask must be boolean [batch,time,K]")
        if sequence_mask.shape != (batch, windows) or sequence_mask.dtype != torch.bool:
            raise ValueError("sequence_mask must be boolean [batch,time]")
        if joint_radar_mask.shape != (batch, windows, 3) or joint_radar_mask.dtype != torch.bool:
            raise ValueError("joint_radar_mask must be boolean [batch,time,3]")
        if proposer_anchor_available.shape != (batch, windows) or proposer_anchor_available.dtype != torch.bool:
            raise ValueError("proposer_anchor_available must be boolean [batch,time]")
        if classical_rr_available.shape != (batch, windows) or classical_rr_available.dtype != torch.bool:
            raise ValueError("classical_rr_available must be boolean [batch,time]")
        for name, value in (
            ("proposer_anchor_bpm", proposer_anchor_bpm),
            ("proposer_anchor_std_bpm", proposer_anchor_std_bpm),
            ("classical_rr_bpm", classical_rr_bpm),
        ):
            if value.shape != (batch, windows):
                raise ValueError(f"{name} shape drifted")
        numeric_inputs = (
            node_features,
            candidate_rr_bpm,
            proposer_anchor_bpm,
            proposer_anchor_std_bpm,
            classical_rr_bpm,
        )
        if not all(
            torch.is_floating_point(value) and value.dtype == torch.float32
            for value in numeric_inputs
        ):
            raise ValueError("model numeric inputs must use canonical float32")
        input_tensors = (
            *numeric_inputs,
            node_feature_availability,
            candidate_mask,
            sequence_mask,
            joint_radar_mask,
            proposer_anchor_available,
            classical_rr_available,
        )
        if any(value.device != node_features.device for value in input_tensors):
            raise ValueError("all model inputs must share one device")
        if reset_mask is None:
            reset_mask = torch.zeros_like(sequence_mask)
        elif reset_mask.shape != sequence_mask.shape or reset_mask.dtype != torch.bool:
            raise ValueError("reset_mask must be boolean [batch,time]")
        elif reset_mask.device != node_features.device:
            raise ValueError("reset_mask must share the model input device")
        if (reset_mask & ~sequence_mask).any():
            raise ValueError("reset can only occur on a real sequence position")

        features = node_features
        device, dtype = features.device, features.dtype
        candidate_rr_bpm = candidate_rr_bpm.to(device=device, dtype=dtype)
        candidate_mask = candidate_mask.to(device=device) & sequence_mask.to(device=device).unsqueeze(-1)
        sequence_mask = sequence_mask.to(device=device)
        availability = node_feature_availability.to(device=device) & sequence_mask[
            ..., None, None
        ]
        joint_radar_mask = joint_radar_mask.to(device=device) & sequence_mask.unsqueeze(-1)
        reset_mask = reset_mask.to(device=device)
        proposer_anchor_bpm = proposer_anchor_bpm.to(device=device, dtype=dtype)
        proposer_anchor_std_bpm = proposer_anchor_std_bpm.to(device=device, dtype=dtype)
        proposer_anchor_available = proposer_anchor_available.to(device=device) & sequence_mask
        classical_rr_bpm = classical_rr_bpm.to(device=device, dtype=dtype)
        classical_rr_available = classical_rr_available.to(device=device) & sequence_mask

        valid_candidate = (
            torch.isfinite(candidate_rr_bpm)
            & (candidate_rr_bpm >= self.rr_min_bpm)
            & (candidate_rr_bpm <= self.rr_max_bpm)
        )
        declared_candidate_mask = candidate_mask
        midpoint = 0.5 * (self.rr_min_bpm + self.rr_max_bpm)
        ceiling_rr_bpm = torch.where(
            valid_candidate,
            candidate_rr_bpm,
            torch.full_like(candidate_rr_bpm, midpoint),
        )
        canonical_ceiling = build_structural_availability_mask(
            ceiling_rr_bpm,
            declared_candidate_mask,
            joint_radar_mask,
            rr_min_bpm=self.rr_min_bpm,
            rr_max_bpm=self.rr_max_bpm,
        )
        if (availability & ~canonical_ceiling).any():
            raise ValueError(
                "explicit node availability exceeds the canonical structural ceiling"
            )
        if not torch.equal(availability[..., 0], declared_candidate_mask):
            raise ValueError(
                "candidate_bpm availability must exactly match candidate_mask"
            )
        finite_available_features = torch.isfinite(
            torch.where(availability, features, torch.zeros_like(features))
        ).all(dim=-1)
        candidate_integrity_valid = valid_candidate & finite_available_features
        source_integrity_failed = (
            declared_candidate_mask & ~candidate_integrity_valid
        ).any(dim=-1)
        node_mask = declared_candidate_mask & candidate_integrity_valid
        availability = availability & node_mask.unsqueeze(-1)
        candidate_rr_bpm = torch.where(
            node_mask, candidate_rr_bpm, torch.zeros_like(candidate_rr_bpm)
        )
        features = torch.where(availability, features, torch.zeros_like(features))
        if not torch.isfinite(features).all() or torch.count_nonzero(features.masked_select(~availability)):
            raise RuntimeError("structural sanitization failed")

        anchor_valid = (
            torch.isfinite(proposer_anchor_bpm)
            & torch.isfinite(proposer_anchor_std_bpm)
            & (proposer_anchor_bpm >= self.rr_min_bpm)
            & (proposer_anchor_bpm <= self.rr_max_bpm)
            & (proposer_anchor_std_bpm > 0.0)
        )
        declared_anchor_available = proposer_anchor_available
        source_integrity_failed = source_integrity_failed | (
            declared_anchor_available & ~anchor_valid
        )
        proposer_anchor_available = declared_anchor_available & anchor_valid
        classical_valid = (
            torch.isfinite(classical_rr_bpm)
            & (classical_rr_bpm >= self.rr_min_bpm)
            & (classical_rr_bpm <= self.rr_max_bpm)
        )
        declared_classical_available = classical_rr_available
        source_integrity_failed = source_integrity_failed | (
            declared_classical_available & ~classical_valid
        )
        classical_available = declared_classical_available & classical_valid
        proposer_anchor_bpm = torch.where(
            proposer_anchor_available, proposer_anchor_bpm, torch.zeros_like(proposer_anchor_bpm)
        )
        proposer_anchor_std_bpm = torch.where(
            proposer_anchor_available, proposer_anchor_std_bpm, torch.ones_like(proposer_anchor_std_bpm)
        )
        classical_rr_bpm = torch.where(
            classical_available, classical_rr_bpm, torch.zeros_like(classical_rr_bpm)
        )
        return (
            features,
            candidate_rr_bpm,
            node_mask,
            sequence_mask,
            joint_radar_mask,
            proposer_anchor_bpm,
            proposer_anchor_std_bpm,
            proposer_anchor_available,
            classical_rr_bpm,
            classical_available,
            reset_mask,
            availability,
            source_integrity_failed,
        )

    def forward(
        self,
        node_features: Tensor,
        candidate_rr_bpm: Tensor,
        candidate_mask: Tensor,
        sequence_mask: Tensor,
        *,
        node_feature_availability: Tensor,
        joint_radar_mask: Tensor,
        proposer_anchor_bpm: Tensor,
        proposer_anchor_std_bpm: Tensor,
        proposer_anchor_available: Tensor,
        classical_rr_bpm: Tensor,
        classical_rr_available: Tensor,
        state: RiskRouterState | None = None,
        reset_mask: Tensor | None = None,
        deployment_mode: bool | None = None,
    ) -> dict[str, Tensor | RiskRouterState | Mapping[str, Tensor]]:
        self._assert_checkpoint_source_receipt(self._checkpoint_source_receipt)
        self._assert_runtime_structure_receipt(self._runtime_structure_receipt)
        self._assert_runtime_behavior_contract()
        if deployment_mode is None:
            deployment_mode = not self.training
        if type(deployment_mode) is not bool:
            raise ValueError("deployment_mode must be boolean or None")
        if self.training and deployment_mode:
            raise RuntimeError(
                "hard deployment routing is forbidden while the model is training"
            )
        (
            node_features,
            candidate_rr_bpm,
            candidate_mask,
            sequence_mask,
            joint_radar_mask,
            proposer_anchor_bpm,
            proposer_anchor_std_bpm,
            anchor_available,
            classical_rr_bpm,
            classical_available,
            reset_mask,
            availability,
            source_integrity_failed,
        ) = self._validate_and_sanitize(
            node_features,
            node_feature_availability,
            candidate_rr_bpm,
            candidate_mask,
            sequence_mask,
            joint_radar_mask,
            proposer_anchor_bpm,
            proposer_anchor_std_bpm,
            proposer_anchor_available,
            classical_rr_bpm,
            classical_rr_available,
            reset_mask,
        )
        batch, windows, candidates, _ = node_features.shape
        dtype = node_features.dtype
        nodes, layout_diagnostics = self.encoder(
            node_features, availability, candidate_rr_bpm
        )
        relations = build_directed_harmonic_relations(
            candidate_rr_bpm,
            candidate_mask,
            near_tolerance_bpm=self.near_relation_tolerance_bpm,
            ratio_tolerance_bpm=self.ratio_relation_tolerance_bpm,
        )
        edge_weights = build_directed_harmonic_edge_weights(
            candidate_rr_bpm,
            candidate_mask,
            log_ratio_bandwidth=self.edge_log_ratio_bandwidth,
            near_tolerance_bpm=self.near_relation_tolerance_bpm,
            ratio_tolerance_bpm=self.ratio_relation_tolerance_bpm,
        )
        flat_nodes = nodes.reshape(batch * windows, candidates, self.hidden_channels)
        flat_edge_weights = edge_weights.reshape(
            batch * windows, candidates, candidates, -1
        )
        flat_mask = candidate_mask.reshape(batch * windows, candidates)
        graph_rates: list[Tensor] = []
        graph_state_integrity = torch.ones(
            (batch * windows,), device=nodes.device, dtype=torch.bool
        )
        for block in self.graph:
            flat_nodes, rates, block_state_finite = block(
                flat_nodes,
                flat_edge_weights,
                flat_mask,
                simulation_steps=self.simulation_steps,
            )
            block_execution_finite = (
                block_state_finite
                & torch.isfinite(flat_nodes).all(dim=(-2, -1))
                & torch.isfinite(rates).all(dim=(-2, -1))
            )
            graph_state_integrity = (
                graph_state_integrity & block_execution_finite
            )
            block_mask = block_execution_finite[:, None, None]
            flat_nodes = torch.where(
                block_mask, flat_nodes, torch.zeros_like(flat_nodes)
            )
            rates = torch.where(
                block_mask, rates, torch.zeros_like(rates)
            )
            denominator = (flat_mask.to(dtype).sum(-1) * self.hidden_channels).clamp_min(1.0)
            graph_rates.append(rates.sum(dim=(-2, -1)) / denominator)
        nodes = flat_nodes.reshape(batch, windows, candidates, self.hidden_channels)
        graph_state_integrity = graph_state_integrity.reshape(batch, windows)

        attention_pool, candidate_attention = self.pool(nodes, candidate_mask)
        count = candidate_mask.to(dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean_pool = nodes.sum(dim=2) / count
        episode = self.episode_projection(torch.cat((attention_pool, mean_pool), dim=-1))
        midpoint = 0.5 * (self.rr_min_bpm + self.rr_max_bpm)
        half_range = 0.5 * (self.rr_max_bpm - self.rr_min_bpm)
        source_values = torch.stack(
            (
                (proposer_anchor_bpm - midpoint) / half_range,
                torch.log1p(proposer_anchor_std_bpm).clamp(max=4.0) / 4.0,
                anchor_available.to(dtype),
                (classical_rr_bpm - midpoint) / half_range,
                classical_available.to(dtype),
            ),
            dim=-1,
        )
        source_values[..., :2] *= anchor_available.unsqueeze(-1).to(dtype)
        source_values[..., 3] *= classical_available.to(dtype)
        episode = episode + self.radar_context(joint_radar_mask.to(dtype))
        episode = episode + self.source_context(source_values)
        episode *= sequence_mask.unsqueeze(-1).to(dtype)
        temporal, temporal_rates, final_state, temporal_state_integrity = self.temporal(
            episode, sequence_mask, reset_mask, state
        )
        execution_state_integrity = (
            graph_state_integrity & temporal_state_integrity
        )
        source_integrity_failed = source_integrity_failed | (
            sequence_mask & ~execution_state_integrity
        )

        candidate_context = torch.cat(
            (nodes, temporal.unsqueeze(2).expand(-1, -1, candidates, -1)), dim=-1
        )
        candidate_value_raw = self.candidate_value_head(candidate_context).squeeze(-1)
        anchor_value_raw = self.anchor_value_head(temporal).squeeze(-1)
        # Routing-only objectives are confined to route/factor heads.  Detaching
        # their shared evidence input prevents an optimizer step on a routing
        # loss from indirectly changing calibrated-risk representations.
        candidate_route_preference = self.candidate_route_head(
            candidate_context.detach()
        ).squeeze(-1)
        anchor_route_preference = self.anchor_route_head(temporal.detach()).squeeze(-1)
        candidate_risk_raw = self.candidate_risk_head(candidate_context)
        anchor_risk_raw = self.anchor_risk_head(temporal)
        candidate_preactivation_finite = (
            execution_state_integrity.unsqueeze(-1)
            & torch.isfinite(candidate_context).all(dim=-1)
            & torch.isfinite(candidate_value_raw)
            & torch.isfinite(candidate_route_preference)
            & torch.isfinite(candidate_risk_raw).all(dim=-1)
        )
        anchor_preactivation_finite = (
            execution_state_integrity
            & torch.isfinite(temporal).all(dim=-1)
            & torch.isfinite(anchor_value_raw)
            & torch.isfinite(anchor_route_preference)
            & torch.isfinite(anchor_risk_raw).all(dim=-1)
        )
        source_integrity_failed = source_integrity_failed | (
            candidate_mask & ~candidate_preactivation_finite
        ).any(dim=-1)
        source_integrity_failed = source_integrity_failed | (
            anchor_available & ~anchor_preactivation_finite
        )
        candidate_value_raw = torch.where(
            candidate_preactivation_finite,
            candidate_value_raw,
            torch.zeros_like(candidate_value_raw),
        )
        candidate_route_preference = torch.where(
            candidate_preactivation_finite,
            candidate_route_preference,
            torch.zeros_like(candidate_route_preference),
        )
        candidate_risk_raw = torch.where(
            candidate_preactivation_finite.unsqueeze(-1),
            candidate_risk_raw,
            torch.zeros_like(candidate_risk_raw),
        )
        anchor_value_raw = torch.where(
            anchor_preactivation_finite,
            anchor_value_raw,
            torch.zeros_like(anchor_value_raw),
        )
        anchor_route_preference = torch.where(
            anchor_preactivation_finite,
            anchor_route_preference,
            torch.zeros_like(anchor_route_preference),
        )
        anchor_risk_raw = torch.where(
            anchor_preactivation_finite.unsqueeze(-1),
            anchor_risk_raw,
            torch.zeros_like(anchor_risk_raw),
        )
        candidate_residual = self.candidate_residual_limit_bpm * torch.tanh(
            candidate_value_raw
        )
        anchor_residual = self.anchor_residual_limit_bpm * torch.tanh(
            anchor_value_raw
        )
        candidate_mean = (candidate_rr_bpm + candidate_residual).clamp(
            self.rr_min_bpm, self.rr_max_bpm
        )
        corrected_anchor = (proposer_anchor_bpm + anchor_residual).clamp(
            self.rr_min_bpm, self.rr_max_bpm
        )
        candidate_scale = (0.25 + F.softplus(candidate_risk_raw[..., 0])).clamp(max=12.0)
        anchor_scale = (0.25 + F.softplus(anchor_risk_raw[..., 0])).clamp(max=12.0)
        candidate_expected_abs = 0.05 + F.softplus(candidate_risk_raw[..., 1])
        anchor_expected_abs = 0.05 + F.softplus(anchor_risk_raw[..., 1])

        candidate_mean = torch.where(
            candidate_mask, candidate_mean, torch.zeros_like(candidate_mean)
        )
        candidate_residual = torch.where(
            candidate_mask,
            candidate_residual,
            torch.zeros_like(candidate_residual),
        )
        candidate_scale = torch.where(
            candidate_mask, candidate_scale, torch.zeros_like(candidate_scale)
        )
        candidate_expected_abs = torch.where(
            candidate_mask,
            candidate_expected_abs,
            torch.zeros_like(candidate_expected_abs),
        )
        corrected_anchor = torch.where(
            anchor_available, corrected_anchor, torch.zeros_like(corrected_anchor)
        )
        anchor_residual = torch.where(
            anchor_available, anchor_residual, torch.zeros_like(anchor_residual)
        )
        anchor_scale = torch.where(
            anchor_available, anchor_scale, torch.zeros_like(anchor_scale)
        )
        anchor_expected_abs = torch.where(
            anchor_available,
            anchor_expected_abs,
            torch.zeros_like(anchor_expected_abs),
        )

        sequence_float = sequence_mask.unsqueeze(-1).to(dtype)
        raw_factor_logits = self.factor_head(temporal.detach())
        factor_finite = torch.isfinite(raw_factor_logits).all(dim=-1)
        source_integrity_failed = source_integrity_failed | (
            sequence_mask & ~factor_finite
        )
        factor_active = sequence_mask & factor_finite
        factor_logits = torch.where(
            factor_active.unsqueeze(-1),
            raw_factor_logits,
            torch.zeros_like(raw_factor_logits),
        )
        factor_probability = factor_logits.softmax(dim=-1) * factor_active.unsqueeze(
            -1
        ).to(dtype)
        factor_centers = classical_rr_bpm.unsqueeze(-1) * candidate_rr_bpm.new_tensor(FACTOR_CLASSES)
        affinity = torch.exp(
            -(candidate_rr_bpm.unsqueeze(-1) - factor_centers.unsqueeze(-2)).abs()
            / self.factor_affinity_bandwidth_bpm
        )
        affinity_mask = candidate_mask.unsqueeze(-1) & classical_available[..., None, None]
        affinity = torch.where(affinity_mask, affinity, torch.zeros_like(affinity))
        factor_prior = 0.25 * (affinity * factor_probability.unsqueeze(-2)).sum(dim=-1)

        candidate_tail2 = candidate_risk_raw[..., 2]
        candidate_tail5_conditional = candidate_risk_raw[..., 3]
        anchor_tail2 = anchor_risk_raw[..., 2]
        anchor_tail5_conditional = anchor_risk_raw[..., 3]
        (
            candidate_tail2_probability,
            candidate_tail5_probability,
            candidate_tail5,
        ) = _ordered_tail_outputs(candidate_tail2, candidate_tail5_conditional)
        (
            anchor_tail2_probability,
            anchor_tail5_probability,
            anchor_tail5,
        ) = _ordered_tail_outputs(anchor_tail2, anchor_tail5_conditional)
        candidate_risk = (
            candidate_expected_abs
            + self.tail2_risk_weight * candidate_tail2_probability
            + self.tail5_risk_weight * candidate_tail5_probability
        )
        anchor_risk = (
            anchor_expected_abs
            + self.tail2_risk_weight * anchor_tail2_probability
            + self.tail5_risk_weight * anchor_tail5_probability
        )
        route_temperature = self._route_temperature.to(
            device=candidate_risk.device, dtype=candidate_risk.dtype
        )
        if (
            route_temperature.numel() != 1
            or not torch.isfinite(route_temperature).all()
            or route_temperature.item() <= 0.0
        ):
            raise RuntimeError("stored route temperature is invalid")
        # Calibration heads learn only from their proper scoring rules below.
        # Stop routing gradients at calibrated risk so the model cannot lower
        # its routing loss by merely understating uncertainty/tail risk.
        candidate_logits = (
            candidate_route_preference
            + factor_prior
            - candidate_risk.detach() / route_temperature
        )
        anchor_logit = (
            anchor_route_preference - anchor_risk.detach() / route_temperature
        )

        # A structurally declared source is deployable only when every value
        # consumed by its expert/risk/route path is finite.  This check is made
        # before softmax or argmax; masking a NaN afterwards is too late because
        # it can already poison normalization and selection.
        candidate_finite = (
            candidate_mask
            & factor_finite.unsqueeze(-1)
            & candidate_preactivation_finite
        )
        for value in (
            candidate_mean,
            candidate_residual,
            candidate_scale,
            candidate_expected_abs,
            candidate_tail2,
            candidate_tail5_conditional,
            candidate_tail2_probability,
            candidate_tail5_probability,
            candidate_tail5,
            candidate_risk,
            candidate_route_preference,
            factor_prior,
            candidate_logits,
        ):
            candidate_finite = candidate_finite & torch.isfinite(value)
        anchor_finite = anchor_available & anchor_preactivation_finite
        for value in (
            corrected_anchor,
            anchor_residual,
            anchor_scale,
            anchor_expected_abs,
            anchor_tail2,
            anchor_tail5_conditional,
            anchor_tail2_probability,
            anchor_tail5_probability,
            anchor_tail5,
            anchor_risk,
            anchor_route_preference,
            anchor_logit,
        ):
            anchor_finite = anchor_finite & torch.isfinite(value)
        source_integrity_failed = source_integrity_failed | (
            candidate_mask & ~candidate_finite
        ).any(dim=-1)
        source_integrity_failed = source_integrity_failed | (
            anchor_available & ~anchor_finite
        )
        candidate_mask = candidate_finite
        anchor_available = anchor_finite

        def candidate_or_zero(value: Tensor) -> Tensor:
            return torch.where(candidate_mask, value, torch.zeros_like(value))

        def anchor_or_zero(value: Tensor) -> Tensor:
            return torch.where(anchor_available, value, torch.zeros_like(value))

        candidate_mean = candidate_or_zero(candidate_mean)
        candidate_residual = candidate_or_zero(candidate_residual)
        candidate_scale = candidate_or_zero(candidate_scale)
        candidate_expected_abs = candidate_or_zero(candidate_expected_abs)
        candidate_tail2 = candidate_or_zero(candidate_tail2)
        candidate_tail5_conditional = candidate_or_zero(
            candidate_tail5_conditional
        )
        candidate_tail2_probability = candidate_or_zero(
            candidate_tail2_probability
        )
        candidate_tail5_probability = candidate_or_zero(
            candidate_tail5_probability
        )
        candidate_tail5 = candidate_or_zero(candidate_tail5)
        corrected_anchor = anchor_or_zero(corrected_anchor)
        anchor_residual = anchor_or_zero(anchor_residual)
        anchor_scale = anchor_or_zero(anchor_scale)
        anchor_expected_abs = anchor_or_zero(anchor_expected_abs)
        anchor_tail2 = anchor_or_zero(anchor_tail2)
        anchor_tail5_conditional = anchor_or_zero(anchor_tail5_conditional)
        anchor_tail2_probability = anchor_or_zero(anchor_tail2_probability)
        anchor_tail5_probability = anchor_or_zero(anchor_tail5_probability)
        anchor_tail5 = anchor_or_zero(anchor_tail5)
        candidate_logits = torch.where(
            candidate_mask,
            candidate_logits,
            torch.full_like(candidate_logits, -1.0e4),
        )
        anchor_logit = torch.where(
            anchor_available,
            anchor_logit,
            torch.full_like(anchor_logit, -1.0e4),
        )

        expert_mask = torch.cat((anchor_available.unsqueeze(-1), candidate_mask), dim=-1)
        expert_logits = torch.cat((anchor_logit.unsqueeze(-1), candidate_logits), dim=-1)
        expert_probability = _masked_softmax(expert_logits, expert_mask, dim=-1)
        expert_means = torch.cat((corrected_anchor.unsqueeze(-1), candidate_mean), dim=-1)
        expert_scales = torch.cat((anchor_scale.unsqueeze(-1), candidate_scale), dim=-1)
        expert_expected_abs = torch.cat(
            (anchor_expected_abs.unsqueeze(-1), candidate_expected_abs), dim=-1
        )
        expert_tail2_logits = torch.cat((anchor_tail2.unsqueeze(-1), candidate_tail2), dim=-1)
        expert_tail5_logits = torch.cat((anchor_tail5.unsqueeze(-1), candidate_tail5), dim=-1)
        expert_tail5_conditional_logits = torch.cat(
            (
                anchor_tail5_conditional.unsqueeze(-1),
                candidate_tail5_conditional,
            ),
            dim=-1,
        )
        expert_tail2_logits = torch.where(
            expert_mask, expert_tail2_logits, torch.zeros_like(expert_tail2_logits)
        )
        expert_tail5_logits = torch.where(
            expert_mask, expert_tail5_logits, torch.zeros_like(expert_tail5_logits)
        )
        expert_tail5_conditional_logits = torch.where(
            expert_mask,
            expert_tail5_conditional_logits.float(),
            torch.zeros_like(expert_tail5_conditional_logits, dtype=torch.float32),
        )
        expert_tail2_probability = torch.cat(
            (
                anchor_tail2_probability.unsqueeze(-1),
                candidate_tail2_probability,
            ),
            dim=-1,
        )
        expert_tail2_probability = torch.where(
            expert_mask,
            expert_tail2_probability,
            torch.zeros_like(expert_tail2_probability),
        )
        expert_tail5_probability = torch.cat(
            (
                anchor_tail5_probability.unsqueeze(-1),
                candidate_tail5_probability,
            ),
            dim=-1,
        )
        expert_tail5_probability = torch.where(
            expert_mask,
            expert_tail5_probability,
            torch.zeros_like(expert_tail5_probability),
        )

        # This is the differentiable training estimate.  The deployment path
        # below is deliberately hard and is never consumed by the loss API.
        training_soft_rr = (expert_probability * expert_means).sum(dim=-1)
        training_soft_rr = training_soft_rr * sequence_mask.to(
            training_soft_rr.dtype
        )

        any_expert = expert_mask.any(dim=-1) & sequence_mask
        hard_outputs: dict[str, Tensor] = {}
        if deployment_mode:
            selected = expert_logits.argmax(dim=-1)
            gather = selected.clamp_min(0).unsqueeze(-1)
            selected_rr = expert_means.gather(-1, gather).squeeze(-1)
            selected_scale = expert_scales.gather(-1, gather).squeeze(-1)
            selected_probability = expert_probability.gather(-1, gather).squeeze(-1)
            hard_expert_finite = (
                torch.isfinite(selected_rr)
                & torch.isfinite(selected_scale)
                & torch.isfinite(selected_probability)
            )
            hard_expert_available = any_expert & hard_expert_finite
            source_integrity_failed = source_integrity_failed | (
                any_expert & ~hard_expert_finite
            )
            selected = torch.where(
                hard_expert_available, selected, torch.full_like(selected, -1)
            )
            classical_fallback = (
                ~hard_expert_available & classical_available & sequence_mask
            )
            source_available = hard_expert_available | classical_fallback
            source_rr = torch.where(
                hard_expert_available,
                selected_rr,
                torch.where(
                    classical_fallback,
                    classical_rr_bpm,
                    torch.zeros_like(classical_rr_bpm),
                ),
            )
            source_scale = torch.where(
                hard_expert_available,
                selected_scale,
                torch.where(
                    classical_fallback,
                    torch.full_like(classical_rr_bpm, 12.0),
                    torch.zeros_like(classical_rr_bpm),
                ),
            )
            source_code = torch.where(
                hard_expert_available,
                selected,
                torch.where(
                    classical_fallback,
                    torch.full_like(selected, -2),
                    torch.full_like(selected, -1),
                ),
            )
            selected_probability = torch.where(
                hard_expert_available,
                selected_probability,
                torch.zeros_like(selected_probability),
            )
            hard_outputs = {
                "selected_expert_index": selected,
                "deployment_hard_selected_expert_index": selected,
                "selected_source_code": source_code,
                "selected_probability": selected_probability,
                "source_rr_bpm": source_rr,
                "source_scale_bpm": source_scale,
                "source_available": source_available,
                "deployment_hard_rr_bpm": source_rr,
                "deployment_hard_scale_bpm": source_scale,
                "deployment_hard_available": source_available,
            }

        graph_spike_sequence = torch.stack(
            [value.reshape(batch, windows) for value in graph_rates], dim=-1
        )
        spike_sequence = torch.cat((graph_spike_sequence, temporal_rates), dim=-1)
        spike_denominator = sequence_mask.to(dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
        spike_rates = (spike_sequence * sequence_mask.unsqueeze(-1).to(dtype)).sum(dim=1) / spike_denominator
        quality_available = (any_expert | classical_available) & sequence_mask
        raw_quality_logit = self.quality_head(temporal.detach()).squeeze(-1)
        quality_head_finite = torch.isfinite(raw_quality_logit)
        source_integrity_failed = source_integrity_failed | (
            quality_available & ~quality_head_finite
        )
        quality_trusted = (
            quality_available & quality_head_finite & ~source_integrity_failed
        )
        quality_logit = torch.where(
            quality_trusted,
            raw_quality_logit,
            torch.zeros_like(raw_quality_logit),
        )
        quality = torch.where(
            quality_trusted,
            quality_logit.sigmoid(),
            torch.zeros_like(quality_logit),
        )
        result: dict[str, Tensor | RiskRouterState | Mapping[str, Tensor]] = {
            "sequence_mask": sequence_mask,
            "candidate_mask": candidate_mask,
            "expert_mask": expert_mask,
            "expert_logits": expert_logits,
            "expert_probabilities": expert_probability,
            "expert_mean_bpm": expert_means,
            "expert_scale_bpm": expert_scales,
            "expert_expected_abs_error_bpm": expert_expected_abs,
            "expert_tail2_logits": expert_tail2_logits,
            "expert_tail2_probabilities": expert_tail2_probability,
            "expert_tail5_logits": expert_tail5_logits,
            "expert_tail5_conditional_logits": expert_tail5_conditional_logits,
            "expert_tail5_probabilities": expert_tail5_probability,
            "candidate_mean_bpm": candidate_mean,
            "candidate_residual_bpm": candidate_residual,
            "corrected_anchor_rr_bpm": corrected_anchor,
            "anchor_residual_bpm": anchor_residual,
            "factor_logits": factor_logits,
            "factor_probabilities": factor_probability,
            "factor_affinity": affinity,
            "training_soft_rr_bpm": training_soft_rr,
            "training_soft_available": any_expert,
            "classical_rr_bpm": classical_rr_bpm,
            "classical_rr_available": classical_available,
            "quality_logit": quality_logit,
            "quality": quality,
            "quality_available": quality_available,
            "quality_trusted": quality_trusted,
            "source_integrity_failed": source_integrity_failed,
            "node_embeddings": nodes,
            "directed_relations": relations,
            "directed_edge_weights": edge_weights,
            "candidate_attention": candidate_attention,
            "temporal_state_sequence": temporal,
            "graph_state_integrity": graph_state_integrity,
            "temporal_state_integrity": temporal_state_integrity,
            "execution_state_integrity": execution_state_integrity,
            "spike_sequence": spike_sequence,
            "spike_rates": spike_rates,
            "spike_rate": spike_rates.mean(),
            "layout_diagnostics": layout_diagnostics,
            "state": final_state,
        }
        result.update(hard_outputs)
        return result


def soft_risk_routing_loss(
    output: Mapping[str, Tensor | RiskRouterState | Mapping[str, Tensor]],
    target_rr_bpm: Tensor,
    valid_mask: Tensor,
    *,
    sample_weight: Tensor | None = None,
    equivalence_tolerance_bpm: float = 0.35,
    tail2_weight: float = 0.5,
    tail5_weight: float = 1.5,
    smooth_tail_bpm: float = 0.25,
    quality_bce_weight: float = 0.15,
    target_rr_min_bpm: float = 6.0,
    target_rr_max_bpm: float = 45.0,
    training_stage: str | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Train expert values, calibrated risks, and soft routing without argmax.

    This is the only target-consuming API in the module.  Near-duplicate
    experts within ``equivalence_tolerance_bpm`` of the best expert share the
    listwise target mass, avoiding arbitrary candidate-index supervision.  It
    requires ``training_soft_rr_bpm`` and deliberately never reads any
    ``deployment_hard_*`` or legacy hard ``source_*`` output.
    """

    equivalence_tolerance_bpm = _nonnegative_real(
        "equivalence_tolerance_bpm", equivalence_tolerance_bpm
    )
    tail2_weight = _nonnegative_real("tail2_weight", tail2_weight)
    tail5_weight = _nonnegative_real("tail5_weight", tail5_weight)
    smooth_tail_bpm = _positive_real("smooth_tail_bpm", smooth_tail_bpm)
    quality_bce_weight = _nonnegative_real(
        "quality_bce_weight", quality_bce_weight
    )
    target_rr_min_bpm = _positive_real("target_rr_min_bpm", target_rr_min_bpm)
    target_rr_max_bpm = _positive_real("target_rr_max_bpm", target_rr_max_bpm)
    if target_rr_min_bpm >= target_rr_max_bpm:
        raise ValueError("target RR bounds are invalid")
    if training_stage is None:
        active_loss_names = _ALL_LOSS_COMPONENTS
    elif training_stage in _TRAINING_STAGE_ACTIVE_LOSSES:
        active_loss_names = _TRAINING_STAGE_ACTIVE_LOSSES[training_stage]
    else:
        raise ValueError(
            f"unknown V8R5 loss training stage {training_stage!r}; expected "
            f"{sorted(_TRAINING_STAGE_ACTIVE_LOSSES)}"
        )
    if not isinstance(target_rr_bpm, Tensor) or not isinstance(valid_mask, Tensor):
        raise TypeError("target_rr_bpm and valid_mask must be tensors")
    if (
        target_rr_bpm.ndim != 2
        or target_rr_bpm.shape != valid_mask.shape
        or target_rr_bpm.numel() < 1
    ):
        raise ValueError("target and valid_mask must share non-empty [batch,time]")
    if not torch.is_floating_point(target_rr_bpm):
        raise ValueError("target_rr_bpm must be floating point")
    if valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean")
    if valid_mask.device != target_rr_bpm.device:
        raise ValueError("target_rr_bpm and valid_mask must share a device")
    finite_input_target = torch.isfinite(target_rr_bpm)
    valid_target_value = (
        finite_input_target
        & (target_rr_bpm >= target_rr_min_bpm)
        & (target_rr_bpm <= target_rr_max_bpm)
    )
    if (valid_mask & ~valid_target_value).any():
        raise ValueError("valid target rows must be finite and within locked RR bounds")

    names = (
        "expert_mask",
        "expert_logits",
        "expert_probabilities",
        "expert_mean_bpm",
        "expert_scale_bpm",
        "expert_expected_abs_error_bpm",
        "expert_tail2_logits",
        "expert_tail5_logits",
        "sequence_mask",
        "training_soft_rr_bpm",
        "classical_rr_bpm",
        "classical_rr_available",
        "quality_logit",
        "quality_available",
    )
    try:
        tensors = tuple(output[name] for name in names)
    except KeyError as error:
        raise ValueError(f"risk loss is missing model output {error.args[0]!r}") from error
    if not all(isinstance(value, Tensor) for value in tensors):
        raise TypeError("risk loss received a non-tensor model output")
    (
        expert_mask,
        expert_logits,
        expert_probability,
        expert_means,
        expert_scales,
        predicted_abs,
        tail2_logits,
        tail5_logits,
        sequence_mask,
        training_soft_rr,
        classical_rr_bpm,
        classical_rr_available,
        quality_logit,
        quality_available,
    ) = tensors
    if any(
        value.dtype != torch.bool
        for value in (
            expert_mask,
            sequence_mask,
            classical_rr_available,
            quality_available,
        )
    ):
        raise ValueError("loss availability and sequence masks must remain boolean")
    expert_shape = expert_means.shape
    if len(expert_shape) != 3 or expert_shape[-1] < 1:
        raise ValueError("expert outputs must be [batch,time,experts]")
    if any(
        value.shape != expert_shape
        for value in (
            expert_mask,
            expert_logits,
            expert_probability,
            expert_scales,
            predicted_abs,
            tail2_logits,
            tail5_logits,
        )
    ):
        raise ValueError("expert output shapes drifted")
    if expert_shape[:-1] != target_rr_bpm.shape:
        raise ValueError("expert/target time shapes drifted")
    if any(
        value.shape != target_rr_bpm.shape
        for value in (
            sequence_mask,
            training_soft_rr,
            classical_rr_bpm,
            classical_rr_available,
            quality_logit,
            quality_available,
        )
    ):
        raise ValueError("row-level output shapes drifted")
    if any(value.device != target_rr_bpm.device for value in tensors):
        raise ValueError("targets and model outputs must share one device")
    floating_outputs = (
        expert_logits,
        expert_probability,
        expert_means,
        expert_scales,
        predicted_abs,
        tail2_logits,
        tail5_logits,
        training_soft_rr,
        classical_rr_bpm,
        quality_logit,
    )
    if not all(torch.is_floating_point(value) for value in floating_outputs):
        raise ValueError("risk-loss model outputs must be floating point")
    if not all(torch.isfinite(value).all() for value in floating_outputs):
        raise ValueError("risk-loss model outputs must be finite")
    if (expert_mask & ~sequence_mask.unsqueeze(-1)).any():
        raise ValueError("expert_mask cannot activate padded sequence rows")
    if (classical_rr_available & ~sequence_mask).any():
        raise ValueError("classical RR cannot be available on padded rows")
    expected_quality_available = expert_mask.any(dim=-1) | classical_rr_available
    if not torch.equal(quality_available, expected_quality_available):
        raise ValueError("quality availability differs from deployable source availability")
    if (
        classical_rr_available
        & (
            (classical_rr_bpm < target_rr_min_bpm)
            | (classical_rr_bpm > target_rr_max_bpm)
        )
    ).any():
        raise ValueError("available classical RR is outside the locked loss bounds")
    if torch.count_nonzero(quality_logit.masked_select(~quality_available)):
        raise ValueError("unavailable quality logits must be exact zero")
    if (expert_scales.masked_select(expert_mask) <= 0.0).any():
        raise ValueError("available expert scales must be positive")
    if (predicted_abs.masked_select(expert_mask) < 0.0).any():
        raise ValueError("predicted absolute errors must be non-negative")
    if (expert_probability < 0.0).any() or (expert_probability > 1.0).any():
        raise ValueError("expert probabilities must lie in [0,1]")
    if torch.count_nonzero(expert_probability.masked_select(~expert_mask)):
        raise ValueError("masked expert probabilities must be exact zero")
    probability_sum = expert_probability.float().sum(dim=-1)
    expected_sum = expert_mask.any(dim=-1).to(probability_sum.dtype)
    if not torch.allclose(probability_sum, expected_sum, rtol=2.0e-4, atol=2.0e-4):
        raise ValueError("expert probabilities are not row-normalized")

    expert_mask = expert_mask.bool()
    expert_logits = expert_logits.float()
    expert_probability = expert_probability.float()
    expert_means = expert_means.float()
    expert_scales = expert_scales.float()
    predicted_abs = predicted_abs.float()
    tail2_logits = tail2_logits.float()
    tail5_logits = tail5_logits.float()
    sequence_mask = sequence_mask.bool()
    training_soft_rr = training_soft_rr.float()
    classical_rr_bpm = classical_rr_bpm.float()
    classical_rr_available = classical_rr_available.bool()
    quality_logit = quality_logit.float()
    quality_available = quality_available.bool()
    # Replace every unusable target before the dtype conversion and before any
    # subtraction.  A finite float64 sentinel can otherwise become float32 inf
    # and poison a nominally zero-weight row via inf*0 -> NaN.
    target_input_active = valid_mask & valid_target_value
    target = torch.where(
        target_input_active,
        target_rr_bpm,
        torch.zeros_like(target_rr_bpm),
    ).float()
    label_active = target_input_active & sequence_mask
    expert_active = label_active & expert_mask.any(dim=-1)
    quality_active = label_active & quality_available
    if sample_weight is None:
        base_weight = torch.ones_like(target)
    else:
        if not isinstance(sample_weight, Tensor):
            raise TypeError("sample_weight must be a tensor")
        if (
            sample_weight.shape != target.shape
            or sample_weight.device != target.device
            or not torch.is_floating_point(sample_weight)
            or not torch.isfinite(sample_weight).all()
            or (sample_weight < 0.0).any()
        ):
            raise ValueError(
                "sample_weight must be floating, finite, non-negative, and match target"
            )
        weight64 = sample_weight.to(dtype=torch.float64)
        maximum_weight = weight64.amax()
        base_weight = torch.where(
            maximum_weight > 0.0,
            weight64 / maximum_weight.clamp_min(torch.finfo(weight64.dtype).tiny),
            weight64,
        ).float()
    expert_weight = base_weight * expert_active.to(base_weight.dtype)
    quality_weight = base_weight * quality_active.to(base_weight.dtype)
    expert_denominator = expert_weight.sum().clamp_min(1.0e-8)
    quality_denominator = quality_weight.sum().clamp_min(1.0e-8)

    error = (expert_means - target.unsqueeze(-1)).abs()
    safe_error = torch.where(expert_mask, error, torch.zeros_like(error))
    # Smooth indicators approximate the declared >2 and >5 failure fractions;
    # absolute error already carries the magnitude term.
    soft_tail2 = torch.sigmoid((safe_error - 2.0) / smooth_tail_bpm)
    soft_tail5 = torch.sigmoid((safe_error - 5.0) / smooth_tail_bpm)
    deployment_cost = safe_error + tail2_weight * soft_tail2 + tail5_weight * soft_tail5
    expected_cost_per = (expert_probability * deployment_cost).sum(dim=-1)

    masked_error = error.masked_fill(~expert_mask, float("inf"))
    minimum_error = masked_error.amin(dim=-1, keepdim=True)
    equivalence = expert_mask & (masked_error <= minimum_error + equivalence_tolerance_bpm)
    equivalence_target = equivalence.to(expert_logits.dtype)
    equivalence_target /= equivalence_target.sum(dim=-1, keepdim=True).clamp_min(1.0)
    log_probability = F.log_softmax(expert_logits.masked_fill(~expert_mask, -1.0e4), dim=-1)
    equivalence_ce_per = -(equivalence_target * log_probability).sum(dim=-1)

    value_each = F.smooth_l1_loss(
        expert_means, target.unsqueeze(-1).expand_as(expert_means), beta=0.25, reduction="none"
    )
    value_per = (value_each * equivalence_target).sum(dim=-1)

    abs_calibration = F.smooth_l1_loss(
        predicted_abs, safe_error.detach(), beta=0.25, reduction="none"
    )
    tail2_target = (safe_error.detach() > 2.0).to(tail2_logits.dtype)
    tail5_target = (safe_error.detach() > 5.0).to(tail5_logits.dtype)
    tail2_bce = F.binary_cross_entropy_with_logits(tail2_logits, tail2_target, reduction="none")
    tail5_bce = F.binary_cross_entropy_with_logits(tail5_logits, tail5_target, reduction="none")
    safe_scales = torch.where(expert_mask, expert_scales, torch.ones_like(expert_scales))
    standardized = (expert_means - target.unsqueeze(-1)) / safe_scales
    scale_nll = 0.5 * standardized.square() + safe_scales.log()
    detached_selection = expert_logits.detach().masked_fill(
        ~expert_mask, -1.0e4
    ).argmax(dim=-1, keepdim=True)
    selected_error = safe_error.detach().gather(
        -1, detached_selection
    ).squeeze(-1)
    fallback_error = (classical_rr_bpm - target).abs().detach()
    deployment_error = torch.where(
        expert_mask.any(dim=-1), selected_error, fallback_error
    )
    quality_target = (
        (deployment_error <= 2.0) & quality_available
    ).to(quality_logit.dtype)
    quality_bce = F.binary_cross_entropy_with_logits(
        quality_logit, quality_target, reduction="none"
    )

    def row_mean(value: Tensor) -> Tensor:
        return (value * expert_weight).sum() / expert_denominator

    def quality_row_mean(value: Tensor) -> Tensor:
        return (value * quality_weight).sum() / quality_denominator

    def expert_row_mean(value: Tensor) -> Tensor:
        # Normalize each row by its available expert count first.  Otherwise a
        # row with 13 candidates receives 13x the calibration influence of a
        # row with only one deployable expert.
        count = expert_mask.to(expert_weight.dtype).sum(dim=-1).clamp_min(1.0)
        per_row = (value * expert_mask.to(value.dtype)).sum(dim=-1) / count
        return row_mean(per_row)

    components = {
        "soft_expected_deployment_cost": row_mean(expected_cost_per),
        "equivalence_set_cross_entropy": row_mean(equivalence_ce_per),
        "equivalence_value_smooth_l1": row_mean(value_per),
        "expected_abs_error_calibration": expert_row_mean(abs_calibration),
        "tail2_bce": expert_row_mean(tail2_bce),
        "tail5_bce": expert_row_mean(tail5_bce),
        "scale_nll": expert_row_mean(scale_nll),
        "quality_bce": quality_row_mean(quality_bce),
        "supervised_weight": expert_weight.sum(),
        "quality_supervised_weight": quality_weight.sum(),
    }
    component_weights = {
        "soft_expected_deployment_cost": 1.0,
        "equivalence_set_cross_entropy": 0.35,
        "equivalence_value_smooth_l1": 0.50,
        "expected_abs_error_calibration": 0.20,
        "tail2_bce": 0.15,
        "tail5_bce": 0.20,
        "scale_nll": 0.05,
        "quality_bce": quality_bce_weight,
    }
    total = sum(
        component_weights[name] * components[name] for name in active_loss_names
    )
    return total, components


__all__ = [
    "AxisRiskRouterSNNV8R5",
    "DIRECTED_RELATIONS",
    "RiskRouterState",
    "build_directed_harmonic_relations",
    "build_directed_harmonic_edge_weights",
    "build_structural_availability_mask",
    "soft_risk_routing_loss",
    "validate_v8r5_cache_contract",
]
