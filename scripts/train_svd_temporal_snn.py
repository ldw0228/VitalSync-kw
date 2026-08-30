#!/usr/bin/env python3
"""Train the window-end SVD temporal SNN with a locked grouped protocol.

The cached component sequence is ordered within a 32-second window, but its
global detrending, normalization, and SVD basis use that complete window.
Consequently the model consumes only the past 32 seconds at prediction time,
while neither the representation nor a sliced component prefix is an
end-to-end raw-stream prefix-causal feature.

For outer fold ``f``, fold ``(f + 1) % 6`` is the validation identity fold and
the remaining four folds are the only weight-fitting identities.  Signal
normalization, action margins, divisor temperatures, model weights, early
stopping, and promotion are all fit or selected without the outer-test rows.
The outer-test loader is created only after ``selection_lock.json`` is written,
and a durable evaluation marker prevents a fold from being evaluated twice.

CPU smoke example::

    python scripts/train_svd_temporal_snn.py --fold 0 --preset tiny \
        --epochs 1 --device cpu --no-amp --workers 0 \
        --batch-size 2 --eval-batch-size 2 --smoke-max-batches 1 \
        --no-verify-file-hashes --bootstrap-samples 20 \
        --output-dir /tmp/svd_temporal_smoke
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for search_path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts import train_svd_snn as shared  # noqa: E402
from snn_rr.models import apply_radar_dropout, gaussian_soft_targets  # noqa: E402
from snn_rr.svd_temporal_models import TemporalSourceSeparatedRRSNN  # noqa: E402


SCHEMA_VERSION = 1
N_FOLDS = shared.N_FOLDS
FoldSplit = shared.FoldSplit


@dataclass(slots=True)
class TemporalSessionArrays:
    session_id: str
    component_signals: np.ndarray
    attributes: np.ndarray
    metadata: pd.DataFrame
    manifest: dict[str, Any]
    component_signals_sha256: str
    radar_timing_valid_mask: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class _TemporalTrainingViewReceipt:
    schema_version: int
    base_receipt_id: int
    experiment_object_id: int
    base_experiment_object_id: int
    sessions_list_object_id: int
    provenance_object_id: int
    provenance_sha256: str
    session_object_ids: tuple[int, ...]
    source_pipeline_sha256: str
    arrays_for_position_function_id: int
    structural_mask_function_id: int


_ISSUED_TEMPORAL_TRAINING_VIEW_RECEIPTS: dict[
    int,
    tuple[
        _TemporalTrainingViewReceipt,
        "TemporalAlignedExperiment",
        shared.AlignedSVDExperiment,
    ],
] = {}


def _current_temporal_training_source_sha256() -> str:
    paths = (
        *shared._training_source_paths(),
        Path(__file__).resolve(),
        SOURCE_ROOT / "snn_rr" / "svd_temporal_models.py",
    )
    return hashlib.sha256(
        "".join(shared.sha256_file(path) for path in paths).encode("utf-8")
    ).hexdigest()


@dataclass(slots=True)
class TemporalAlignedExperiment:
    cache_root: Path
    oof_csv: Path
    oof_npz: Path
    metadata: pd.DataFrame
    sessions: list[TemporalSessionArrays]
    root_manifest: dict[str, Any]
    provenance: dict[str, Any]
    training_authority_receipt: shared._SVDTrainingAuthorityReceipt | None = None
    _base_authority_experiment: shared.AlignedSVDExperiment | None = None
    _temporal_view_receipt: _TemporalTrainingViewReceipt | None = None

    def arrays_for_position(self, position: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.metadata.iloc[int(position)]
        session = self.sessions[int(row["_session_slot"])]
        local_row = int(row["_local_row"])
        return session.component_signals[local_row], session.attributes[local_row]

    def structural_radar_mask_for_position(
        self, position: int
    ) -> np.ndarray | None:
        row = self.metadata.iloc[int(position)]
        session = self.sessions[int(row["_session_slot"])]
        if session.radar_timing_valid_mask is None:
            return None
        local_row = int(row["_local_row"])
        timing = np.asarray(
            session.radar_timing_valid_mask[local_row], dtype=np.bool_
        )
        if timing.ndim != 2 or timing.shape[0] != 3:
            raise RuntimeError(
                f"invalid structural radar timing mask for {session.session_id}"
            )
        return np.all(timing, axis=1)


@dataclass(frozen=True, slots=True)
class TemporalSignalNormalizer:
    """Per-variant robust transform fit exclusively on training identities."""

    center: np.ndarray
    scale: np.ndarray
    clip: np.ndarray
    fit_positions_sha256: str
    sampled_values_per_variant: int

    def transform(self, component_signals: np.ndarray) -> np.ndarray:
        values = np.asarray(component_signals, dtype=np.float32)
        if values.ndim != 4 or values.shape[1] != len(self.center):
            raise ValueError("component signals must have shape [radar, variant, component, time]")
        center = self.center.reshape(1, -1, 1, 1)
        scale = self.scale.reshape(1, -1, 1, 1)
        clip = self.clip.reshape(1, -1, 1, 1)
        normalized = (np.nan_to_num(values) - center) / scale
        return np.clip(normalized, -clip, clip).astype(np.float32, copy=False)

    def record(self) -> dict[str, Any]:
        return {
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "clip": self.clip.tolist(),
            "fit_positions_sha256": self.fit_positions_sha256,
            "sampled_values_per_variant": int(self.sampled_values_per_variant),
            "fit_scope": "outer-training identities only",
        }


@dataclass(frozen=True, slots=True)
class TrainOnlyActionCalibration:
    gate_margin_bpm: float
    gate_temperature_bpm: float
    divisor_temperature_bpm: float
    fit_positions_sha256: str
    fit_identity_count: int

    def record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "fit_scope": "outer-training references and deployment-time predictions only",
        }


@dataclass(frozen=True, slots=True)
class TemporalPredictionResult:
    position: np.ndarray
    cache_index: np.ndarray
    target: np.ndarray
    base_prediction: np.ndarray
    candidate_prediction: np.ndarray
    map_prediction: np.ndarray
    rr_std: np.ndarray
    source_prediction: np.ndarray
    source_std: np.ndarray
    mixture_gate: np.ndarray
    divisor_probabilities: np.ndarray
    residual_rr: np.ndarray
    candidate_std: np.ndarray
    quality: np.ndarray
    spike_rate: np.ndarray
    radar_weights: np.ndarray
    posterior_entropy: np.ndarray
    posterior_probability: np.ndarray


def _positions_digest(positions: Sequence[int] | np.ndarray) -> str:
    values = np.asarray(positions, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Tensor):
        return _json_ready(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    """Strict JSON write that represents undefined audit metrics as null."""

    shared.atomic_write_json(path, _json_ready(value))


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _canonical_content_sha256(document: Mapping[str, Any]) -> str:
    payload = dict(document)
    payload.pop("content_sha256", None)
    return _canonical_sha256(payload)


def _manifest_session_ids(value: Any, *, location: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{location} must be a non-empty session-ID list")
    session_ids: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip() or item != item.strip():
            raise RuntimeError(f"{location} contains an invalid session ID")
        session_ids.append(item)
    if len(set(session_ids)) != len(session_ids):
        raise RuntimeError(f"{location} contains duplicate session IDs")
    return session_ids


def _validate_acquisition_v2_training_scope(
    root_manifest: Mapping[str, Any],
    session_manifests: Mapping[str, Mapping[str, Any]],
) -> None:
    """Require exact full-cohort provenance for acquisition-v2 training.

    Historical SVD caches without a version-2 acquisition indicator retain
    their existing loader behavior.  Once any root or child manifest carries a
    v2 binding, however, stripping or contradicting the root scope contract is
    a hard failure rather than a downgrade to legacy behavior.
    """

    root_contract = root_manifest.get("canonical_acquisition_contract")
    root_sessions_value = root_manifest.get("sessions")
    root_sessions = (
        root_sessions_value if isinstance(root_sessions_value, list) else []
    )
    indicated_v2 = bool(
        isinstance(root_contract, dict)
        and root_contract.get("schema_version")
        == "snn_rr.feature_cache_acquisition.v2"
        or root_manifest.get(
            "canonical_acquisition_reconstruction_content_sha256"
        )
        is not None
    )
    for item in root_sessions:
        indicated_v2 = bool(
            indicated_v2
            or (
                isinstance(item, Mapping)
                and shared._session_has_acquisition_v2_indicator(item)
            )
        )
    for manifest in session_manifests.values():
        indicated_v2 = bool(
            indicated_v2
            or shared._session_has_acquisition_v2_indicator(manifest)
        )
    if not indicated_v2:
        return

    if not isinstance(root_contract, dict) or root_contract.get(
        "schema_version"
    ) != "snn_rr.feature_cache_acquisition.v2":
        raise RuntimeError(
            "acquisition-v2 SVD input lacks its canonical root acquisition contract"
        )
    declared_root_content = root_manifest.get("content_sha256")
    if (
        not isinstance(declared_root_content, str)
        or declared_root_content != _canonical_content_sha256(root_manifest)
    ):
        raise RuntimeError("acquisition-v2 SVD root content hash mismatch")

    expected = _manifest_session_ids(
        root_manifest.get("expected_session_ids"),
        location="SVD expected_session_ids",
    )
    selected = _manifest_session_ids(
        root_manifest.get("selected_session_ids"),
        location="SVD selected_session_ids",
    )
    if root_manifest.get("expected_session_ids_sha256") != _canonical_sha256(
        expected
    ):
        raise RuntimeError("SVD expected-session ID hash mismatch")
    if root_manifest.get("selected_session_ids_sha256") != _canonical_sha256(
        selected
    ):
        raise RuntimeError("SVD selected-session ID hash mismatch")
    subjects_filter_applied = root_manifest.get("subjects_filter_applied")
    if type(subjects_filter_applied) is not bool:
        raise RuntimeError("SVD subjects_filter_applied must be an explicit boolean")

    if not isinstance(root_sessions_value, list) or any(
        not isinstance(item, dict) for item in root_sessions_value
    ):
        raise RuntimeError("acquisition-v2 SVD root session catalogue is malformed")
    catalogue_ids = _manifest_session_ids(
        [item.get("session_id") for item in root_sessions_value],
        location="SVD root catalogue session IDs",
    )
    if catalogue_ids != selected:
        raise RuntimeError("SVD selected-session IDs differ from its root catalogue")

    selection_is_full = bool(
        not subjects_filter_applied and selected == expected
    )
    derived_scope = "full_cohort" if selection_is_full else "diagnostic_subset"
    if root_manifest.get("selection_scope") != derived_scope:
        raise RuntimeError("SVD selection_scope does not match its selection evidence")

    canonical_expected = _manifest_session_ids(
        root_contract.get("expected_usable_session_ids"),
        location="canonical acquisition expected usable-session IDs",
    )
    canonical_cache = _manifest_session_ids(
        root_contract.get("cache_usable_session_ids"),
        location="canonical acquisition cache usable-session IDs",
    )
    if root_contract.get(
        "expected_usable_session_ids_sha256"
    ) != _canonical_sha256(canonical_expected):
        raise RuntimeError("canonical acquisition expected-session hash mismatch")
    if root_contract.get("cache_usable_session_ids_sha256") != _canonical_sha256(
        canonical_cache
    ):
        raise RuntimeError("canonical acquisition cache-session hash mismatch")
    if canonical_expected != expected or canonical_cache != expected:
        raise RuntimeError(
            "scientific SVD training requires exact canonical expected-session coverage"
        )

    canonical_full = bool(
        root_contract.get("selection_scope") == "full_cohort"
        and root_contract.get("full_cohort_complete") is True
        and root_contract.get("reconstruction_full_cohort_complete") is True
    )
    all_results_ok = bool(root_sessions_value) and all(
        item.get("status") == "ok" for item in root_sessions_value
    )
    derived_complete = bool(
        selection_is_full and all_results_ok and canonical_full
    )
    if type(root_manifest.get("full_cohort_complete")) is not bool or (
        root_manifest.get("full_cohort_complete") != derived_complete
    ):
        raise RuntimeError(
            "SVD full_cohort_complete does not match its bound cohort evidence"
        )

    if set(session_manifests) != set(selected):
        raise RuntimeError("SVD child manifest set differs from selected sessions")
    bindings_scientific = True
    for item in root_sessions_value:
        session_id = str(item["session_id"])
        child = session_manifests[session_id]
        declared_child_content = child.get("content_sha256")
        if (
            not isinstance(declared_child_content, str)
            or declared_child_content != _canonical_content_sha256(child)
            or item.get("content_sha256") != declared_child_content
        ):
            raise RuntimeError(f"SVD child manifest content mismatch: {session_id}")
        root_binding = item.get("canonical_acquisition_binding")
        child_binding = child.get("canonical_acquisition_binding")
        if root_binding != child_binding:
            raise RuntimeError(
                f"SVD root/child acquisition binding mismatch: {session_id}"
            )
        bindings_scientific = bool(
            bindings_scientific
            and isinstance(child_binding, dict)
            and child_binding.get("schema_version")
            == "snn_rr.feature_cache_acquisition.v2"
            and child_binding.get("scientific_eligible") is True
        )

    derived_scientific = bool(
        derived_complete
        and root_contract.get("mode") == "strict"
        and root_contract.get("scientific_eligible") is True
        and bindings_scientific
    )
    if type(root_manifest.get("scientific_eligible")) is not bool or (
        root_manifest.get("scientific_eligible") != derived_scientific
    ):
        raise RuntimeError(
            "SVD scientific_eligible does not match its acquisition evidence"
        )
    if not derived_scientific:
        raise RuntimeError(
            "acquisition-v2 temporal training requires a strict, scientifically "
            "eligible, untargeted full-cohort SVD cache"
        )


def load_temporal_aligned_experiment(
    cache_root: Path,
    oof_csv: Path,
    oof_npz: Path | None = None,
    *,
    base_oof_provenance: Path | None = None,
    verify_file_hashes: bool = True,
    historical_legacy_reproduction: bool = False,
) -> TemporalAlignedExperiment:
    """Bind valid-only component signals to the already frozen base OOF."""

    base = shared.load_aligned_experiment(
        cache_root,
        oof_csv,
        oof_npz,
        base_oof_provenance=base_oof_provenance,
        verify_file_hashes=verify_file_hashes,
        load_legacy_component_signals=True,
        historical_legacy_reproduction=historical_legacy_reproduction,
    )
    _validate_acquisition_v2_training_scope(
        base.root_manifest,
        {session.session_id: session.manifest for session in base.sessions},
    )
    if not bool(base.root_manifest.get("valid_only", False)):
        raise RuntimeError("temporal training requires a valid-reference-only cache")
    sessions: list[TemporalSessionArrays] = []
    component_hashes: dict[str, str] = {}
    for source_session in base.sessions:
        signal_path = base.cache_root / source_session.session_id / "component_signals.npy"
        if not signal_path.is_file():
            raise FileNotFoundError(
                f"component_signals.npy is missing for {source_session.session_id}"
            )
        component_signals = source_session.component_signals
        if component_signals is None:
            raise RuntimeError(
                "shared SVD loader did not bind component signals for "
                f"{source_session.session_id}"
            )
        expected_shape = (
            len(source_session.metadata),
            *source_session.attributes.shape[1:4],
            320,
        )
        if component_signals.shape != expected_shape:
            raise RuntimeError(
                f"invalid component signal shape for {source_session.session_id}: "
                f"expected {expected_shape}, got {component_signals.shape}"
            )
        declared_shape = tuple(source_session.manifest.get("component_signals_shape", ()))
        if declared_shape and declared_shape != expected_shape:
            raise RuntimeError(
                f"component signal manifest mismatch for {source_session.session_id}"
            )
        if not bool(source_session.manifest.get("valid_only", False)):
            raise RuntimeError(
                f"session {source_session.session_id} is not declared valid-only"
            )
        if source_session.manifest.get("feature_value_label_inputs", []) != []:
            raise RuntimeError(
                f"session {source_session.session_id} declares label-derived SVD values"
            )
        expected_row_labels = [
            "canonical_metadata.reference_valid (row_selection_only)"
        ]
        if (
            source_session.manifest.get("target_dependent_row_selection") is not True
            or source_session.manifest.get("label_inputs") != expected_row_labels
        ):
            raise RuntimeError(
                f"session {source_session.session_id} does not disclose its "
                "valid-only target-derived row filter"
            )
        causality_scope = source_session.manifest.get("causality_scope")
        if not isinstance(causality_scope, dict) or (
            causality_scope.get("within_window_prefix_causal_representation")
            is not False
            or causality_scope.get("streaming_prefix_causality_claim_allowed")
            is not False
        ):
            raise RuntimeError(
                f"session {source_session.session_id} lacks the bounded SVD "
                "causality disclosure"
            )
        bound_signal_sha = source_session.files_sha256.get("component_signals")
        signal_sha = (
            bound_signal_sha
            if bound_signal_sha is not None
            else (
                shared.sha256_file(signal_path)
                if verify_file_hashes
                else "not_computed_verify_file_hashes_false"
            )
        )
        component_hashes[source_session.session_id] = signal_sha
        sessions.append(
            TemporalSessionArrays(
                session_id=source_session.session_id,
                component_signals=component_signals,
                attributes=source_session.attributes,
                metadata=source_session.metadata,
                manifest=source_session.manifest,
                component_signals_sha256=signal_sha,
                radar_timing_valid_mask=source_session.radar_timing_valid_mask,
            )
        )
    provenance = dict(base.provenance)
    provenance.update(
        {
            "component_signals_status": (
                "window_end_label_free_values_with_target_derived_row_selection"
            ),
            "component_signals_sha256": component_hashes,
            "valid_only_alignment_enforced": True,
            "nominal_time_samples": 320,
            "causality_scope": (
                "past_32s_window_end_only_not_raw_stream_prefix_causal"
            ),
        }
    )
    experiment = TemporalAlignedExperiment(
        cache_root=base.cache_root,
        oof_csv=base.oof_csv,
        oof_npz=base.oof_npz,
        metadata=base.metadata,
        sessions=sessions,
        root_manifest=base.root_manifest,
        provenance=provenance,
        training_authority_receipt=base.training_authority_receipt,
        _base_authority_experiment=base,
    )
    view_receipt = _TemporalTrainingViewReceipt(
        schema_version=1,
        base_receipt_id=id(base.training_authority_receipt),
        experiment_object_id=id(experiment),
        base_experiment_object_id=id(base),
        sessions_list_object_id=id(experiment.sessions),
        provenance_object_id=id(experiment.provenance),
        provenance_sha256=shared._canonical_sha256(experiment.provenance),
        session_object_ids=tuple(id(session) for session in experiment.sessions),
        source_pipeline_sha256=_current_temporal_training_source_sha256(),
        arrays_for_position_function_id=id(
            TemporalAlignedExperiment.arrays_for_position
        ),
        structural_mask_function_id=id(
            TemporalAlignedExperiment.structural_radar_mask_for_position
        ),
    )
    _ISSUED_TEMPORAL_TRAINING_VIEW_RECEIPTS[id(view_receipt)] = (
        view_receipt,
        experiment,
        base,
    )
    experiment._temporal_view_receipt = view_receipt
    return experiment


# Audit-friendly aliases.
load_aligned_experiment = load_temporal_aligned_experiment
make_outer_split = shared.make_outer_split
promotion_decision = shared.promotion_decision


def _assert_temporal_training_authority(
    experiment: TemporalAlignedExperiment,
    *,
    historical_legacy_reproduction: bool = False,
) -> None:
    """Replay shared authority and bind the temporal view to that exact load."""

    base = experiment._base_authority_experiment
    receipt = experiment.training_authority_receipt
    view_receipt = experiment._temporal_view_receipt
    issued_view = (
        _ISSUED_TEMPORAL_TRAINING_VIEW_RECEIPTS.get(id(view_receipt))
        if type(view_receipt) is _TemporalTrainingViewReceipt
        else None
    )
    if (
        type(experiment) is not TemporalAlignedExperiment
        or type(base) is not shared.AlignedSVDExperiment
        or receipt is None
        or receipt is not base.training_authority_receipt
        or type(view_receipt) is not _TemporalTrainingViewReceipt
        or view_receipt.schema_version != 1
        or issued_view is None
        or issued_view[0] is not view_receipt
        or issued_view[1] is not experiment
        or issued_view[2] is not base
        or view_receipt.base_receipt_id != id(receipt)
        or view_receipt.experiment_object_id != id(experiment)
        or view_receipt.base_experiment_object_id != id(base)
        or view_receipt.source_pipeline_sha256
        != _current_temporal_training_source_sha256()
        or view_receipt.arrays_for_position_function_id
        != id(TemporalAlignedExperiment.arrays_for_position)
        or view_receipt.structural_mask_function_id
        != id(TemporalAlignedExperiment.structural_radar_mask_for_position)
    ):
        raise RuntimeError(
            "temporal SVD training requires the loader-issued shared authority receipt"
        )
    shared._assert_training_cache_authority(
        base,
        historical_legacy_reproduction=historical_legacy_reproduction,
    )
    if not (
        experiment.cache_root.resolve() == base.cache_root.resolve()
        and experiment.oof_csv.resolve() == base.oof_csv.resolve()
        and experiment.oof_npz.resolve() == base.oof_npz.resolve()
        and experiment.metadata is base.metadata
        and experiment.root_manifest is base.root_manifest
        and experiment.provenance.get("row_binding_sha256")
        == base.provenance.get("row_binding_sha256")
        and type(experiment.sessions) is list
        and all(type(session) is TemporalSessionArrays for session in experiment.sessions)
        and id(experiment.sessions) == view_receipt.sessions_list_object_id
        and id(experiment.provenance) == view_receipt.provenance_object_id
        and shared._canonical_sha256(experiment.provenance)
        == view_receipt.provenance_sha256
        and tuple(id(session) for session in experiment.sessions)
        == view_receipt.session_object_ids
        and len(experiment.sessions) == len(base.sessions)
    ):
        raise RuntimeError("temporal SVD view differs from its authorized base load")
    for temporal, source in zip(experiment.sessions, base.sessions, strict=True):
        if not (
            temporal.session_id == source.session_id
            and temporal.component_signals is source.component_signals
            and temporal.attributes is source.attributes
            and temporal.metadata is source.metadata
            and temporal.manifest is source.manifest
            and temporal.radar_timing_valid_mask is source.radar_timing_valid_mask
            and temporal.component_signals_sha256
            == source.files_sha256.get("component_signals")
        ):
            raise RuntimeError(
                f"temporal SVD session view is not source-bound: {temporal.session_id}"
            )


def fit_temporal_signal_normalizer(
    experiment: TemporalAlignedExperiment,
    train_positions: Sequence[int] | np.ndarray,
    *,
    max_samples_per_variant: int = 200_000,
    clip_quantile: float = 0.999,
    seed: int = 20260828,
) -> TemporalSignalNormalizer:
    positions = np.asarray(train_positions, dtype=np.int64)
    if not len(positions):
        raise ValueError("normalizer requires at least one training row")
    if max_samples_per_variant < len(positions) or not 0.95 <= clip_quantile < 1.0:
        raise ValueError("normalizer sample limit/clip quantile is invalid")
    first, _ = experiment.arrays_for_position(int(positions[0]))
    if first.ndim != 4:
        raise RuntimeError("component signal cache rows must be four-dimensional")
    variants = first.shape[1]
    samples_per_row = max(1, int(max_samples_per_variant) // len(positions))
    rng = np.random.default_rng(int(seed))
    samples: list[list[np.ndarray]] = [[] for _ in range(variants)]
    for position in positions:
        values, _ = experiment.arrays_for_position(int(position))
        values = np.asarray(values, dtype=np.float32)
        radar_available = np.any(np.abs(values) > 1.0e-8, axis=(1, 2, 3))
        for variant in range(variants):
            flattened = values[radar_available, variant].reshape(-1)
            flattened = flattened[np.isfinite(flattened)]
            if not len(flattened):
                continue
            take = min(samples_per_row, len(flattened))
            indices = rng.choice(len(flattened), size=take, replace=False)
            samples[variant].append(flattened[indices])
    center = np.zeros(variants, dtype=np.float32)
    scale = np.ones(variants, dtype=np.float32)
    clip = np.full(variants, 8.0, dtype=np.float32)
    sampled_count = math.inf
    for variant, parts in enumerate(samples):
        if not parts:
            raise RuntimeError(f"training rows contain no signal for variant {variant}")
        values = np.concatenate(parts).astype(np.float64, copy=False)
        sampled_count = min(sampled_count, len(values))
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median))) * 1.4826
        if not math.isfinite(mad) or mad < 1.0e-4:
            rms = float(np.sqrt(np.mean(np.square(values - median))))
            mad = rms if math.isfinite(rms) and rms >= 1.0e-4 else 1.0
        normalized_abs = np.abs((values - median) / mad)
        threshold = float(np.quantile(normalized_abs, float(clip_quantile)))
        center[variant] = median
        scale[variant] = mad
        clip[variant] = np.clip(threshold, 3.0, 20.0)
    return TemporalSignalNormalizer(
        center=center,
        scale=scale,
        clip=clip,
        fit_positions_sha256=_positions_digest(positions),
        sampled_values_per_variant=int(sampled_count),
    )


def _classical_values(row: pd.Series) -> np.ndarray:
    return np.asarray(
        [
            row.get("classical_rr_bpm", np.nan),
            row.get("radar_peak_1_bpm", np.nan),
            row.get("radar_peak_2_bpm", np.nan),
            row.get("radar_peak_3_bpm", np.nan),
        ],
        dtype=np.float32,
    )


def fit_train_only_action_calibration(
    metadata: pd.DataFrame,
    train_positions: Sequence[int] | np.ndarray,
    *,
    rr_min: float = 6.0,
    rr_max: float = 45.0,
) -> TrainOnlyActionCalibration:
    positions = np.asarray(train_positions, dtype=np.int64)
    if not len(positions):
        raise ValueError("action calibration requires training rows")
    improvements: list[float] = []
    divisor_gaps: list[float] = []
    for position in positions:
        row = metadata.iloc[int(position)]
        target = float(row["rr_bpm"])
        base_error = abs(float(row["prediction_bpm"]) - target)
        seeds = _classical_values(row)
        means = seeds[:, None] * np.asarray((1.0, 2.0, 3.0, 4.0))[None, :]
        valid = (
            np.isfinite(means)
            & np.isfinite(seeds[:, None])
            & (seeds[:, None] > 0)
            & (means >= rr_min)
            & (means <= rr_max)
        )
        errors = np.where(valid, np.abs(means - target), np.inf)
        per_divisor = errors.min(axis=0)
        finite = np.sort(per_divisor[np.isfinite(per_divisor)])
        if len(finite):
            improvement = base_error - float(finite[0])
            if improvement > 0:
                improvements.append(improvement)
        if len(finite) >= 2:
            divisor_gaps.append(float(finite[1] - finite[0]))
    gate_margin = (
        float(np.quantile(improvements, 0.25)) if improvements else 0.05
    )
    median_gap = float(np.median(divisor_gaps)) if divisor_gaps else 0.75
    gate_margin = float(np.clip(gate_margin, 0.05, 0.50))
    gate_temperature = float(np.clip(0.5 * median_gap, 0.25, 1.50))
    divisor_temperature = float(np.clip(median_gap, 0.25, 2.00))
    identities = metadata.iloc[positions]["identity"].astype(str)
    return TrainOnlyActionCalibration(
        gate_margin_bpm=gate_margin,
        gate_temperature_bpm=gate_temperature,
        divisor_temperature_bpm=divisor_temperature,
        fit_positions_sha256=_positions_digest(positions),
        fit_identity_count=int(identities.nunique()),
    )


class TemporalSVDDataset(Dataset[dict[str, Tensor]]):
    """Expose deployment-time component evidence plus loss-only reference fields."""

    def __init__(
        self,
        experiment: TemporalAlignedExperiment,
        positions: Sequence[int] | np.ndarray,
        normalizer: TemporalSignalNormalizer,
    ) -> None:
        self.experiment = experiment
        self.positions = np.asarray(positions, dtype=np.int64)
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        position = int(self.positions[int(item)])
        row = self.experiment.metadata.iloc[position]
        raw_signals, raw_attributes = self.experiment.arrays_for_position(position)
        raw_signals = np.asarray(raw_signals)
        structural_mask = self.experiment.structural_radar_mask_for_position(position)
        if structural_mask is None:
            # Legacy caches have no timing authority and retain their original
            # numeric-availability fallback.
            radar_mask = np.any(np.abs(raw_signals) > 1.0e-8, axis=(1, 2, 3))
        else:
            radar_mask = np.asarray(structural_mask, dtype=np.bool_)
            if radar_mask.shape != (raw_signals.shape[0],):
                raise RuntimeError("structural radar mask has an invalid shape")
        signals = self.normalizer.transform(raw_signals)
        signals[~radar_mask] = 0
        attributes = np.asarray(raw_attributes, dtype=np.float32).copy()
        attributes[~radar_mask] = 0
        classical = _classical_values(row)
        if structural_mask is not None:
            classical[1:][~radar_mask] = 0
            if not bool(radar_mask.all()):
                # The leading candidate is the cached three-view fusion and
                # cannot survive a missing contributing radar.
                classical[0] = 0
        reference_sigma = float(row.get("reference_sigma_bpm", 1.0))
        reference_quality = float(row.get("reference_quality", 1.0))
        return {
            "component_signals": torch.from_numpy(signals),
            "attributes": torch.from_numpy(attributes),
            "base_prediction": torch.tensor(
                float(row["prediction_bpm"]), dtype=torch.float32
            ),
            "base_std": torch.tensor(
                max(0.25, float(row.get("rr_std_bpm", 1.5))), dtype=torch.float32
            ),
            "classical_rr": torch.from_numpy(classical),
            "radar_mask": torch.from_numpy(radar_mask),
            # Loss/evaluation only: forward_temporal_model cannot access these.
            "rr": torch.tensor(float(row["rr_bpm"]), dtype=torch.float32),
            "reference_valid": torch.tensor(bool(row.get("reference_valid", True))),
            "reference_quality": torch.tensor(reference_quality, dtype=torch.float32),
            "reference_sigma": torch.tensor(reference_sigma, dtype=torch.float32),
            "observable": torch.tensor(
                bool(row.get("radar_observable", True)), dtype=torch.float32
            ),
            "position": torch.tensor(position, dtype=torch.int64),
            "cache_index": torch.tensor(int(row["cache_index"]), dtype=torch.int64),
        }


def _worker_seed(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_temporal_loader(
    experiment: TemporalAlignedExperiment,
    positions: Sequence[int] | np.ndarray,
    normalizer: TemporalSignalNormalizer,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
    seed: int,
    train: bool,
    samples_per_epoch: int | None = None,
    rr_bin_width: float = 3.0,
    rr_balance_power: float = 0.65,
    tail_min_bpm: float = 25.0,
    tail_max_bpm: float = 35.0,
    tail_boost: float = 2.0,
) -> DataLoader[dict[str, Tensor]]:
    positions = np.asarray(positions, dtype=np.int64)
    dataset = TemporalSVDDataset(experiment, positions, normalizer)
    generator = torch.Generator().manual_seed(int(seed))
    sampler = None
    if train:
        weights = shared.identity_rr_tail_sample_weights(
            experiment.metadata,
            positions,
            rr_bin_width=rr_bin_width,
            rr_balance_power=rr_balance_power,
            tail_min_bpm=tail_min_bpm,
            tail_max_bpm=tail_max_bpm,
            tail_boost=tail_boost,
        )
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=int(samples_per_epoch or len(positions)),
            replacement=True,
            generator=generator,
        )
    loader_generator = torch.Generator().manual_seed(int(seed) + 104_729)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        shuffle=False,
        num_workers=int(workers),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(workers > 0),
        worker_init_fn=_worker_seed,
        generator=loader_generator,
        drop_last=bool(train and len(dataset) >= batch_size),
    )


def apply_coupled_temporal_radar_dropout(
    component_signals: Tensor,
    attributes: Tensor,
    radar_mask: Tensor,
    *,
    p: float,
    training: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    dropped, available = apply_radar_dropout(
        component_signals,
        radar_mask,
        p=float(p),
        training=bool(training),
        ensure_one=True,
    )
    attribute_shape = (*available.shape, 1, 1, 1)
    attributes = attributes * available.view(attribute_shape).to(dtype=attributes.dtype)
    return dropped, attributes, available


def forward_temporal_model(
    model: nn.Module,
    batch: Mapping[str, Tensor],
    device: torch.device,
    *,
    radar_dropout_p: float = 0.0,
    training: bool = False,
) -> Mapping[str, Tensor]:
    """Forward an explicit deployment allow-list; target fields are unreachable."""

    def move(name: str, *, dtype: torch.dtype | None = None) -> Tensor:
        value = batch[name].to(device, non_blocking=True)
        return value.to(dtype=dtype) if dtype is not None else value

    signals = move("component_signals", dtype=torch.float32)
    attributes = move("attributes", dtype=torch.float32)
    radar_mask = move("radar_mask").bool()
    signals, attributes, radar_mask = apply_coupled_temporal_radar_dropout(
        signals,
        attributes,
        radar_mask,
        p=radar_dropout_p,
        training=training,
    )
    return model(
        signals,
        attributes,
        move("base_prediction", dtype=torch.float32),
        move("base_std", dtype=torch.float32),
        move("classical_rr", dtype=torch.float32),
        radar_mask,
    )


def _oracle_divisor_targets(
    classical_rr: Tensor,
    target: Tensor,
    *,
    rr_min: float,
    rr_max: float,
    temperature: float,
    max_residual_bpm: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    multipliers = classical_rr.new_tensor((1.0, 2.0, 3.0, 4.0))
    means = classical_rr.unsqueeze(-1) * multipliers.view(1, 1, 4)
    valid = (
        torch.isfinite(classical_rr).unsqueeze(-1)
        & (classical_rr.unsqueeze(-1) > 0)
        & torch.isfinite(means)
        & (means >= float(rr_min))
        & (means <= float(rr_max))
    )
    costs = (means - target[:, None, None]).abs().masked_fill(~valid, 1.0e4)
    best_cost, best_seed = costs.min(dim=1)
    candidate_valid = valid.any(dim=1)
    logits = (-best_cost / float(temperature)).masked_fill(~candidate_valid, -1.0e4)
    soft = logits.softmax(dim=1) * candidate_valid.to(dtype=logits.dtype)
    soft = soft / soft.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    selected_mean = means.gather(
        1, best_seed.unsqueeze(1)
    ).squeeze(1)
    selected_mean = torch.where(candidate_valid, selected_mean, target.unsqueeze(1))
    residual_target = (target.unsqueeze(1) - selected_mean).clamp(
        -float(max_residual_bpm), float(max_residual_bpm)
    )
    return soft, best_cost, residual_target, candidate_valid


def compute_temporal_multitask_loss(
    output: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    rr_bins: Tensor,
    *,
    action_calibration: TrainOnlyActionCalibration | None = None,
    posterior_nll_weight: float = 1.0,
    source_nll_weight: float = 0.35,
    crps_weight: float = 0.15,
    mae_weight: float = 0.40,
    divisor_weight: float = 0.25,
    residual_weight: float = 0.15,
    uncertainty_weight: float = 0.08,
    gate_weight: float = 0.15,
    action_regret_weight: float = 0.20,
    safe_gate_weight: float = 0.25,
    quality_weight: float = 0.03,
    spike_sparsity_weight: float = 5.0e-4,
    max_residual_bpm: float = 1.5,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Posterior, soft divisor/action, uncertainty and safety objective."""

    calibration = action_calibration or TrainOnlyActionCalibration(
        gate_margin_bpm=0.05,
        gate_temperature_bpm=0.5,
        divisor_temperature_bpm=0.75,
        fit_positions_sha256="default_not_fit",
        fit_identity_count=0,
    )
    probability = output["probabilities"].float()
    source_probability = output["source_probabilities"].float()
    base_probability = output["base_probabilities"].float()
    bins = rr_bins.to(device=probability.device, dtype=torch.float32)
    target = batch["rr"].to(probability.device).float()
    valid = batch["reference_valid"].to(probability.device).bool() & torch.isfinite(target)
    zero = probability.sum() * 0.0
    if valid.any():
        weight = batch["reference_quality"].to(probability.device).float()[valid].clamp(
            0.25, 1.0
        )
        sigma = batch["reference_sigma"].to(probability.device).float()[valid].clamp(
            0.30, 2.50
        )
        soft_target = gaussian_soft_targets(target[valid], bins, sigma=sigma)
        log_probability = probability[valid].clamp_min(1.0e-8).log()
        log_source = source_probability[valid].clamp_min(1.0e-8).log()
        posterior_nll_per = -(soft_target * log_probability).sum(dim=1)
        source_nll_per = -(soft_target * log_source).sum(dim=1)
        posterior_nll = (posterior_nll_per * weight).sum() / weight.sum()
        source_nll = (source_nll_per * weight).sum() / weight.sum()

        empirical_cdf = (
            bins.view(1, -1) >= target[valid].unsqueeze(1)
        ).to(dtype=probability.dtype)
        crps_per = (
            probability[valid].cumsum(dim=1) - empirical_cdf
        ).square().sum(dim=1) * float(bins[1] - bins[0])
        crps = (crps_per * weight).sum() / weight.sum()
        mae_per = F.smooth_l1_loss(
            output["expected_rr"].float()[valid],
            target[valid],
            beta=1.0,
            reduction="none",
        )
        mae = (mae_per * weight).sum() / weight.sum()

        classical = batch["classical_rr"].to(probability.device).float()[valid]
        divisor_target, divisor_cost, residual_target, candidate_valid = (
            _oracle_divisor_targets(
                classical,
                target[valid],
                rr_min=float(bins[0]),
                rr_max=float(bins[-1]),
                temperature=calibration.divisor_temperature_bpm,
                max_residual_bpm=float(max_residual_bpm),
            )
        )
        predicted_divisor = output["divisor_probabilities"].float()[valid]
        divisor_ce_per = -(
            divisor_target * predicted_divisor.clamp_min(1.0e-8).log()
        ).sum(dim=1)
        usable = candidate_valid.any(dim=1)
        usable_weight = weight * usable.to(dtype=weight.dtype)
        denominator = usable_weight.sum().clamp_min(1.0e-8)
        divisor_ce = (divisor_ce_per * usable_weight).sum() / denominator
        divisor_expected_cost = (
            predicted_divisor * divisor_cost.clamp_max(100.0)
        ).sum(dim=1)
        divisor_best_cost = divisor_cost.min(dim=1).values.clamp_max(100.0)
        divisor_regret = (
            (divisor_expected_cost - divisor_best_cost) * usable_weight
        ).sum() / denominator

        residual_prediction = output["residual_rr_per_divisor"].float()[valid]
        residual_per = F.smooth_l1_loss(
            residual_prediction, residual_target, beta=0.5, reduction="none"
        )
        residual_per = (residual_per * divisor_target).sum(dim=1)
        residual = (residual_per * usable_weight).sum() / denominator

        candidate_std = output["candidate_std_rr"].float()[valid].clamp_min(0.1)
        candidate_center = output["candidate_centers_rr"].float()[valid]
        gaussian_nll = 0.5 * (
            ((target[valid, None] - candidate_center) / candidate_std).square()
            + 2.0 * candidate_std.log()
        )
        gaussian_nll = (gaussian_nll * divisor_target).sum(dim=1)
        uncertainty_nll = (gaussian_nll * usable_weight).sum() / denominator

        absolute_grid_error = (
            bins.view(1, -1) - target[valid].unsqueeze(1)
        ).abs()
        base_risk = (base_probability[valid] * absolute_grid_error).sum(dim=1)
        source_risk = (source_probability[valid] * absolute_grid_error).sum(dim=1)
        final_risk = (probability[valid] * absolute_grid_error).sum(dim=1)
        best_candidate_cost = divisor_cost.min(dim=1).values
        gate_target = torch.sigmoid(
            (
                base_risk.detach()
                - best_candidate_cost.detach()
                - calibration.gate_margin_bpm
            )
            / calibration.gate_temperature_bpm
        ) * usable.to(dtype=weight.dtype)
        gate_bce_per = F.binary_cross_entropy_with_logits(
            output["mixture_gate_logits"].float()[valid], gate_target, reduction="none"
        )
        gate_bce = (gate_bce_per * weight).sum() / weight.sum()
        action_regret_per = final_risk - torch.minimum(base_risk, source_risk)
        action_regret = (action_regret_per * weight).sum() / weight.sum()
        safe_gate_per = output["mixture_gate"].float()[valid] * F.relu(
            source_risk - base_risk + calibration.gate_margin_bpm
        )
        safe_gate = (safe_gate_per * weight).sum() / weight.sum()
        oracle_source_fraction = gate_target.mean()
    else:
        posterior_nll = source_nll = crps = mae = zero
        divisor_ce = divisor_regret = residual = uncertainty_nll = zero
        gate_bce = action_regret = safe_gate = oracle_source_fraction = zero

    observable = batch["observable"].to(probability.device).float()
    quality_bce = F.binary_cross_entropy_with_logits(
        output["quality_logits"].float(), observable
    )
    spike_rate = output["spike_rate"].float().mean()
    total = (
        float(posterior_nll_weight) * posterior_nll
        + float(source_nll_weight) * source_nll
        + float(crps_weight) * crps
        + float(mae_weight) * mae
        + float(divisor_weight) * (divisor_ce + 0.25 * divisor_regret)
        + float(residual_weight) * residual
        + float(uncertainty_weight) * uncertainty_nll
        + float(gate_weight) * gate_bce
        + float(action_regret_weight) * action_regret
        + float(safe_gate_weight) * safe_gate
        + float(quality_weight) * quality_bce
        + float(spike_sparsity_weight) * spike_rate
    )
    components = {
        "loss": total.detach(),
        "posterior_nll": posterior_nll.detach(),
        "source_nll": source_nll.detach(),
        "crps": crps.detach(),
        "mae": mae.detach(),
        "divisor_ce": divisor_ce.detach(),
        "divisor_regret": divisor_regret.detach(),
        "residual": residual.detach(),
        "uncertainty_nll": uncertainty_nll.detach(),
        "gate_bce": gate_bce.detach(),
        "action_regret": action_regret.detach(),
        "safe_gate": safe_gate.detach(),
        "quality_bce": quality_bce.detach(),
        "spike_rate": spike_rate.detach(),
        "oracle_source_fraction": oracle_source_fraction.detach(),
        "valid_fraction": valid.float().mean().detach(),
    }
    return total, components


compute_multitask_loss = compute_temporal_multitask_loss


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=bool(enabled and device.type == "cuda"),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    amp_scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    amp: bool,
    gradient_clip: float,
    radar_dropout_p: float,
    action_calibration: TrainOnlyActionCalibration,
    loss_kwargs: Mapping[str, float],
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    totals: defaultdict[str, float] = defaultdict(float)
    examples = 0
    for batch_number, batch in enumerate(loader):
        if max_batches is not None and batch_number >= max_batches:
            break
        count = len(batch["rr"])
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            output = forward_temporal_model(
                model,
                batch,
                device,
                radar_dropout_p=radar_dropout_p,
                training=True,
            )
            loss, components = compute_temporal_multitask_loss(
                output,
                batch,
                model.rr_bins,
                action_calibration=action_calibration,
                **dict(loss_kwargs),
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite temporal SNN loss")
        amp_scaler.scale(loss).backward()
        amp_scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
        amp_scaler.step(optimizer)
        amp_scaler.update()
        for name, value in components.items():
            totals[name] += float(value) * count
        examples += count
    if not examples:
        raise RuntimeError("training loader yielded no batches")
    return {name: value / examples for name, value in totals.items()}


@torch.inference_mode()
def predict_loader(
    model: nn.Module,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
    *,
    amp: bool,
    max_batches: int | None = None,
) -> TemporalPredictionResult:
    model.eval()
    values: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    for batch_number, batch in enumerate(loader):
        if max_batches is not None and batch_number >= max_batches:
            break
        with _autocast(device, amp):
            output = forward_temporal_model(model, batch, device)
        mapping = {
            "position": batch["position"],
            "cache_index": batch["cache_index"],
            "target": batch["rr"],
            "base_prediction": batch["base_prediction"],
            "candidate_prediction": output["expected_rr"],
            "map_prediction": output["map_rr"],
            "rr_std": output["rr_std"],
            "source_prediction": output["source_expected_rr"],
            "source_std": output["source_std"],
            "mixture_gate": output["mixture_gate"],
            "divisor_probabilities": output["divisor_probabilities"],
            "residual_rr": output["residual_rr"],
            "candidate_std": output["candidate_std_rr"],
            "quality": output["quality"],
            "spike_rate": output["spike_rate_per_sample"],
            "radar_weights": output["radar_weights"],
            "posterior_entropy": output["posterior_entropy"],
            "posterior_probability": output["probabilities"],
        }
        for name, value in mapping.items():
            values[name].append(value.detach().float().cpu().numpy())
    if not values:
        raise RuntimeError("prediction loader yielded no batches")
    arrays = {
        name: np.concatenate(parts).astype(
            np.int64 if name in {"position", "cache_index"} else np.float32
        )
        for name, parts in values.items()
    }
    order = np.argsort(arrays["position"], kind="stable")
    return TemporalPredictionResult(
        **{name: arrays[name][order] for name in TemporalPredictionResult.__dataclass_fields__}
    )


def _prediction_arrays(
    result: TemporalPredictionResult, *, promoted: bool | None = None
) -> dict[str, np.ndarray]:
    arrays = {
        name: np.asarray(getattr(result, name))
        for name in TemporalPredictionResult.__dataclass_fields__
    }
    arrays["posterior_probability"] = arrays["posterior_probability"].astype(np.float16)
    if promoted is not None:
        arrays["prediction_final"] = (
            result.candidate_prediction if promoted else result.base_prediction
        ).astype(np.float32)
        arrays["promoted"] = np.full(len(result.position), bool(promoted), dtype=bool)
    return arrays


def _load_prediction(path: Path) -> TemporalPredictionResult:
    _binding, payload = shared._snapshot_training_file_payload(path)
    return _load_prediction_payload(payload)


def _load_prediction_payload(payload: bytes) -> TemporalPredictionResult:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        missing = sorted(
            set(TemporalPredictionResult.__dataclass_fields__) - set(archive.files)
        )
        if missing:
            raise RuntimeError(
                f"temporal prediction archive is missing fields: {missing}"
            )
        return TemporalPredictionResult(
            **{
                name: np.asarray(archive[name])
                for name in TemporalPredictionResult.__dataclass_fields__
            }
        )


def _prediction_report(
    result: TemporalPredictionResult,
    metadata: pd.DataFrame,
    *,
    promoted: bool | None = None,
    tail_min_bpm: float = 25.0,
    tail_max_bpm: float = 35.0,
) -> dict[str, Any]:
    identities = metadata.iloc[result.position]["identity"].astype(str).to_numpy()
    report: dict[str, Any] = {
        "n": int(len(result.position)),
        "candidate": shared.evaluation_snapshot(
            result.target,
            result.candidate_prediction,
            identities,
            high_min_bpm=tail_min_bpm,
            high_max_bpm=tail_max_bpm,
        ),
        "base": shared.evaluation_snapshot(
            result.target,
            result.base_prediction,
            identities,
            high_min_bpm=tail_min_bpm,
            high_max_bpm=tail_max_bpm,
        ),
    }
    if promoted is not None:
        final = result.candidate_prediction if promoted else result.base_prediction
        report["locked_final"] = shared.evaluation_snapshot(
            result.target,
            final,
            identities,
            high_min_bpm=tail_min_bpm,
            high_max_bpm=tail_max_bpm,
        )
        report["promoted"] = bool(promoted)
    return report


def _source_binding(experiment: TemporalAlignedExperiment) -> str:
    payload = json.dumps(
        _json_ready(experiment.provenance),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalizer_signature(normalizer: TemporalSignalNormalizer) -> str:
    return shared.stable_signature(normalizer.record(), length=32)


def _action_signature(calibration: TrainOnlyActionCalibration) -> str:
    return shared.stable_signature(calibration.record(), length=32)


def _checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    amp_scaler: torch.amp.GradScaler,
    loader: DataLoader[Any],
    epoch: int,
    best_epoch: int,
    best_score: float,
    stale_epochs: int,
    fold: int,
    split: FoldSplit,
    run_signature: str,
    source_binding_sha256: str,
    model_kwargs: Mapping[str, Any],
    normalizer: TemporalSignalNormalizer,
    action_calibration: TrainOnlyActionCalibration,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": SCHEMA_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "amp_scaler_state": amp_scaler.state_dict(),
        "rng_state": shared.capture_rng_state(loader),
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "stale_epochs": int(stale_epochs),
        "fold": int(fold),
        "split": {
            "train_identities": list(split.train_identities),
            "validation_identities": list(split.validation_identities),
            "test_identities": list(split.test_identities),
        },
        "run_signature": run_signature,
        "source_binding_sha256": source_binding_sha256,
        "model_kwargs": dict(model_kwargs),
        "signal_normalizer": normalizer.record(),
        "signal_normalizer_signature": _normalizer_signature(normalizer),
        "action_calibration": action_calibration.record(),
        "action_calibration_signature": _action_signature(action_calibration),
        "resume_reproducibility": {
            "python_numpy_torch_cuda_loader_sampler_rng_captured": True,
            "bitwise_gpu_guarantee": False,
        },
    }


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    fold: int,
    split: FoldSplit,
    run_signature: str,
    source_binding_sha256: str,
    normalizer: TemporalSignalNormalizer,
    action_calibration: TrainOnlyActionCalibration,
) -> None:
    expected = {
        "fold": int(fold),
        "run_signature": run_signature,
        "source_binding_sha256": source_binding_sha256,
        "signal_normalizer_signature": _normalizer_signature(normalizer),
        "action_calibration_signature": _action_signature(action_calibration),
    }
    for name, value in expected.items():
        if checkpoint.get(name) != value:
            raise RuntimeError(f"temporal resume checkpoint {name} mismatch")
    declared = checkpoint.get("split", {})
    for name, identities in (
        ("train_identities", split.train_identities),
        ("validation_identities", split.validation_identities),
        ("test_identities", split.test_identities),
    ):
        if tuple(declared.get(name, ())) != tuple(identities):
            raise RuntimeError(f"temporal resume checkpoint {name} mismatch")


def _model_kwargs(
    args: argparse.Namespace, experiment: TemporalAlignedExperiment
) -> dict[str, Any]:
    first = experiment.sessions[0].component_signals
    cell_types = tuple(
        value.strip().lower() for value in str(args.cell_types).split(",") if value.strip()
    )
    kwargs: dict[str, Any] = {
        "num_variants": int(first.shape[2]),
        "num_components": int(first.shape[3]),
        "num_radars": int(first.shape[1]),
        "compressor_channels": int(args.compressor_channels),
        "hidden_channels": int(args.hidden_channels),
        "cell_types": cell_types,
        "beta": float(args.beta),
        "max_residual_bpm": float(args.max_residual_bpm),
        "initial_gate_bias": float(args.initial_gate_bias),
        "dropout": float(args.dropout),
    }
    if args.preset == "tiny":
        kwargs.update(compressor_channels=8, hidden_channels=12)
    elif args.preset == "compact":
        kwargs.update(
            compressor_channels=min(24, int(args.compressor_channels)),
            hidden_channels=min(48, int(args.hidden_channels)),
        )
    return kwargs


def _fold_split_record(
    experiment: TemporalAlignedExperiment,
    fold: int,
    split: FoldSplit,
) -> dict[str, Any]:
    return {
        "outer_test_fold": int(fold),
        "validation_fold": int(split.validation_fold),
        "weight_fit_folds": sorted(
            set(range(N_FOLDS)) - {fold, split.validation_fold}
        ),
        "train_rows": int(len(split.train)),
        "validation_rows": int(len(split.validation)),
        "test_rows": int(len(split.test)),
        "train_identities": list(split.train_identities),
        "validation_identities": list(split.validation_identities),
        "test_identities": list(split.test_identities),
        "identity_overlap_asserted_empty": True,
        "test_policy": (
            "outer-test rows define the partition only; no test label, metric, "
            "protocol, normalizer statistic, threshold, or prediction is accessed "
            "before selection lock"
        ),
    }


def _fold_row_authority_sha256(
    experiment: TemporalAlignedExperiment, split: FoldSplit
) -> str:
    columns = ["cache_index", "identity", "fold", "rr_bpm", "prediction_bpm"]
    return shared._dataframe_training_sha256(
        experiment.metadata.iloc[np.asarray(split.test, dtype=np.int64)]
        .loc[:, columns]
        .copy()
    )


def _validate_temporal_prediction_binding(
    result: TemporalPredictionResult,
    *,
    experiment: TemporalAlignedExperiment,
    fold: int,
    split: FoldSplit,
) -> None:
    if type(result) is not TemporalPredictionResult:
        raise RuntimeError("temporal fold result has an invalid exact type")
    position = np.asarray(result.position)
    if (
        position.ndim != 1
        or position.dtype.kind not in "iu"
        or not len(position)
        or len(np.unique(position)) != len(position)
        or np.any(position < 0)
        or np.any(position >= len(experiment.metadata))
    ):
        raise RuntimeError("temporal fold result positions are invalid")
    position = position.astype(np.int64, copy=False)
    permitted = set(map(int, np.asarray(split.test, dtype=np.int64)))
    if not set(map(int, position)) <= permitted:
        raise RuntimeError("temporal fold result contains rows outside its outer fold")
    metadata = experiment.metadata.iloc[position]
    if not np.all(metadata["fold"].to_numpy(dtype=np.int64) == int(fold)):
        raise RuntimeError("temporal fold result row/fold binding mismatch")
    result_cache_index = np.asarray(result.cache_index)
    if result_cache_index.dtype.kind not in "iu" or not np.array_equal(
        result_cache_index.astype(np.int64, copy=False),
        metadata["cache_index"].to_numpy(dtype=np.int64),
    ):
        raise RuntimeError("temporal fold result cache_index binding mismatch")
    shared._assert_close_numbers(
        result.target,
        metadata["rr_bpm"],
        "temporal fold result target",
        atol=5e-5,
    )
    shared._assert_close_numbers(
        result.base_prediction,
        metadata["prediction_bpm"],
        "temporal fold result base prediction",
        atol=5e-5,
    )
    count = len(position)
    for name in TemporalPredictionResult.__dataclass_fields__:
        value = np.asarray(getattr(result, name))
        if value.ndim < 1 or len(value) != count:
            raise RuntimeError(f"temporal fold result {name} row count mismatch")
        if value.dtype.hasobject or (
            value.dtype.kind in "fc" and not np.isfinite(value).all()
        ):
            raise RuntimeError(f"temporal fold result {name} is non-finite")


def _loss_kwargs(args: argparse.Namespace) -> dict[str, float]:
    names = (
        "posterior_nll_weight",
        "source_nll_weight",
        "crps_weight",
        "mae_weight",
        "divisor_weight",
        "residual_weight",
        "uncertainty_weight",
        "gate_weight",
        "action_regret_weight",
        "safe_gate_weight",
        "quality_weight",
        "spike_sparsity_weight",
        "max_residual_bpm",
    )
    return {name: float(getattr(args, name)) for name in names}


def _completed_fold_result(
    fold_dir: Path,
    *,
    experiment: TemporalAlignedExperiment,
    fold: int,
    split: FoldSplit,
    split_record: Mapping[str, Any],
    run_signature: str,
    source_binding_sha256: str,
) -> tuple[TemporalPredictionResult, bool, dict[str, Any]] | None:
    completion_path = fold_dir / "test_evaluation_manifest.json"
    prediction_path = fold_dir / "test_predictions.npz"
    report_path = fold_dir / "test_predictions.json"
    if not completion_path.is_file():
        return None
    if not (prediction_path.is_file() and report_path.is_file()):
        raise RuntimeError("test completion marker exists without complete test artifacts")
    completion = shared._read_strict_json(
        completion_path, label="temporal completed-fold manifest"
    )
    expected_context = {
        "outer_fold": int(fold),
        "run_signature": run_signature,
        "source_binding_sha256": source_binding_sha256,
        "split_sha256": _canonical_sha256(dict(split_record)),
        "test_row_authority_sha256": _fold_row_authority_sha256(
            experiment, split
        ),
    }
    for name, value in expected_context.items():
        if completion.get(name) != value:
            raise RuntimeError(f"completed outer-test artifact failed {name} binding")
    if type(completion.get("test_fold_evaluation_invocations")) is not int or (
        completion.get("test_fold_evaluation_invocations") != 1
    ):
        raise RuntimeError("outer test evaluation count is not exactly one")
    selection_lock_path = fold_dir / "selection_lock.json"
    if not selection_lock_path.is_file():
        raise RuntimeError("test completion marker lacks its validation selection lock")
    selection_binding, selection_payload = shared._snapshot_training_file_payload(
        selection_lock_path
    )
    prediction_binding, prediction_payload = shared._snapshot_training_file_payload(
        prediction_path
    )
    report_binding, report_payload = shared._snapshot_training_file_payload(
        report_path
    )
    expected_hashes = {
        "selection_lock_sha256": selection_binding.sha256,
        "test_predictions_npz_sha256": prediction_binding.sha256,
        "test_predictions_json_sha256": report_binding.sha256,
    }
    for name, value in expected_hashes.items():
        if completion.get(name) != value:
            raise RuntimeError(f"completed outer-test artifact failed {name} binding")
    selection_lock = shared._decode_strict_json_payload(
        selection_payload, label="temporal completed-fold selection lock"
    )
    checkpoint_path = fold_dir / "temporal_best.pt"
    if not checkpoint_path.is_file():
        raise RuntimeError("completed outer-test artifact lacks its best checkpoint")
    checkpoint_binding, checkpoint_payload = shared._snapshot_training_file_payload(
        checkpoint_path
    )
    checkpoint_sha256 = checkpoint_binding.sha256
    for name, value in {
        "outer_fold": int(fold),
        "run_signature": run_signature,
        "source_binding_sha256": source_binding_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }.items():
        if selection_lock.get(name) != value or completion.get(name) != value:
            raise RuntimeError(
                f"completed outer-test selection failed {name} binding"
            )
    checkpoint = torch.load(
        io.BytesIO(checkpoint_payload), map_location="cpu", weights_only=True
    )
    checkpoint_split = (
        checkpoint.get("split") if isinstance(checkpoint, Mapping) else None
    )
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("fold") != int(fold)
        or checkpoint.get("run_signature") != run_signature
        or checkpoint.get("source_binding_sha256") != source_binding_sha256
        or not isinstance(checkpoint_split, Mapping)
        or tuple(checkpoint_split.get("train_identities", ()))
        != tuple(split.train_identities)
        or tuple(checkpoint_split.get("validation_identities", ()))
        != tuple(split.validation_identities)
        or tuple(checkpoint_split.get("test_identities", ()))
        != tuple(split.test_identities)
    ):
        raise RuntimeError("completed outer-test checkpoint context mismatch")

    if prediction_binding.sha256 != completion.get("test_predictions_npz_sha256"):
        raise RuntimeError("completed outer-test prediction snapshot mismatch")
    test = _load_prediction_payload(prediction_payload)
    _validate_temporal_prediction_binding(
        test, experiment=experiment, fold=fold, split=split
    )
    report = shared._decode_strict_json_payload(
        report_payload, label="temporal completed-fold prediction report"
    )
    if (
        report.get("outer_fold") != int(fold)
        or report.get("run_signature") != run_signature
        or report.get("source_binding_sha256") != source_binding_sha256
        or report.get("checkpoint_sha256") != checkpoint_sha256
        or report.get("selection_lock_sha256")
        != completion.get("selection_lock_sha256")
        or report.get("split") != dict(split_record)
        or report.get("selection") != selection_lock.get("decision")
    ):
        raise RuntimeError("completed outer-test report context mismatch")
    promoted = bool(report["promoted"])
    selection_decision = selection_lock.get("decision")
    if (
        type(report.get("promoted")) is not bool
        or not isinstance(selection_decision, Mapping)
        or type(selection_decision.get("promoted")) is not bool
        or promoted != selection_decision.get("promoted")
        or selection_lock.get("locked_final_action")
        != ("candidate" if promoted else "base_only_fallback")
        or type(completion.get("test_rows_expected")) is not int
        or completion.get("test_rows_expected") != len(split.test)
        or type(completion.get("test_rows_evaluated")) is not int
        or completion.get("test_rows_evaluated") != len(test.position)
    ):
        raise RuntimeError("completed outer-test decision/row binding mismatch")
    return test, promoted, {
        "split": report.get("split"),
        "selection": report.get("selection"),
        "test": report,
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "test_reused_without_evaluation": True,
    }


def train_fold(
    args: argparse.Namespace,
    experiment: TemporalAlignedExperiment,
    fold: int,
    device: torch.device,
    run_signature: str,
) -> tuple[TemporalPredictionResult, bool, dict[str, Any]]:
    _assert_temporal_training_authority(
        experiment,
        historical_legacy_reproduction=bool(
            getattr(args, "historical_legacy_reproduction", False)
        ),
    )
    fold_dir = Path(args.output_dir) / f"fold_{fold}"
    split = shared.make_outer_split(experiment.metadata, fold)
    split_record = _fold_split_record(experiment, fold, split)
    source_binding_sha256 = _source_binding(experiment)
    completed = _completed_fold_result(
        fold_dir,
        experiment=experiment,
        fold=fold,
        split=split,
        split_record=split_record,
        run_signature=run_signature,
        source_binding_sha256=source_binding_sha256,
    )
    if completed is not None:
        if not args.resume:
            raise RuntimeError(
                f"fold {fold} outer test was already evaluated; pass --resume to reuse it"
            )
        return completed
    fold_dir.mkdir(parents=True, exist_ok=True)
    test_started_path = fold_dir / "test_evaluation_started.json"
    if test_started_path.exists():
        raise RuntimeError(
            "an outer-test evaluation was already started but did not atomically complete; "
            "use a new output directory rather than evaluating the test fold twice"
        )

    atomic_write_json(fold_dir / "split.json", split_record)

    normalizer = fit_temporal_signal_normalizer(
        experiment,
        split.train,
        max_samples_per_variant=args.normalizer_max_samples,
        clip_quantile=args.normalizer_clip_quantile,
        seed=args.seed + 1009 * fold,
    )
    action_calibration = fit_train_only_action_calibration(
        experiment.metadata, split.train
    )
    atomic_write_json(fold_dir / "signal_normalizer.json", normalizer.record())
    atomic_write_json(
        fold_dir / "train_only_action_calibration.json", action_calibration.record()
    )
    if normalizer.fit_positions_sha256 != _positions_digest(split.train):
        raise RuntimeError("normalizer scope does not bind the outer-training rows")
    if action_calibration.fit_positions_sha256 != _positions_digest(split.train):
        raise RuntimeError("action calibration scope does not bind the outer-training rows")

    train_loader = make_temporal_loader(
        experiment,
        split.train,
        normalizer,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        seed=args.seed + 1009 * fold,
        train=True,
        samples_per_epoch=args.samples_per_epoch,
        rr_bin_width=args.rr_balance_bin_width,
        rr_balance_power=args.rr_balance_power,
        tail_min_bpm=args.tail_min_bpm,
        tail_max_bpm=args.tail_max_bpm,
        tail_boost=args.tail_boost,
    )
    validation_loader = make_temporal_loader(
        experiment,
        split.validation,
        normalizer,
        batch_size=args.eval_batch_size,
        workers=args.workers,
        device=device,
        seed=args.seed + 1009 * fold + 1,
        train=False,
    )

    model_kwargs = _model_kwargs(args, experiment)
    model = TemporalSourceSeparatedRRSNN(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, args.patience // 3),
        min_lr=1.0e-6,
    )
    amp_scaler = torch.amp.GradScaler(
        device.type, enabled=args.amp and device.type == "cuda"
    )
    best_score = math.inf
    best_epoch = -1
    stale_epochs = 0
    start_epoch = 0
    last_path = fold_dir / "temporal_last.pt"
    best_path = fold_dir / "temporal_best.pt"
    resume_path = Path(args.resume_from) if args.resume_from else last_path
    if args.resume and resume_path.is_file():
        _resume_binding, resume_payload = shared._snapshot_training_file_payload(
            resume_path
        )
        checkpoint = torch.load(
            io.BytesIO(resume_payload), map_location=device, weights_only=True
        )
        _validate_checkpoint(
            checkpoint,
            fold=fold,
            split=split,
            run_signature=run_signature,
            source_binding_sha256=source_binding_sha256,
            normalizer=normalizer,
            action_calibration=action_calibration,
        )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        amp_scaler.load_state_dict(checkpoint["amp_scaler_state"])
        shared.restore_rng_state(checkpoint["rng_state"], train_loader)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_score = float(checkpoint["best_score"])
        stale_epochs = int(checkpoint["stale_epochs"])
    elif args.resume and args.resume_from:
        raise FileNotFoundError(resume_path)

    history_path = fold_dir / "history.jsonl"
    for epoch in range(start_epoch, args.epochs):
        training = train_one_epoch(
            model,
            train_loader,
            optimizer,
            amp_scaler,
            device,
            amp=args.amp,
            gradient_clip=args.gradient_clip,
            radar_dropout_p=args.radar_dropout,
            action_calibration=action_calibration,
            loss_kwargs=_loss_kwargs(args),
            max_batches=args.smoke_max_batches,
        )
        validation = predict_loader(
            model,
            validation_loader,
            device,
            amp=args.amp,
            max_batches=args.smoke_max_batches,
        )
        identities = (
            experiment.metadata.iloc[validation.position]["identity"].astype(str).to_numpy()
        )
        validation_metrics = shared.evaluation_snapshot(
            validation.target,
            validation.candidate_prediction,
            identities,
            high_min_bpm=args.tail_min_bpm,
            high_max_bpm=args.tail_max_bpm,
        )
        score = float(validation_metrics["identity_macro"]["macro_mae"])
        scheduler.step(score)
        improved = score < best_score - args.min_delta
        if improved:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        checkpoint = _checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            amp_scaler=amp_scaler,
            loader=train_loader,
            epoch=epoch,
            best_epoch=best_epoch,
            best_score=best_score,
            stale_epochs=stale_epochs,
            fold=fold,
            split=split,
            run_signature=run_signature,
            source_binding_sha256=source_binding_sha256,
            model_kwargs=model_kwargs,
            normalizer=normalizer,
            action_calibration=action_calibration,
        )
        shared.atomic_torch_save(checkpoint, last_path)
        if improved:
            shared.atomic_torch_save(checkpoint, best_path)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    _json_ready(
                        {
                            "epoch": epoch,
                            "train": training,
                            "validation": validation_metrics,
                            "score": score,
                            "best_score": best_score,
                            "best_epoch": best_epoch,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        }
                    ),
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
        if stale_epochs >= args.patience:
            break
    if not best_path.is_file():
        raise RuntimeError("no temporal best checkpoint was produced")
    best_binding, best_payload = shared._snapshot_training_file_payload(best_path)
    best = torch.load(
        io.BytesIO(best_payload), map_location=device, weights_only=True
    )
    _validate_checkpoint(
        best,
        fold=fold,
        split=split,
        run_signature=run_signature,
        source_binding_sha256=source_binding_sha256,
        normalizer=normalizer,
        action_calibration=action_calibration,
    )
    _assert_temporal_training_authority(
        experiment,
        historical_legacy_reproduction=bool(
            getattr(args, "historical_legacy_reproduction", False)
        ),
    )
    model.load_state_dict(best["model_state"])

    validation = predict_loader(
        model,
        validation_loader,
        device,
        amp=args.amp,
        max_batches=args.smoke_max_batches,
    )
    validation_identities = (
        experiment.metadata.iloc[validation.position]["identity"].astype(str).to_numpy()
    )
    selection = shared.promotion_decision(
        validation.target,
        validation.candidate_prediction,
        validation.base_prediction,
        validation_identities,
        minimum_macro_mae_improvement=args.promotion_min_improvement,
        noninferiority_tolerance=args.promotion_noninferiority_tolerance,
        high_min_bpm=args.tail_min_bpm,
        high_max_bpm=args.tail_max_bpm,
    )
    if args.smoke_max_batches is not None:
        selection["promoted"] = False
        selection["smoke_override"] = "promotion disabled for truncated smoke evaluation"
    promoted = bool(selection["promoted"])
    shared.atomic_save_npz(
        fold_dir / "validation_predictions.npz", **_prediction_arrays(validation)
    )
    atomic_write_json(
        fold_dir / "validation_predictions.json",
        _prediction_report(
            validation,
            experiment.metadata,
            tail_min_bpm=args.tail_min_bpm,
            tail_max_bpm=args.tail_max_bpm,
        ),
    )
    selection_lock = {
        "lock_created_utc": datetime.now(timezone.utc).isoformat(),
        "outer_fold": int(fold),
        "best_epoch": int(best["best_epoch"]),
        "best_validation_macro_mae": float(best["best_score"]),
        "run_signature": run_signature,
        "checkpoint_sha256": best_binding.sha256,
        "source_binding_sha256": source_binding_sha256,
        "signal_normalizer_signature": _normalizer_signature(normalizer),
        "action_calibration_signature": _action_signature(action_calibration),
        "decision": selection,
        "locked_final_action": "candidate" if promoted else "base_only_fallback",
        "test_loader_constructed": False,
        "test_predictions_generated": False,
        "test_labels_or_metrics_used_for_selection": False,
        "maximum_permitted_test_evaluations": 1,
    }
    atomic_write_json(fold_dir / "selection_lock.json", selection_lock)
    selection_lock_sha256 = shared.sha256_file(fold_dir / "selection_lock.json")

    # From here onward test evaluation is a one-way state transition.  The
    # durable marker is written before even constructing the test loader.  A
    # crash refuses a second look and requires a fresh output directory.
    atomic_write_json(
        test_started_path,
        {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "outer_fold": int(fold),
            "selection_lock_sha256": selection_lock_sha256,
            "test_fold_evaluation_invocation": 1,
            "test_loader_constructed_after_this_marker": True,
        },
    )
    test_loader = make_temporal_loader(
        experiment,
        split.test,
        normalizer,
        batch_size=args.eval_batch_size,
        workers=args.workers,
        device=device,
        seed=args.seed + 1009 * fold + 2,
        train=False,
    )
    test = predict_loader(
        model,
        test_loader,
        device,
        amp=args.amp,
        max_batches=args.smoke_max_batches,
    )
    _assert_temporal_training_authority(
        experiment,
        historical_legacy_reproduction=bool(
            getattr(args, "historical_legacy_reproduction", False)
        ),
    )
    shared.atomic_save_npz(
        fold_dir / "test_predictions.npz",
        **_prediction_arrays(test, promoted=promoted),
    )
    test_report = _prediction_report(
        test,
        experiment.metadata,
        promoted=promoted,
        tail_min_bpm=args.tail_min_bpm,
        tail_max_bpm=args.tail_max_bpm,
    )
    test_report.update(
        {
            "outer_fold": int(fold),
            "run_signature": run_signature,
            "source_binding_sha256": source_binding_sha256,
            "selection_lock_sha256": selection_lock_sha256,
            "selection": selection,
            "split": split_record,
            "checkpoint_sha256": best_binding.sha256,
            "candidate_saved_even_when_rejected": True,
            "test_evaluated_once_after_validation_lock": True,
        }
    )
    atomic_write_json(fold_dir / "test_predictions.json", test_report)
    atomic_write_json(
        fold_dir / "test_evaluation_manifest.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "outer_fold": int(fold),
            "run_signature": run_signature,
            "source_binding_sha256": source_binding_sha256,
            "split_sha256": _canonical_sha256(split_record),
            "test_row_authority_sha256": _fold_row_authority_sha256(
                experiment, split
            ),
            "checkpoint_sha256": best_binding.sha256,
            "test_fold_evaluation_invocations": 1,
            "test_rows_expected": int(len(split.test)),
            "test_rows_evaluated": int(len(test.position)),
            "complete_outer_test_evaluation": bool(
                args.smoke_max_batches is None and len(test.position) == len(split.test)
            ),
            "selection_lock_sha256": selection_lock_sha256,
            "test_predictions_npz_sha256": shared.sha256_file(
                fold_dir / "test_predictions.npz"
            ),
            "test_predictions_json_sha256": shared.sha256_file(
                fold_dir / "test_predictions.json"
            ),
            "validation_selection_completed_before_test_loader_construction": True,
            "test_metrics_used_for_model_or_action_selection": False,
        },
    )
    return test, promoted, {
        "split": split_record,
        "selection": selection,
        "test": test_report,
        "checkpoint_sha256": best_binding.sha256,
        "test_reused_without_evaluation": False,
    }


def _build_run_config(
    args: argparse.Namespace, experiment: TemporalAlignedExperiment
) -> dict[str, Any]:
    arguments = {
        key: value
        for key, value in vars(args).items()
        if key not in {"resume", "resume_from"}
    }
    sources = (
        *shared._training_source_paths(),
        Path(__file__).resolve(),
        SOURCE_ROOT / "snn_rr" / "svd_temporal_models.py",
    )
    config = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim_classification": (
            "retrospective_scientific_noncommercial"
            if experiment.training_authority_receipt is not None
            and experiment.training_authority_receipt.acquisition_v2
            else "historical_noncommercial_reproduction"
        ),
        "commercial_claim_allowed": False,
        "arguments": arguments,
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): shared.sha256_file(path)
            for path in sources
        },
        "data_provenance": experiment.provenance,
        "split_protocol": {
            "outer_test": "f",
            "validation": "(f+1)%6",
            "weight_fit": "remaining four frozen identity folds",
            "normalization_and_action_thresholds": "outer-training identities only",
            "promotion": "validation identities only",
            "test": "constructed and evaluated once only after durable selection lock",
        },
        "test_evaluation_contract": {
            "maximum_evaluations_per_outer_fold": 1,
            "repeated_resume_behavior": "reuse saved test artifact without model inference",
            "incomplete_started_evaluation_behavior": "refuse a second evaluation",
        },
        "component_signals_input": True,
        "chronological_spike_time": True,
        "raw_to_component_prefix_causal": False,
        "prediction_time_context": "past_complete_32s_window",
    }
    signature_payload = {
        "arguments": arguments,
        "source_sha256": config["source_sha256"],
        "data_provenance": experiment.provenance,
        "split_protocol": config["split_protocol"],
        "test_evaluation_contract": config["test_evaluation_contract"],
    }
    config["run_signature"] = shared.stable_signature(signature_payload)
    return config


def _metric_bundle(
    target: np.ndarray,
    prediction: np.ndarray,
    identities: np.ndarray,
    folds: np.ndarray,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    tail_min_bpm: float,
    tail_max_bpm: float,
) -> dict[str, Any]:
    result = shared.grouped_oof_metrics(
        target,
        prediction,
        identities,
        fold_ids=folds,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    result["tail_25_35"] = shared.evaluation_snapshot(
        target,
        prediction,
        identities,
        high_min_bpm=tail_min_bpm,
        high_max_bpm=tail_max_bpm,
    )["high_25_35"]
    return result


def write_oof(
    args: argparse.Namespace,
    experiment: TemporalAlignedExperiment,
    fold_results: Mapping[
        int, tuple[TemporalPredictionResult, bool, dict[str, Any]]
    ],
    run_signature: str,
) -> dict[str, Any]:
    _assert_temporal_training_authority(
        experiment,
        historical_legacy_reproduction=bool(
            getattr(args, "historical_legacy_reproduction", False)
        ),
    )
    if not isinstance(run_signature, str) or not run_signature:
        raise RuntimeError("temporal OOF publication requires a run signature")
    count = len(experiment.metadata)
    base = pd.to_numeric(
        experiment.metadata["prediction_bpm"], errors="raise"
    ).to_numpy(np.float32)
    candidate = np.full(count, np.nan, dtype=np.float32)
    final = np.full(count, np.nan, dtype=np.float32)
    rr_std = np.full(count, np.nan, dtype=np.float32)
    source = np.full(count, np.nan, dtype=np.float32)
    gate = np.full(count, np.nan, dtype=np.float32)
    entropy = np.full(count, np.nan, dtype=np.float32)
    divisor = np.full((count, 4), np.nan, dtype=np.float32)
    residual = np.full(count, np.nan, dtype=np.float32)
    candidate_std = np.full((count, 4), np.nan, dtype=np.float32)
    quality = np.full(count, np.nan, dtype=np.float32)
    spike = np.full(count, np.nan, dtype=np.float32)
    radar_weights = np.full((count, 3), np.nan, dtype=np.float32)
    posterior = np.full((count, 157), np.nan, dtype=np.float16)
    promoted = np.zeros(count, dtype=bool)
    evaluated = np.zeros(count, dtype=bool)
    for fold, value in fold_results.items():
        if (
            type(fold) is not int
            or not 0 <= fold < N_FOLDS
            or not isinstance(value, tuple)
            or len(value) != 3
        ):
            raise RuntimeError("temporal OOF fold result catalogue is invalid")
        result, did_promote, report = value
        if type(did_promote) is not bool or not isinstance(report, Mapping):
            raise RuntimeError("temporal OOF fold decision/report is invalid")
        split = shared.make_outer_split(experiment.metadata, fold)
        _validate_temporal_prediction_binding(
            result, experiment=experiment, fold=fold, split=split
        )
        test_report = report.get("test")
        if (
            not isinstance(test_report, Mapping)
            or report.get("split") != _fold_split_record(experiment, fold, split)
            or test_report.get("run_signature") != run_signature
            or test_report.get("outer_fold") != fold
        ):
            raise RuntimeError("temporal OOF fold report context mismatch")
        position = np.asarray(result.position, dtype=np.int64)
        if evaluated[position].any():
            raise RuntimeError("OOF row was predicted by more than one outer model")
        candidate[position] = result.candidate_prediction
        final[position] = (
            result.candidate_prediction if did_promote else result.base_prediction
        )
        rr_std[position] = result.rr_std
        source[position] = result.source_prediction
        gate[position] = result.mixture_gate
        entropy[position] = result.posterior_entropy
        divisor[position] = result.divisor_probabilities
        residual[position] = result.residual_rr
        candidate_std[position] = result.candidate_std
        quality[position] = result.quality
        spike[position] = result.spike_rate
        radar_weights[position] = result.radar_weights
        if result.posterior_probability.shape[1] != posterior.shape[1]:
            raise RuntimeError("OOF posterior RR grid changed across folds")
        posterior[position] = result.posterior_probability.astype(np.float16)
        promoted[position] = did_promote
        evaluated[position] = True
    positions = np.flatnonzero(evaluated)
    if not len(positions):
        raise RuntimeError("no outer fold was evaluated")
    metadata = experiment.metadata
    target = pd.to_numeric(metadata["rr_bpm"], errors="raise").to_numpy(np.float32)
    folds = metadata["fold"].to_numpy(np.int16)
    identities = metadata["identity"].astype(str).to_numpy()
    metric_kwargs = {
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "tail_min_bpm": args.tail_min_bpm,
        "tail_max_bpm": args.tail_max_bpm,
    }
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_signature": run_signature,
        "complete_six_fold_oof": bool(evaluated.all()),
        "evaluated_rows": int(evaluated.sum()),
        "expected_rows": int(count),
        "evaluated_folds": sorted(fold_results),
        "promoted_folds": sorted(
            fold for fold, value in fold_results.items() if value[1]
        ),
        "locked_final": _metric_bundle(
            target[positions],
            final[positions],
            identities[positions],
            folds[positions],
            **metric_kwargs,
        ),
        "candidate": _metric_bundle(
            target[positions],
            candidate[positions],
            identities[positions],
            folds[positions],
            **metric_kwargs,
        ),
        "base": _metric_bundle(
            target[positions],
            base[positions],
            identities[positions],
            folds[positions],
            **metric_kwargs,
        ),
        "fold_reports": {str(fold): value[2] for fold, value in fold_results.items()},
        "commercial_safety_policy": "validation-rejected folds are exact base fallback",
        "test_evaluation_policy": "exactly one inference pass per evaluated outer fold",
    }
    _assert_temporal_training_authority(
        experiment,
        historical_legacy_reproduction=bool(
            getattr(args, "historical_legacy_reproduction", False)
        ),
    )
    output_dir = Path(args.output_dir)
    shared.atomic_save_npz(
        output_dir / "temporal_oof.npz",
        index=metadata["cache_index"].to_numpy(np.int64),
        target=target,
        fold=folds,
        evaluated=evaluated,
        prediction_base=base,
        prediction_candidate=candidate,
        prediction_final=final,
        rr_std=rr_std,
        source_prediction=source,
        mixture_gate=gate,
        posterior_entropy=entropy,
        divisor_probabilities=divisor,
        residual_rr=residual,
        candidate_std=candidate_std,
        quality=quality,
        spike_rate=spike,
        radar_weights=radar_weights,
        posterior_probability=posterior,
        promoted=promoted,
        run_signature=np.asarray(run_signature),
    )
    table = metadata.loc[:, list(shared.BINDING_COLUMNS) + ["fold"]].copy()
    table["prediction_base_bpm"] = base
    table["prediction_candidate_bpm"] = candidate
    table["prediction_locked_final_bpm"] = final
    table["candidate_rr_std_bpm"] = rr_std
    table["source_prediction_bpm"] = source
    table["mixture_gate"] = gate
    table["posterior_entropy"] = entropy
    for index in range(4):
        table[f"divisor_probability_x{index + 1}"] = divisor[:, index]
        table[f"candidate_std_x{index + 1}_bpm"] = candidate_std[:, index]
    table["residual_rr_bpm"] = residual
    table["quality"] = quality
    table["spike_rate"] = spike
    table["promoted"] = promoted
    table["evaluated"] = evaluated
    temporary = output_dir / "temporal_oof.csv.tmp"
    table.to_csv(temporary, index=False)
    temporary.replace(output_dir / "temporal_oof.csv")
    atomic_write_json(output_dir / "metrics.json", metrics)
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--svd-cache",
        type=Path,
        default=PROJECT_ROOT / "artifacts/cache/svd_components_v1",
    )
    parser.add_argument(
        "--base-oof-csv",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/runs/ensemble_structured_exact/ensemble_oof.csv",
    )
    parser.add_argument(
        "--base-oof-npz",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/runs/ensemble_structured_exact/ensemble_oof.npz",
    )
    parser.add_argument(
        "--base-oof-provenance",
        type=Path,
        help=(
            "authority JSON for scientific acquisition-v2 base predictions; "
            "defaults only to provenance.json beside --base-oof-csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/runs/svd_temporal_snn",
    )
    parser.add_argument("--fold", default="all")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--verify-file-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "verify legacy-cache payload hashes when available; acquisition-v2 "
            "authority hashes are always mandatory"
        ),
    )
    parser.add_argument(
        "--historical-legacy-reproduction",
        action="store_true",
        help=(
            "explicitly authorize noncommercial reproduction from a non-v2 "
            "historical cache; never grants acquisition-v2 authority"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--preset", choices=("tiny", "compact", "full"), default="compact")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--smoke-max-batches", type=int)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=2.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=2.0)
    parser.add_argument("--compressor-channels", type=int, default=24)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--cell-types", default="lif,plif,alif")
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--radar-dropout", type=float, default=0.15)
    parser.add_argument("--max-residual-bpm", type=float, default=1.5)
    parser.add_argument("--initial-gate-bias", type=float, default=-8.0)
    parser.add_argument("--normalizer-max-samples", type=int, default=200_000)
    parser.add_argument("--normalizer-clip-quantile", type=float, default=0.999)
    parser.add_argument("--posterior-nll-weight", type=float, default=1.0)
    parser.add_argument("--source-nll-weight", type=float, default=0.35)
    parser.add_argument("--crps-weight", type=float, default=0.15)
    parser.add_argument("--mae-weight", type=float, default=0.40)
    parser.add_argument("--divisor-weight", type=float, default=0.25)
    parser.add_argument("--residual-weight", type=float, default=0.15)
    parser.add_argument("--uncertainty-weight", type=float, default=0.08)
    parser.add_argument("--gate-weight", type=float, default=0.15)
    parser.add_argument("--action-regret-weight", type=float, default=0.20)
    parser.add_argument("--safe-gate-weight", type=float, default=0.25)
    parser.add_argument("--quality-weight", type=float, default=0.03)
    parser.add_argument("--spike-sparsity-weight", type=float, default=5.0e-4)
    parser.add_argument("--rr-balance-bin-width", type=float, default=3.0)
    parser.add_argument("--rr-balance-power", type=float, default=0.65)
    parser.add_argument("--tail-min-bpm", type=float, default=25.0)
    parser.add_argument("--tail-max-bpm", type=float, default=35.0)
    parser.add_argument("--tail-boost", type=float, default=2.0)
    parser.add_argument("--promotion-min-improvement", type=float, default=0.05)
    parser.add_argument("--promotion-noninferiority-tolerance", type=float, default=0.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(argv)
    positive = (
        args.epochs,
        args.patience,
        args.batch_size,
        args.eval_batch_size,
        args.bootstrap_samples,
        args.normalizer_max_samples,
    )
    if any(value < 1 for value in positive):
        parser.error("epochs, patience, batch sizes, bootstrap and sample limits must be positive")
    if args.smoke_max_batches is not None and args.smoke_max_batches < 1:
        parser.error("--smoke-max-batches must be positive")
    if args.resume_from is not None and args.fold == "all":
        parser.error("--resume-from requires a single --fold")
    if not 0.0 <= args.radar_dropout <= 1.0:
        parser.error("--radar-dropout must be in [0, 1]")
    if not args.cell_types.strip():
        parser.error("--cell-types cannot be empty")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    experiment = load_temporal_aligned_experiment(
        args.svd_cache,
        args.base_oof_csv,
        args.base_oof_npz,
        base_oof_provenance=args.base_oof_provenance,
        verify_file_hashes=args.verify_file_hashes,
        historical_legacy_reproduction=args.historical_legacy_reproduction,
    )
    _assert_temporal_training_authority(
        experiment,
        historical_legacy_reproduction=args.historical_legacy_reproduction,
    )
    if (
        experiment.training_authority_receipt is not None
        and not experiment.training_authority_receipt.acquisition_v2
    ):
        output_dir = Path(args.output_dir)
        if output_dir.exists() and not args.resume:
            raise RuntimeError(
                "historical legacy reproduction requires a new, non-existing "
                "versioned output directory"
            )
        if args.resume and not (output_dir / "run_config.json").is_file():
            raise RuntimeError(
                "historical legacy resume requires its existing run_config.json"
            )
    run_config = _build_run_config(args, experiment)
    existing_run_config_path = Path(args.output_dir) / "run_config.json"
    if args.resume and existing_run_config_path.is_file():
        existing_run_config = shared._read_strict_json(
            existing_run_config_path, label="temporal SVD resume run config"
        )
        if existing_run_config.get("run_signature") != run_config.get(
            "run_signature"
        ):
            raise RuntimeError(
                "temporal SVD resume run signature differs from output root"
            )
    _assert_temporal_training_authority(
        experiment,
        historical_legacy_reproduction=args.historical_legacy_reproduction,
    )

    # The complete shared/temporal authority replay precedes RNG, device
    # discovery, output creation, normalization, and model construction.
    shared.seed_everything(args.seed, deterministic=args.deterministic)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if not existing_run_config_path.is_file():
        atomic_write_json(existing_run_config_path, run_config)
    folds = shared.parse_fold_selection(args.fold)
    results: dict[
        int, tuple[TemporalPredictionResult, bool, dict[str, Any]]
    ] = {}
    for fold in folds:
        shared.seed_everything(
            args.seed + 1009 * fold, deterministic=args.deterministic
        )
        results[fold] = train_fold(
            args, experiment, fold, device, str(run_config["run_signature"])
        )
    metrics = write_oof(
        args, experiment, results, str(run_config["run_signature"])
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).resolve()),
                "run_signature": run_config["run_signature"],
                "evaluated_folds": folds,
                "promoted_folds": metrics["promoted_folds"],
                "locked_final_mae": metrics["locked_final"]["overall"]["mae"],
                "locked_final_macro_mae": metrics["locked_final"]["identity_macro"][
                    "macro_mae"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
