"""Fail-closed authority for externally prescribed identity splits.

The ordinary training entry point derives rotating folds from the cache.  A
nested or prospective experiment sometimes needs an explicitly prescribed
train/validation/prediction partition instead.  This module validates that
partition against immutable cache and fold-assignment artifacts before it is
allowed to produce row indices.

The manifest itself is content-addressed.  ``content_sha256`` is the SHA-256
of canonical JSON after removing only that field.  Referenced JSON artifacts
are additionally bound by their exact file SHA-256, so formatting or content
changes both fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


IDENTITY_ROLES = ("train", "validation", "prediction", "excluded")
SCHEMA_VERSION = 1


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a JSON-compatible value with one platform-independent form."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_content_sha256(document: Mapping[str, Any]) -> str:
    """Hash a split manifest after excluding only its self-hash field."""

    payload = dict(document)
    payload.pop("content_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase hexadecimal SHA-256")
    return digest


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _identity_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"identities.{label} must be a JSON array")
    identities: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip() or item != item.strip():
            raise ValueError(
                f"identities.{label} entries must be non-empty, trimmed strings"
            )
        identities.append(item)
    if len(set(identities)) != len(identities):
        raise ValueError(f"identities.{label} contains duplicates")
    if identities != sorted(identities):
        raise ValueError(f"identities.{label} must be sorted canonically")
    return tuple(identities)


def _resolve_reference(path_value: Any, manifest_path: Path, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label}.path must be a non-empty string")
    referenced = Path(path_value).expanduser()
    if not referenced.is_absolute():
        referenced = manifest_path.parent / referenced
    resolved = referenced.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} artifact is missing: {resolved}")
    return resolved


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON number {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path} ({exc})") from exc
    return _require_mapping(value, label)


def _verify_reference(
    binding: Mapping[str, Any], manifest_path: Path, label: str
) -> tuple[Path, str, Mapping[str, Any]]:
    path = _resolve_reference(binding.get("path"), manifest_path, label)
    expected = _require_sha256(binding.get("sha256"), f"{label}.sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {actual}"
        )
    return path, actual, _read_json(path, label)


def _fold_identity_map(document: Mapping[str, Any]) -> dict[str, int]:
    raw: Any = document.get("identity_to_fold", document)
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("fold_assignments must contain a non-empty identity mapping")
    result: dict[str, int] = {}
    for identity, fold in raw.items():
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("fold assignment identities must be non-empty strings")
        if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
            raise ValueError(f"fold assignment for {identity!r} must be a non-negative integer")
        result[identity] = int(fold)
    return result


def _session_identity_map(
    metadata: pd.DataFrame, cache_manifest: Mapping[str, Any]
) -> dict[str, str]:
    required = {"session_id", "identity", "reference_valid"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"cache metadata lacks split columns: {missing}")
    if metadata.empty:
        raise ValueError("cache metadata is empty")
    if metadata[["session_id", "identity"]].isna().any().any():
        raise ValueError("cache session_id/identity contains missing values")

    session_ids = metadata["session_id"].astype(str)
    identities = metadata["identity"].astype(str)
    result: dict[str, str] = {}
    for session_id, rows in metadata.assign(
        _session=session_ids, _identity=identities
    ).groupby("_session", sort=False):
        unique = rows["_identity"].unique().tolist()
        if len(unique) != 1:
            raise ValueError(
                f"cache session {session_id!r} crosses identities: {sorted(unique)}"
            )
        result[str(session_id)] = str(unique[0])

    raw_sessions = cache_manifest.get("sessions")
    if not isinstance(raw_sessions, list):
        raise ValueError("cache manifest sessions must be an array")
    available: list[str] = []
    for item in raw_sessions:
        if not isinstance(item, Mapping):
            raise ValueError("cache manifest session entries must be objects")
        if item.get("status") == "ok":
            session_id = item.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise ValueError("cache manifest has an invalid successful session_id")
            available.append(session_id)
    if len(set(available)) != len(available):
        raise ValueError("cache manifest repeats a successful session_id")
    if set(available) != set(result):
        missing_metadata = sorted(set(available) - set(result))
        unknown_metadata = sorted(set(result) - set(available))
        raise ValueError(
            "cache manifest/metadata session cover mismatch "
            f"(missing_metadata={missing_metadata}, unknown_metadata={unknown_metadata})"
        )
    return result


@dataclass(frozen=True, slots=True)
class ExplicitIdentitySplit:
    """Validated row indices for one externally prescribed split."""

    train_index: np.ndarray
    validation_index: np.ndarray
    prediction_index: np.ndarray
    split: dict[str, list[str]]


@dataclass(frozen=True, slots=True)
class IdentitySplitAuthority:
    """An immutable manifest validated against the current feature cache."""

    manifest_path: Path
    manifest_file_sha256: str
    content_sha256: str
    fold_id: int
    train_identities: tuple[str, ...]
    validation_identities: tuple[str, ...]
    prediction_identities: tuple[str, ...]
    excluded_identities: tuple[str, ...]
    scaler_identities: tuple[str, ...]
    fold_assignments_path: Path
    fold_assignments_sha256: str
    cache_manifest_path: Path
    cache_manifest_sha256: str
    identity_to_fold: Mapping[str, int]
    session_to_identity: Mapping[str, str]

    def checkpoint_provenance(self) -> dict[str, Any]:
        """Small, stable binding embedded in run and checkpoint artifacts."""

        return {
            "mode": "custom_identity_split",
            "schema_version": SCHEMA_VERSION,
            "fold_id": self.fold_id,
            "split_manifest_content_sha256": self.content_sha256,
            "split_manifest_file_sha256": self.manifest_file_sha256,
            "fold_assignments_sha256": self.fold_assignments_sha256,
            "cache_manifest_sha256": self.cache_manifest_sha256,
            "train_identities": list(self.train_identities),
            "validation_identities": list(self.validation_identities),
            "prediction_identities": list(self.prediction_identities),
            "excluded_identities": list(self.excluded_identities),
            "scaler_identities": list(self.scaler_identities),
        }

    def run_provenance(self) -> dict[str, Any]:
        value = self.checkpoint_provenance()
        value.update(
            split_manifest_path=str(self.manifest_path),
            fold_assignments_path=str(self.fold_assignments_path),
            cache_manifest_path=str(self.cache_manifest_path),
            session_to_identity=dict(sorted(self.session_to_identity.items())),
        )
        return value

    def explicit_indices(
        self, metadata: pd.DataFrame, *, include_invalid: bool
    ) -> ExplicitIdentitySplit:
        """Build indices without consulting rotating-fold helpers."""

        identities = metadata["identity"].astype(str).to_numpy()
        valid = metadata["reference_valid"].to_numpy(dtype=bool)
        train_mask = np.isin(identities, self.train_identities)
        if not include_invalid:
            train_mask &= valid
        validation_mask = np.isin(identities, self.validation_identities) & valid
        prediction_mask = np.isin(identities, self.prediction_identities) & valid
        excluded_mask = np.isin(identities, self.excluded_identities)

        train = np.flatnonzero(train_mask)
        validation = np.flatnonzero(validation_mask)
        prediction = np.flatnonzero(prediction_mask)
        if min(len(train), len(validation), len(prediction)) == 0:
            raise ValueError("custom train/validation/prediction row split is empty")
        if (
            np.intersect1d(train, validation).size
            or np.intersect1d(train, prediction).size
            or np.intersect1d(validation, prediction).size
            or np.intersect1d(train, np.flatnonzero(excluded_mask)).size
            or np.intersect1d(validation, np.flatnonzero(excluded_mask)).size
            or np.intersect1d(prediction, np.flatnonzero(excluded_mask)).size
        ):
            raise RuntimeError("custom split row leakage detected")

        observed_train = set(identities[train].tolist())
        observed_validation = set(identities[validation].tolist())
        observed_prediction = set(identities[prediction].tolist())
        if observed_train != set(self.train_identities):
            raise ValueError("not every custom train identity contributes a selected row")
        if observed_validation != set(self.validation_identities):
            raise ValueError("not every custom validation identity has a valid-reference row")
        if observed_prediction != set(self.prediction_identities):
            raise ValueError("not every custom prediction identity has a valid-reference row")
        self.validate_scaler_indices(metadata, train)
        return ExplicitIdentitySplit(
            train_index=train,
            validation_index=validation,
            prediction_index=prediction,
            split={
                "train_identities": list(self.train_identities),
                "validation_identities": list(self.validation_identities),
                "prediction_identities": list(self.prediction_identities),
                "excluded_identities": list(self.excluded_identities),
                "scaler_identities": list(self.scaler_identities),
            },
        )

    def validate_scaler_indices(
        self, metadata: pd.DataFrame, indices: Sequence[int] | np.ndarray
    ) -> None:
        positions = np.asarray(indices, dtype=np.int64)
        if positions.ndim != 1 or len(positions) == 0:
            raise ValueError("scaler indices must be a non-empty vector")
        if positions.min() < 0 or positions.max() >= len(metadata):
            raise ValueError("scaler indices are outside cache metadata")
        actual = set(metadata.iloc[positions]["identity"].astype(str).tolist())
        expected = set(self.scaler_identities)
        if actual != expected:
            raise ValueError(
                "scaler identity binding mismatch "
                f"(expected={sorted(expected)}, actual={sorted(actual)})"
            )
        if actual & set(self.excluded_identities):
            raise RuntimeError("excluded identity reached auxiliary scaler")


def load_identity_split_authority(
    path: str | Path,
    *,
    metadata: pd.DataFrame,
    cache_dir: str | Path,
) -> IdentitySplitAuthority:
    """Load and validate a version-1 custom identity split manifest."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"identity split manifest is missing: {manifest_path}")
    document = _read_json(manifest_path, "identity split manifest")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"identity split schema_version must equal {SCHEMA_VERSION}"
        )
    expected_content = _require_sha256(
        document.get("content_sha256"), "content_sha256"
    )
    observed_content = canonical_content_sha256(document)
    if observed_content != expected_content:
        raise ValueError(
            "identity split manifest canonical content SHA-256 mismatch: "
            f"expected {expected_content}, observed {observed_content}"
        )

    fold_id = document.get("fold_id")
    if isinstance(fold_id, bool) or not isinstance(fold_id, int) or fold_id < 0:
        raise ValueError("fold_id must be a non-negative integer")
    identities_document = _require_mapping(document.get("identities"), "identities")
    role_values = {
        role: _identity_tuple(identities_document.get(role), role)
        for role in IDENTITY_ROLES
    }
    scaler_identities = _identity_tuple(
        identities_document.get("scaler"), "scaler"
    )
    if not role_values["train"] or not role_values["validation"] or not role_values["prediction"]:
        raise ValueError("train, validation and prediction identities must be non-empty")
    if set(scaler_identities) != set(role_values["train"]):
        raise ValueError("scaler identities must exactly equal train identities")
    for left_index, left in enumerate(IDENTITY_ROLES):
        for right in IDENTITY_ROLES[left_index + 1 :]:
            overlap = set(role_values[left]) & set(role_values[right])
            if overlap:
                raise ValueError(
                    f"identity roles {left}/{right} overlap: {sorted(overlap)}"
                )

    fold_binding = _require_mapping(
        document.get("fold_assignments"), "fold_assignments"
    )
    fold_path, fold_sha, fold_document = _verify_reference(
        fold_binding, manifest_path, "fold_assignments"
    )
    identity_to_fold = _fold_identity_map(fold_document)

    cache_binding = _require_mapping(document.get("cache"), "cache")
    # ``manifest_path`` is the descriptive field requested by the schema;
    # normalize it into the generic path verifier without accepting aliases.
    if "manifest_path" not in cache_binding or "manifest_sha256" not in cache_binding:
        raise ValueError("cache must contain manifest_path and manifest_sha256")
    cache_reference = {
        "path": cache_binding["manifest_path"],
        "sha256": cache_binding["manifest_sha256"],
    }
    cache_path, cache_sha, cache_document = _verify_reference(
        cache_reference, manifest_path, "cache"
    )
    expected_cache_path = (Path(cache_dir).expanduser().resolve() / "manifest.json")
    if cache_path != expected_cache_path:
        raise ValueError(
            "identity split cache path does not match --cache-dir "
            f"({cache_path} != {expected_cache_path})"
        )

    session_to_identity = _session_identity_map(metadata, cache_document)
    metadata_identities = set(metadata["identity"].astype(str).tolist())
    canonical_identities = set(identity_to_fold)
    if metadata_identities != canonical_identities:
        raise ValueError(
            "fold assignments/cache identity cover mismatch "
            f"(cache_only={sorted(metadata_identities - canonical_identities)}, "
            f"fold_only={sorted(canonical_identities - metadata_identities)})"
        )
    partition = set().union(*(set(role_values[role]) for role in IDENTITY_ROLES))
    if partition != canonical_identities:
        raise ValueError(
            "custom identity roles do not exactly cover canonical identities "
            f"(missing={sorted(canonical_identities - partition)}, "
            f"unknown={sorted(partition - canonical_identities)})"
        )

    return IdentitySplitAuthority(
        manifest_path=manifest_path,
        manifest_file_sha256=sha256_file(manifest_path),
        content_sha256=observed_content,
        fold_id=int(fold_id),
        train_identities=role_values["train"],
        validation_identities=role_values["validation"],
        prediction_identities=role_values["prediction"],
        excluded_identities=role_values["excluded"],
        scaler_identities=scaler_identities,
        fold_assignments_path=fold_path,
        fold_assignments_sha256=fold_sha,
        cache_manifest_path=cache_path,
        cache_manifest_sha256=cache_sha,
        identity_to_fold=identity_to_fold,
        session_to_identity=session_to_identity,
    )
