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
import io
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import weakref

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


def _parse_json_bytes(payload: bytes, path: Path, label: str) -> Mapping[str, Any]:
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
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path} ({exc})") from exc
    return _require_mapping(value, label)


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_regular_file_snapshot(
    path: Path, label: str
) -> tuple[bytes, str, int]:
    """Read one unaliased regular inode and bind the consumed bytes.

    The path/open/fd/path signature checks reject symlinks, hard-link aliases,
    replacement, and in-place mutation during capture.  Callers parse only the
    returned private byte string, never the namespace path a second time.
    """

    source = path.expanduser().absolute()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        before_path = os.stat(source, follow_symlinks=False)
        descriptor = os.open(source, flags)
        before_fd = os.fstat(descriptor)
        if not (
            stat.S_ISREG(before_path.st_mode)
            and stat.S_ISREG(before_fd.st_mode)
            and before_path.st_nlink == before_fd.st_nlink == 1
            and (before_path.st_dev, before_path.st_ino)
            == (before_fd.st_dev, before_fd.st_ino)
        ):
            raise ValueError(f"{label} must be an unaliased regular file: {source}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        payload = b"".join(chunks)
        after_fd = os.fstat(descriptor)
        after_path = os.stat(source, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{label} cannot be snapshotted: {source} ({exc})") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    signature = _stat_signature(before_fd)
    if not (
        signature == _stat_signature(after_fd) == _stat_signature(after_path)
        and before_fd.st_size == len(payload)
    ):
        raise ValueError(f"{label} changed while being snapshotted: {source}")
    return payload, digest.hexdigest(), len(payload)


def _read_regular_json_snapshot(
    path: Path, label: str
) -> tuple[Mapping[str, Any], str]:
    """Parse and hash the exact same regular-file bytes from one open inode."""

    payload, digest, _ = _read_regular_file_snapshot(path, label)
    return _parse_json_bytes(payload, path, label), digest


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    """Compatibility wrapper returning a one-inode JSON snapshot."""

    return _read_regular_json_snapshot(path, label)[0]


def _verify_reference(
    binding: Mapping[str, Any], manifest_path: Path, label: str
) -> tuple[Path, str, Mapping[str, Any]]:
    path = _resolve_reference(binding.get("path"), manifest_path, label)
    expected = _require_sha256(binding.get("sha256"), f"{label}.sha256")
    document, actual = _read_regular_json_snapshot(path, label)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {actual}"
        )
    return path, actual, document


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


def _successful_session_ids(cache_manifest: Mapping[str, Any]) -> tuple[str, ...]:
    raw_sessions = cache_manifest.get("sessions")
    if not isinstance(raw_sessions, list):
        raise ValueError("cache manifest sessions must be an array")
    available: list[str] = []
    for item in raw_sessions:
        if not isinstance(item, Mapping):
            raise ValueError("cache manifest session entries must be objects")
        if item.get("status") == "ok":
            session_id = item.get("session_id")
            if (
                not isinstance(session_id, str)
                or not session_id
                or session_id != session_id.strip()
                or len(Path(session_id).parts) != 1
                or session_id in {".", ".."}
            ):
                raise ValueError("cache manifest has an invalid successful session_id")
            available.append(session_id)
    if not available:
        raise ValueError("cache manifest has no successful sessions")
    if len(set(available)) != len(available):
        raise ValueError("cache manifest repeats a successful session_id")
    return tuple(available)


def _split_role_columns(
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if type(metadata) is not pd.DataFrame:
        raise TypeError("cache metadata must be an exact pandas DataFrame")
    required = {"session_id", "identity", "reference_valid"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"cache metadata lacks split columns: {missing}")
    if metadata.empty:
        raise ValueError("cache metadata is empty")
    session_values = metadata["session_id"].tolist()
    identity_values = metadata["identity"].tolist()
    reference_values = metadata["reference_valid"].tolist()
    if any(
        type(value) is not str or not value or value != value.strip()
        for value in session_values
    ):
        raise ValueError("cache session_id must contain exact non-empty strings")
    if any(
        type(value) is not str or not value or value != value.strip()
        for value in identity_values
    ):
        raise ValueError("cache identity must contain exact non-empty strings")
    if any(type(value) not in {bool, np.bool_} for value in reference_values):
        raise ValueError("cache reference_valid must contain exact booleans")
    return (
        np.asarray(session_values, dtype=object),
        np.asarray(identity_values, dtype=object),
        np.asarray(reference_values, dtype=np.bool_),
    )


def _session_identity_map(
    metadata: pd.DataFrame, cache_manifest: Mapping[str, Any]
) -> dict[str, str]:
    session_ids, identities, _ = _split_role_columns(metadata)

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

    available = _successful_session_ids(cache_manifest)
    if set(available) != set(result):
        missing_metadata = sorted(set(available) - set(result))
        unknown_metadata = sorted(set(result) - set(available))
        raise ValueError(
            "cache manifest/metadata session cover mismatch "
            f"(missing_metadata={missing_metadata}, unknown_metadata={unknown_metadata})"
        )
    return result


@dataclass(frozen=True, slots=True)
class _MetadataFileBinding:
    session_id: str
    path: Path
    sha256: str
    bytes: int

    def provenance(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "path": str(self.path),
            "sha256": self.sha256,
            "bytes": self.bytes,
        }


def _metadata_row_roles_sha256(metadata: pd.DataFrame) -> str:
    session_ids, identities, reference_valid = _split_role_columns(metadata)
    rows = [
        [position, str(session_id), str(identity), bool(valid)]
        for position, (session_id, identity, valid) in enumerate(
            zip(session_ids, identities, reference_valid, strict=True)
        )
    ]
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": "snn_rr.identity_split_metadata_row_roles.v1",
                "columns": ["row_position", "session_id", "identity", "reference_valid"],
                "rows": rows,
            }
        )
    ).hexdigest()


def _load_authoritative_metadata_snapshot(
    cache_root: Path, cache_manifest: Mapping[str, Any]
) -> tuple[pd.DataFrame, tuple[_MetadataFileBinding, ...], str]:
    """Parse every metadata CSV from the exact bytes used for its digest."""

    root = cache_root.expanduser().resolve()
    frames: list[pd.DataFrame] = []
    bindings: list[_MetadataFileBinding] = []
    for session_id in _successful_session_ids(cache_manifest):
        session_root = (root / session_id).resolve()
        metadata_path = (session_root / "metadata.csv").resolve()
        try:
            session_root.relative_to(root)
            metadata_path.relative_to(session_root)
        except ValueError as error:
            raise ValueError(
                f"cache metadata path escapes its root for {session_id}"
            ) from error
        payload, digest, byte_count = _read_regular_file_snapshot(
            metadata_path, f"cache metadata {session_id}"
        )
        try:
            frame = pd.read_csv(io.BytesIO(payload))
        except Exception as error:
            raise ValueError(
                f"cache metadata {session_id} cannot be parsed from its bound bytes"
            ) from error
        frame_sessions, _, _ = _split_role_columns(frame)
        if set(frame_sessions.tolist()) != {session_id}:
            raise ValueError(
                f"cache metadata file/session mismatch for {session_id}"
            )
        frames.append(frame)
        bindings.append(
            _MetadataFileBinding(
                session_id=session_id,
                path=metadata_path,
                sha256=digest,
                bytes=byte_count,
            )
        )
    authoritative = pd.concat(frames, ignore_index=True)
    source_content_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": "snn_rr.identity_split_metadata_sources.v1",
                "files": [binding.provenance() for binding in bindings],
            }
        )
    ).hexdigest()
    return authoritative, tuple(bindings), source_content_sha256


@dataclass(frozen=True, slots=True)
class _AuthorityRuntimeBinding:
    authority_ref: weakref.ReferenceType[Any]
    authority_claim_sha256: str
    metadata_object: pd.DataFrame
    metadata_snapshot: pd.DataFrame


# A split authority is useful only when issued by the loader for the exact
# metadata object that the trainer will consume.  Keeping the issuing object in
# this process-local strong registry makes dataclass construction/copy and
# receipt transplant fail before indices or scaler roles can be produced.
_AUTHORITY_RUNTIME_REGISTRY: dict[int, _AuthorityRuntimeBinding] = {}


def _authority_claim_sha256(authority: Any) -> str:
    """Bind every authority field whose mutation could change split output."""

    return hashlib.sha256(
        canonical_json_bytes(
            {
                "manifest_path": str(authority.manifest_path),
                "manifest_file_sha256": authority.manifest_file_sha256,
                "content_sha256": authority.content_sha256,
                "fold_id": authority.fold_id,
                "train_identities": list(authority.train_identities),
                "validation_identities": list(authority.validation_identities),
                "prediction_identities": list(authority.prediction_identities),
                "excluded_identities": list(authority.excluded_identities),
                "scaler_identities": list(authority.scaler_identities),
                "fold_assignments_path": str(authority.fold_assignments_path),
                "fold_assignments_sha256": authority.fold_assignments_sha256,
                "cache_manifest_path": str(authority.cache_manifest_path),
                "cache_manifest_sha256": authority.cache_manifest_sha256,
                "identity_to_fold": dict(sorted(authority.identity_to_fold.items())),
                "session_to_identity": dict(
                    sorted(authority.session_to_identity.items())
                ),
                "metadata_source_content_sha256": (
                    authority.metadata_source_content_sha256
                ),
                "metadata_row_roles_sha256": authority.metadata_row_roles_sha256,
                "metadata_row_count": authority.metadata_row_count,
                "metadata_file_bindings": [
                    binding.provenance()
                    for binding in authority.metadata_file_bindings
                ],
            }
        )
    ).hexdigest()


def _register_authority(
    authority: Any, metadata: pd.DataFrame, metadata_snapshot: pd.DataFrame
) -> None:
    key = id(authority)

    def discard(reference: weakref.ReferenceType[Any]) -> None:
        current = _AUTHORITY_RUNTIME_REGISTRY.get(key)
        if current is not None and current.authority_ref is reference:
            _AUTHORITY_RUNTIME_REGISTRY.pop(key, None)

    authority_ref = weakref.ref(authority, discard)
    _AUTHORITY_RUNTIME_REGISTRY[key] = _AuthorityRuntimeBinding(
        authority_ref=authority_ref,
        authority_claim_sha256=_authority_claim_sha256(authority),
        metadata_object=metadata,
        metadata_snapshot=metadata_snapshot.copy(deep=True),
    )


def _require_live_authority(authority: Any) -> _AuthorityRuntimeBinding:
    if type(authority) is not IdentitySplitAuthority:
        raise RuntimeError("identity split authority must have the exact issued type")
    binding = _AUTHORITY_RUNTIME_REGISTRY.get(id(authority))
    if binding is None or binding.authority_ref() is not authority:
        raise RuntimeError("identity split authority was not issued by the loader")
    if _authority_claim_sha256(authority) != binding.authority_claim_sha256:
        raise RuntimeError("identity split authority changed after loader issuance")
    return binding


def _require_bound_metadata(authority: Any, metadata: pd.DataFrame) -> None:
    binding = _require_live_authority(authority)
    if type(metadata) is not pd.DataFrame or metadata is not binding.metadata_object:
        raise RuntimeError(
            "identity split authority requires the exact loader-bound metadata object"
        )
    if not metadata.equals(binding.metadata_snapshot):
        raise RuntimeError("loader-bound cache metadata changed after authority issuance")
    observed_roles = _metadata_row_roles_sha256(metadata)
    if observed_roles != authority.metadata_row_roles_sha256:
        raise RuntimeError("loader-bound metadata row roles changed after authority issuance")


@dataclass(frozen=True, slots=True)
class ExplicitIdentitySplit:
    """Validated row indices for one externally prescribed split."""

    train_index: np.ndarray
    validation_index: np.ndarray
    prediction_index: np.ndarray
    split: dict[str, list[str]]


@dataclass(frozen=True, slots=True, weakref_slot=True)
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
    metadata_source_content_sha256: str
    metadata_row_roles_sha256: str
    metadata_row_count: int
    metadata_file_bindings: tuple[_MetadataFileBinding, ...]

    def checkpoint_provenance(self) -> dict[str, Any]:
        """Small, stable binding embedded in run and checkpoint artifacts."""

        runtime = _require_live_authority(self)
        _require_bound_metadata(self, runtime.metadata_object)
        return {
            "mode": "custom_identity_split",
            "schema_version": SCHEMA_VERSION,
            "authority_receipt_version": 2,
            "fold_id": self.fold_id,
            "split_manifest_content_sha256": self.content_sha256,
            "split_manifest_file_sha256": self.manifest_file_sha256,
            "fold_assignments_sha256": self.fold_assignments_sha256,
            "cache_manifest_sha256": self.cache_manifest_sha256,
            "metadata_source_content_sha256": self.metadata_source_content_sha256,
            "metadata_row_roles_sha256": self.metadata_row_roles_sha256,
            "metadata_row_count": self.metadata_row_count,
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
            metadata_files=[
                binding.provenance() for binding in self.metadata_file_bindings
            ],
        )
        return value

    def explicit_indices(
        self, metadata: pd.DataFrame, *, include_invalid: bool
    ) -> ExplicitIdentitySplit:
        """Build indices without consulting rotating-fold helpers."""

        _require_bound_metadata(self, metadata)
        _, identities, valid = _split_role_columns(metadata)
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
        for positions in (train, validation, prediction):
            positions.setflags(write=False)
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
        _require_bound_metadata(self, metadata)
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
    document, manifest_file_sha256 = _read_regular_json_snapshot(
        manifest_path, "identity split manifest"
    )
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
    (
        authoritative_metadata,
        metadata_file_bindings,
        metadata_source_content_sha256,
    ) = _load_authoritative_metadata_snapshot(
        Path(cache_dir).expanduser().resolve(), cache_document
    )
    metadata_snapshot = metadata.copy(deep=True)
    if not metadata_snapshot.equals(authoritative_metadata):
        raise ValueError(
            "caller metadata differs from the exact cache metadata byte snapshot"
        )
    metadata_row_roles_sha256 = _metadata_row_roles_sha256(metadata_snapshot)
    authoritative_row_roles_sha256 = _metadata_row_roles_sha256(
        authoritative_metadata
    )
    if metadata_row_roles_sha256 != authoritative_row_roles_sha256:
        raise RuntimeError("cache metadata row-role snapshot binding drifted")
    _, metadata_identity_values, _ = _split_role_columns(metadata_snapshot)
    metadata_identities = set(metadata_identity_values.tolist())
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

    # Persistent replacement during parsing is rejected.  A swap-and-revert
    # cannot make provenance lie about consumed bytes because every returned
    # digest came from the same snapshot that was parsed.
    for current_path, expected_sha256, label in (
        (manifest_path, manifest_file_sha256, "identity split manifest"),
        (fold_path, fold_sha, "fold assignments"),
        (cache_path, cache_sha, "cache manifest"),
    ):
        if sha256_file(current_path) != expected_sha256:
            raise ValueError(f"{label} changed while split authority was loaded")
    for binding in metadata_file_bindings:
        if (
            binding.path.stat().st_size != binding.bytes
            or sha256_file(binding.path) != binding.sha256
        ):
            raise ValueError(
                f"cache metadata {binding.session_id} changed while split authority was loaded"
            )
    if not metadata.equals(metadata_snapshot):
        raise RuntimeError("caller metadata changed while split authority was loaded")

    authority = IdentitySplitAuthority(
        manifest_path=manifest_path,
        manifest_file_sha256=manifest_file_sha256,
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
        identity_to_fold=MappingProxyType(dict(identity_to_fold)),
        session_to_identity=MappingProxyType(dict(session_to_identity)),
        metadata_source_content_sha256=metadata_source_content_sha256,
        metadata_row_roles_sha256=metadata_row_roles_sha256,
        metadata_row_count=len(metadata_snapshot),
        metadata_file_bindings=metadata_file_bindings,
    )
    _register_authority(authority, metadata, metadata_snapshot)
    return authority
