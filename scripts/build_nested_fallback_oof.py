#!/usr/bin/env python3
"""Export a label-free fallback OOF table from one strict nested stack.

Only the five discovery-owned partitions are exported.  The unopened outer
test partition stays absent from the CSV, and the input stack's internal
content signature is verified before any output is published.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snn_rr.split_authority import (  # noqa: E402
    canonical_content_sha256,
    sha256_file,
)


SCHEMA_VERSION = 1
STACK_FORMAT_VERSION = 1
STACK_CLASSIFICATION = "retrospective_strict_nested_proposer_stack"
OUTPUT_COLUMNS = (
    "cache_index",
    "session_id",
    "identity",
    "protocol",
    "window_number",
    "window_start_s",
    "window_end_s",
    "fold",
    "nested_role",
    "proposer_fold_id",
    "outer_fold",
    "seed",
    "prediction_bpm",
    "rr_std_bpm",
)
SEMANTIC_COLUMNS = (
    "cache_index",
    "session_id",
    "identity",
    "protocol",
    "window_number",
    "window_start_s",
    "window_end_s",
    "fold",
    "nested_role",
    "proposer_fold_id",
)
REQUIRED_STACK_FIELDS = {
    "cache_index",
    "session_id",
    "identity",
    "protocol",
    "window_number",
    "window_start_s",
    "window_end_s",
    "fold",
    "proposal_available",
    "nested_role",
    "proposer_fold_id",
    "prediction",
    "rr_std",
    "outer_fold",
    "seed",
    "strict_nested",
    "outer_test_opened",
    "content_signature_sha256",
    "provenance_json",
}
ROW_PROPOSER_FIELDS = (
    "prediction",
    "map_prediction",
    "rr_std",
    "uncertainty",
    "quality",
    "alias_probability",
    "posterior_entropy",
    "spike_rate",
    "topk_rr",
    "topk_probability",
    "posterior_probability",
    "radar_weights",
)
KNOWN_REFERENCE_AUDIT_FIELDS = {"reference_valid", "reference_rr_bpm"}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _array_signature(
    arrays: Mapping[str, Any], provenance_without_signature: Mapping[str, Any]
) -> str:
    """Reproduce the nested-stack builder's content signature exactly."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes())
    digest.update(
        json.dumps(
            provenance_without_signature,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _scalar(array: Any, name: str) -> Any:
    value = np.asarray(array)
    if value.ndim != 0:
        raise RuntimeError(f"nested stack field {name} must be scalar")
    return value.item()


def _require_bool_scalar(array: Any, name: str, expected: bool) -> None:
    value = _scalar(array, name)
    if not isinstance(value, (bool, np.bool_)) or bool(value) is not expected:
        raise RuntimeError(f"nested stack field {name} must equal {expected}")


def _require_int_scalar(array: Any, name: str) -> int:
    value = _scalar(array, name)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise RuntimeError(f"nested stack field {name} must be an integer scalar")
    return int(value)


def _parse_provenance(array: Any) -> dict[str, Any]:
    raw = _scalar(array, "provenance_json")
    if not isinstance(raw, str):
        raise RuntimeError("nested stack provenance_json must be a string scalar")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("nested stack provenance_json is invalid") from exc
    if not isinstance(value, dict):
        raise RuntimeError("nested stack provenance must be an object")
    return value


def _forbidden_extra_fields(names: set[str]) -> list[str]:
    """Find undeclared target/label arrays, while allowing stack audit targets."""

    forbidden: list[str] = []
    exact = {
        "label",
        "labels",
        "target",
        "targets",
        "rr_bpm",
        "ground_truth",
        "ground_truth_rr_bpm",
    }
    for name in names - KNOWN_REFERENCE_AUDIT_FIELDS:
        lowered = name.lower()
        if (
            lowered in exact
            or lowered.startswith(
                ("label_", "target_", "ground_truth_", "reference_", "ref_")
            )
            or lowered.endswith(("_label", "_target"))
        ):
            forbidden.append(name)
    return sorted(forbidden)


def _validate_content_signature(
    arrays: Mapping[str, np.ndarray], provenance: Mapping[str, Any]
) -> str:
    field_signature = _scalar(arrays["content_signature_sha256"], "content_signature_sha256")
    provenance_signature = provenance.get("content_signature_sha256")
    if not isinstance(field_signature, str) or field_signature != provenance_signature:
        raise RuntimeError("nested stack content signature fields disagree")
    unsigned_provenance = dict(provenance)
    unsigned_provenance.pop("content_signature_sha256", None)
    unsigned_arrays = {
        name: value
        for name, value in arrays.items()
        if name not in {"content_signature_sha256", "provenance_json"}
    }
    observed = _array_signature(unsigned_arrays, unsigned_provenance)
    if observed != field_signature:
        raise RuntimeError("nested stack content signature mismatch (tamper detected)")
    return field_signature


def _validate_row_array(array: Any, name: str, row_count: int) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 1 or value.shape != (row_count,):
        raise RuntimeError(f"nested stack row field has the wrong shape: {name}")
    return value


def load_strict_nested_fallback(stack_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate a nested stack and return only its label-free available rows."""

    source = stack_path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"nested stack does not exist: {source}")
    source_sha256 = sha256_file(source)
    try:
        with np.load(source, allow_pickle=False) as archive:
            fields = set(archive.files)
            missing = sorted(REQUIRED_STACK_FIELDS - fields)
            if missing:
                raise RuntimeError(f"nested stack is missing fields: {missing}")
            forbidden = _forbidden_extra_fields(fields)
            if forbidden:
                raise RuntimeError(
                    f"nested stack declares forbidden label/target fields: {forbidden}"
                )
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"invalid nested stack archive: {source} ({exc})") from exc

    provenance = _parse_provenance(arrays["provenance_json"])
    content_signature = _validate_content_signature(arrays, provenance)
    _require_bool_scalar(arrays["strict_nested"], "strict_nested", True)
    _require_bool_scalar(arrays["outer_test_opened"], "outer_test_opened", False)
    outer_fold = _require_int_scalar(arrays["outer_fold"], "outer_fold")
    seed = _require_int_scalar(arrays["seed"], "seed")
    if outer_fold < 0:
        raise RuntimeError("nested stack outer_fold must be non-negative")

    required_provenance = {
        "format_version": STACK_FORMAT_VERSION,
        "classification": STACK_CLASSIFICATION,
        "strict_nested": True,
        "outer_test_opened": False,
        "commercial_performance_claim_eligible": False,
        "target_consulted_for_stitching": False,
        "outer_fold": outer_fold,
        "seed": seed,
    }
    for name, expected in required_provenance.items():
        if provenance.get(name) != expected:
            raise RuntimeError(f"nested stack provenance mismatch: {name}")
    test_identity_value = provenance.get("outer_test_identities")
    if not isinstance(test_identity_value, list) or not test_identity_value:
        raise RuntimeError("nested stack provenance lacks outer-test identities")
    if any(not isinstance(item, str) or not item for item in test_identity_value):
        raise RuntimeError("nested stack outer-test identities are invalid")
    test_identities = set(test_identity_value)

    cache_index_raw = np.asarray(arrays["cache_index"])
    if cache_index_raw.ndim != 1 or not np.issubdtype(cache_index_raw.dtype, np.integer):
        raise RuntimeError("nested stack cache_index must be a one-dimensional integer array")
    cache_index = cache_index_raw.astype(np.int64, copy=False)
    row_count = len(cache_index)
    if row_count == 0 or not np.array_equal(cache_index, np.arange(row_count, dtype=np.int64)):
        raise RuntimeError("nested stack cache_index must be unique and contiguous")

    row: dict[str, np.ndarray] = {}
    for name in (
        "session_id",
        "identity",
        "protocol",
        "window_number",
        "window_start_s",
        "window_end_s",
        "fold",
        "proposal_available",
        "nested_role",
        "proposer_fold_id",
        "prediction",
        "rr_std",
    ):
        row[name] = _validate_row_array(arrays[name], name, row_count)

    for name in ("session_id", "identity", "protocol", "nested_role"):
        values = row[name].astype(str)
        if any(not value or value != value.strip() for value in values):
            raise RuntimeError(f"nested stack has empty or untrimmed semantics: {name}")
        row[name] = values
    for name in ("window_number", "fold", "proposer_fold_id"):
        if not np.issubdtype(row[name].dtype, np.integer):
            raise RuntimeError(f"nested stack semantic field must be integral: {name}")
        row[name] = row[name].astype(np.int64, copy=False)
    if (row["window_number"] < 0).any():
        raise RuntimeError("nested stack window_number must be non-negative")
    for name in ("window_start_s", "window_end_s"):
        row[name] = row[name].astype(np.float64, copy=False)
        if not np.isfinite(row[name]).all():
            raise RuntimeError(f"nested stack has non-finite window semantics: {name}")
    if (row["window_end_s"] <= row["window_start_s"]).any():
        raise RuntimeError("nested stack window intervals are invalid")

    semantic_keys = pd.MultiIndex.from_arrays(
        [row["session_id"], row["window_number"]],
        names=["session_id", "window_number"],
    )
    if semantic_keys.has_duplicates:
        raise RuntimeError("nested stack has duplicate session/window semantics")
    identity_folds = pd.DataFrame(
        {"identity": row["identity"], "fold": row["fold"]}
    ).groupby("identity", sort=False)["fold"].nunique()
    if (identity_folds != 1).any():
        raise RuntimeError("nested stack maps one identity to multiple canonical folds")

    available_raw = row["proposal_available"]
    if available_raw.dtype.kind != "b":
        raise RuntimeError("nested stack proposal_available must be boolean")
    available = available_raw.astype(bool, copy=False)
    outer_test = row["fold"] == outer_fold
    identity_test = np.isin(row["identity"], tuple(sorted(test_identities)))
    if not np.array_equal(outer_test, identity_test):
        raise RuntimeError("outer-test identity and canonical-fold bindings disagree")
    if not np.array_equal(available, ~outer_test):
        raise RuntimeError("outer-test availability or discovery partition cover is invalid")
    if available[outer_test].any():
        raise RuntimeError("outer-test proposal availability is forbidden")
    if set(row["nested_role"][outer_test]) != {"outer_test_unavailable"}:
        raise RuntimeError("outer-test rows have an invalid nested role")
    if (row["proposer_fold_id"][outer_test] != -1).any():
        raise RuntimeError("outer-test rows unexpectedly bind a proposer fold")
    allowed_roles = {"hcs_train_oof", "hcs_validation"}
    if not set(row["nested_role"][available]).issubset(allowed_roles):
        raise RuntimeError("available rows have an invalid discovery role")
    if (row["proposer_fold_id"][available] < 0).any():
        raise RuntimeError("available rows lack a proposer-fold binding")

    prediction = row["prediction"].astype(np.float64, copy=False)
    rr_std = row["rr_std"].astype(np.float64, copy=False)
    if not np.isfinite(prediction[available]).all() or (prediction[available] <= 0).any():
        raise RuntimeError("available fallback predictions must be finite and positive")
    if not np.isfinite(rr_std[available]).all() or (rr_std[available] <= 0).any():
        raise RuntimeError("available fallback standard deviations must be finite and positive")
    for name in ROW_PROPOSER_FIELDS:
        if name not in arrays:
            continue
        value = np.asarray(arrays[name])
        if value.ndim < 1 or value.shape[0] != row_count:
            raise RuntimeError(f"nested stack proposer field has the wrong shape: {name}")
        if not np.isfinite(value[available]).all():
            raise RuntimeError(f"available proposer field contains non-finite data: {name}")

    if provenance.get("row_count") != row_count:
        raise RuntimeError("nested stack provenance row_count mismatch")
    if provenance.get("available_rows") != int(available.sum()):
        raise RuntimeError("nested stack provenance available_rows mismatch")
    if provenance.get("outer_test_rows") != int(outer_test.sum()):
        raise RuntimeError("nested stack provenance outer_test_rows mismatch")

    selected = np.flatnonzero(available)
    frame = pd.DataFrame(
        {
            "cache_index": cache_index[selected],
            "session_id": row["session_id"][selected],
            "identity": row["identity"][selected],
            "protocol": row["protocol"][selected],
            "window_number": row["window_number"][selected],
            "window_start_s": row["window_start_s"][selected],
            "window_end_s": row["window_end_s"][selected],
            "fold": row["fold"][selected],
            "nested_role": row["nested_role"][selected],
            "proposer_fold_id": row["proposer_fold_id"][selected],
            "outer_fold": np.full(len(selected), outer_fold, dtype=np.int64),
            "seed": np.full(len(selected), seed, dtype=np.int64),
            "prediction_bpm": prediction[selected],
            "rr_std_bpm": rr_std[selected],
        },
        columns=OUTPUT_COLUMNS,
    )
    if any(
        "reference" in column.lower()
        or "target" in column.lower()
        or "label" in column.lower()
        for column in frame.columns
    ):
        raise AssertionError("fallback output schema includes a label/reference field")

    if sha256_file(source) != source_sha256:
        raise RuntimeError("nested stack changed while it was being validated")
    audit = {
        "source_path": str(source),
        "source_sha256": source_sha256,
        "source_content_signature_sha256": content_signature,
        "outer_fold": outer_fold,
        "seed": seed,
        "source_rows": row_count,
        "exported_rows": len(frame),
        "excluded_outer_test_rows": int(outer_test.sum()),
        "outer_test_identities": sorted(test_identities),
    }
    return frame, audit


def _temporary_path(destination: Path) -> Path:
    return destination.with_name(
        f".{destination.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp"
    )


def _publish_exclusive(temporary: Path, destination: Path) -> None:
    """Atomically publish without ever replacing an existing artifact."""

    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise FileExistsError(f"output already exists and is immutable: {destination}") from exc


def _row_binding_sha256(frame: pd.DataFrame) -> str:
    payload = frame.loc[:, SEMANTIC_COLUMNS].to_csv(
        index=False, lineterminator="\n", float_format="%.17g"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_fallback_artifacts(
    *, stack_path: Path, output_path: Path, provenance_path: Path | None = None
) -> dict[str, Any]:
    output = output_path.expanduser().resolve()
    if output.suffix.lower() != ".csv":
        raise ValueError("fallback output must use a .csv filename")
    sidecar = (
        provenance_path.expanduser().resolve()
        if provenance_path is not None
        else output.with_name(f"{output.name}.provenance.json")
    )
    source = stack_path.expanduser().resolve()
    if len({source, output, sidecar}) != 3:
        raise ValueError("source stack, CSV output, and provenance sidecar must differ")
    if output.exists() or sidecar.exists():
        existing = output if output.exists() else sidecar
        raise FileExistsError(f"output already exists and is immutable: {existing}")

    frame, audit = load_strict_nested_fallback(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    csv_temporary = _temporary_path(output)
    sidecar_temporary = _temporary_path(sidecar)
    published: list[Path] = []
    try:
        frame.to_csv(
            csv_temporary,
            index=False,
            lineterminator="\n",
            float_format="%.17g",
        )
        csv_sha256 = sha256_file(csv_temporary)
        provenance: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "strict_nested_fallback_oof",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "classification": "retrospective_strict_nested_discovery_fallback",
            "strict_nested": True,
            "outer_test_opened": False,
            "commercial_performance_claim_eligible": False,
            "target_consulted_for_fallback": False,
            "label_or_reference_fields_exported": False,
            "fallback_source_fields": {
                "prediction_bpm": "prediction",
                "rr_std_bpm": "rr_std",
            },
            "source_stack": {
                "path": audit["source_path"],
                "sha256": audit["source_sha256"],
                "content_signature_sha256": audit[
                    "source_content_signature_sha256"
                ],
            },
            "outer_fold": audit["outer_fold"],
            "seed": audit["seed"],
            "source_rows": audit["source_rows"],
            "exported_rows": audit["exported_rows"],
            "excluded_outer_test_rows": audit["excluded_outer_test_rows"],
            "outer_test_identities": audit["outer_test_identities"],
            "row_binding_sha256": _row_binding_sha256(frame),
            "output_csv": {
                "path": str(output),
                "sha256": csv_sha256,
                "columns": list(OUTPUT_COLUMNS),
            },
            "source_code_sha256": {
                str(Path(__file__).resolve().relative_to(PROJECT_ROOT)): sha256_file(
                    Path(__file__).resolve()
                )
            },
        }
        provenance["content_sha256"] = canonical_content_sha256(provenance)
        sidecar_temporary.write_text(
            json.dumps(
                provenance,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        if sha256_file(source) != audit["source_sha256"]:
            raise RuntimeError("nested stack changed before output publication")

        _publish_exclusive(csv_temporary, output)
        published.append(output)
        _publish_exclusive(sidecar_temporary, sidecar)
        published.append(sidecar)
    except Exception:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        csv_temporary.unlink(missing_ok=True)
        sidecar_temporary.unlink(missing_ok=True)

    return {
        "output": str(output),
        "output_sha256": sha256_file(output),
        "provenance": str(sidecar),
        "provenance_sha256": sha256_file(sidecar),
        "provenance_content_sha256": provenance["content_sha256"],
        "rows": len(frame),
        "excluded_outer_test_rows": audit["excluded_outer_test_rows"],
        "outer_test_opened": False,
        "strict_nested": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--provenance-output",
        type=Path,
        help="defaults to <output>.provenance.json",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    return write_fallback_artifacts(
        stack_path=args.stack,
        output_path=args.output,
        provenance_path=args.provenance_output,
    )


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(_canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
