from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

from snn_rr.split_authority import canonical_content_sha256, sha256_file


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_nested_fallback_oof.py"
_SPEC = importlib.util.spec_from_file_location("build_nested_fallback_oof", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_BUILD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BUILD
_SPEC.loader.exec_module(_BUILD)


def _base_stack() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    row_count = 6
    available = np.array([True, True, True, False, True, True], dtype=bool)
    role = np.array(
        [
            "hcs_train_oof",
            "hcs_train_oof",
            "hcs_train_oof",
            "outer_test_unavailable",
            "hcs_validation",
            "hcs_train_oof",
        ],
        dtype=np.str_,
    )
    arrays: dict[str, np.ndarray] = {
        "cache_index": np.arange(row_count, dtype=np.int64),
        "session_id": np.array([f"S{index}" for index in range(row_count)]),
        "identity": np.array([f"I{index}" for index in range(row_count)]),
        "protocol": np.array(["rest", "rest", "move", "move", "rest", "move"]),
        "window_number": np.zeros(row_count, dtype=np.int32),
        "window_start_s": np.arange(row_count, dtype=np.float64) * 4.0,
        "window_end_s": np.arange(row_count, dtype=np.float64) * 4.0 + 32.0,
        "fold": np.arange(row_count, dtype=np.int16),
        "reference_valid": np.array([True, True, False, False, True, True]),
        "reference_rr_bpm": np.array([10.0, 11.0, np.nan, np.nan, 14.0, 15.0], dtype=np.float32),
        "proposal_available": available,
        "nested_role": role,
        "proposer_fold_id": np.array([300, 301, 302, -1, 350, 305], dtype=np.int16),
        "prediction": np.array([10.1, 11.1, 12.1, 0.0, 14.1, 15.1], dtype=np.float32),
        "map_prediction": np.array([10.2, 11.2, 12.2, 0.0, 14.2, 15.2], dtype=np.float32),
        "rr_std": np.array([0.5, 0.6, 0.7, 0.0, 0.9, 1.0], dtype=np.float32),
        "uncertainty": np.array([0.2, 0.2, 0.3, 0.0, 0.4, 0.5], dtype=np.float32),
        "quality": np.array([0.8, 0.8, 0.7, 0.0, 0.6, 0.5], dtype=np.float32),
        "alias_probability": np.array([0.1, 0.1, 0.2, 0.0, 0.2, 0.3], dtype=np.float32),
        "posterior_entropy": np.array([0.2, 0.2, 0.3, 0.0, 0.4, 0.5], dtype=np.float32),
        "spike_rate": np.array([0.1, 0.1, 0.1, 0.0, 0.2, 0.2], dtype=np.float32),
        "topk_rr": np.tile(np.array([[12.0, 24.0]], dtype=np.float32), (row_count, 1)),
        "topk_probability": np.tile(np.array([[0.8, 0.2]], dtype=np.float32), (row_count, 1)),
        "posterior_probability": np.tile(np.array([[0.8, 0.2]], dtype=np.float32), (row_count, 1)),
        "radar_weights": np.tile(np.array([[0.2, 0.3, 0.5]], dtype=np.float32), (row_count, 1)),
        "posterior_rr_grid_bpm": np.array([12.0, 24.0], dtype=np.float32),
        "outer_fold": np.asarray(3, dtype=np.int16),
        "seed": np.asarray(17, dtype=np.int64),
        "strict_nested": np.asarray(True),
        "outer_test_opened": np.asarray(False),
    }
    provenance: dict[str, Any] = {
        "format_version": 1,
        "classification": "retrospective_strict_nested_proposer_stack",
        "strict_nested": True,
        "outer_test_opened": False,
        "commercial_performance_claim_eligible": False,
        "target_consulted_for_stitching": False,
        "outer_fold": 3,
        "seed": 17,
        "row_count": row_count,
        "available_rows": int(available.sum()),
        "outer_test_rows": int((~available).sum()),
        "outer_test_identities": ["I3"],
    }
    return arrays, provenance


def _write_signed_stack(
    path: Path,
    arrays: dict[str, np.ndarray],
    provenance: dict[str, Any],
) -> None:
    unsigned = {
        name: value
        for name, value in arrays.items()
        if name not in {"content_signature_sha256", "provenance_json"}
    }
    unsigned_provenance = dict(provenance)
    unsigned_provenance.pop("content_signature_sha256", None)
    signature = _BUILD._array_signature(unsigned, unsigned_provenance)
    signed_provenance = dict(unsigned_provenance)
    signed_provenance["content_signature_sha256"] = signature
    signed = dict(unsigned)
    signed["content_signature_sha256"] = np.asarray(signature)
    signed["provenance_json"] = np.asarray(
        json.dumps(signed_provenance, sort_keys=True, separators=(",", ":"))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **signed)


def _fixture(path: Path) -> Path:
    arrays, provenance = _base_stack()
    _write_signed_stack(path, arrays, provenance)
    return path


def test_exports_only_available_label_free_rows_with_bound_provenance(tmp_path: Path) -> None:
    stack = _fixture(tmp_path / "strict_stack.npz")
    output = tmp_path / "nested_fallback.csv"
    result = _BUILD.write_fallback_artifacts(stack_path=stack, output_path=output)

    sidecar = tmp_path / "nested_fallback.csv.provenance.json"
    frame = pd.read_csv(output)
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    assert list(frame.columns) == list(_BUILD.OUTPUT_COLUMNS)
    assert frame["cache_index"].tolist() == [0, 1, 2, 4, 5]
    assert set(frame["identity"]) == {"I0", "I1", "I2", "I4", "I5"}
    assert "I3" not in set(frame["identity"])
    assert frame["prediction_bpm"].tolist() == pytest.approx([10.1, 11.1, 12.1, 14.1, 15.1])
    assert frame["rr_std_bpm"].tolist() == pytest.approx([0.5, 0.6, 0.7, 0.9, 1.0])
    assert not any(
        token in column.lower()
        for column in frame.columns
        for token in ("reference", "target", "label")
    )
    assert provenance["target_consulted_for_fallback"] is False
    assert provenance["label_or_reference_fields_exported"] is False
    assert provenance["outer_test_opened"] is False
    assert provenance["outer_test_identities"] == ["I3"]
    assert provenance["source_stack"]["sha256"] == sha256_file(stack)
    assert provenance["output_csv"]["sha256"] == sha256_file(output)
    assert canonical_content_sha256(provenance) == provenance["content_sha256"]
    assert result["output_sha256"] == sha256_file(output)
    assert result["provenance_sha256"] == sha256_file(sidecar)

    with pytest.raises(FileExistsError, match="immutable"):
        _BUILD.write_fallback_artifacts(stack_path=stack, output_path=output)


def test_unsigned_array_tamper_is_rejected(tmp_path: Path) -> None:
    stack = _fixture(tmp_path / "strict_stack.npz")
    with np.load(stack, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    arrays["prediction"][0] += 7.0
    np.savez_compressed(stack, **arrays)
    with pytest.raises(RuntimeError, match="content signature mismatch"):
        _BUILD.load_strict_nested_fallback(stack)


def test_missing_schema_field_is_rejected(tmp_path: Path) -> None:
    arrays, provenance = _base_stack()
    arrays.pop("window_start_s")
    stack = tmp_path / "missing.npz"
    _write_signed_stack(stack, arrays, provenance)
    with pytest.raises(RuntimeError, match="missing fields.*window_start_s"):
        _BUILD.load_strict_nested_fallback(stack)


Mutation = Callable[[dict[str, np.ndarray], dict[str, Any]], None]


def _duplicate_cache_index(arrays: dict[str, np.ndarray], _: dict[str, Any]) -> None:
    arrays["cache_index"][1] = 0


def _duplicate_semantic_row(arrays: dict[str, np.ndarray], _: dict[str, Any]) -> None:
    arrays["session_id"][1] = arrays["session_id"][0]


def _open_outer_test(arrays: dict[str, np.ndarray], provenance: dict[str, Any]) -> None:
    arrays["proposal_available"][3] = True
    provenance["available_rows"] += 1
    provenance["outer_test_rows"] -= 1


def _nonfinite_prediction(arrays: dict[str, np.ndarray], _: dict[str, Any]) -> None:
    arrays["prediction"][0] = np.nan


def _invalid_standard_deviation(arrays: dict[str, np.ndarray], _: dict[str, Any]) -> None:
    arrays["rr_std"][0] = 0.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_duplicate_cache_index, "unique and contiguous"),
        (_duplicate_semantic_row, "duplicate session/window"),
        (_open_outer_test, "outer-test availability"),
        (_nonfinite_prediction, "predictions must be finite"),
        (_invalid_standard_deviation, "standard deviations must be finite"),
    ],
)
def test_validly_signed_invalid_rows_fail_closed(
    tmp_path: Path, mutation: Mutation, message: str
) -> None:
    arrays, provenance = _base_stack()
    mutation(arrays, provenance)
    stack = tmp_path / f"{mutation.__name__}.npz"
    _write_signed_stack(stack, arrays, provenance)
    with pytest.raises(RuntimeError, match=message):
        _BUILD.load_strict_nested_fallback(stack)


def test_target_consultation_and_label_derived_arrays_are_rejected(tmp_path: Path) -> None:
    arrays, provenance = _base_stack()
    provenance["target_consulted_for_stitching"] = True
    stack = tmp_path / "target_consulted.npz"
    _write_signed_stack(stack, arrays, provenance)
    with pytest.raises(RuntimeError, match="target_consulted_for_stitching"):
        _BUILD.load_strict_nested_fallback(stack)

    arrays, provenance = _base_stack()
    arrays["target_rr_bpm"] = np.arange(6, dtype=np.float32)
    stack = tmp_path / "target_array.npz"
    _write_signed_stack(stack, arrays, provenance)
    with pytest.raises(RuntimeError, match="forbidden label/target fields"):
        _BUILD.load_strict_nested_fallback(stack)


def test_sidecar_collision_prevents_partial_csv_publication(tmp_path: Path) -> None:
    stack = _fixture(tmp_path / "strict_stack.npz")
    output = tmp_path / "nested_fallback.csv"
    sidecar = tmp_path / "reserved.json"
    sidecar.write_text("reserved\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable"):
        _BUILD.write_fallback_artifacts(
            stack_path=stack,
            output_path=output,
            provenance_path=sidecar,
        )
    assert not output.exists()
    assert sidecar.read_text(encoding="utf-8") == "reserved\n"
