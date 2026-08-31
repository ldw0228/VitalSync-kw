from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import pytest

from snn_rr.acquisition_contract import (
    AcquisitionContractError,
    V3_SYNC_SIGNALS_ARTIFACT_SCHEMA,
    build_v3_sync_signals_artifact_binding,
    validate_v3_sync_signals_artifact,
)
from snn_rr.radar_timing import canonical_ndarray_sha256


def _arrays() -> dict[str, np.ndarray]:
    valid = np.asarray(
        [[False, True, True], [True, True, True], [True, False, True]],
        dtype=bool,
    )
    reasons = np.where(valid, 0, 1).astype(np.uint8)
    return {
        "radar_times_s": np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
        "radar_motion_robust_z": np.asarray([0.0, 1.5, 2.5], dtype=np.float32),
        "radar_motion_valid_mask": np.asarray([False, True, True], dtype=bool),
        "rsp_marker_times_s": np.asarray([1.25], dtype=np.float64),
        "radar_marker_times_s": np.asarray([0.25], dtype=np.float64),
        "radar_resample_valid_mask": valid,
        "radar_invalid_reason_mask": reasons,
    }


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, np.ndarray]]:
    root = tmp_path / "reconstruction"
    path = root / "sessions" / "S01_TEST" / "sync_signals.npz"
    path.parent.mkdir(parents=True)
    arrays = _arrays()
    np.savez_compressed(path, **arrays)
    binding = build_v3_sync_signals_artifact_binding(
        path,
        artifact_relative_path="sessions/S01_TEST/sync_signals.npz",
    )
    session: dict[str, object] = {
        "session_id": "S01_TEST",
        "sync_raw_replay_verified": False,
        "protocol_raw_replay_verified": False,
        "sync_signals_artifact": binding,
        "sensor_summary": {
            "radar": {
                "feature_resampling": {
                    "content_hashes": {
                        "output_times_sha256": canonical_ndarray_sha256(
                            arrays["radar_times_s"]
                        ),
                        "valid_mask_sha256": canonical_ndarray_sha256(
                            arrays["radar_resample_valid_mask"]
                        ),
                        "invalid_reason_mask_sha256": canonical_ndarray_sha256(
                            arrays["radar_invalid_reason_mask"]
                        ),
                    }
                }
            }
        },
        "synchronization": {
            "authorized": False,
            "radar_markers": [{"time_s": 0.25}],
            "rsp_markers": [{"time_s": 1.25}],
        },
    }
    return root, session, arrays


def test_v3_sync_signals_binds_exact_bytes_and_every_array(tmp_path: Path) -> None:
    root, session, arrays = _fixture(tmp_path)

    validated = validate_v3_sync_signals_artifact(
        session, reconstruction_root=root
    )

    assert validated == session["sync_signals_artifact"]
    assert validated["schema"] == V3_SYNC_SIGNALS_ARTIFACT_SCHEMA
    assert validated["diagnostic_only"] is True
    assert validated["scientific_authority"] is False
    assert set(validated["arrays"]) == set(arrays)
    for name, array in arrays.items():
        evidence = validated["arrays"][name]
        assert evidence["dtype"] == array.dtype.name
        assert evidence["shape"] == list(array.shape)
        assert evidence["canonical_sha256"] == canonical_ndarray_sha256(array)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("array_hash", "bytes/array evidence"),
        ("array_shape", "bytes/array evidence"),
        ("arrays_hash", "bytes/array evidence"),
        ("artifact_bytes_bool", "binding claims"),
        ("artifact_path", "binding claims"),
    ],
)
def test_v3_sync_signals_rejects_resealed_binding_tamper(
    tmp_path: Path, mutation: str, message: str
) -> None:
    root, session, _ = _fixture(tmp_path)
    tampered = deepcopy(session)
    binding = tampered["sync_signals_artifact"]
    if mutation == "array_hash":
        binding["arrays"]["radar_times_s"]["canonical_sha256"] = "0" * 64
    elif mutation == "array_shape":
        binding["arrays"]["radar_times_s"]["shape"] = [999]
    elif mutation == "arrays_hash":
        binding["arrays_sha256"] = "0" * 64
    elif mutation == "artifact_bytes_bool":
        binding["artifact_bytes"] = True
    else:
        binding["artifact"] = "sessions/S99_OTHER/sync_signals.npz"

    with pytest.raises(AcquisitionContractError, match=message):
        validate_v3_sync_signals_artifact(tampered, reconstruction_root=root)


def test_v3_sync_signals_rejects_changed_npz_bytes(tmp_path: Path) -> None:
    root, session, arrays = _fixture(tmp_path)
    path = root / "sessions/S01_TEST/sync_signals.npz"
    changed = {name: value.copy() for name, value in arrays.items()}
    changed["radar_motion_robust_z"][1] += np.float32(1.0)
    np.savez_compressed(path, **changed)

    with pytest.raises(AcquisitionContractError, match="consume.*sync-signals"):
        validate_v3_sync_signals_artifact(session, reconstruction_root=root)


@pytest.mark.parametrize(
    "field",
    ("synchronization.authorized", "sync_raw_replay_verified", "protocol_raw_replay_verified"),
)
def test_v3_sync_signals_never_grants_replay_authority(
    tmp_path: Path, field: str
) -> None:
    root, session, _ = _fixture(tmp_path)
    if field == "synchronization.authorized":
        session["synchronization"]["authorized"] = True
    else:
        session[field] = True

    with pytest.raises(AcquisitionContractError, match="cannot carry replay authority"):
        validate_v3_sync_signals_artifact(session, reconstruction_root=root)


def test_v3_sync_signals_rejects_hardlink_and_symlink_aliases(tmp_path: Path) -> None:
    root, session, _ = _fixture(tmp_path)
    path = root / "sessions/S01_TEST/sync_signals.npz"
    alias = path.with_name("alias.npz")
    os.link(path, alias)
    with pytest.raises(AcquisitionContractError, match="consume.*sync-signals"):
        validate_v3_sync_signals_artifact(session, reconstruction_root=root)

    alias.unlink()
    target = path.with_name("target.npz")
    path.replace(target)
    path.symlink_to(target.name)
    with pytest.raises(AcquisitionContractError, match="consume.*sync-signals"):
        validate_v3_sync_signals_artifact(session, reconstruction_root=root)


@pytest.mark.parametrize("cross_link", ("times", "valid_mask", "reason_mask", "marker"))
def test_v3_sync_signals_rejects_session_graph_transplant(
    tmp_path: Path, cross_link: str
) -> None:
    root, session, _ = _fixture(tmp_path)
    tampered = deepcopy(session)
    hashes = tampered["sensor_summary"]["radar"]["feature_resampling"][
        "content_hashes"
    ]
    if cross_link == "times":
        hashes["output_times_sha256"] = "0" * 64
    elif cross_link == "valid_mask":
        hashes["valid_mask_sha256"] = "0" * 64
    elif cross_link == "reason_mask":
        hashes["invalid_reason_mask_sha256"] = "0" * 64
    else:
        tampered["synchronization"]["radar_markers"][0]["time_s"] = 9.0

    with pytest.raises(AcquisitionContractError, match="cross-link"):
        validate_v3_sync_signals_artifact(tampered, reconstruction_root=root)


@pytest.mark.parametrize("mutation", ("extra_member", "wrong_dtype", "mask_mismatch"))
def test_v3_sync_signals_builder_rejects_invalid_npz_schema(
    tmp_path: Path, mutation: str
) -> None:
    path = tmp_path / "sync_signals.npz"
    arrays = _arrays()
    if mutation == "extra_member":
        arrays["unexpected"] = np.zeros(1, dtype=np.float32)
    elif mutation == "wrong_dtype":
        arrays["radar_times_s"] = arrays["radar_times_s"].astype(np.float32)
    else:
        arrays["radar_invalid_reason_mask"] = np.zeros((3, 3), dtype=np.uint8)
    np.savez_compressed(path, **arrays)

    with pytest.raises(AcquisitionContractError, match="members|dtype|shape/finite/mask"):
        build_v3_sync_signals_artifact_binding(
            path,
            artifact_relative_path="sessions/S01_TEST/sync_signals.npz",
        )
