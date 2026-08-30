from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "reconstruct_acquisition.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "reconstruct_acquisition_publication", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_RECON = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RECON
_SPEC.loader.exec_module(_RECON)


def _document(context: dict[str, object]) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": _RECON.SCHEMA_VERSION,
        "session_id": "S24_TEST",
        "usable": False,
        "reconstruction_context": context,
        "scientific_eligible": False,
        "eligibility": {
            "measured_timing_eligible": False,
            "alignment_eligible": False,
            "stage_metric_eligible": False,
            "range_feature_eligible": False,
            "strict_cache_eligible": False,
        },
        "reason": "missing paired three-radar/BIOPAC recording",
    }
    value["content_sha256"] = _RECON.canonical_content_sha256(value)
    return value


def test_publication_barrier_rejects_child_changed_after_worker(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "reconstruction"
    session_dir = output_root / "sessions" / "S24_TEST"
    session_dir.mkdir(parents=True)
    context = {"pipeline_sha256": "1" * 64}
    original = _document(context)
    manifest_path = session_dir / "session_manifest.json"
    manifest_path.write_text(json.dumps(original), encoding="utf-8")
    subject = SimpleNamespace(subject_id="S24_TEST", usable=False)

    verified = _RECON._validate_session_publication(
        subject,
        original,
        dataset_root=tmp_path / "dataset",
        output_root=output_root,
        approval_dir=None,
        reconstruction_context=context,
    )
    assert verified == original

    changed = dict(original)
    changed["reason"] = "changed after worker"
    changed["content_sha256"] = _RECON.canonical_content_sha256(changed)
    manifest_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="in-memory/session manifest mismatch"):
        _RECON._validate_session_publication(
            subject,
            original,
            dataset_root=tmp_path / "dataset",
            output_root=output_root,
            approval_dir=None,
            reconstruction_context=context,
        )


def test_publication_barrier_rejects_unresealed_child_tamper(tmp_path: Path) -> None:
    output_root = tmp_path / "reconstruction"
    session_dir = output_root / "sessions" / "S24_TEST"
    session_dir.mkdir(parents=True)
    context = {"pipeline_sha256": "1" * 64}
    original = _document(context)
    tampered = dict(original)
    tampered["reason"] = "unsealed tamper"
    (session_dir / "session_manifest.json").write_text(
        json.dumps(tampered), encoding="utf-8"
    )
    subject = SimpleNamespace(subject_id="S24_TEST", usable=False)
    with pytest.raises(RuntimeError, match="content hash mismatch"):
        _RECON._validate_session_publication(
            subject,
            original,
            dataset_root=tmp_path / "dataset",
            output_root=output_root,
            approval_dir=None,
            reconstruction_context=context,
        )


def test_unusable_cached_reconstruction_reuses_without_sync_receipt(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "reconstruction"
    subject = SimpleNamespace(subject_id="S24_KHJ", usable=False)
    context = {"pipeline_sha256": "1" * 64}
    kwargs = {
        "dataset_root": tmp_path / "dataset",
        "output_root": output_root,
        "sync_config": SimpleNamespace(),
        "protocol_config": SimpleNamespace(),
        "session_record": None,
        "approval_dir": None,
        "build_range_tracks": False,
        "build_review_plot": False,
        "layout_maximum_frames": 1,
        "reconstruction_context": context,
    }

    created = _RECON.reconstruct_subject(subject, force=True, **kwargs)
    reused = _RECON.reconstruct_subject(subject, force=False, **kwargs)

    assert reused == created
    assert not (output_root / "sessions" / "S24_KHJ" / "sync_receipt.json").exists()


_RESET_WARNING = "unwrapped 1 relative-timestamp counter reset(s)"


def _warning_config() -> SimpleNamespace:
    return SimpleNamespace(
        radar_metadata_warning_allowlist={"S07_KDM": [_RESET_WARNING]}
    )


def test_radar_metadata_warning_policy_accepts_only_exact_per_view_content() -> None:
    evidence = _RECON._radar_metadata_warning_evidence(
        session_id="S07_KDM",
        warnings_by_view=[[_RESET_WARNING], [_RESET_WARNING], [_RESET_WARNING]],
        sync_config=_warning_config(),
    )

    assert evidence["metadata_warnings_eligible"] is True
    assert evidence["metadata_warning_policy"]["session_allowlist_declared"] is True
    assert [
        view["exact_match"] for view in evidence["metadata_warning_views"]
    ] == [True, True, True]
    assert len(evidence["metadata_warning_evidence_sha256"]) == 64


@pytest.mark.parametrize(
    "warnings_by_view",
    [
        [[], [_RESET_WARNING], [_RESET_WARNING]],
        [[_RESET_WARNING], ["different metadata warning"], [_RESET_WARNING]],
        [
            [_RESET_WARNING],
            [_RESET_WARNING],
            [_RESET_WARNING, "unexpected extra warning"],
        ],
    ],
)
def test_radar_metadata_warning_policy_rejects_missing_changed_or_extra_content(
    warnings_by_view: list[list[str]],
) -> None:
    evidence = _RECON._radar_metadata_warning_evidence(
        session_id="S07_KDM",
        warnings_by_view=warnings_by_view,
        sync_config=_warning_config(),
    )

    assert evidence["metadata_warnings_eligible"] is False
    assert any(
        view["exact_match"] is False
        for view in evidence["metadata_warning_views"]
    )


def test_radar_metadata_warning_policy_unlisted_session_requires_no_warnings() -> None:
    clean = _RECON._radar_metadata_warning_evidence(
        session_id="S02_RJS",
        warnings_by_view=[[], [], []],
        sync_config=_warning_config(),
    )
    unlisted_warning = _RECON._radar_metadata_warning_evidence(
        session_id="S02_RJS",
        warnings_by_view=[[], ["arbitrary warning"], []],
        sync_config=_warning_config(),
    )

    assert clean["metadata_warnings_eligible"] is True
    assert clean["metadata_warning_policy"]["session_allowlist_declared"] is False
    assert unlisted_warning["metadata_warnings_eligible"] is False
    assert (
        clean["metadata_warning_evidence_sha256"]
        != unlisted_warning["metadata_warning_evidence_sha256"]
    )
