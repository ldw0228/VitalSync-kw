from __future__ import annotations

import importlib.util
import json
import os
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


def test_v3_staging_is_private_atomic_and_no_replace(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    final = tmp_path / "acquisition_v3"
    first = _RECON._prepare_v3_staging_root(
        final, protected_sources=[source]
    )
    second = _RECON._prepare_v3_staging_root(
        final, protected_sources=[source]
    )
    assert not final.exists()
    assert os.stat(first).st_mode & 0o777 == 0o700
    (first / "manifest.json").write_text("{}", encoding="utf-8")
    (second / "manifest.json").write_text('{"other":true}', encoding="utf-8")
    _RECON._fsync_v3_staging_tree(first)
    _RECON._publish_v3_root_noreplace(first, final)
    assert (final / "manifest.json").read_text(encoding="utf-8") == "{}"
    with pytest.raises(FileExistsError, match="concurrently published"):
        _RECON._publish_v3_root_noreplace(second, final)
    assert second.is_dir()


@pytest.mark.parametrize("digest", ["A" * 64, "1" * 63, "g" * 64, True])
def test_v3_path_snapshot_cannot_claim_executed_loader_binding(
    digest: object,
) -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _RECON.v3_diagnostic_execution_source_generation(digest)  # type: ignore[arg-type]

    diagnostic = _RECON.v3_diagnostic_execution_source_generation("1" * 64)
    assert diagnostic["guard_scope"] == "post_import_path_snapshot_only"
    assert diagnostic["binds_actual_loader_compiled_bytes"] is False
    assert diagnostic["complete_private_import_closure"] is False
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["scientific_authority"] is False


def test_pipeline_path_snapshot_rejects_symlink_and_hardlink(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    symlink = tmp_path / "symlink.py"
    symlink.symlink_to(source)
    with pytest.raises(ValueError, match="cannot snapshot|unalias"):
        _RECON._sha256_paths([symlink])

    hardlink = tmp_path / "hardlink.py"
    os.link(source, hardlink)
    with pytest.raises(ValueError, match="unalias"):
        _RECON._sha256_paths([source])


def test_v3_staging_rejects_overlap_existing_and_dangling_symlink(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    with pytest.raises(ValueError, match="overlaps protected source"):
        _RECON._prepare_v3_staging_root(
            source / "derived", protected_sources=[source]
        )

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="must be absent"):
        _RECON._prepare_v3_staging_root(existing, protected_sources=[source])

    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    with pytest.raises(FileExistsError, match="must be absent"):
        _RECON._prepare_v3_staging_root(dangling, protected_sources=[source])


def test_v3_force_fails_before_creating_session_output(tmp_path: Path) -> None:
    output_root = tmp_path / "unpublished"
    subject = SimpleNamespace(subject_id="S24_KHJ", usable=False)
    with pytest.raises(ValueError, match="forbids --force"):
        _RECON.reconstruct_subject(
            subject,
            dataset_root=tmp_path / "raw",
            output_root=output_root,
            sync_config=SimpleNamespace(),
            protocol_config=SimpleNamespace(),
            session_record=None,
            approval_dir=None,
            build_range_tracks=False,
            build_review_plot=False,
            layout_maximum_frames=1,
            reconstruction_context={"pipeline_sha256": "1" * 64},
            force=True,
            schema_version=_RECON.SCHEMA_VERSION_V3,
        )
    assert not output_root.exists()


def test_v3_full_cohort_root_keeps_30_selected_and_projects_29_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The producer catalogue includes the excluded session; consumers project it."""

    cohort_path = _RECON.PROJECT_ROOT / "configs/acquisition_cohort_v1.yaml"
    cohort = _RECON.load_acquisition_cohort_authority(cohort_path)
    expected_ids = cohort.expected_session_ids
    expected_usable_ids = cohort.expected_usable_session_ids
    assert len(expected_ids) == 30
    assert len(expected_usable_ids) == 29

    dataset_root = tmp_path / "raw"
    dataset_root.mkdir()
    sync_path = tmp_path / "sync.yaml"
    protocol_path = tmp_path / "protocol.yaml"
    spreadsheet_path = tmp_path / "issues.xlsx"
    local_cohort_path = tmp_path / "cohort.yaml"
    for path in (sync_path, protocol_path, spreadsheet_path, local_cohort_path):
        path.write_bytes(b"fixture")
    final_root = tmp_path / "reconstruction_v3"

    identities = cohort.session_identity_map
    usable_set = set(expected_usable_ids)
    subjects = tuple(
        SimpleNamespace(subject_id=session_id, usable=session_id in usable_set)
        for session_id in expected_ids
    )
    dataset = SimpleNamespace(
        subjects=subjects,
        to_dict=lambda: {"session_ids": list(expected_ids)},
    )
    args = SimpleNamespace(
        schema_version="v3",
        force=False,
        dataset_root=dataset_root,
        sync_config=sync_path,
        protocol_config=protocol_path,
        spreadsheet=spreadsheet_path,
        cohort_authority=local_cohort_path,
        output_dir=final_root,
        approval_dir=None,
        subjects=None,
        skip_range_tracks=True,
        skip_review_plots=True,
        layout_maximum_frames=1,
    )

    monkeypatch.setattr(_RECON, "parse_args", lambda: args)
    monkeypatch.setattr(
        _RECON, "load_synchronization_config", lambda *args, **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(
        _RECON, "load_protocol_config", lambda *args, **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(
        _RECON, "load_acquisition_cohort_authority", lambda *args, **kwargs: cohort
    )
    monkeypatch.setattr(
        _RECON, "load_dataset_issue_records", lambda *args, **kwargs: ()
    )
    monkeypatch.setattr(_RECON, "records_by_session", lambda records: {})
    monkeypatch.setattr(_RECON, "build_dataset_manifest", lambda root: dataset)
    monkeypatch.setattr(
        _RECON, "_validate_dataset_against_cohort_authority", lambda *args: None
    )
    monkeypatch.setattr(_RECON, "_sha256_file", lambda path: "a" * 64)
    monkeypatch.setattr(_RECON, "_sha256_paths", lambda paths: "b" * 64)
    monkeypatch.setattr(_RECON, "_atomic_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        _RECON, "_session_rows", lambda document: ({}, [], [])
    )
    monkeypatch.setattr(_RECON, "_fsync_v3_staging_tree", lambda root: None)

    def fake_reconstruct(subject: SimpleNamespace, **kwargs: object) -> dict[str, object]:
        usable = subject.usable is True
        document: dict[str, object] = {
            "schema_version": _RECON.SCHEMA_VERSION_V3,
            "session_id": subject.subject_id,
            "physical_identity": identities[subject.subject_id],
            "usable": usable,
            "scientific_eligible": False,
            "raw_consumed_bytes_verified": usable,
            "timing_adjudicated": usable,
            "sync_raw_replay_verified": False,
            "protocol_raw_replay_verified": False,
            "synchronization": {"authorized": False},
            "eligibility": {"measured_timing_eligible": usable},
        }
        document["content_sha256"] = _RECON.canonical_content_sha256(document)
        session_dir = (
            Path(kwargs["output_root"])
            / "sessions"
            / str(subject.subject_id)
        )
        session_dir.mkdir(parents=True, exist_ok=True)
        _RECON._atomic_json(session_dir / "session_manifest.json", document)
        return document

    monkeypatch.setattr(_RECON, "reconstruct_subject", fake_reconstruct)
    monkeypatch.setattr(
        _RECON,
        "_validate_session_publication",
        lambda subject, document, **kwargs: document,
    )

    assert _RECON.main() == 0
    root = json.loads((final_root / "manifest.json").read_text(encoding="utf-8"))
    entries = root["sessions"]
    usable_entries = tuple(
        entry["session_id"] for entry in entries if entry["usable"] is True
    )

    assert tuple(root["expected_session_ids"]) == expected_ids
    assert tuple(root["selected_session_ids"]) == expected_ids
    assert tuple(entry["session_id"] for entry in entries) == expected_ids
    assert usable_entries == expected_usable_ids
    assert root["dataset_session_count"] == 30
    assert root["selected_session_count"] == 30
    assert root["session_count"] == 30
    assert root["dataset_usable_session_count"] == 29
    assert root["usable_session_count"] == 29
    assert root["raw_consumed_bytes_verified"] is True
    assert root["timing_adjudicated"] is True
    assert root["sync_raw_replay_verified"] is False
    assert root["protocol_raw_replay_verified"] is False
    assert root["scientific_eligible"] is False
    assert root["execution_source_generation"] == {
        "schema": "snn_rr.acquisition_execution_source_diagnostic.v1",
        "guard_scope": "post_import_path_snapshot_only",
        "pipeline_path_snapshot_sha256": "b" * 64,
        "binds_actual_loader_compiled_bytes": False,
        "complete_private_import_closure": False,
        "diagnostic_only": True,
        "scientific_authority": False,
    }
    assert "actual_loader_compiled_bytes_unbound" in root[
        "strict_failure_reasons"
    ]
    excluded = next(entry for entry in entries if entry["session_id"] == "S24_KHJ")
    assert excluded["usable"] is False
    assert excluded["raw_consumed_bytes_verified"] is False
    assert excluded["timing_adjudicated"] is False
    assert excluded["sync_authorized"] is False
