from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/build_locked_hcs_targets_after_release_lock.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_locked_hcs_targets_after_release_lock", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def _write_json(path: Path, value: dict[str, Any], *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if immutable:
        path.chmod(0o444)


class Fixture:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = tmp_path / "locked"
        self.target = self.root / "canonical_targets.npz"
        self.target_receipt = self.root / "canonical_targets_receipt.json"
        self.wrapper_receipt = self.root / "release_receipt.json"
        self.lock_path = self.root / "pretarget_release_lock.json"
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.write_text("sealed lock bytes", encoding="utf-8")
        self.lock = {
            "classification": "locked_hcs_pretarget_release_lock",
            "content_sha256": "c" * 64,
            "locations": {
                "primary_root": str((self.root / "primary").resolve()),
                "canonical_target": str(self.target.resolve()),
                "canonical_target_receipt": str(self.target_receipt.resolve()),
                "release_receipt": str(self.wrapper_receipt.resolve()),
            },
        }
        self.events: list[str] = []
        self.build_calls = 0

        def validate(path: Path, *, require_target_absence: bool = True) -> dict[str, Any]:
            self.events.append(f"validate:{require_target_absence}")
            return self.lock

        def build_targets(**kwargs: Any) -> dict[str, Any]:
            self.events.append("target_builder")
            self.build_calls += 1
            assert kwargs["output"] == self.target.resolve()
            assert kwargs["receipt"] == self.target_receipt.resolve()
            assert self.events[-3:] == [
                "validate:True",
                "runtime_reverify",
                "target_builder",
            ]
            self.target.write_bytes(b"canonical target")
            self.target.chmod(0o444)
            document: dict[str, Any] = {
                "schema_version": 1,
                "classification": "retrospective_locked_hcs_canonical_target_artifact_receipt",
                "target_artifact_created_once": True,
                "target_artifact_overwrite_allowed": False,
                "commercial_claim_authorized": False,
                "target_artifact": BUILD.TARGETS.bind_file(self.target),
                "orchestrator_command": list(kwargs["orchestrator_command"]),
            }
            document["content_sha256"] = BUILD.TARGETS.canonical_json_sha256(document)
            _write_json(self.target_receipt, document, immutable=True)
            return document

        monkeypatch.setattr(BUILD.RELEASE, "validate_release_lock", validate)
        monkeypatch.setattr(
            BUILD.RELEASE,
            "reverify_runtime_inputs_from_lock",
            lambda document: self.events.append("runtime_reverify") or {"status": "verified"},
        )
        monkeypatch.setattr(BUILD.TARGETS, "build_targets", build_targets)

    def run(self) -> dict[str, Any]:
        return BUILD.build_targets_after_release_lock(
            release_lock_path=self.lock_path,
            cache_dir=self.root / "cache",
            fold_assignments=self.root / "folds.json",
            output=self.target,
            receipt=self.target_receipt,
            release_receipt=self.wrapper_receipt,
            orchestrator_command=[
                str(Path(BUILD.sys.executable).resolve()),
                str(Path(BUILD.__file__).resolve()),
                "--synthetic",
            ],
        )


def test_release_is_revalidated_before_target_builder_and_receipt_is_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Fixture(tmp_path, monkeypatch)
    result = fixture.run()
    assert fixture.events == [
        "validate:False",
        "validate:True",
        "runtime_reverify",
        "target_builder",
    ]
    assert result["classification"] == "locked_hcs_canonical_targets_built_after_pretarget_release"
    assert result["release_lock_revalidated_before_target_builder_call"] is True
    assert result["target_metadata_access_before_release_validation"] is False
    assert result["pretarget_release_lock"] == BUILD.RELEASE.bind_file(fixture.lock_path)
    assert fixture.wrapper_receipt.stat().st_mode & 0o777 == 0o444


def test_failed_release_validation_proves_target_builder_is_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Fixture(tmp_path, monkeypatch)

    def reject(path: Path, *, require_target_absence: bool = True) -> dict[str, Any]:
        raise BUILD.RELEASE.PretargetReleaseLockError("boundary incomplete")

    monkeypatch.setattr(BUILD.RELEASE, "validate_release_lock", reject)
    with pytest.raises(BUILD.ReleasedTargetBuildError, match="release validation failed"):
        fixture.run()
    assert fixture.build_calls == 0
    assert not fixture.target.exists()


def test_failed_immediate_runtime_reverify_proves_target_builder_is_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Fixture(tmp_path, monkeypatch)

    def reject(document: dict[str, Any]) -> dict[str, Any]:
        fixture.events.append("runtime_reverify")
        raise BUILD.RELEASE.PretargetReleaseLockError("RF payload changed")

    monkeypatch.setattr(BUILD.RELEASE, "reverify_runtime_inputs_from_lock", reject)
    with pytest.raises(BUILD.ReleasedTargetBuildError, match="runtime payload closure"):
        fixture.run()
    assert fixture.build_calls == 0
    assert fixture.events == ["validate:False", "validate:True", "runtime_reverify"]
    assert not fixture.target.exists()


def test_complete_publication_is_idempotent_without_second_target_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Fixture(tmp_path, monkeypatch)
    first = fixture.run()
    fixture.events.clear()
    second = fixture.run()
    assert first == second
    assert fixture.build_calls == 1
    assert fixture.events == ["validate:False"]


def test_resume_after_target_receipt_publishes_only_missing_wrapper_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.target.write_bytes(b"canonical target")
    fixture.target.chmod(0o444)
    document: dict[str, Any] = {
        "schema_version": 1,
        "classification": "retrospective_locked_hcs_canonical_target_artifact_receipt",
        "target_artifact_created_once": True,
        "target_artifact_overwrite_allowed": False,
        "commercial_claim_authorized": False,
        "target_artifact": BUILD.TARGETS.bind_file(fixture.target),
        "orchestrator_command": [
            str(Path(BUILD.sys.executable).resolve()),
            str(Path(BUILD.__file__).resolve()),
            "--previous-attempt",
        ],
    }
    document["content_sha256"] = BUILD.TARGETS.canonical_json_sha256(document)
    _write_json(fixture.target_receipt, document, immutable=True)
    result = fixture.run()
    assert result["canonical_target"] == BUILD.TARGETS.bind_file(fixture.target)
    assert fixture.build_calls == 0
    assert fixture.events == ["validate:False"]


def test_partial_target_publication_is_fail_closed_and_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.target.write_bytes(b"orphaned")
    with pytest.raises(BUILD.ReleasedTargetBuildError, match="partial canonical"):
        fixture.run()
    assert fixture.target.read_bytes() == b"orphaned"
    assert fixture.build_calls == 0


def test_direct_target_builder_receipt_cannot_be_adopted_as_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.target.write_bytes(b"canonical target")
    document: dict[str, Any] = {
        "schema_version": 1,
        "classification": "retrospective_locked_hcs_canonical_target_artifact_receipt",
        "target_artifact_created_once": True,
        "target_artifact_overwrite_allowed": False,
        "commercial_claim_authorized": False,
        "target_artifact": BUILD.TARGETS.bind_file(fixture.target),
        "orchestrator_command": ["python", str(Path(BUILD.TARGETS.__file__).resolve())],
    }
    document["content_sha256"] = BUILD.TARGETS.canonical_json_sha256(document)
    _write_json(fixture.target_receipt, document, immutable=True)
    fixture.target.chmod(0o444)
    with pytest.raises(BUILD.ReleasedTargetBuildError, match="not executed through"):
        fixture.run()
    assert fixture.build_calls == 0
    assert not fixture.wrapper_receipt.exists()


@pytest.mark.parametrize("role", ["target", "target_receipt"])
def test_resume_rejects_writable_canonical_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.run()
    path = getattr(fixture, role)
    path.chmod(0o644)
    with pytest.raises(BUILD.ReleasedTargetBuildError, match="mode must be exactly 0444"):
        fixture.run()
    assert fixture.build_calls == 1


@pytest.mark.parametrize("role", ["target", "target_receipt"])
def test_resume_rejects_locked_path_symlink_before_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    fixture = Fixture(tmp_path, monkeypatch)
    fixture.run()
    path = getattr(fixture, role)
    external = tmp_path / f"external_{role}"
    external.write_bytes(path.read_bytes())
    external.chmod(0o444)
    path.unlink()
    path.symlink_to(external)
    with pytest.raises(BUILD.ReleasedTargetBuildError, match="must not be a symlink"):
        fixture.run()
    assert fixture.build_calls == 1
