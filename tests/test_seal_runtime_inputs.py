from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/seal_runtime_inputs.py"
SPEC = importlib.util.spec_from_file_location("seal_runtime_inputs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SEAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEAL)


def _fixture_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source.py"
    source.write_text("VERSION = 1\n", encoding="utf-8")
    tree = tmp_path / "cache"
    tree.mkdir()
    (tree / "maps.npy").write_bytes(b"maps")
    (tree / "metadata.csv").write_text("rr_bpm\n12.0\n", encoding="utf-8")
    binding = tmp_path / "plan.json"
    binding.write_text('{"plan": 1}\n', encoding="utf-8")
    return source, tree, binding


def test_snapshot_is_content_hashed_and_verifies_every_payload(tmp_path: Path) -> None:
    source, tree, binding = _fixture_inputs(tmp_path)
    document = SEAL.inventory(
        sources=[source], trees=[tree], bindings=[binding]
    )
    assert document["content_sha256"] == SEAL.canonical_sha256(document)
    assert document["post_launch_attestation"] is True
    assert document["attestation_phase"] == "post_launch"
    assert document["commercial_claim_authorized"] is False
    assert document["input_trees"][0]["file_count"] == 2

    output = tmp_path / "runtime_seal.json"
    SEAL.atomic_json(output, document)
    result = SEAL.verify(output)
    assert result["status"] == "verified"
    assert result["verified_files"] == 4


def test_verify_rejects_payload_or_tree_membership_drift(tmp_path: Path) -> None:
    source, tree, binding = _fixture_inputs(tmp_path)
    output = tmp_path / "runtime_seal.json"
    SEAL.atomic_json(
        output,
        SEAL.inventory(sources=[source], trees=[tree], bindings=[binding]),
    )

    payload = tree / "maps.npy"
    payload.write_bytes(b"MAPS")
    with pytest.raises(RuntimeError, match="runtime input changed after seal"):
        SEAL.verify(output)

    payload.write_bytes(b"maps")
    # Restoring bytes is insufficient because the attestation also binds mtime.
    document = SEAL.inventory(sources=[source], trees=[tree], bindings=[binding])
    second = tmp_path / "runtime_seal_2.json"
    SEAL.atomic_json(second, document)
    (tree / "unexpected.bin").write_bytes(b"new")
    with pytest.raises(RuntimeError, match="tree membership changed"):
        SEAL.verify(second)


def test_seal_is_immutable_and_duplicate_sources_fail_closed(tmp_path: Path) -> None:
    source, tree, binding = _fixture_inputs(tmp_path)
    with pytest.raises(RuntimeError, match="duplicates"):
        SEAL.inventory(
            sources=[source, source], trees=[tree], bindings=[binding]
        )

    output = tmp_path / "runtime_seal.json"
    document = SEAL.inventory(sources=[source], trees=[tree], bindings=[binding])
    SEAL.atomic_json(output, document)
    tampered = dict(document)
    tampered["classification"] = "tampered"
    tampered["content_sha256"] = SEAL.canonical_sha256(tampered)
    with pytest.raises(RuntimeError, match="refusing to replace immutable"):
        SEAL.atomic_json(output, tampered)


def test_verify_rejects_document_hash_tamper_before_touching_inputs(
    tmp_path: Path,
) -> None:
    source, tree, binding = _fixture_inputs(tmp_path)
    document = SEAL.inventory(sources=[source], trees=[tree], bindings=[binding])
    document["runtime"]["python"] = "forged"
    output = tmp_path / "runtime_seal.json"
    output.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RuntimeError, match="content hash mismatch"):
        SEAL.verify(output)


def test_prelaunch_phase_is_explicit_and_hash_bound(tmp_path: Path) -> None:
    source, tree, binding = _fixture_inputs(tmp_path)
    document = SEAL.inventory(
        sources=[source],
        trees=[tree],
        bindings=[binding],
        post_launch_attestation=False,
    )
    assert document["post_launch_attestation"] is False
    assert document["attestation_phase"] == "prelaunch"
    assert document["content_sha256"] == SEAL.canonical_sha256(document)
