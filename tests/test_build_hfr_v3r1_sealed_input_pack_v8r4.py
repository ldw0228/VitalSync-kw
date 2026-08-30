from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts/build_hfr_v3r1_sealed_input_pack_v8r4.py"
SPEC = importlib.util.spec_from_file_location("v8r4_pack_builder", MODULE_PATH)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def write_json(path: Path, value: dict[str, Any], *, immutable: bool = False) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if immutable:
        path.chmod(0o444)


def add_content_hash(value: dict[str, Any]) -> dict[str, Any]:
    value = dict(value)
    value["content_sha256"] = builder.canonical_content_sha256(value)
    return value


def make_authorization(
    root: Path,
    *,
    scopes: tuple[str, ...] = (builder.PREDICTION_SCOPE,),
    name: str = "promotion_authorization.json",
) -> tuple[Any, Path]:
    path = root / name
    document = add_content_hash(
        {
            "classification": "adaptive_v3r1_v8r4_promotion_authorization",
            "campaign_id": builder.CAMPAIGN_ID,
            "campaign_revision": builder.PACK_REVISION,
            "authorized_now": True,
            "authorized_scopes": list(scopes),
        }
    )
    write_json(path, document, immutable=True)
    authorization = builder.validate_promotion_authorization(
        path,
        expected_sha256=sha256(path),
        expected_bytes=path.stat().st_size,
        required_scope=scopes[0],
    )
    return authorization, path


METADATA_COLUMNS = (
    "cache_index",
    "fold",
    "session_id",
    "identity",
    "window_number",
    "rr_bpm",
    "reference_valid",
    "classical_rr_bpm",
    "protocol",
    "reference_quality",
)


def write_metadata(
    path: Path,
    *,
    outer_fold: int,
    cache_index: np.ndarray,
    folds: np.ndarray,
) -> None:
    payload = bytearray(b",".join(name.encode("ascii") for name in METADATA_COLUMNS) + b"\n")
    seen = {fold: 0 for fold in range(6)}
    for row, (index, fold) in enumerate(zip(cache_index, folds, strict=True)):
        window = seen[int(fold)]
        seen[int(fold)] += 1
        protected = int(fold) == outer_fold
        identity = b"\xffPOISON_OUTER_ID" if protected else f"I{fold}".encode()
        rr = b"NOT_A_NUMERIC_REFERENCE" if protected else f"{12 + fold:.1f}".encode()
        valid = b"POISON_VALIDITY" if protected else b"True"
        quality = b"\xfePOISON_OUTER_QUALITY" if protected else b"0.9"
        fields = (
            str(int(index)).encode(),
            str(int(fold)).encode(),
            f"S{fold}".encode(),
            identity,
            str(window).encode(),
            rr,
            valid,
            f"{10 + fold:.1f}".encode(),
            b"POISON_OUTER_PROTOCOL" if protected else b"Rest",
            quality,
        )
        payload.extend(b",".join(fields) + b"\n")
    path.write_bytes(bytes(payload))


def make_source_unit(
    root: Path,
    *,
    outer_fold: int = 3,
    seed: int = 20260828,
    cache_index: np.ndarray | None = None,
    folds: np.ndarray | None = None,
    tamper_manifest_for: str | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    unit_root = root / f"legacy_outer_{outer_fold}_seed_{seed}"
    cache = unit_root / "cache"
    cache.mkdir(parents=True)
    rows = 12
    if folds is None:
        folds = np.repeat(np.arange(6, dtype=np.int16), 2)
    folds = np.asarray(folds, dtype=np.int16)
    if cache_index is None:
        cache_index = np.arange(len(folds), dtype=np.int64)
    cache_index = np.asarray(cache_index, dtype=np.int64)
    rows = len(folds)
    write_metadata(
        cache / "metadata.csv",
        outer_fold=outer_fold,
        cache_index=cache_index,
        folds=folds,
    )
    (cache / "feature_names.json").write_text(
        json.dumps({"node_feature_names": [f"f{i}" for i in range(571)]}) + "\n",
        encoding="utf-8",
    )
    candidate = np.linspace(8.0, 20.0, rows * 2, dtype=np.float32).reshape(rows, 2)
    node = np.arange(rows * 2 * 571, dtype=np.float32).reshape(rows, 2, 571) / 100.0
    candidate_mask = np.ones((rows, 2), dtype=np.bool_)
    radar = np.ones((rows, 3), dtype=np.bool_)
    np.save(cache / "candidate_bpm.npy", candidate, allow_pickle=False)
    np.save(cache / "node_features.npy", node, allow_pickle=False)
    np.save(cache / "candidate_mask.npy", candidate_mask, allow_pickle=False)
    np.save(cache / "joint_radar_mask.npy", radar, allow_pickle=False)

    stack_path = unit_root / "strict_stack.npz"
    available = folds != outer_fold
    prediction = np.where(available, 12.0 + folds, np.nan).astype(np.float32)
    rr_std = np.where(available, 1.25, np.nan).astype(np.float32)
    # Forbidden object entries are deliberate poison.  The producer must not
    # open them, and every output must remain loadable with allow_pickle=False.
    np.savez_compressed(
        stack_path,
        cache_index=cache_index,
        fold=folds,
        proposal_available=available.astype(np.bool_),
        nested_role=np.where(available, "safe_nonouter", "outer_test_unavailable"),
        prediction=prediction,
        rr_std=rr_std,
        outer_fold=np.asarray(outer_fold, np.int16),
        seed=np.asarray(seed, np.int64),
        strict_nested=np.asarray(True, np.bool_),
        outer_test_opened=np.asarray(False, np.bool_),
        identity=np.asarray([object() for _ in range(rows)], dtype=object),
        reference_rr_bpm=np.asarray([object() for _ in range(rows)], dtype=object),
        quality=np.asarray([object() for _ in range(rows)], dtype=object),
    )

    outputs: dict[str, Any] = {}
    logical_files = {
        "feature_names": "feature_names.json",
        "metadata": "metadata.csv",
        "node_features": "node_features.npy",
        "candidate_bpm": "candidate_bpm.npy",
        "candidate_mask": "candidate_mask.npy",
        "joint_radar_mask": "joint_radar_mask.npy",
    }
    for logical, filename in logical_files.items():
        file_path = cache / filename
        outputs[logical] = {
            "filename": filename,
            "sha256": sha256(file_path),
            "bytes": file_path.stat().st_size,
        }
    if tamper_manifest_for:
        outputs[tamper_manifest_for]["sha256"] = "0" * 64
    manifest = add_content_hash(
        {"format_version": 1, "complete": True, "outputs": outputs}
    )
    manifest_path = cache / "manifest.json"
    write_json(manifest_path, manifest)
    unit = {
        "outer_fold": outer_fold,
        "seed": seed,
        "artifacts": {
            "cache_manifest": bind(manifest_path),
            "strict_stack": bind(stack_path),
        },
    }
    return unit, manifest_path, stack_path


def make_index(
    root: Path,
    units: list[dict[str, Any]],
    *,
    immutable: bool = False,
) -> Path:
    document = add_content_hash(
        {
            "status": "complete",
            "outer_test_opened": False,
            "completed_units": len(units),
            "units": units,
        }
    )
    path = root / "legacy_index.json"
    write_json(path, document, immutable=immutable)
    return path


def load_unit_source(root: Path, unit: dict[str, Any]) -> tuple[Any, Any]:
    index = make_index(root, [unit])
    sources, index_binding = builder.load_training_index(
        root,
        index,
        expected_sha256=sha256(index),
        expected_bytes=index.stat().st_size,
        require_exact_matrix=False,
    )
    assert len(sources) == 1
    return sources[0], index_binding


def all_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def test_discovery_pack_excludes_poison_and_never_full_loads_npz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit, _, _ = make_source_unit(tmp_path, outer_fold=3)
    source, index_binding = load_unit_source(tmp_path, unit)
    output = tmp_path / "sealed"
    conversions: list[tuple[str, int]] = []
    indexed_reads: list[tuple[str, tuple[int, ...]]] = []
    original_take = builder.IndexedNpy.take

    def audited_take(self: Any, positions: np.ndarray) -> np.ndarray:
        indexed_reads.append((self.label, tuple(map(int, positions))))
        return original_take(self, positions)

    def forbidden_np_load(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("full-entry np.load is forbidden in producer")

    monkeypatch.setattr(builder, "NPZ_ROW_CONVERSION_HOOK", lambda name, row: conversions.append((name, row)))
    monkeypatch.setattr(builder.IndexedNpy, "take", audited_take)
    monkeypatch.setattr(builder.np, "load", forbidden_np_load)
    result = builder.build_unit_pack(source, index_binding=index_binding, output_root=output)

    nonouter = tuple(index for index in range(12) if index // 2 != 3)
    for name in ("proposal_available", "prediction", "rr_std"):
        assert tuple(row for field, row in conversions if field == name) == nonouter
    assert all(position in nonouter for _, positions in indexed_reads for position in positions)
    assert result["partition"]["discovery_outer_rows"] == 0
    assert result["partition"]["outer_prediction_pack_rows"] == 0
    assert result["preselection_prediction_boundary"]["outer_prediction_pack_absent"] is True
    assert not (output / "units/outer_3_seed_20260828/outer_predict_input.npz").exists()

    metadata = output / "units/outer_3_seed_20260828/discovery_cache/metadata.csv"
    raw = metadata.read_bytes()
    assert b"POISON" not in raw and b"\xff" not in raw and b"\xfe" not in raw
    with metadata.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(rows[0]) == builder.NONOUTER_METADATA_COLUMNS
    assert [int(row["cache_index"]) for row in rows] == list(nonouter)
    assert {int(row["fold"]) for row in rows} == {0, 1, 2, 4, 5}
    for path in all_files(output):
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
        assert path.stat().st_nlink == 1


def test_discovery_stack_and_manifest_match_consumer_exact_schema(tmp_path: Path) -> None:
    unit, _, _ = make_source_unit(tmp_path, outer_fold=4)
    source, index_binding = load_unit_source(tmp_path, unit)
    output = tmp_path / "sealed"
    builder.build_unit_pack(source, index_binding=index_binding, output_root=output)
    unit_root = output / "units/outer_4_seed_20260828"
    manifest = json.loads((unit_root / "discovery_cache/manifest.json").read_text())
    assert set(manifest) == {
        "schema_version", "classification", "campaign_id", "campaign_revision",
        "format_version", "complete", "outer_fold", "partition",
        "source_combined_cache_open_authorized_by_consumer",
        "outer_test_rows_physically_present", "outer_prediction_pack_absent",
        "inputs", "outputs", "content_sha256",
    }
    assert set(manifest["inputs"]) == {"source_combined_cache", "proposer_stack"}
    assert set(manifest["outputs"]) == set(builder.CACHE_FILES) | {
        "local_to_global_cache_index"
    }
    with np.load(unit_root / "discovery_proposer_stack.npz", allow_pickle=False) as archive:
        assert set(archive.files) == set(builder.DISCOVERY_STACK_FIELDS)
        assert all(not archive[name].dtype.hasobject for name in archive.files)
        assert archive["classification"].item() == builder.NONOUTER_STACK_CLASSIFICATION
        assert not archive["outer_rows_present"].item()
        assert set(map(int, archive["fold"])) == {0, 1, 2, 3, 5}
        assert all("outer" not in value and "test" not in value for value in archive["nested_role"])
    local_global = np.load(
        unit_root / "discovery_cache/local_to_global_cache_index.npy",
        allow_pickle=False,
    )
    assert local_global.dtype == np.int64
    assert np.array_equal(local_global, np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 10, 11]))
    partition = json.loads((unit_root / "PARTITION_MANIFEST.json").read_text())
    assert set(partition) == {
        "schema_version", "classification", "campaign_id", "campaign_revision",
        "outer_fold", "seed", "legacy_row_count", "partition", "legacy_inputs",
        "outputs", "integration_interface", "protected_outer_access",
        "preselection_prediction_boundary", "serialization", "claim_boundary",
        "content_sha256",
    }
    assert partition["classification"] == (
        "adaptive_v3r1_v8r4_sealed_nonouter_partition"
    )
    assert set(partition["outputs"]) == {
        "discovery_cache_manifest", "discovery_proposer_stack",
        "discovery_local_to_global_map",
    }
    assert partition["protected_outer_access"]["target_free_columns_decoded_after_partition"] == []
    assert partition["preselection_prediction_boundary"] == {
        "outer_prediction_pack_absent": True,
        "outer_prediction_path_bound": False,
        "outer_prediction_values_materialized": False,
        "promotion_authorization_required_before_prediction_pack": True,
    }
    # These are the pre-promotion producer bytes from the original discovery
    # implementation.  Cross-root equality alone would not catch a deterministic
    # ABI drift; fixed hashes prove the default path stayed byte-identical while
    # authorization fields were added only to the promoted path.
    assert {
        relative: (sha256(unit_root / relative), (unit_root / relative).stat().st_size)
        for relative in (
            "discovery_cache/manifest.json",
            "discovery_proposer_stack.npz",
            "discovery_cache/metadata.csv",
            "discovery_cache/local_to_global_cache_index.npy",
            "discovery_cache/node_features.npy",
            "discovery_cache/candidate_bpm.npy",
            "discovery_cache/candidate_mask.npy",
            "discovery_cache/joint_radar_mask.npy",
        )
    } == {
        "discovery_cache/manifest.json": (
            "0a80fae9b702040a294229d952c9846ea9297de4ef6fce4592e933927d161dce",
            2117,
        ),
        "discovery_proposer_stack.npz": (
            "d30a9f15eb78ce21438633e6e838b761dc0170cf921be90ff96e9321a2a8f2e1",
            4286,
        ),
        "discovery_cache/metadata.csv": (
            "d5a48d0d81ef38c8a12266bd054fea511c17622fafb5b09ae9ec5ff3a3d6bd04",
            363,
        ),
        "discovery_cache/local_to_global_cache_index.npy": (
            "e23da9389920059a38dfff944a3c35f947d99cc004ebca9558fe00fa42929de7",
            208,
        ),
        "discovery_cache/node_features.npy": (
            "654a710908d3c4be231cb2093896ad62c6e13dbe7062d69f42521861911516e1",
            45808,
        ),
        "discovery_cache/candidate_bpm.npy": (
            "23f5fd85282bfed57416b37425ed66b75afa8bee02aaabeb5d98a8c318ea660c",
            208,
        ),
        "discovery_cache/candidate_mask.npy": (
            "183a5edf78766b635195ccd300b79d9b36700a333cb7c887a385827b75e4b6ad",
            148,
        ),
        "discovery_cache/joint_radar_mask.npy": (
            "08a0d0dedc9c7fb6c9e7542d83f7815ae401d286b4d00ac7bd65eb2f62deb60b",
            158,
        ),
    }


@pytest.mark.parametrize(
    "indices,folds,error",
    [
        (np.asarray([0, 1, 1, 3, 4, 5]), np.arange(6), "contiguous"),
        (np.asarray([1, 0, 2, 3, 4, 5]), np.arange(6), "contiguous"),
        (np.arange(5), np.asarray([0, 1, 2, 3, 4]), "empty V8R4 partition"),
    ],
)
def test_row_duplicate_permutation_and_missing_fold_fail_closed(
    tmp_path: Path, indices: np.ndarray, folds: np.ndarray, error: str
) -> None:
    unit, _, _ = make_source_unit(
        tmp_path, outer_fold=5, cache_index=indices, folds=folds
    )
    source, index_binding = load_unit_source(tmp_path, unit)
    with pytest.raises(builder.PackError, match=error):
        builder.build_unit_pack(
            source, index_binding=index_binding, output_root=tmp_path / "sealed"
        )


def test_manifest_tamper_symlink_and_hardlink_are_rejected(tmp_path: Path) -> None:
    tamper_root = tmp_path / "tamper"
    unit, _, _ = make_source_unit(
        tamper_root, outer_fold=3, tamper_manifest_for="candidate_bpm"
    )
    source, index_binding = load_unit_source(tamper_root, unit)
    with pytest.raises(builder.PackError, match="SHA-256 drifted"):
        builder.build_unit_pack(
            source, index_binding=index_binding, output_root=tamper_root / "sealed"
        )

    symlink_root = tmp_path / "symlink"
    unit, manifest, _ = make_source_unit(symlink_root, outer_fold=3)
    real = manifest.with_suffix(".real")
    manifest.rename(real)
    manifest.symlink_to(real.name)
    unit["artifacts"]["cache_manifest"] = bind(real) | {"path": str(manifest)}
    index = make_index(symlink_root, [unit])
    sources, index_binding = builder.load_training_index(
        symlink_root, index, expected_sha256=sha256(index),
        expected_bytes=index.stat().st_size, require_exact_matrix=False,
    )
    with pytest.raises(builder.PackError, match="symlink"):
        builder.build_unit_pack(
            sources[0], index_binding=index_binding,
            output_root=symlink_root / "sealed",
        )

    hardlink_root = tmp_path / "hardlink"
    unit, _, stack_path = make_source_unit(hardlink_root, outer_fold=3)
    os.link(stack_path, stack_path.with_suffix(".alias"))
    source, index_binding = load_unit_source(hardlink_root, unit)
    with pytest.raises(builder.PackError, match="hard link"):
        builder.build_unit_pack(
            source, index_binding=index_binding, output_root=hardlink_root / "sealed"
        )


def test_create_once_resume_is_byte_deterministic_and_detects_output_attack(
    tmp_path: Path,
) -> None:
    unit, _, _ = make_source_unit(tmp_path, outer_fold=3)
    source, index_binding = load_unit_source(tmp_path, unit)
    output = tmp_path / "sealed"
    builder.build_unit_pack(source, index_binding=index_binding, output_root=output)
    before = {path.relative_to(output): sha256(path) for path in all_files(output)}
    builder.build_unit_pack(source, index_binding=index_binding, output_root=output)
    after = {path.relative_to(output): sha256(path) for path in all_files(output)}
    assert before == after

    protected = output / "units/outer_3_seed_20260828/discovery_cache/candidate_mask.npy"
    alias = tmp_path / "malicious-hardlink"
    os.link(protected, alias)
    with pytest.raises(builder.PackError, match="create-once|hard link"):
        builder.build_unit_pack(source, index_binding=index_binding, output_root=output)


def test_deterministic_bytes_across_roots_and_symlink_output_fail_closed(
    tmp_path: Path,
) -> None:
    unit, _, _ = make_source_unit(tmp_path, outer_fold=4)
    source, index_binding = load_unit_source(tmp_path, unit)
    first = tmp_path / "sealed_a"
    second = tmp_path / "sealed_b"
    builder.build_unit_pack(source, index_binding=index_binding, output_root=first)
    builder.build_unit_pack(source, index_binding=index_binding, output_root=second)
    first_hashes = {
        path.relative_to(first): sha256(path) for path in all_files(first)
    }
    second_hashes = {
        path.relative_to(second): sha256(path) for path in all_files(second)
    }
    assert first_hashes == second_hashes

    attacked = tmp_path / "attacked"
    target = tmp_path / "attacker_target"
    target.mkdir()
    attacked.symlink_to(target, target_is_directory=True)
    with pytest.raises(builder.PackError, match="symlink|unsafe"):
        builder.build_unit_pack(
            source, index_binding=index_binding, output_root=attacked
        )


def test_prediction_scope_requires_authorization_before_any_output(tmp_path: Path) -> None:
    output = tmp_path / "prediction"
    code = builder.main(
        [
            "--project-root", str(tmp_path),
            "--scope", "prediction",
            "--outer-fold", "3",
            "--seed", "20260828",
            "--output-root", str(output),
        ]
    )
    assert code == 2
    assert not output.exists()


def test_authorized_prediction_exact_allowlist_and_outer_only_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit, _, _ = make_source_unit(tmp_path, outer_fold=3)
    source, index_binding = load_unit_source(tmp_path, unit)
    authorization, _ = make_authorization(
        tmp_path,
        name="authorization.json",
    )
    conversions: list[tuple[str, int]] = []
    indexed: list[tuple[str, tuple[int, ...]]] = []
    original_take = builder.IndexedNpy.take

    def audited_take(self: Any, positions: np.ndarray) -> np.ndarray:
        indexed.append((self.label, tuple(map(int, positions))))
        return original_take(self, positions)

    monkeypatch.setattr(builder, "NPZ_ROW_CONVERSION_HOOK", lambda name, row: conversions.append((name, row)))
    monkeypatch.setattr(builder.IndexedNpy, "take", audited_take)
    result = builder.build_authorized_prediction_pack(
        source,
        authorization=authorization,
        index_binding=index_binding,
        output_root=tmp_path / "authorized_prediction_outer_3",
    )
    outer = (6, 7)
    for name in ("proposal_available", "prediction", "rr_std"):
        assert tuple(row for field, row in conversions if field == name) == outer
    assert all(position in outer for _, positions in indexed for position in positions)
    path = Path(result["manifest_binding"]["path"]).parent / "outer_predict_input.npz"
    with np.load(path, allow_pickle=False) as archive:
        assert tuple(archive.files) == builder.OUTER_PREDICT_FIELDS
        assert all(not archive[name].dtype.hasobject for name in archive.files)
        assert np.array_equal(archive["cache_index"], np.asarray(outer))
        assert archive["session_reset"].tolist() == [True, False]
        assert not any(
            token in name.lower()
            for name in archive.files
            for token in builder.OUTER_FORBIDDEN_TOKENS
        )
        assert not (set(archive.files) & set(builder.OUTER_FORBIDDEN_EXACT_FIELDS))


def test_promotion_authorization_is_exact_immutable_and_scoped(tmp_path: Path) -> None:
    path = tmp_path / "promotion.json"
    document = add_content_hash(
        {
            "classification": "adaptive_v3r1_v8r4_promotion_authorization",
            "campaign_id": builder.CAMPAIGN_ID,
            "campaign_revision": builder.PACK_REVISION,
            "authorized_now": True,
            "authorized_scopes": ["outer_prediction_pack"],
        }
    )
    write_json(path, document, immutable=True)
    authorization = builder.validate_promotion_authorization(
        path,
        expected_sha256=sha256(path),
        expected_bytes=path.stat().st_size,
        required_scope="outer_prediction_pack",
    )
    assert authorization.scopes == {"outer_prediction_pack"}
    with pytest.raises(builder.PackError, match="does not grant"):
        builder.validate_promotion_authorization(
            path, expected_sha256=sha256(path), expected_bytes=path.stat().st_size,
            required_scope="promotion_training_pack",
        )
    path.chmod(0o644)
    with pytest.raises(builder.PackError, match="0444"):
        builder.validate_promotion_authorization(
            path, expected_sha256=sha256(path), expected_bytes=path.stat().st_size,
            required_scope="outer_prediction_pack",
        )


def test_discovery_matrix_is_two_unmountable_three_seed_shards(tmp_path: Path) -> None:
    units = []
    legacy = tmp_path / "legacy"
    for outer_fold in range(6):
        for seed in builder.SEEDS:
            unit, _, _ = make_source_unit(
                legacy, outer_fold=outer_fold, seed=seed
            )
            units.append(unit)
    index = make_index(tmp_path, units)
    output = tmp_path / "split"
    result = builder.build_pack_matrix(
        project_root=tmp_path,
        training_index=index,
        output_root=output,
        expected_index_sha256=sha256(index),
        expected_index_bytes=index.stat().st_size,
        require_exact_matrix=True,
    )
    assert result["exact_outer_fold_cover"] is True
    assert result["target_bearing_pack_directories_bound_by_aggregator"] is False
    for outer_fold in (3, 4):
        shard = output / f"discovery_shard_outer_{outer_fold}"
        document = json.loads(
            (shard / "V8R4_NONOUTER_TRAINING_INDEX.json").read_text()
        )
        assert document["outer_fold"] == outer_fold
        assert document["unit_count"] == document["completed_units"] == 3
        assert [unit["seed"] for unit in document["units"]] == list(builder.SEEDS)
        assert {unit["outer_fold"] for unit in document["units"]} == {outer_fold}
        assert document["cross_outer_shard_mounted"] is False
    aggregator = json.loads(
        (output / "V8R4_DISCOVERY_SHARD_AGGREGATOR.json").read_text()
    )
    assert {entry["outer_fold"] for entry in aggregator["shards"]} == {3, 4}
    assert all(set(entry) == {
        "outer_fold", "index_manifest", "runtime_must_mount_only_this_shard"
    } for entry in aggregator["shards"])
    assert not any("cache_manifest" in entry for entry in aggregator["shards"])
    assert not any(
        path.name in {
            builder.PREDICTION_INDEX_FILENAME,
            "OUTER_PREDICTION_PACK_MANIFEST.json",
            "outer_predict_input.npz",
        }
        for path in output.rglob("*")
    )


def test_real_discovery_shard_index_hashes_are_unchanged_without_rebuild() -> None:
    candidates = (MODULE_PATH.parents[1], Path.cwd())
    project = next(
        (
            candidate
            for candidate in candidates
            if (candidate / "artifacts/runs").is_dir()
        ),
        None,
    )
    if project is None:
        pytest.skip("frozen real discovery packs are not present")
    expected = {
        3: (
            "db204cdba72e9f3023c58ef37c1761cfd4ec2f4310449f8eaeeef7003afadb9b",
            3172,
        ),
        4: (
            "1ce1b1a0154c609b1fa1693ff5702b3e81be178f8ddbdd7ec25f559c39029f0a",
            3172,
        ),
    }
    base = project / (
        "artifacts/runs/directed_harmonic_factor_expert_snn_v3r1/"
        "v8r4_split_inputs"
    )
    for outer_fold, binding in expected.items():
        index = (
            base
            / f"discovery_shard_outer_{outer_fold}"
            / "V8R4_NONOUTER_TRAINING_INDEX.json"
        )
        if not index.is_file():
            pytest.skip("frozen real discovery packs are not present")
        assert (sha256(index), index.stat().st_size) == binding


def test_promotion_training_binds_authorization_everywhere_and_aggregates_indexes(
    tmp_path: Path,
) -> None:
    units: list[dict[str, Any]] = []
    legacy = tmp_path / "legacy"
    for outer_fold in range(builder.N_FOLDS):
        for seed in builder.SEEDS:
            unit, _, _ = make_source_unit(
                legacy,
                outer_fold=outer_fold,
                seed=seed,
            )
            units.append(unit)
    index = make_index(tmp_path, units)
    authorization, _ = make_authorization(
        tmp_path,
        scopes=(builder.PROMOTION_TRAINING_SCOPE, builder.PREDICTION_SCOPE),
    )
    exact_authorization = authorization.binding.as_dict()
    output = tmp_path / "promotion_training"

    for outer_fold in builder.PROMOTION_TRAINING_FOLDS:
        result = builder.build_pack_matrix(
            project_root=tmp_path,
            training_index=index,
            output_root=output,
            expected_index_sha256=sha256(index),
            expected_index_bytes=index.stat().st_size,
            require_exact_matrix=True,
            selected_outer_fold=outer_fold,
            selected_seed=None,
            promotion_authorization=authorization,
        )
        assert result["campaign_revision"] == builder.PACK_REVISION
        assert result["promotion_scope"] == builder.PROMOTION_TRAINING_SCOPE
        assert result["promotion_authorization"] == exact_authorization
        assert result["exact_three_seed_cover"] is True
        shard = output / f"promotion_training_shard_outer_{outer_fold}"
        shard_index = json.loads(
            (shard / "V8R4_NONOUTER_TRAINING_INDEX.json").read_text()
        )
        assert set(shard_index) == {
            "schema_version", "classification", "campaign_id", "campaign_revision",
            "outer_fold", "seeds", "unit_count", "completed_units", "status",
            "outer_test_opened",
            "combined_target_bearing_cache_consumer_access_authorized",
            "physical_nonouter_training_packs", "outer_prediction_packs_absent",
            "cross_outer_shard_mounted", "promotion_scope",
            "promotion_authorization", "units", "content_sha256",
        }
        assert shard_index["outer_fold"] == outer_fold
        assert shard_index["campaign_revision"] == builder.PACK_REVISION
        assert shard_index["promotion_scope"] == builder.PROMOTION_TRAINING_SCOPE
        assert shard_index["promotion_authorization"] == exact_authorization
        assert [row["seed"] for row in shard_index["units"]] == list(builder.SEEDS)
        for row in shard_index["units"]:
            unit_root = shard / row["relative_path"]
            cache = json.loads(
                (unit_root / "discovery_cache/manifest.json").read_text()
            )
            partition = json.loads(
                (unit_root / "PARTITION_MANIFEST.json").read_text()
            )
            for document in (cache, partition):
                assert document["campaign_revision"] == builder.PACK_REVISION
                assert document["promotion_scope"] == builder.PROMOTION_TRAINING_SCOPE
                assert document["promotion_authorization"] == exact_authorization
                assert document["content_sha256"] == (
                    builder.canonical_content_sha256(document)
                )
            assert set(cache) == {
                "schema_version", "classification", "campaign_id",
                "campaign_revision", "format_version", "complete", "outer_fold",
                "partition", "source_combined_cache_open_authorized_by_consumer",
                "outer_test_rows_physically_present", "outer_prediction_pack_absent",
                "inputs", "outputs", "promotion_scope", "promotion_authorization",
                "content_sha256",
            }
            assert set(partition) == {
                "schema_version", "classification", "campaign_id",
                "campaign_revision", "outer_fold", "seed", "legacy_row_count",
                "partition", "legacy_inputs", "outputs", "integration_interface",
                "protected_outer_access", "preselection_prediction_boundary",
                "serialization", "claim_boundary", "promotion_scope",
                "promotion_authorization", "content_sha256",
            }

    aggregator_path = output / builder.PROMOTION_TRAINING_AGGREGATOR_FILENAME
    aggregator = json.loads(aggregator_path.read_text())
    assert aggregator["classification"] == (
        builder.PROMOTION_TRAINING_AGGREGATOR_CLASSIFICATION
    )
    assert aggregator["authorized_outer_folds"] == list(
        builder.PROMOTION_TRAINING_FOLDS
    )
    assert aggregator["seeds"] == list(builder.SEEDS)
    assert aggregator["exact_outer_fold_seed_cover"] is True
    assert aggregator["target_bearing_pack_directories_bound_by_aggregator"] is False
    assert aggregator["promotion_authorization"] == exact_authorization
    assert [row["outer_fold"] for row in aggregator["shards"]] == list(
        builder.PROMOTION_TRAINING_FOLDS
    )
    assert all(
        set(row) == {
            "outer_fold",
            "index_manifest",
            "runtime_must_mount_only_this_shard",
        }
        for row in aggregator["shards"]
    )
    for path in all_files(output):
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
        assert path.stat().st_nlink == 1
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o555
        for path in sorted(path for path in output.rglob("*") if path.is_dir())
    )


def test_promotion_training_rejects_scope_or_mutable_replay_before_output(
    tmp_path: Path,
) -> None:
    unit, _, _ = make_source_unit(
        tmp_path / "legacy",
        outer_fold=0,
        seed=builder.SEEDS[0],
    )
    index = make_index(tmp_path, [unit])
    prediction_only, _ = make_authorization(
        tmp_path,
        scopes=(builder.PREDICTION_SCOPE,),
        name="prediction_only.json",
    )
    denied_output = tmp_path / "denied_scope"
    with pytest.raises(builder.PackError, match="does not grant"):
        builder.build_pack_matrix(
            project_root=tmp_path,
            training_index=index,
            output_root=denied_output,
            expected_index_sha256=sha256(index),
            expected_index_bytes=index.stat().st_size,
            require_exact_matrix=False,
            selected_outer_fold=0,
            selected_seed=builder.SEEDS[0],
            promotion_authorization=prediction_only,
        )
    assert not denied_output.exists()

    training, training_path = make_authorization(
        tmp_path,
        scopes=(builder.PROMOTION_TRAINING_SCOPE,),
        name="training.json",
    )
    training_path.chmod(0o644)
    mutable_output = tmp_path / "mutable_replay"
    with pytest.raises(builder.PackError, match="0444"):
        builder.build_pack_matrix(
            project_root=tmp_path,
            training_index=index,
            output_root=mutable_output,
            expected_index_sha256=sha256(index),
            expected_index_bytes=index.stat().st_size,
            require_exact_matrix=False,
            selected_outer_fold=0,
            selected_seed=builder.SEEDS[0],
            promotion_authorization=training,
        )
    assert not mutable_output.exists()


def test_prediction_shard_is_exact_three_seed_target_free_abi_and_create_once(
    tmp_path: Path,
) -> None:
    units: list[dict[str, Any]] = []
    for seed in builder.SEEDS:
        unit, _, _ = make_source_unit(
            tmp_path / "legacy",
            outer_fold=3,
            seed=seed,
        )
        units.append(unit)
    index = make_index(tmp_path, units)
    sources, index_binding = builder.load_training_index(
        tmp_path,
        index,
        expected_sha256=sha256(index),
        expected_bytes=index.stat().st_size,
        require_exact_matrix=False,
    )
    authorization, _ = make_authorization(tmp_path)
    output = tmp_path / "prediction_shard_outer_3"
    result = builder.build_authorized_prediction_shard(
        sources,
        authorization=authorization,
        index_binding=index_binding,
        output_root=output,
        outer_fold=3,
    )
    assert result["status"] == "complete"
    assert result["unit_count"] == result["completed_units"] == 3
    assert result["physical_target_free_outer_prediction_packs"] is True
    assert result["outer_prediction_packs_absent"] is False
    assert result[
        "combined_target_bearing_cache_consumer_access_authorized"
    ] is False
    assert "physical_nonouter_training_packs" not in result
    assert result["promotion_authorization"] == authorization.binding.as_dict()

    index_path = output / builder.PREDICTION_INDEX_FILENAME
    document = json.loads(index_path.read_text())
    assert set(document) == {
        "schema_version", "classification", "campaign_id", "campaign_revision",
        "outer_fold", "seeds", "unit_count", "completed_units", "status",
        "outer_test_opened",
        "combined_target_bearing_cache_consumer_access_authorized",
        "physical_target_free_outer_prediction_packs",
        "outer_prediction_packs_absent", "cross_outer_shard_mounted",
        "promotion_authorization", "units", "content_sha256",
    }
    assert document["classification"] == builder.PREDICTION_INDEX_CLASSIFICATION
    assert document["campaign_revision"] == builder.PACK_REVISION
    assert document["seeds"] == list(builder.SEEDS)
    assert [(row["outer_fold"], row["seed"]) for row in document["units"]] == [
        (3, seed) for seed in builder.SEEDS
    ]
    for row in document["units"]:
        assert set(row["artifacts"]) == {
            "prediction_pack_manifest",
            "outer_predict_input",
        }
        unit_root = output / row["relative_path"]
        expected_paths = {
            "prediction_pack_manifest": (
                unit_root / "OUTER_PREDICTION_PACK_MANIFEST.json"
            ),
            "outer_predict_input": unit_root / "outer_predict_input.npz",
        }
        for role, path in expected_paths.items():
            binding = row["artifacts"][role]
            assert binding["path"] == path.relative_to(output).as_posix()
            assert binding["sha256"] == sha256(path)
            assert binding["bytes"] == path.stat().st_size
        manifest = json.loads(
            expected_paths["prediction_pack_manifest"].read_text()
        )
        assert manifest["promotion_authorization"] == authorization.binding.as_dict()
        assert manifest["fields"] == list(builder.OUTER_PREDICT_FIELDS)
        assert manifest["exact_allowlist"] is True
        with np.load(expected_paths["outer_predict_input"], allow_pickle=False) as archive:
            assert tuple(archive.files) == builder.OUTER_PREDICT_FIELDS
            assert all(not archive[name].dtype.hasobject for name in archive.files)
            assert not any(
                token in field.lower()
                for field in archive.files
                for token in ("target", "reference", "identity", "protocol", "quality", "qc")
            )

    before = {path.relative_to(output): sha256(path) for path in all_files(output)}
    builder.build_authorized_prediction_shard(
        sources,
        authorization=authorization,
        index_binding=index_binding,
        output_root=output,
        outer_fold=3,
    )
    after = {path.relative_to(output): sha256(path) for path in all_files(output)}
    assert before == after
    for path in all_files(output):
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
        assert path.stat().st_nlink == 1
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o555
        for path in (output, *sorted(path for path in output.rglob("*") if path.is_dir()))
    )


def test_filtered_prediction_build_never_publishes_final_index(tmp_path: Path) -> None:
    units: list[dict[str, Any]] = []
    for seed in builder.SEEDS:
        unit, _, _ = make_source_unit(
            tmp_path / "legacy",
            outer_fold=5,
            seed=seed,
        )
        units.append(unit)
    index = make_index(tmp_path, units)
    sources, index_binding = builder.load_training_index(
        tmp_path,
        index,
        expected_sha256=sha256(index),
        expected_bytes=index.stat().st_size,
        require_exact_matrix=False,
    )
    authorization, _ = make_authorization(tmp_path)
    output = tmp_path / "filtered_prediction"
    result = builder.build_authorized_prediction_shard(
        sources,
        authorization=authorization,
        index_binding=index_binding,
        output_root=output,
        outer_fold=5,
        selected_seed=builder.SEEDS[1],
    )
    assert result["status"] == "filtered_unit_complete"
    assert result["unit_count"] == result["completed_units"] == 1
    assert result["index_binding"] is None
    assert not (output / builder.PREDICTION_INDEX_FILENAME).exists()
    assert (
        output
        / f"units/outer_5_seed_{builder.SEEDS[1]}/outer_predict_input.npz"
    ).is_file()


def test_model_bound_successor_pack_copies_exact_models_and_is_idempotent(
    tmp_path: Path,
) -> None:
    units: list[dict[str, Any]] = []
    for seed in builder.SEEDS:
        unit, _, _ = make_source_unit(
            tmp_path / "legacy", outer_fold=2, seed=seed
        )
        units.append(unit)
    index = make_index(tmp_path, units)
    sources, index_binding = builder.load_training_index(
        tmp_path,
        index,
        expected_sha256=sha256(index),
        expected_bytes=index.stat().st_size,
        require_exact_matrix=False,
    )
    authorization, _ = make_authorization(tmp_path)
    selection = add_content_hash(
        {
            "classification": "adaptive_v3r1_v8r4_global_discovery_selection_lock",
            "campaign_id": builder.CAMPAIGN_ID,
            "campaign_revision": builder.PACK_REVISION,
            "selected_variant": "H1_log_factor",
            "promotion_eligible": True,
            "promotion_authorized": True,
            "outer_test_features_or_targets_used": False,
            "commercial_claim_authorized": False,
        }
    )
    selection_path = tmp_path / "DISCOVERY_SELECTION_LOCK.json"
    write_json(selection_path, selection, immutable=True)
    model_sources: dict[int, Any] = {}
    for seed in builder.SEEDS:
        source_root = tmp_path / "models" / str(seed)
        source_root.mkdir(parents=True)
        checkpoint = source_root / "best.pt"
        scaler = source_root / "scaler.json"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        scaler.write_text('{"scale":1}\n', encoding="utf-8")
        signature = hashlib.sha256(f"signature-{seed}".encode()).hexdigest()
        receipt = add_content_hash(
            {
                "campaign_id": builder.CAMPAIGN_ID,
                "campaign_revision": builder.PACK_REVISION,
                "outer_fold": 2,
                "seed": seed,
                "variant": "H1_log_factor",
                "scientific_signature_sha256": signature,
                "outer_test_opened": False,
                "commercial_claim_authorized": False,
            }
        )
        receipt_path = source_root / "completion_receipt.json"
        write_json(receipt_path, receipt)
        for path in (checkpoint, scaler, receipt_path):
            path.chmod(0o444)
        model_sources[seed] = SimpleNamespace(
            kind="local_training",
            receipt_path=receipt_path,
            checkpoint=checkpoint,
            scaler=scaler,
            scientific_signature_sha256=signature,
            artifacts={"best.pt": bind(checkpoint), "scaler.json": bind(scaler)},
            receipt=receipt,
        )
    output = tmp_path / "model_bound_outer_2"
    result = builder.build_model_bound_prediction_shard(
        sources,
        model_sources=model_sources,
        authorization=authorization,
        index_binding=index_binding,
        selection_lock_path=selection_path,
        selected_variant="H1_log_factor",
        output_root=output,
        outer_fold=2,
    )
    assert result["classification"] == builder.MODEL_BOUND_PREDICTION_INDEX_CLASSIFICATION
    assert result["status"] == "complete"
    assert result["model_source_shard_seal_binding"] is not None
    for row in result["units"]:
        unit_root = output / row["relative_path"]
        assert {path.name for path in unit_root.iterdir()} == {
            "OUTER_PREDICTION_PACK_MANIFEST.json",
            builder.MODEL_BOUND_PREDICTION_MANIFEST_FILENAME,
            builder.MODEL_SOURCE_CAPABILITY_FILENAME,
            "outer_predict_input.npz",
            "model_checkpoint.pt",
            "model_scaler.json",
        }
        seed = int(row["seed"])
        assert (unit_root / "model_checkpoint.pt").read_bytes() == (
            model_sources[seed].checkpoint.read_bytes()
        )
        assert (unit_root / "model_scaler.json").read_bytes() == (
            model_sources[seed].scaler.read_bytes()
        )
    before = {path.relative_to(output): sha256(path) for path in all_files(output)}
    replay = builder.build_model_bound_prediction_shard(
        sources,
        model_sources=model_sources,
        authorization=authorization,
        index_binding=index_binding,
        selection_lock_path=selection_path,
        selected_variant="H1_log_factor",
        output_root=output,
        outer_fold=2,
    )
    assert replay["content_sha256"] == result["content_sha256"]
    assert before == {
        path.relative_to(output): sha256(path) for path in all_files(output)
    }
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o444
        and path.stat().st_nlink == 1
        for path in all_files(output)
    )
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o555
        for path in (output, *sorted(path for path in output.rglob("*") if path.is_dir()))
    )


def test_model_bound_successor_rejects_source_tamper_without_replacing_output(
    tmp_path: Path,
) -> None:
    unit, _, _ = make_source_unit(tmp_path / "legacy", outer_fold=1)
    source, index_binding = load_unit_source(tmp_path, unit)
    authorization, _ = make_authorization(tmp_path)
    selection_path = tmp_path / "selection.json"
    write_json(
        selection_path,
        add_content_hash(
            {
                "classification": "adaptive_v3r1_v8r4_global_discovery_selection_lock",
                "campaign_id": builder.CAMPAIGN_ID,
                "campaign_revision": builder.PACK_REVISION,
                "selected_variant": "H0_no_factor",
                "promotion_eligible": True,
                "promotion_authorized": True,
                "outer_test_features_or_targets_used": False,
                "commercial_claim_authorized": False,
            }
        ),
        immutable=True,
    )
    checkpoint = tmp_path / "best.pt"
    scaler = tmp_path / "scaler.json"
    checkpoint.write_bytes(b"model")
    scaler.write_bytes(b"{}")
    signature = "a" * 64
    receipt = add_content_hash(
        {
            "campaign_id": builder.CAMPAIGN_ID,
            "campaign_revision": builder.PACK_REVISION,
            "outer_fold": 1,
            "seed": builder.SEEDS[0],
            "variant": "H0_no_factor",
            "scientific_signature_sha256": signature,
            "outer_test_opened": False,
            "commercial_claim_authorized": False,
        }
    )
    receipt_path = tmp_path / "receipt.json"
    write_json(receipt_path, receipt)
    model_source = SimpleNamespace(
        kind="local_training",
        receipt_path=receipt_path,
        checkpoint=checkpoint,
        scaler=scaler,
        scientific_signature_sha256=signature,
        artifacts={"best.pt": bind(checkpoint), "scaler.json": bind(scaler)},
        receipt=receipt,
    )
    for path in (checkpoint, scaler, receipt_path):
        path.chmod(0o444)
    checkpoint.chmod(0o644)
    with pytest.raises(builder.PackError, match="immutable mode 0444"):
        builder.build_model_bound_prediction_pack(
            source,
            model_source=model_source,
            authorization=authorization,
            index_binding=index_binding,
            selection_lock_path=selection_path,
            selected_variant="H0_no_factor",
            output_root=tmp_path / "output",
        )
    assert not (tmp_path / "output").exists()
