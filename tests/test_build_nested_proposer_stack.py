from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from snn_rr.split_authority import (
    canonical_content_sha256,
    load_identity_split_authority,
    sha256_file,
)


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_nested_proposer_stack.py"
_SPEC = importlib.util.spec_from_file_location("build_nested_proposer_stack", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_BUILD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BUILD
_SPEC.loader.exec_module(_BUILD)
build_nested_stack = _BUILD.build_nested_stack


SEED = 17
OUTER = 3
FOLDS = (0, 1, 2, 3, 4, 5)
PREDICTION_FOLDS = (0, 1, 2, 5, 4)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _make_fixture(root: Path) -> dict[str, Path]:
    cache = root / "cache"
    sessions = []
    identities = [f"I{fold}" for fold in FOLDS]
    for fold, identity in zip(FOLDS, identities, strict=True):
        session_id = f"S{fold}_{identity}"
        frame = pd.DataFrame(
            {
                "session_id": [session_id, session_id],
                "identity": [identity, identity],
                "protocol": ["rest", "move"],
                "window_number": [0, 1],
                "window_start_s": [0.0, 4.0],
                "window_end_s": [32.0, 36.0],
                "reference_valid": [True, fold % 2 == 0],
                "rr_bpm": [12.0 + fold, 13.0 + fold],
            }
        )
        directory = cache / session_id
        directory.mkdir(parents=True)
        frame.to_csv(directory / "metadata.csv", index=False)
        sessions.append(
            {"session_id": session_id, "status": "ok", "window_count": len(frame)}
        )
    cache_manifest = cache / "manifest.json"
    _json(cache_manifest, {"sessions": sessions})
    fold_path = root / "fold_assignments.json"
    _json(fold_path, {"identity_to_fold": {identity: fold for fold, identity in zip(FOLDS, identities, strict=True)}})

    manifest_root = root / "manifests"
    unit_descriptions: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    cache_sha = sha256_file(cache_manifest)
    fold_sha = sha256_file(fold_path)
    metadata = pd.concat(
        [pd.read_csv(cache / f"S{fold}_I{fold}" / "metadata.csv") for fold in FOLDS],
        ignore_index=True,
    )
    role_names = ["hcs_train_oof"] * 4 + ["hcs_validation"]
    stems = [f"inner_pred_{fold}" for fold in PREDICTION_FOLDS[:4]] + ["validation_pred_4"]
    for unit_number, (prediction_fold, role, stem) in enumerate(
        zip(PREDICTION_FOLDS, role_names, stems, strict=True)
    ):
        prediction_identity = f"I{prediction_fold}"
        other = [identity for identity in identities if identity != prediction_identity]
        document = {
            "schema_version": 1,
            "fold_id": 300 + unit_number,
            "identities": {
                "train": sorted(other[:2]),
                "validation": sorted(other[2:3]),
                "prediction": [prediction_identity],
                "excluded": sorted(other[3:]),
                "scaler": sorted(other[:2]),
            },
            "fold_assignments": {"path": str(fold_path), "sha256": fold_sha},
            "cache": {"manifest_path": str(cache_manifest), "manifest_sha256": cache_sha},
        }
        document["content_sha256"] = canonical_content_sha256(document)
        manifest_path = manifest_root / f"outer_{OUTER}" / f"{stem}.json"
        _json(manifest_path, document)

        unit_dir = root / "runs" / f"seed_{SEED}" / f"outer_{OUTER}" / stem
        fold_dir = unit_dir / f"fold_{document['fold_id']}"
        fold_dir.mkdir(parents=True)
        provenance = load_identity_split_authority(
            manifest_path, metadata=metadata, cache_dir=cache
        ).checkpoint_provenance()
        checkpoint_path = fold_dir / "snn_best.pt"
        torch.save(
            {
                "format_version": 2,
                "model_type": "snn",
                "fold": document["fold_id"],
                "run_signature": "unit-run",
                "split_authority_provenance": provenance,
            },
            checkpoint_path,
        )
        _json(
            unit_dir / "run_config.json",
            {
                "run_signature": "unit-run",
                "arguments": {
                    "seed": SEED,
                    "identity_split_manifest_sha256": document["content_sha256"],
                },
            },
        )
        checkpoint_sha = sha256_file(checkpoint_path)
        positions = np.flatnonzero(metadata["identity"].to_numpy() == prediction_identity)
        rows = metadata.iloc[positions]
        prediction_provenance = {
            "strict_nested_role": "prediction",
            "labels_forwarded_to_model": False,
            "checkpoint_sha256": checkpoint_sha,
            "split_manifest_content_sha256": document["content_sha256"],
        }
        inference_signature = _canonical_hash(prediction_provenance)
        prediction_provenance["inference_signature_sha256"] = inference_signature
        n = len(positions)
        prediction_path = fold_dir / "snn_prediction_all_windows.npz"
        np.savez_compressed(
            prediction_path,
            cache_index=positions,
            session_id=rows["session_id"].astype(str).to_numpy(dtype=np.str_),
            identity=rows["identity"].astype(str).to_numpy(dtype=np.str_),
            protocol=rows["protocol"].astype(str).to_numpy(dtype=np.str_),
            window_number=rows["window_number"].to_numpy(dtype=np.int32),
            prediction=np.full(n, 10.0 + prediction_fold, dtype=np.float32),
            map_prediction=np.full(n, 10.5 + prediction_fold, dtype=np.float32),
            rr_std=np.full(n, 0.5, dtype=np.float32),
            uncertainty=np.full(n, 0.2, dtype=np.float32),
            quality=np.full(n, 0.8, dtype=np.float32),
            alias_probability=np.full(n, 0.1, dtype=np.float32),
            posterior_entropy=np.full(n, 0.3, dtype=np.float32),
            spike_rate=np.full(n, 0.05, dtype=np.float32),
            topk_rr=np.tile(np.array([[12.0, 24.0]], dtype=np.float32), (n, 1)),
            topk_probability=np.tile(np.array([[0.8, 0.2]], dtype=np.float32), (n, 1)),
            posterior_probability=np.tile(np.array([[0.2, 0.8]], dtype=np.float32), (n, 1)),
            radar_weights=np.tile(np.array([[0.2, 0.3, 0.5]], dtype=np.float32), (n, 1)),
            posterior_rr_grid_bpm=np.array([12.0, 24.0], dtype=np.float32),
            checkpoint_sha256=np.asarray(checkpoint_sha),
            split_manifest_file_sha256=np.asarray(sha256_file(manifest_path)),
            split_manifest_content_sha256=np.asarray(document["content_sha256"]),
            fold_assignments_sha256=np.asarray(fold_sha),
            cache_manifest_sha256=np.asarray(cache_sha),
            inference_signature_sha256=np.asarray(inference_signature),
            strict_nested_prediction_role=np.asarray(True),
            provenance_json=np.asarray(json.dumps(prediction_provenance, sort_keys=True, separators=(",", ":"))),
        )
        unit_descriptions.append(
            {
                "manifest": f"outer_{OUTER}/{stem}.json",
                "manifest_content_sha256": document["content_sha256"],
                "prediction_fold": prediction_fold,
                "role": role,
            }
        )
        records.append(
            {
                "seed": SEED,
                "outer_fold": OUTER,
                "role": role,
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "output_dir": str(unit_dir),
                "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
                "all_window_prediction": {
                    "path": str(prediction_path),
                    "sha256": sha256_file(prediction_path),
                },
            }
        )

    unit_descriptions.append(
        {
            "manifest": f"outer_{OUTER}/test_pred_{OUTER}.json",
            "manifest_content_sha256": "0" * 64,
            "prediction_fold": OUTER,
            "role": "hcs_test_open_only_after_policy_lock",
        }
    )
    plan = {
        "schema_version": 1,
        "classification": "test",
        "outer_folds": {
            str(OUTER): {
                "hcs_training_folds": [0, 1, 2, 5],
                "outer_test_fold": OUTER,
                "outer_validation_fold": 4,
                "units": unit_descriptions,
            }
        },
    }
    plan["content_sha256"] = canonical_content_sha256(plan)
    plan_path = manifest_root / "plan.json"
    _json(plan_path, plan)
    index_path = root / "discovery_index.json"
    _json(
        index_path,
        {
            "schema_version": 1,
            "outer_test_opened": False,
            "records": records,
        },
    )
    return {
        "cache": cache,
        "plan": plan_path,
        "index": index_path,
        "first_manifest": Path(records[0]["manifest"]),
        "first_checkpoint": Path(records[0]["checkpoint"]["path"]),
        "first_prediction": Path(records[0]["all_window_prediction"]["path"]),
    }


def _build(paths: dict[str, Path]) -> dict[str, Any]:
    return build_nested_stack(
        discovery_index_path=paths["index"],
        plan_path=paths["plan"],
        cache_dir=paths["cache"],
        outer_fold=OUTER,
        seed=SEED,
    )


def test_exact_cover_keeps_outer_test_explicitly_unavailable(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path)
    result = _build(paths)
    identity = result["identity"].astype(str)
    test = identity == "I3"
    assert result["strict_nested"].item() is True
    assert result["outer_test_opened"].item() is False
    assert np.all(~result["proposal_available"][test])
    assert np.all(result["proposal_available"][~test])
    assert np.all(result["nested_role"][test] == "outer_test_unavailable")
    assert np.all(result["prediction"][test] == 0.0)
    assert np.all(~result["reference_valid"][test])
    assert np.isnan(result["reference_rr_bpm"][test]).all()
    assert set(result["nested_role"][~test]) == {"hcs_train_oof", "hcs_validation"}
    provenance = json.loads(result["provenance_json"].item())
    assert provenance["target_consulted_for_stitching"] is False
    assert provenance["outer_test_identities"] == ["I3"]
    assert len(provenance["source_units"]) == 5


def _convert_to_flat_campaign_plan(
    plan_path: Path, *, seeds: tuple[int, ...] = (SEED,)
) -> None:
    document = json.loads(plan_path.read_text(encoding="utf-8"))
    nested_units = [
        unit
        for unit in document["outer_folds"][str(OUTER)]["units"]
        if unit["role"] in {"hcs_train_oof", "hcs_validation"}
    ]
    flat_units = []
    for seed in seeds:
        for source in nested_units:
            unit = dict(source)
            unit.pop("prediction_fold", None)
            unit.update({"outer_fold": OUTER, "seed": seed})
            flat_units.append(unit)
    document["outer_folds"] = [OUTER]
    document["seeds"] = list(seeds)
    document["units"] = flat_units
    document.pop("content_sha256")
    document["content_sha256"] = canonical_content_sha256(document)
    _json(plan_path, document)


def test_flat_hash_complete_campaign_plan_is_normalized_without_seed_selection(
    tmp_path: Path,
) -> None:
    paths = _make_fixture(tmp_path)
    _convert_to_flat_campaign_plan(paths["plan"])
    result = _build(paths)
    assert result["strict_nested"].item() is True
    assert result["outer_test_opened"].item() is False


def test_flat_campaign_plan_rejects_manifest_semantic_drift_across_seeds(
    tmp_path: Path,
) -> None:
    paths = _make_fixture(tmp_path)
    _convert_to_flat_campaign_plan(paths["plan"], seeds=(SEED, SEED + 1))
    document = json.loads(paths["plan"].read_text(encoding="utf-8"))
    document["units"][-1]["role"] = "hcs_train_oof"
    document.pop("content_sha256")
    document["content_sha256"] = canonical_content_sha256(document)
    _json(paths["plan"], document)
    with pytest.raises(RuntimeError, match="manifest semantics differ across seeds"):
        _build(paths)


def test_cache_builder_semantic_fields_match_canonical_cache(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path)
    result = _build(paths)
    assert result["fold"].dtype == np.int16
    assert np.array_equal(result["fold"], np.repeat(np.arange(6), 2))
    assert result["window_start_s"].dtype == np.float64
    assert result["window_end_s"].dtype == np.float64
    assert np.array_equal(result["window_start_s"], np.tile([0.0, 4.0], 6))
    assert np.array_equal(result["window_end_s"], np.tile([32.0, 36.0], 6))


def test_duplicate_and_missing_units_fail_closed(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path)
    document = json.loads(paths["index"].read_text())
    document["records"].append(document["records"][0])
    _json(paths["index"], document)
    with pytest.raises(RuntimeError, match="duplicate discovery unit"):
        _build(paths)

    paths = _make_fixture(tmp_path / "missing")
    document = json.loads(paths["index"].read_text())
    document["records"].pop()
    _json(paths["index"], document)
    with pytest.raises(RuntimeError, match="missing discovery units"):
        _build(paths)


@pytest.mark.parametrize("target", ["first_manifest", "first_checkpoint", "first_prediction"])
def test_bound_artifact_tamper_is_rejected(tmp_path: Path, target: str) -> None:
    paths = _make_fixture(tmp_path)
    with paths[target].open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        _build(paths)


def test_test_prediction_record_is_rejected_even_when_five_units_exist(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path)
    document = json.loads(paths["index"].read_text())
    test_record = dict(document["records"][0])
    test_record["role"] = "hcs_test_open_only_after_policy_lock"
    test_record["manifest"] = str(paths["plan"].parent / f"outer_{OUTER}/test_pred_{OUTER}.json")
    document["records"].append(test_record)
    _json(paths["index"], document)
    with pytest.raises(RuntimeError, match="outer-test prediction artifact"):
        _build(paths)
