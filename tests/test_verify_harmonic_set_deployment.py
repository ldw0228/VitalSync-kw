from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import stat
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from snn_rr.harmonic_set_models import HarmonicCandidateSetEpisodeSNN


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "verify_harmonic_set_deployment.py"
)
_SPEC = importlib.util.spec_from_file_location("verify_harmonic_set_deployment", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = VERIFY
_SPEC.loader.exec_module(VERIFY)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _synthetic_locked_artifacts(
    tmp_path: Path, *, anchor_enabled: bool = False
) -> tuple[Path, Path]:
    cache = tmp_path / "cache"
    run = tmp_path / "run"
    cache.mkdir()
    run.mkdir()
    rows, candidates, features = 10, 3, 5
    rng = np.random.default_rng(20260828)
    raw_node = rng.normal(size=(rows, candidates, features)).astype(np.float32)
    candidate_rr = np.tile(np.asarray([9.0, 18.0, 27.0], np.float32), (rows, 1))
    candidate_rr += np.arange(rows, dtype=np.float32)[:, None] * 0.01
    candidate_mask = np.ones((rows, candidates), dtype=bool)
    radar_mask = np.ones((rows, 3), dtype=bool)
    np.save(cache / "node_features.npy", raw_node)
    np.save(cache / "candidate_bpm.npy", candidate_rr)
    np.save(cache / "candidate_mask.npy", candidate_mask)
    np.save(cache / "joint_radar_mask.npy", radar_mask)
    metadata = pd.DataFrame({
        "cache_index": np.arange(rows),
        "session_id": ["S1"] * 5 + ["S2"] * 5,
        "window_number": list(range(5)) * 2,
        # These are intentionally present but the verifier must never request
        # them from pandas or pass them into the model.
        "rr_bpm": np.full(rows, 999.0),
        "reference_valid": np.ones(rows, bool),
        "identity": ["forbidden"] * rows,
        "fold": np.zeros(rows, int),
    })
    metadata.to_csv(cache / "metadata.csv", index=False)
    feature_document = {
        "node_feature_names": [
            "global_candidate",
            "rf_radar1_power",
            "rf_radar2_power",
            "svd_radar3_power",
            "global_context",
        ],
        "forward_arrays": [
            "node_features", "candidate_bpm", "candidate_mask", "joint_radar_mask"
        ],
        "forbidden_target_qc_forward_fields": ["rr_bpm", "reference_valid"],
    }
    _write_json(cache / "feature_names.json", feature_document)
    filenames = {
        "node_features": "node_features.npy",
        "candidate_bpm": "candidate_bpm.npy",
        "candidate_mask": "candidate_mask.npy",
        "joint_radar_mask": "joint_radar_mask.npy",
        "metadata": "metadata.csv",
        "feature_names": "feature_names.json",
    }
    outputs = {
        name: {
            "filename": filename,
            "sha256": VERIFY.sha256_file(cache / filename),
            "bytes": (cache / filename).stat().st_size,
        }
        for name, filename in filenames.items()
    }
    cache_manifest = {
        "format_version": 1,
        "complete": True,
        "row_count": rows,
        "outputs": outputs,
        "model_boundary": {
            "target_qc_excluded_from_candidate_and_feature_construction": True,
        },
    }
    cache_manifest["content_sha256"] = VERIFY.canonical_json_sha256(cache_manifest)
    _write_json(cache / "manifest.json", cache_manifest)
    cache_manifest_sha = VERIFY.sha256_file(cache / "manifest.json")

    torch.manual_seed(19)
    model_config = {
        "node_features": features,
        "hidden_channels": 8,
        "attention_heads": 1,
        "graph_blocks": 2,
        "dropout": 0.0,
    }
    if anchor_enabled:
        model_config.update({
            "anchor_enabled": True,
            "anchor_max_residual_bpm": 12.0,
            "anchor_minimum_scale_bpm": 0.25,
            "anchor_maximum_scale_bpm": 12.0,
            "anchor_initial_scale_bpm": 1.5,
            "anchor_distance_weight": 1.0,
            "anchor_source_mode": "learned_blend",
        })
    model = HarmonicCandidateSetEpisodeSNN(**model_config)
    forward_allowlist = [
        "node_features", "candidate_rr", "candidate_mask", "radar_mask",
        "sequence_mask", "causal_state", "reset_mask",
    ]
    if anchor_enabled:
        forward_allowlist += ["anchor_rr", "anchor_std", "anchor_available"]
    fallback = tmp_path / "strict_nested_fallback.csv"
    if anchor_enabled:
        pd.DataFrame({
            "cache_index": np.arange(rows),
            "prediction_bpm": np.linspace(12.0, 13.0, rows),
            "rr_std_bpm": np.linspace(0.8, 1.2, rows),
            "target_rr_bpm": np.full(rows, 1234.0),
        }).to_csv(fallback, index=False)
    fallback_sha = VERIFY.sha256_file(fallback) if anchor_enabled else None
    effective = {
        "model": model_config,
        "data_bindings": {
            "cache_manifest_sha256": cache_manifest_sha,
            **(
                {"fallback_oof_sha256": fallback_sha}
                if anchor_enabled
                else {}
            ),
        },
        "forward_allowlist": forward_allowlist,
    }
    effective_sha = VERIFY.canonical_json_sha256(effective)
    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": model_config,
        "effective_configuration_sha256": effective_sha,
        "seed": 7,
        "fold": 0,
        "adaptive_iteration": 2,
    }
    torch.save(checkpoint, run / "best_checkpoint.pt")
    _write_json(
        run / "scaler.json",
        {
            "center": [0.0] * features,
            "scale": [1.0] * features,
            "fit_positions_sha256": "synthetic",
        },
    )
    _write_json(run / "fallback_policy.json", {"policy": {"synthetic": True}})
    source = tmp_path / "locked_source.py"
    source.write_text("LOCKED = True\n", encoding="utf-8")
    source_bindings = {
        "synthetic_source": {
            "path": str(source.resolve()),
            "sha256": VERIFY.sha256_file(source),
        }
    }
    run_manifest = {
        "schema_version": 1,
        "retrospective_only": True,
        "commercial_claim_authorized": False,
        "model_config": model_config,
        "iteration_effective_configuration": effective,
        "iteration_effective_configuration_sha256": effective_sha,
        "input_bindings": {
            "cache_manifest_sha256": cache_manifest_sha,
            **(
                {
                    "fallback_oof_path": str(fallback.resolve()),
                    "fallback_oof_sha256": fallback_sha,
                }
                if anchor_enabled
                else {}
            ),
        },
        "source_and_config_bindings": source_bindings,
    }
    _write_json(run / "run_manifest.json", run_manifest)
    lock = {
        "schema_version": 1,
        "retrospective_only": True,
        "outer_fold": 0,
        "seed": 7,
        "adaptive_iteration": 2,
        "effective_configuration_sha256": effective_sha,
        "source_bindings": source_bindings,
        "checkpoint_sha256": VERIFY.sha256_file(run / "best_checkpoint.pt"),
        "scaler_sha256": VERIFY.sha256_file(run / "scaler.json"),
        "policy_sha256": VERIFY.sha256_file(run / "fallback_policy.json"),
        "run_manifest_sha256": VERIFY.sha256_file(run / "run_manifest.json"),
        "cache_manifest_sha256": cache_manifest_sha,
        **(
            {"fallback_oof_sha256": fallback_sha}
            if anchor_enabled
            else {}
        ),
    }
    _write_json(run / "selection_lock.json", lock)
    return run, cache


def test_whole_chunk_stream_reset_and_all_radar_masks_are_strict(tmp_path: Path) -> None:
    run, cache = _synthetic_locked_artifacts(tmp_path)
    bindings = VERIFY.validate_locked_artifacts(run, cache)
    stream = VERIFY.load_deployment_stream(
        cache, run / "scaler.json", maximum_sessions=2
    )
    assert stream.session_lengths == (5, 5)
    model = VERIFY.load_model(
        run / "best_checkpoint.pt", bindings["model_config"], torch.device("cpu")
    )
    batch = stream.forward_batch()
    parity = VERIFY.parity_verification(
        model, batch, [(2, 1, 3), (4, 2)], atol=2e-6, rtol=2e-6
    )
    reset = VERIFY.session_reset_verification(
        model, batch, stream.session_lengths, atol=2e-6, rtol=2e-6
    )
    robustness = VERIFY.robustness_verification(
        model, stream, windows=3, atol=2e-6, rtol=2e-6
    )
    assert parity["passed"] and reset["passed"] and robustness["passed"]
    assert len(robustness["seven_nonempty_radar_masks"]) == 7
    assert robustness["no_candidate_structural_fallback_route"]
    assert robustness["corrupt_nan_inf_inputs_finite_and_unavailable"]


def test_lock_and_cache_tamper_are_rejected_before_inference(tmp_path: Path) -> None:
    run, cache = _synthetic_locked_artifacts(tmp_path)
    VERIFY.validate_locked_artifacts(run, cache)
    with (cache / "candidate_bpm.npy").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(RuntimeError, match="tamper"):
        VERIFY.validate_locked_artifacts(run, cache)


def test_optional_i3_anchor_is_hash_bound_label_free_and_streaming_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, cache = _synthetic_locked_artifacts(tmp_path, anchor_enabled=True)
    bindings = VERIFY.validate_locked_artifacts(run, cache)
    assert bindings["anchor_enabled"] is True
    assert bindings["anchor_input"]["sha256"]
    original_read_csv = VERIFY.pd.read_csv

    def guarded_read_csv(*args, **kwargs):
        requested = set(kwargs.get("usecols", ()))
        assert requested in (
            set(VERIFY.DEPLOYMENT_METADATA_COLUMNS),
            {"cache_index", "prediction_bpm", "rr_std_bpm"},
        )
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(VERIFY.pd, "read_csv", guarded_read_csv)
    stream = VERIFY.load_deployment_stream(
        cache,
        run / "scaler.json",
        maximum_sessions=2,
        anchor_input_path=Path(bindings["anchor_input"]["path"]),
    )
    batch = stream.forward_batch()
    assert batch.anchor_rr is not None
    assert batch.anchor_std is not None
    assert batch.anchor_available is not None and batch.anchor_available.all()
    model = VERIFY.load_model(
        run / "best_checkpoint.pt", bindings["model_config"], torch.device("cpu")
    )
    whole = VERIFY.run_chunk_schedule(model, batch, (batch.windows,))
    one = VERIFY.run_chunk_schedule(model, batch, (1,))
    comparison = VERIFY.compare_outputs(whole, one, atol=2e-6, rtol=2e-6)
    assert comparison["passed"]
    assert "raw_anchor_rr" in whole and "candidate_source_rr" in whole
    robustness = VERIFY.robustness_verification(
        model, stream, windows=2, atol=2e-6, rtol=2e-6
    )
    assert robustness["anchor_only_route_checked"]
    assert robustness["anchor_only_route_matches_availability"]


def test_end_to_end_reports_are_immutable_hashed_and_noncommercial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, cache = _synthetic_locked_artifacts(tmp_path)
    output = tmp_path / "verification"
    original_read_csv = VERIFY.pd.read_csv

    def guarded_read_csv(*args, **kwargs):
        assert kwargs.get("usecols") == list(VERIFY.DEPLOYMENT_METADATA_COLUMNS)
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(VERIFY.pd, "read_csv", guarded_read_csv)
    args = VERIFY.parse_args([
        "--run-dir", str(run),
        "--cache", str(cache),
        "--output-dir", str(output),
        "--devices", "cpu",
        "--maximum-sessions", "2",
        "--chunk-schedules", "2,3;4,1",
        "--random-schedules", "0",
        "--robustness-windows", "2",
        "--warmup-repeats", "1",
        "--benchmark-repeats", "2",
        "--cold-repeats", "1",
        "--benchmark-chunk-windows", "3",
    ])
    result = VERIFY.verify(args)
    assert result["status"] == "passed"
    report = json.loads(
        (output / "deployment_verification.json").read_text(encoding="utf-8")
    )
    assert report["commercial_claim_authorized"] is False
    assert report["label_access"]["target_or_test_labels_accessed"] is False
    assert {row["operation"] for row in report["benchmarks"]} == {
        "checkpoint_load_model_init_plus_first_window_cold",
        "stateful_one_window_warm",
        "stateless_chunk_warm",
    }
    hashes = json.loads((output / "artifact_hashes.json").read_text(encoding="utf-8"))
    for filename, binding in hashes["artifacts"].items():
        assert binding["sha256"] == VERIFY.sha256_file(output / filename)
    for path in output.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
    with pytest.raises(RuntimeError, match="already exists"):
        VERIFY.verify(args)
