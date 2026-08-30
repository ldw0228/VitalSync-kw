from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_e2e.py"
_SPEC = importlib.util.spec_from_file_location("snn_rr_benchmark_e2e", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
BENCH = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = BENCH
_SPEC.loader.exec_module(BENCH)


SMALL_DATA_CONFIG = {
    "radar_hz": 40.0,
    "model_hz": 10.0,
    "window_seconds": 4.0,
    "respiration_band_hz": [0.08, 0.80],
    "rr_range_bpm": [6.0, 45.0],
    "range_pool": 2,
    "fft_size": 256,
}


def _prepared_for_feature(
    feature: BENCH.FeatureBundle,
    *,
    tail_dim: int = 3,
    map_branch: str = "both",
) -> BENCH.PreparedCheckpoint:
    aux_dim = len(feature.base_aux) + tail_dim
    return BENCH.PreparedCheckpoint(
        path=Path("dummy.pt"),
        checkpoint={},
        run_config={},
        model_type="snn",
        model_kwargs={
            "aux_dim": aux_dim,
            "aux_base_dim": len(feature.base_aux),
            "input_branches": 2 if map_branch == "both" else 1,
        },
        aux_center=np.zeros(aux_dim, dtype=np.float32),
        aux_scale=np.ones(aux_dim, dtype=np.float32),
        map_branch=map_branch,
        expected_aux_dim=aux_dim,
    )


def test_summary_reports_linear_p50_and_p95() -> None:
    summary = BENCH.summarize_latency([1.0, 2.0, 3.0, 4.0])

    assert summary["repeats"] == 4
    assert summary["p50_ms"] == pytest.approx(2.5)
    assert summary["p95_ms"] == pytest.approx(3.85)
    assert summary["samples_ms"] == [1.0, 2.0, 3.0, 4.0]


def test_synthetic_preprocessing_and_model_input_layout_are_deterministic() -> None:
    frames = int(SMALL_DATA_CONFIG["radar_hz"] * SMALL_DATA_CONFIG["window_seconds"])
    raw_a = BENCH.synthetic_raw_window(frames=frames, seed=31)
    raw_b = BENCH.synthetic_raw_window(frames=frames, seed=31)
    np.testing.assert_array_equal(raw_a, raw_b)

    feature = BENCH.preprocess_raw_window(raw_a, SMALL_DATA_CONFIG)
    assert raw_a.shape == (3, frames, 182)
    assert feature.radar_map.shape[0] == 3
    assert feature.radar_map.shape[1] == len(feature.pooled_frequencies_hz)
    assert feature.radar_map.shape[2] == 182
    assert feature.radar_map.dtype == np.float16
    assert np.isfinite(feature.base_aux).all()

    prepared = _prepared_for_feature(feature, tail_dim=3)
    radar_map, aux, mask = BENCH.construct_numpy_model_inputs(
        feature,
        prepared,
        np.asarray([0.25, -0.5, 1.0], dtype=np.float32),
    )
    assert radar_map.shape == feature.radar_map.shape
    assert aux.shape == (prepared.expected_aux_dim,)
    np.testing.assert_array_equal(aux[-3:], [0.25, -0.5, 1.0])
    np.testing.assert_array_equal(mask, [True, True, True])

    raw_only = replace(
        prepared,
        map_branch="raw",
        model_kwargs={**prepared.model_kwargs, "input_branches": 1},
    )
    raw_map, _, _ = BENCH.construct_numpy_model_inputs(
        feature,
        raw_only,
        np.asarray([0.25, -0.5, 1.0], dtype=np.float32),
    )
    assert raw_map.shape[-1] == feature.radar_map.shape[-1] // 2


def test_checkpoint_loader_preserves_structured_and_harmonic_options(
    tmp_path: Path,
) -> None:
    kwargs = {
        "num_radars": 3,
        "rr_min": 6.0,
        "rr_max": 45.0,
        "num_rr_bins": 17,
        "radar_dropout_p": 0.0,
        "aux_dim": 45,
        "structured_auxiliary": True,
        "aux_base_dim": 45,
        "harmonic_auxiliary": True,
        "auxiliary_frequency_min_hz": 0.10,
        "auxiliary_frequency_max_hz": 0.80,
        "input_branches": 2,
        "spatial_channels": (8, 12),
        "hidden_channels": 16,
        "num_spiking_blocks": 1,
        "simulation_steps": 2,
        "dropout": 0.0,
    }
    model = BENCH.TRAIN.build_model("snn", kwargs)
    checkpoint_path = tmp_path / "snn_best.pt"
    torch.save(
        {
            "model_type": "snn",
            "model_kwargs": kwargs,
            "model_state": model.state_dict(),
            "aux_center": torch.zeros(kwargs["aux_dim"]),
            "aux_scale": torch.ones(kwargs["aux_dim"]),
        },
        checkpoint_path,
    )

    prepared = BENCH.prepare_checkpoint(checkpoint_path)
    loaded = BENCH.build_model(prepared, torch.device("cpu"))
    options = BENCH.pipeline_option_summary(prepared)

    assert loaded.structured_auxiliary is True
    assert loaded.harmonic_auxiliary is True
    assert options["serialized_special_options"] == {
        "structured_auxiliary": True,
        "harmonic_auxiliary": True,
    }
    assert options["model_kwargs_loaded_verbatim"] is True


def test_checkpoint_loader_rejects_run_config_signature_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    fold_dir = run_dir / "fold_0"
    fold_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(
        json.dumps({"run_signature": "config-signature", "arguments": {}}),
        encoding="utf-8",
    )
    checkpoint_path = fold_dir / "snn_best.pt"
    torch.save(
        {
            "model_type": "snn",
            "model_kwargs": {"aux_dim": 2},
            "model_state": {},
            "aux_center": torch.zeros(2),
            "aux_scale": torch.ones(2),
            "run_signature": "checkpoint-signature",
        },
        checkpoint_path,
    )

    with pytest.raises(RuntimeError, match="run_signature mismatch"):
        BENCH.prepare_checkpoint(checkpoint_path)


def test_pipeline_config_provenance_requires_training_config_hash_match(
    tmp_path: Path,
) -> None:
    training_config = tmp_path / "training.yaml"
    supplied_config = tmp_path / "supplied.yaml"
    training_config.write_text("data:\n  window_seconds: 32\n", encoding="utf-8")
    supplied_config.write_text(training_config.read_text(encoding="utf-8"), encoding="utf-8")
    prepared = BENCH.PreparedCheckpoint(
        path=tmp_path / "model.pt",
        checkpoint={"run_signature": "same"},
        run_config={
            "run_signature": "same",
            "arguments": {"config": str(training_config)},
        },
        model_type="snn",
        model_kwargs={"aux_dim": 1},
        aux_center=np.zeros(1, dtype=np.float32),
        aux_scale=np.ones(1, dtype=np.float32),
        map_branch="both",
        expected_aux_dim=1,
    )

    audit = BENCH.validate_pipeline_config_provenance(prepared, supplied_config)
    assert audit["verified"] is True

    supplied_config.write_text("data:\n  window_seconds: 16\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs"):
        BENCH.validate_pipeline_config_provenance(prepared, supplied_config)


class _DummyRRModel(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        radar_mask: torch.Tensor | None = None,
        aux: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        assert radar_mask is not None and radar_mask.dtype == torch.bool
        assert aux is not None and aux.ndim == 2
        expected = x.float().mean(dim=(1, 2, 3)) + 18.0
        return {"expected_rr": expected}


def test_cpu_latency_trial_covers_resident_raw_to_forward() -> None:
    frames = int(SMALL_DATA_CONFIG["radar_hz"] * SMALL_DATA_CONFIG["window_seconds"])
    raw = BENCH.synthetic_raw_window(frames=frames, seed=7)
    feature = BENCH.preprocess_raw_window(raw, SMALL_DATA_CONFIG)
    prepared = _prepared_for_feature(feature, tail_dim=2)

    report = BENCH.run_latency_trials(
        model=_DummyRRModel(),
        prepared=prepared,
        data_config=SMALL_DATA_CONFIG,
        history_tail=np.zeros(2, dtype=np.float32),
        device=torch.device("cpu"),
        raw_supplier=lambda: raw,
        include_raw_load=False,
        repeats=2,
        warmup=1,
        amp=True,
    )

    assert report["raw_load_included"] is False
    assert report["amp"] is False
    assert report["stages"]["raw_load_ms"]["samples_ms"] == [0.0, 0.0]
    assert report["stages"]["total_ms"]["repeats"] == 2
    assert report["stages"]["total_ms"]["p95_ms"] > 0.0
    assert report["last_expected_rr_bpm"] > 18.0
