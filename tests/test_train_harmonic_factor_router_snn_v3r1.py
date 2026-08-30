from __future__ import annotations

import copy
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch


def _checkpoint_pickle_side_effect(path: str) -> None:
    Path(path).write_text("executed\n", encoding="utf-8")


class _MaliciousCheckpointValue:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return _checkpoint_pickle_side_effect, (str(self.path),)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    ROOT
    if (ROOT / "configs/harmonic_factor_router_v3.yaml").is_file()
    else Path.cwd().resolve()
)
SCRIPT = ROOT / "scripts/train_harmonic_factor_router_snn_v3r1.py"
SPEC = importlib.util.spec_from_file_location("train_hfr_v3r1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
trainer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trainer
SPEC.loader.exec_module(trainer)
if SOURCE_ROOT != ROOT:
    # The staged script is exercised against the workspace's immutable
    # contract/config/wrapper bytes.  After integration ROOT==SOURCE_ROOT and
    # these assignments are no-ops.
    trainer.PROJECT_ROOT = SOURCE_ROOT
    trainer.CONTRACT_PATH = SOURCE_ROOT / (
        "artifacts/campaigns/directed_harmonic_factor_expert_snn_v3r1/"
        "ADAPTIVE_RETROSPECTIVE_CAMPAIGN_CONTRACT.json"
    )
    trainer.CONFIG_PATH = SOURCE_ROOT / "configs/harmonic_factor_router_v3.yaml"
    trainer.SELECTION_AUTHORITY_PATH = (
        SOURCE_ROOT / "scripts/select_hfr_v3r1_common_variant.py"
    )


def _authorized(*_args: object, **_kwargs: object) -> dict[str, object]:
    return {
        "valid": True,
        "training_authorized": True,
        "promotion_authorized": False,
        "commercial_claim_authorized": False,
        "pretrain_authorization_file_sha256": "a" * 64,
    }


def test_promotion_authority_rejects_alternate_path_before_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replayed = False

    def forbidden_replay() -> object:
        nonlocal replayed
        replayed = True
        raise AssertionError("selection authority must not be opened")

    monkeypatch.setattr(trainer, "_selection_authority_module", forbidden_replay)
    with pytest.raises(RuntimeError, match="canonical path"):
        trainer.validate_phase_authorization(
            phase="promotion",
            outer_fold=0,
            variant="H0_no_factor",
            release_mode="raw_anchor",
            pretrain=_authorized(),
            promotion_authorization=tmp_path / "forged.json",
        )
    assert replayed is False


def test_promotion_authority_replays_with_admitted_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_path = tmp_path / "DISCOVERY_SELECTION_LOCK.json"
    authorization_path = tmp_path / "PROMOTION_AUTHORIZATION.json"
    authorization_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(trainer, "PROMOTION_SELECTION_PATH", selection_path)
    monkeypatch.setattr(trainer, "PROMOTION_AUTHORIZATION_PATH", authorization_path)
    admitted = {"classification": "verified_v8_gpu_admitted_child_lifecycle"}
    observed: dict[str, object] = {}

    def replay(project_root: Path, **kwargs: object) -> tuple[dict[str, object], ...]:
        observed.update(kwargs)
        return (
            {
                "selected_variant": "H0_no_factor",
                "selected_release_mode": "raw_anchor",
            },
            {
                "pretrain_authorization": {"sha256": "a" * 64},
                "discovery_selection_lock": {"sha256": "b" * 64},
                "selected_variant": "H0_no_factor",
                "selected_release_mode": "raw_anchor",
                "fixed_confidence_switch_probability_min": 0.8,
            },
            {
                "selection_lock": {"sha256": "b" * 64},
                "promotion_authorization": {"sha256": "c" * 64},
            },
        )

    monkeypatch.setattr(
        trainer,
        "_selection_authority_module",
        lambda: SimpleNamespace(validate_locked_selection_authorization=replay),
    )
    result = trainer.validate_phase_authorization(
        phase="promotion",
        outer_fold=0,
        variant="H0_no_factor",
        release_mode="raw_anchor",
        pretrain=_authorized(),
        promotion_authorization=authorization_path,
        admitted_binding=admitted,
    )
    assert observed["selection_lock_path"] == selection_path
    assert observed["promotion_authorization_path"] == authorization_path
    assert observed["admitted_binding"] is admitted
    assert result["phase"] == "promotion"


def _synthetic_cache(tmp_path: Path, *, seed: int = 7, outer_fold: int = 3) -> tuple[Path, Path]:
    cache = tmp_path / "cache"
    cache.mkdir()
    rows_per_fold = 3
    legacy_rows = 6 * rows_per_fold
    candidates = 2
    metadata: list[dict[str, object]] = []
    candidate_rr = np.empty((legacy_rows, candidates), np.float32)
    node = np.zeros((legacy_rows, candidates, 571), np.float32)
    candidate_mask = np.ones((legacy_rows, candidates), bool)
    radar = np.ones((legacy_rows, 3), bool)
    anchor = np.empty(legacy_rows, np.float32)
    for fold in range(6):
        for window in range(rows_per_fold):
            index = fold * rows_per_fold + window
            target = 25.5 + 0.2 * fold + 0.05 * window
            metadata.append(
                {
                    "cache_index": index,
                    "fold": fold,
                    "session_id": f"S{fold}",
                    "identity": f"I{fold}",
                    "window_number": window,
                    "rr_bpm": target,
                    "reference_valid": True,
                    "classical_rr_bpm": target,
                }
            )
            candidate_rr[index] = (target - 0.2, target + 1.0)
            anchor[index] = target + 0.1
            # Available values need not be non-zero.  One changing core cell
            # gives the scaler a directly auditable statistic.
            node[index, :, 0] = float(fold)
    full_metadata = pd.DataFrame(metadata)
    safe = full_metadata["fold"].to_numpy(np.int16) != int(outer_fold)
    safe_metadata = full_metadata.loc[safe, list(trainer.NONOUTER_METADATA_COLUMNS)]
    global_cache_index = safe_metadata["cache_index"].to_numpy(np.int64)
    np.save(cache / "node_features.npy", node[safe])
    np.save(cache / "candidate_bpm.npy", candidate_rr[safe])
    np.save(cache / "candidate_mask.npy", candidate_mask[safe])
    np.save(cache / "joint_radar_mask.npy", radar[safe])
    np.save(cache / "local_to_global_cache_index.npy", global_cache_index)
    safe_metadata.to_csv(cache / "metadata.csv", index=False)
    schema = SOURCE_ROOT / (
        "artifacts/cache/harmonic_set_v2_i2r_nested_o3_s20260828_"
        "nms125_base_emap_svd12_m050/feature_names.json"
    )
    shutil.copyfile(schema, cache / "feature_names.json")
    proposer = tmp_path / "proposer.npz"
    safe_folds = safe_metadata["fold"].to_numpy(np.int16)
    validation_fold = (outer_fold + 1) % 6
    nested_role = np.where(safe_folds == validation_fold, "validation", "training")
    np.savez_compressed(
        proposer,
        classification=np.asarray(trainer.NONOUTER_STACK_CLASSIFICATION),
        campaign_revision=np.asarray(trainer.CAMPAIGN_REVISION),
        partition=np.asarray("outer_excluded_training_validation"),
        cache_index=global_cache_index,
        fold=safe_folds,
        prediction=anchor[safe],
        rr_std=np.ones(len(global_cache_index), np.float32),
        proposal_available=np.ones(len(global_cache_index), bool),
        nested_role=np.asarray(nested_role, dtype="<U10"),
        outer_fold=np.asarray(outer_fold, np.int16),
        seed=np.asarray(seed, np.int64),
        outer_test_opened=np.asarray(False),
        outer_rows_present=np.asarray(False),
    )
    outputs = {}
    for logical_name, filename in trainer.REQUIRED_CACHE_OUTPUTS.items():
        path = cache / filename
        outputs[logical_name] = {
            "filename": filename,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    manifest = {
        "schema_version": 1,
        "classification": trainer.NONOUTER_PACK_CLASSIFICATION,
        "campaign_id": trainer.CAMPAIGN_ID,
        "campaign_revision": trainer.CAMPAIGN_REVISION,
        "format_version": 1,
        "complete": True,
        "outer_fold": outer_fold,
        "partition": "outer_excluded_training_validation",
        "source_combined_cache_open_authorized_by_consumer": False,
        "outer_test_rows_physically_present": False,
        "outer_prediction_pack_absent": True,
        "inputs": {
            "source_combined_cache": {"sha256": "0" * 64, "bytes": 1},
            "proposer_stack": {
                "sha256": hashlib.sha256(proposer.read_bytes()).hexdigest(),
                "bytes": proposer.stat().st_size,
            },
        },
        "outputs": outputs,
    }
    manifest["content_sha256"] = trainer.semantic_sha256(manifest)
    (cache / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return cache, proposer


def _train_args(cache: Path, proposer: Path, output: Path) -> object:
    return trainer.parse_args(
        [
            "--mode", "train",
            "--cache", str(cache),
            "--proposer-stack", str(proposer),
            "--output-dir", str(output),
            "--target-sealed-capability-receipt", str(output / "capability.json"),
            "--expected-admitted-context-json", json.dumps(
                {
                    "outer_fold": 3,
                    "seed": 7,
                    "variant": "H0_no_factor",
                    "execution_number": 0,
                    "resume": False,
                },
                sort_keys=True,
            ),
            "--outer-fold", "3",
            "--seed", "7",
            "--variant", "H0_no_factor",
            "--device", "cpu",
            "--no-amp",
            "--epochs", "1",
            "--minimum-epochs", "1",
            "--patience", "1",
            "--warmup-windows", "0",
            "--gradient-accumulation-sessions", "2",
            "--chunk-windows", "3",
            "--smoke-test",
        ]
    )


def _rewrite_cache_manifest(path: Path, document: dict[str, object]) -> None:
    payload = dict(document)
    payload.pop("content_sha256", None)
    payload["content_sha256"] = trainer.semantic_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _refresh_cache_output_binding(cache: Path, logical_name: str) -> None:
    manifest_path = cache / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    filename = trainer.REQUIRED_CACHE_OUTPUTS[logical_name]
    output = cache / filename
    document["outputs"][logical_name] = {
        "filename": filename,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "bytes": output.stat().st_size,
    }
    _rewrite_cache_manifest(manifest_path, document)


def test_cache_manifest_outputs_are_exactly_verified_before_reference_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, proposer = _synthetic_cache(tmp_path)
    del proposer
    manifest, binding = trainer.verify_cache_manifest_outputs(cache, outer_fold=3)
    assert set(binding) == {"manifest", "outputs"}
    assert set(binding["outputs"]) == set(trainer.REQUIRED_CACHE_OUTPUTS)
    for logical_name, filename in trainer.REQUIRED_CACHE_OUTPUTS.items():
        assert binding["outputs"][logical_name] == manifest["outputs"][logical_name]
        assert binding["outputs"][logical_name]["filename"] == filename

    node_path = cache / "node_features.npy"
    with node_path.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        value = stream.read(1)
        stream.seek(-1, os.SEEK_END)
        stream.write(bytes([value[0] ^ 0x01]))
    reference_opened = False

    def forbidden_read_csv(*args: object, **kwargs: object) -> object:
        del args, kwargs
        nonlocal reference_opened
        reference_opened = True
        raise AssertionError("reference CSV opened before cache verification")

    monkeypatch.setattr(trainer.pd, "read_csv", forbidden_read_csv)
    with pytest.raises(RuntimeError, match="SHA-256 drifted"):
        trainer.load_experiment(cache, tmp_path / "unused.npz", outer_fold=3, seed=7)
    assert reference_opened is False


def test_cache_manifest_output_schema_is_exact(tmp_path: Path) -> None:
    cache, _ = _synthetic_cache(tmp_path)
    manifest_path = cache / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["outputs"]["node_features"]["untrusted_extra"] = True
    _rewrite_cache_manifest(manifest_path, document)
    with pytest.raises(RuntimeError, match="binding schema drifted"):
        trainer.verify_cache_manifest_outputs(cache, outer_fold=3)


def test_promotion_training_manifest_authority_is_exact_and_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, proposer = _synthetic_cache(tmp_path)
    authority = tmp_path / "PROMOTION_AUTHORIZATION.json"
    authority_document: dict[str, object] = {
        "schema_version": 1,
        "classification": "adaptive_v3r1_v8r4_promotion_authorization",
        "campaign_id": trainer.CAMPAIGN_ID,
        "campaign_revision": trainer.CAMPAIGN_REVISION,
        "authorized_now": True,
        "authorized_scopes": [
            trainer.PROMOTION_TRAINING_PACK_SCOPE,
            "outer_prediction_pack",
        ],
        "training_authorized": True,
        "promotion_authorized": True,
        "outer_test_targets_authorized": False,
        "commercial_claim_authorized": False,
    }
    authority_document["content_sha256"] = trainer.semantic_sha256(
        authority_document
    )
    authority.write_text(
        json.dumps(authority_document, sort_keys=True) + "\n", encoding="utf-8"
    )
    authority.chmod(0o444)
    monkeypatch.setattr(trainer, "PROMOTION_AUTHORIZATION_PATH", authority)
    manifest_path = cache / "manifest.json"
    manifest_document = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_document.update(
        {
            "promotion_scope": trainer.PROMOTION_TRAINING_PACK_SCOPE,
            "promotion_authorization": {
                "path": str(authority.absolute()),
                "sha256": hashlib.sha256(authority.read_bytes()).hexdigest(),
                "bytes": authority.stat().st_size,
            },
        }
    )
    _rewrite_cache_manifest(manifest_path, manifest_document)

    manifest, binding = trainer.verify_cache_manifest_outputs(cache, outer_fold=3)
    assert manifest["promotion_scope"] == trainer.PROMOTION_TRAINING_PACK_SCOPE
    assert binding["promotion_scope"] == trainer.PROMOTION_TRAINING_PACK_SCOPE
    assert binding["promotion_authorization"] == manifest[
        "promotion_authorization"
    ]
    experiment = trainer.load_experiment(cache, proposer, outer_fold=3, seed=7)
    assert experiment.cache_input_binding["promotion_authorization"] == binding[
        "promotion_authorization"
    ]

    authority.chmod(0o644)
    with pytest.raises(RuntimeError, match="mode must be 0444"):
        trainer.verify_cache_manifest_outputs(cache, outer_fold=3)


def test_promotion_training_manifest_requires_both_exact_extra_fields(
    tmp_path: Path,
) -> None:
    cache, _ = _synthetic_cache(tmp_path)
    manifest_path = cache / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["promotion_scope"] = trainer.PROMOTION_TRAINING_PACK_SCOPE
    _rewrite_cache_manifest(manifest_path, document)
    with pytest.raises(RuntimeError, match="manifest schema drifted"):
        trainer.verify_cache_manifest_outputs(cache, outer_fold=3)


@pytest.mark.parametrize("alias_kind", ("symlink", "hardlink"))
def test_cache_manifest_output_rejects_link_aliases(
    tmp_path: Path, alias_kind: str
) -> None:
    cache, _ = _synthetic_cache(tmp_path)
    source = cache / "candidate_mask.npy"
    target = tmp_path / "candidate_mask_target.npy"
    shutil.copyfile(source, target)
    source.unlink()
    if alias_kind == "symlink":
        source.symlink_to(target)
    else:
        os.link(target, source)
    with pytest.raises(RuntimeError, match="cache input"):
        trainer.verify_cache_manifest_outputs(cache, outer_fold=3)


def test_cache_npy_consumes_the_exact_hashed_bytes_during_same_inode_aba(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, proposer = _synthetic_cache(tmp_path)
    node_path = cache / "node_features.npy"
    original_bytes = node_path.read_bytes()
    changed = np.load(node_path, allow_pickle=False)
    changed[...] = 999.0
    changed_stream = trainer.io.BytesIO()
    np.save(changed_stream, changed, allow_pickle=False)
    changed_bytes = changed_stream.getvalue()
    assert len(changed_bytes) == len(original_bytes)

    original_load = trainer.np.load
    npy_calls = 0

    def adversarial_load(source: object, *args: object, **kwargs: object) -> object:
        nonlocal npy_calls
        if isinstance(source, trainer.io.BytesIO):
            npy_calls += 1
        if npy_calls == 2:
            # Change and restore the already-verified source inode exactly
            # while the node payload is consumed.  The parser must see only
            # its descriptor-captured byte string.
            with node_path.open("r+b") as stream:
                stream.seek(0)
                stream.write(changed_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                return original_load(source, *args, **kwargs)
            finally:
                with node_path.open("r+b") as stream:
                    stream.seek(0)
                    stream.write(original_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
        return original_load(source, *args, **kwargs)

    monkeypatch.setattr(trainer.np, "load", adversarial_load)
    experiment = trainer.load_experiment(cache, proposer, outer_fold=3, seed=7)
    observed = np.asarray(experiment.node_features[[0]], np.float32)
    assert np.count_nonzero(observed == 999.0) == 0
    assert node_path.read_bytes() == original_bytes
    # local-index, four forward arrays, and proposer were all opened from
    # captured bytes rather than a source pathname.
    assert npy_calls == 6


def test_checkpoint_loader_is_weights_only_and_pickle_side_effect_free(
    tmp_path: Path,
) -> None:
    sentinel = tmp_path / "pickle-side-effect.txt"
    checkpoint_path = tmp_path / "malicious.pt"
    torch.save(
        {
            "model_state": {},
            "malicious": _MaliciousCheckpointValue(sentinel),
        },
        checkpoint_path,
    )
    with pytest.raises(RuntimeError, match="weights-only"):
        trainer._load_torch_snapshot(checkpoint_path, map_location="cpu")
    assert not sentinel.exists()


def test_prediction_model_authority_is_terminal_without_sealed_pointer(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(campaign_phase="promotion", outer_fold=3, seed=7)
    with pytest.raises(RuntimeError, match="target-sealed model authority"):
        trainer._prediction_model_authority(
            args,
            pretrain=_authorized(),
            phase_binding={"phase": "promotion"},
        )


def test_identity_balancing_and_outer_split_are_identity_disjoint(tmp_path: Path) -> None:
    cache, proposer = _synthetic_cache(tmp_path)
    experiment = trainer.load_experiment(cache, proposer, outer_fold=3, seed=7)
    train, validation, validation_fold = trainer.split_positions(experiment.metadata, 3)
    assert validation_fold == 4
    identities = experiment.metadata["identity"].astype(str).to_numpy()
    test = np.flatnonzero(experiment.metadata["fold"].to_numpy() == 3)
    assert not (set(identities[train]) & set(identities[validation]))
    assert not (set(identities[train]) & set(identities[test]))
    weight = trainer.identity_balanced_weights(experiment.metadata, train)
    masses = [weight[train][identities[train] == name].sum() for name in sorted(set(identities[train]))]
    assert np.allclose(masses, masses[0])


def test_outer_train_scaler_excludes_validation_and_preserves_exact_zero_masks(tmp_path: Path) -> None:
    cache, proposer = _synthetic_cache(tmp_path)
    experiment = trainer.load_experiment(cache, proposer, outer_fold=3, seed=7)
    train, validation, _ = trainer.split_positions(experiment.metadata, 3)
    # A huge validation-only value must not change the outer-train mean.
    writable = np.load(cache / "node_features.npy")
    writable[validation, :, 0] = 10_000.0
    np.save(cache / "node_features.npy", writable)
    _refresh_cache_output_binding(cache, "node_features")
    experiment = trainer.load_experiment(cache, proposer, outer_fold=3, seed=7)
    scaler = trainer.fit_outer_train_standardizer(experiment, train)
    expected = np.mean(experiment.metadata.iloc[train]["fold"].to_numpy(np.float64))
    assert float(scaler.mean[0]) == pytest.approx(expected)
    candidate = np.asarray(experiment.candidate_rr[validation], np.float32)
    mask = np.asarray(experiment.candidate_mask[validation], bool)
    radar = np.zeros((len(validation), 3), bool)
    availability = trainer._build_availability(candidate, mask, radar)
    transformed = trainer._scaler_transform(
        scaler, np.asarray(experiment.node_features[validation], np.float32), availability
    )
    # RF/SVD are all unavailable and exact zero.  Core remains candidate-bound
    # in the v3r1 feature contract.
    assert np.count_nonzero(transformed[..., 46:]) == 0


def test_commercial_lexicographic_key_prefers_all_gate_pass() -> None:
    passing = {
        "overall_mae_bpm": 0.99,
        "identity_macro_mae_bpm": 0.99,
        "rmse_bpm": 1.7,
        "within_2_fraction": 0.91,
        "over_5_fraction": 0.02,
        "high_rr_25_35_mae_bpm": 1.9,
    }
    failing = {**passing, "within_2_fraction": 0.89, "overall_mae_bpm": 0.8}
    assert trainer.commercial_selection_key(passing, epoch=2)[0] == 0
    assert trainer.commercial_selection_key(failing, epoch=1)[0] == 1
    assert trainer.commercial_selection_key(passing, epoch=2) < trainer.commercial_selection_key(failing, epoch=1)


def test_amp_gradient_scaler_contract_and_rng_replay_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_gradient_scaler(device_type: str, **kwargs: object) -> object:
        captured.update({"device_type": device_type, **kwargs})
        return sentinel

    monkeypatch.setattr(trainer.torch.amp, "GradScaler", fake_gradient_scaler)
    assert trainer.build_gradient_scaler(torch.device("cuda"), True) is sentinel
    assert captured == {
        "device_type": "cuda",
        "enabled": True,
        "init_scale": 8192.0,
    }
    assert trainer._next_amp_gradient_scale(8192.0) == 4096.0
    assert trainer._next_amp_gradient_scale(2.0) == 1.0
    with pytest.raises(RuntimeError, match="cannot be reduced"):
        trainer._next_amp_gradient_scale(1.0)

    torch.manual_seed(123)
    radar_rng = np.random.default_rng(456)
    replay = trainer._capture_group_rng_state(torch.device("cpu"), radar_rng)
    expected_torch = torch.rand(4)
    expected_radar = radar_rng.integers(0, 1000, size=4)
    trainer._restore_group_rng_state(replay, torch.device("cpu"), radar_rng)
    assert torch.equal(torch.rand(4), expected_torch)
    assert np.array_equal(radar_rng.integers(0, 1000, size=4), expected_radar)


def test_amp_overflow_replays_identical_group_without_skipping_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeAmpScaler:
        def __init__(self) -> None:
            self.current_scale = 8192.0

        def scale(self, loss: torch.Tensor) -> torch.Tensor:
            return loss

        def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
            del optimizer

        def step(self, optimizer: torch.optim.Optimizer) -> None:
            optimizer.step()

        def update(self, new_scale: float | None = None) -> None:
            if new_scale is not None:
                self.current_scale = float(new_scale)

        def get_scale(self) -> float:
            return self.current_scale

        def is_enabled(self) -> bool:
            return True

    cache, proposer = _synthetic_cache(tmp_path)
    experiment = trainer.load_experiment(cache, proposer, outer_fold=3, seed=7)
    train, _, _ = trainer.split_positions(experiment.metadata, 3)
    feature_scaler = trainer.fit_outer_train_standardizer(experiment, train)
    row_numpy = trainer.identity_balanced_weights(experiment.metadata, train)
    row_tensor = torch.as_tensor(row_numpy)
    factor_tensor = torch.as_tensor(trainer.factor_class_weights(experiment.metadata, train))
    torch.manual_seed(987)
    replay_model = trainer.build_model("H0_no_factor", torch.device("cpu"))
    initial_state = {
        name: value.detach().clone() for name, value in replay_model.state_dict().items()
    }
    control_model = trainer.build_model("H0_no_factor", torch.device("cpu"))
    control_model.load_state_dict(initial_state)
    replay_optimizer = torch.optim.AdamW(replay_model.parameters(), lr=3.0e-4)
    control_optimizer = torch.optim.AdamW(control_model.parameters(), lr=3.0e-4)
    real_clip = torch.nn.utils.clip_grad_norm_
    calls = 0

    def overflow_once(parameters: object, maximum: float) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 1:
            return torch.tensor(float("inf"))
        return real_clip(parameters, maximum)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", overflow_once)
    torch.manual_seed(654)
    replay_metrics = trainer.run_training_epoch(
        replay_model,
        experiment,
        train,
        feature_scaler,
        replay_optimizer,
        FakeAmpScaler(),
        row_tensor,
        row_numpy,
        factor_tensor,
        torch.device("cpu"),
        seed=7,
        epoch=0,
        variant="H0_no_factor",
        amp=True,
        chunk_windows=3,
        warmup_windows=0,
        accumulation_sessions=2,
        gradient_clip=2.0,
    )
    replay_final_rng = torch.get_rng_state().clone()
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", real_clip)
    torch.manual_seed(654)
    control_metrics = trainer.run_training_epoch(
        control_model,
        experiment,
        train,
        feature_scaler,
        control_optimizer,
        FakeAmpScaler(),
        row_tensor,
        row_numpy,
        factor_tensor,
        torch.device("cpu"),
        seed=7,
        epoch=0,
        variant="H0_no_factor",
        amp=True,
        chunk_windows=3,
        warmup_windows=0,
        accumulation_sessions=2,
        gradient_clip=2.0,
    )
    assert replay_metrics["amp_overflow_retries"] == 1.0
    assert replay_metrics["optimizer_steps"] == control_metrics["optimizer_steps"]
    assert replay_metrics["optimizer_steps"] == 2.0
    assert replay_metrics["forward_passes"] == control_metrics["forward_passes"] == 2.0
    assert replay_metrics["processed_windows"] == control_metrics["processed_windows"] == 12.0
    assert torch.equal(torch.get_rng_state(), replay_final_rng)
    for replay_parameter, control_parameter in zip(
        replay_model.parameters(), control_model.parameters(), strict=True
    ):
        assert torch.equal(replay_parameter, control_parameter)


def test_full_loss_stack_is_finite_and_target_never_enters_forward_kwargs() -> None:
    device = torch.device("cpu")
    model = trainer.build_model("H2_full", device)
    candidate = torch.tensor([[[25.8, 26.8], [25.9, float("nan")]]])
    batch = {
        "position": torch.tensor([[0, 1]]),
        "node_features": torch.zeros(1, 2, 2, 571),
        "candidate_rr": candidate,
        "candidate_mask": torch.tensor([[[True, True], [True, False]]]),
        "joint_radar_mask": torch.ones(1, 2, 3, dtype=torch.bool),
        "sequence_mask": torch.ones(1, 2, dtype=torch.bool),
        "reset_mask": torch.tensor([[True, False]]),
        "anchor_rr": torch.tensor([[26.0, 26.1]]),
        "anchor_std": torch.ones(1, 2),
        "anchor_available": torch.ones(1, 2, dtype=torch.bool),
        "classical_rr": torch.tensor([[26.0, 26.1]]),
        "warmup_mask": torch.zeros(1, 2, dtype=torch.bool),
        "target": torch.tensor([[26.0, float("nan")]]),
        "reference_valid": torch.tensor([[True, False]]),
    }
    output = trainer.forward_model(model, batch, state=None)
    loss, components = trainer.compute_multitask_loss(
        output,
        batch,
        torch.ones(2),
        torch.ones(4),
        variant="H2_full",
    )
    assert torch.isfinite(loss)
    assert set(components) == set(trainer.LOSS_WEIGHTS)
    assert (output["factor_affinity"][0, 0] == 0.0).any()
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    assert torch.isfinite(gradient_norm)


def test_factor_candidate_js_disables_rows_without_candidate_factor_support() -> None:
    device = torch.device("cpu")
    model = trainer.build_model("H2_full", device)
    batch = {
        "position": torch.tensor([[0]]),
        "node_features": torch.zeros(1, 1, 2, 571),
        "candidate_rr": torch.full((1, 1, 2), float("nan")),
        "candidate_mask": torch.zeros(1, 1, 2, dtype=torch.bool),
        "joint_radar_mask": torch.ones(1, 1, 3, dtype=torch.bool),
        "sequence_mask": torch.ones(1, 1, dtype=torch.bool),
        "reset_mask": torch.ones(1, 1, dtype=torch.bool),
        "anchor_rr": torch.tensor([[32.0]]),
        "anchor_std": torch.ones(1, 1),
        "anchor_available": torch.ones(1, 1, dtype=torch.bool),
        "classical_rr": torch.tensor([[32.0]]),
        "warmup_mask": torch.zeros(1, 1, dtype=torch.bool),
        "target": torch.tensor([[32.0]]),
        "reference_valid": torch.ones(1, 1, dtype=torch.bool),
    }
    output = trainer.forward_model(model, batch, state=None)
    loss, components = trainer.compute_multitask_loss(
        output,
        batch,
        torch.ones(1),
        torch.ones(4),
        variant="H2_full",
    )
    assert torch.equal(output["factor_affinity"], torch.zeros_like(output["factor_affinity"]))
    assert components["factor_candidate_js"].item() == pytest.approx(0.0, abs=1.0e-12)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0))


def test_target_free_loader_rejects_reference_or_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    arrays = {
        "cache_index": np.arange(1),
        "node_features": np.zeros((1, 1, 571), np.float32),
        "candidate_rr_bpm": np.full((1, 1), 20.0, np.float32),
        "candidate_mask": np.ones((1, 1), bool),
        "joint_radar_mask": np.ones((1, 3), bool),
        "proposer_anchor_bpm": np.full(1, 20.0, np.float32),
        "proposer_anchor_std_bpm": np.ones(1, np.float32),
        "proposer_anchor_available": np.ones(1, bool),
        "classical_rr_bpm": np.full(1, 20.0, np.float32),
        "session_reset": np.ones(1, bool),
        "reference_rr_bpm": np.full(1, 20.0, np.float32),
    }
    np.savez(path, **arrays)
    with pytest.raises(RuntimeError, match="unknown"):
        trainer.load_sanitized_inference_input(path)


def test_synthetic_train_and_target_free_predict_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache, proposer = _synthetic_cache(tmp_path)
    monkeypatch.setattr(trainer, "validate_pretrain_authorization", _authorized)
    output = tmp_path / "train"
    train_args = _train_args(cache, proposer, output)
    result = trainer.train(
        train_args,
        admitted_binding={
            "phase": "discovery",
            "context": train_args.expected_admitted_context_json,
        },
    )
    assert result["status"] == "checkpoint_locked"
    for name in (
        "run_manifest.json", "scaler.json", "history.json", "last.pt", "best.pt",
        "checkpoint_selection_lock.json", "validation_predictions.npz", "validation_metrics.json",
    ):
        assert (output / name).is_file()
    with np.load(output / "validation_predictions.npz", allow_pickle=False) as validation:
        assert "reference_rr_bpm" in validation.files
        assert "hard_source_bpm" in validation.files

    raw_node = np.load(cache / "node_features.npy")[:2]
    raw_candidate = np.load(cache / "candidate_bpm.npy")[:2]
    sanitized = tmp_path / "sanitized.npz"
    np.savez_compressed(
        sanitized,
        cache_index=np.asarray([100, 101], np.int64),
        node_features=raw_node,
        candidate_rr_bpm=raw_candidate,
        candidate_mask=np.ones((2, 2), bool),
        joint_radar_mask=np.ones((2, 3), bool),
        proposer_anchor_bpm=np.asarray([25.6, 25.65], np.float32),
        proposer_anchor_std_bpm=np.ones(2, np.float32),
        proposer_anchor_available=np.ones(2, bool),
        classical_rr_bpm=np.asarray([25.5, 25.55], np.float32),
        session_reset=np.asarray([True, False]),
    )
    predict_output = tmp_path / "predict"
    predict_args = trainer.parse_args(
        [
            "--mode", "predict",
            "--predict-input", str(sanitized),
            "--checkpoint", str(output / "best.pt"),
            "--scaler", str(output / "scaler.json"),
            "--output-dir", str(predict_output),
            "--target-sealed-capability-receipt", str(predict_output / "capability.json"),
            "--expected-admitted-context-json", json.dumps(
                {
                    "outer_fold": 3,
                    "seed": 7,
                    "variant": "H0_no_factor",
                    "release_mode": "hard_source_argmax",
                    "attempt_number": 0,
                },
                sort_keys=True,
            ),
            "--outer-fold", "3",
            "--seed", "7",
            "--variant", "H0_no_factor",
            "--release-mode", "hard_source_argmax",
            "--device", "cpu",
            "--no-amp",
            "--chunk-windows", "2",
        ]
    )
    authorized_checkpoint, _ = trainer._verified_regular_file(
        output / "best.pt"
    )
    authorized_scaler, _ = trainer._verified_regular_file(output / "scaler.json")
    authorized_input, _ = trainer._verified_regular_file(sanitized)
    authorized_payload, _ = trainer._load_torch_snapshot(
        output / "best.pt", map_location="cpu"
    )
    synthetic_model_authority = {
        "schema_version": 1,
        "index": authorized_checkpoint,
        "model_source_capability": authorized_scaler,
        "checkpoint": authorized_checkpoint,
        "scaler": authorized_scaler,
        "predict_input": authorized_input,
        "scientific_signature_sha256": authorized_payload[
            "scientific_signature_sha256"
        ],
        "source_receipt": {"sha256": "f" * 64},
    }
    monkeypatch.setattr(
        trainer,
        "_prediction_model_authority",
        lambda *_args, **_kwargs: copy.deepcopy(synthetic_model_authority),
    )
    prediction_result = trainer.predict_target_free(
        predict_args,
        admitted_binding={
            "phase": "promotion_prediction",
            "context": predict_args.expected_admitted_context_json,
        },
    )
    assert prediction_result["status"] == "target_free_prediction_complete"
    with np.load(predict_output / "predictions.npz", allow_pickle=False) as predictions:
        assert set(predictions.files) == set(trainer.PREDICTION_KEYS)
        assert not any(
            token in name.lower()
            for name in predictions.files
            for token in ("target", "reference", "identity", "protocol", "fold")
        )
    manifest = json.loads((predict_output / "prediction_manifest.json").read_text())
    assert manifest["target_fields_accepted"] is False
    assert manifest["target_fields_emitted"] is False
    resume_predict_args = trainer.argparse.Namespace(
        **{**vars(predict_args), "resume": True}
    )
    reused = trainer.predict_target_free(
        resume_predict_args,
        admitted_binding={
            "phase": "promotion_prediction",
            "context": resume_predict_args.expected_admitted_context_json,
        },
    )
    assert reused["status"] == "already_predicted"
    original_prediction_raw = (predict_output / "predictions.npz").read_bytes()
    original_assert_bindings = trainer._assert_file_bindings_current
    reuse_swapped = False

    def swap_completed_prediction_before_return(
        bindings: dict[str, dict[str, object]],
    ) -> None:
        nonlocal reuse_swapped
        if set(bindings) == {"prediction_manifest", "predictions"}:
            reuse_swapped = True
            trainer.atomic_save_npz(
                predict_output / "predictions.npz", swapped_arrays
            )
        original_assert_bindings(bindings)

    # Build the alternate payload before the return-barrier injection.
    with np.load(predict_output / "predictions.npz", allow_pickle=False) as archive:
        swapped_arrays = {
            name: np.asarray(archive[name]).copy() for name in archive.files
        }
    swapped_arrays["prediction_bpm"] = (
        swapped_arrays["prediction_bpm"].astype(np.float32) + 1.0
    )
    monkeypatch.setattr(
        trainer,
        "_assert_file_bindings_current",
        swap_completed_prediction_before_return,
    )
    with pytest.raises(
        RuntimeError, match="consumed source drifted|SHA-256 drifted"
    ):
        trainer.predict_target_free(
            resume_predict_args,
            admitted_binding={
                "phase": "promotion_prediction",
                "context": resume_predict_args.expected_admitted_context_json,
            },
        )
    assert reuse_swapped is True
    trainer._atomic_publish_immutable(
        predict_output / "predictions.npz",
        lambda descriptor: trainer._write_all(descriptor, original_prediction_raw),
    )
    monkeypatch.setattr(
        trainer, "_assert_file_bindings_current", original_assert_bindings
    )
    checkpoint_copy = tmp_path / "transplanted-best.pt"
    shutil.copyfile(output / "best.pt", checkpoint_copy)
    transplanted_predict_args = trainer.argparse.Namespace(
        **{
            **vars(resume_predict_args),
            "checkpoint": checkpoint_copy,
        }
    )
    with pytest.raises(RuntimeError, match="differ from sealed authority"):
        trainer.predict_target_free(
            transplanted_predict_args,
            admitted_binding={
                "phase": "promotion_prediction",
                "context": transplanted_predict_args.expected_admitted_context_json,
            },
        )
    original_best_raw = (output / "best.pt").read_bytes()
    same_unit_checkpoint = dict(authorized_payload)
    same_unit_state = {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in authorized_payload["model_state"].items()
    }
    same_unit_tensor = next(
        value
        for value in same_unit_state.values()
        if isinstance(value, torch.Tensor)
        and value.numel()
        and value.dtype.is_floating_point
    )
    same_unit_tensor.reshape(-1)[0] += 1.0
    same_unit_checkpoint["model_state"] = same_unit_state
    trainer.atomic_torch_save(output / "best.pt", same_unit_checkpoint)
    alternate_checkpoint_args = trainer.argparse.Namespace(
        **{
            **vars(predict_args),
            "output_dir": tmp_path / "alternate-checkpoint-predict",
        }
    )
    with pytest.raises(RuntimeError, match="checkpoint campaign binding drifted"):
        trainer.predict_target_free(
            alternate_checkpoint_args,
            admitted_binding={
                "phase": "promotion_prediction",
                "context": alternate_checkpoint_args.expected_admitted_context_json,
            },
        )
    trainer._atomic_publish_immutable(
        output / "best.pt",
        lambda descriptor: trainer._write_all(descriptor, original_best_raw),
    )
    partial_output = tmp_path / "partial-predict"
    partial_output.mkdir()
    partial_prediction = partial_output / "predictions.npz"
    shutil.copyfile(predict_output / "predictions.npz", partial_prediction)
    partial_prediction.chmod(0o444)
    with np.load(predict_output / "predictions.npz", allow_pickle=False) as archive:
        swapped_arrays = {
            name: np.asarray(archive[name]).copy() for name in archive.files
        }
    swapped_arrays["prediction_bpm"] = (
        swapped_arrays["prediction_bpm"].astype(np.float32) + 1.0
    )
    original_capture = trainer._capture_verified_file
    swapped = False

    def swap_after_partial_snapshot(
        path: Path, **kwargs: object
    ) -> tuple[dict[str, object], bytes]:
        nonlocal swapped
        binding, raw = original_capture(path, **kwargs)
        if path.resolve() == partial_prediction.resolve() and not swapped:
            swapped = True
            trainer.atomic_save_npz(partial_prediction, swapped_arrays)
        return binding, raw

    monkeypatch.setattr(trainer, "_capture_verified_file", swap_after_partial_snapshot)
    partial_args = trainer.argparse.Namespace(
        **{
            **vars(resume_predict_args),
            "output_dir": partial_output,
        }
    )
    with pytest.raises(RuntimeError, match="SHA-256 drifted"):
        trainer.predict_target_free(
            partial_args,
            admitted_binding={
                "phase": "promotion_prediction",
                "context": partial_args.expected_admitted_context_json,
            },
        )
    assert swapped is True
    assert not (partial_output / "prediction_manifest.json").exists()
    monkeypatch.setattr(trainer, "_capture_verified_file", original_capture)
    fresh_swap_output = tmp_path / "fresh-swap-predict"
    original_atomic_npz = trainer.atomic_save_npz
    fresh_swapped = False

    def swap_after_fresh_publish(
        path: Path,
        arrays: dict[str, np.ndarray],
        **kwargs: object,
    ) -> dict[str, object]:
        nonlocal fresh_swapped
        binding = original_atomic_npz(path, arrays, **kwargs)
        if path.resolve() == (fresh_swap_output / "predictions.npz").resolve():
            fresh_swapped = True
            original_atomic_npz(path, swapped_arrays)
        return binding

    monkeypatch.setattr(trainer, "atomic_save_npz", swap_after_fresh_publish)
    fresh_swap_args = trainer.argparse.Namespace(
        **{
            **vars(predict_args),
            "output_dir": fresh_swap_output,
        }
    )
    with pytest.raises(RuntimeError, match="SHA-256 drifted"):
        trainer.predict_target_free(
            fresh_swap_args,
            admitted_binding={
                "phase": "promotion_prediction",
                "context": fresh_swap_args.expected_admitted_context_json,
            },
        )
    assert fresh_swapped is True
    assert not (fresh_swap_output / "prediction_manifest.json").exists()
    monkeypatch.setattr(trainer, "atomic_save_npz", original_atomic_npz)
    training_manifest = json.loads((output / "run_manifest.json").read_text())
    scientific = training_manifest["scientific_signature"]
    scientific_sha = trainer.scientific_signature_sha256(scientific)
    assert training_manifest["scientific_signature_sha256"] == scientific_sha
    lock = json.loads((output / "checkpoint_selection_lock.json").read_text())
    assert lock["scientific_signature_sha256"] == scientific_sha
    assert lock["last_checkpoint_sha256"] == trainer.sha256_file(output / "last.pt")
    assert set(lock["completed_output_inventory"]) == (
        trainer._TRAIN_COMPLETED_OUTPUT_FILENAMES
        - {"checkpoint_selection_lock.json"}
    )
    assert set(os.listdir(output)) == trainer._TRAIN_COMPLETED_OUTPUT_FILENAMES
    assert trainer._validate_completed_lock(output) == lock
    original_lock_raw = (output / "checkpoint_selection_lock.json").read_bytes()
    original_assert_binding = trainer._assert_file_binding_current
    lock_swapped = False

    def swap_completed_lock_before_return(binding: dict[str, object]) -> None:
        nonlocal lock_swapped
        if binding.get("path") == str(output / "checkpoint_selection_lock.json"):
            lock_swapped = True
            trainer.atomic_write_json(
                output / "checkpoint_selection_lock.json",
                {**lock, "created_utc": "changed-during-validation"},
            )
        original_assert_binding(binding)

    monkeypatch.setattr(
        trainer, "_assert_file_binding_current", swap_completed_lock_before_return
    )
    with pytest.raises(RuntimeError, match="SHA-256 drifted"):
        trainer._validate_completed_lock(output)
    assert lock_swapped is True
    trainer._atomic_publish_immutable(
        output / "checkpoint_selection_lock.json",
        lambda descriptor: trainer._write_all(descriptor, original_lock_raw),
    )
    monkeypatch.setattr(
        trainer, "_assert_file_binding_current", original_assert_binding
    )
    resumed_context = {
        **train_args.expected_admitted_context_json,
        "resume": True,
    }
    resumed_train_args = trainer.argparse.Namespace(
        **{
            **vars(train_args),
            "resume": True,
            "expected_admitted_context_json": resumed_context,
        }
    )
    completed_reuse = trainer.train(
        resumed_train_args,
        admitted_binding={"phase": "discovery", "context": resumed_context},
    )
    assert completed_reuse["status"] == "already_complete"

    def changed_authority(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            **_authorized(),
            "pretrain_authorization_file_sha256": "b" * 64,
        }

    monkeypatch.setattr(
        trainer, "validate_pretrain_authorization", changed_authority
    )
    with pytest.raises(RuntimeError, match="configuration/input binding differs"):
        trainer.train(
            resumed_train_args,
            admitted_binding={"phase": "discovery", "context": resumed_context},
        )
    monkeypatch.setattr(trainer, "validate_pretrain_authorization", _authorized)
    wrong_current_context = copy.deepcopy(lock["reuse_context"])
    wrong_current_context["seed"] = 8
    with pytest.raises(RuntimeError, match="current exact reuse context"):
        trainer._validate_completed_lock(
            output, expected_reuse_context=wrong_current_context
        )
    transplanted_output = tmp_path / "transplanted-train"
    shutil.copytree(output, transplanted_output)
    with pytest.raises(RuntimeError, match="reuse context binding drifted"):
        trainer._validate_completed_lock(transplanted_output)
    unknown = output / "unknown.bin"
    unknown.write_bytes(b"unknown")
    unknown.chmod(0o444)
    with pytest.raises(RuntimeError, match="inventory drifted"):
        trainer._validate_completed_lock(output)
    unknown.unlink()
    (output / "last.pt").chmod(0o644)
    with pytest.raises(RuntimeError, match="mutable or aliased"):
        trainer._validate_completed_lock(output)
    (output / "last.pt").chmod(0o444)
    alias = tmp_path / "last-alias.pt"
    os.link(output / "last.pt", alias)
    with pytest.raises(RuntimeError, match="mutable or aliased"):
        trainer._validate_completed_lock(output)
    alias.unlink()
    assert trainer._validate_completed_lock(output) == lock
    assert not (
        set(scientific) & trainer.SCIENTIFIC_SIGNATURE_ORCHESTRATION_FIELDS
    )
    verified_cache = scientific["input_bindings"]["verified_cache_inputs"]
    assert verified_cache["manifest"]["sha256"] == trainer.sha256_file(
        cache / "manifest.json"
    )
    assert set(verified_cache["outputs"]) == set(trainer.REQUIRED_CACHE_OUTPUTS)
    amp_contract = training_manifest["effective_configuration"]["optimization"][
        "amp_gradient_scaler"
    ]
    assert amp_contract == {
        "deterministic_same_group_replay": True,
        "failed_group_skip_allowed": False,
        "initial_scale": 8192.0,
        "maximum_same_group_retries": 14,
        "minimum_scale": 1.0,
    }


def test_final_validation_rejects_best_checkpoint_swap_after_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, proposer = _synthetic_cache(tmp_path)
    output = tmp_path / "train"
    args = _train_args(cache, proposer, output)
    monkeypatch.setattr(trainer, "validate_pretrain_authorization", _authorized)
    original_predict = trainer.predict_experiment_positions
    prediction_calls = 0

    def adversarial_predict(*call_args: object, **call_kwargs: object) -> object:
        nonlocal prediction_calls
        result = original_predict(*call_args, **call_kwargs)
        prediction_calls += 1
        if prediction_calls == 2:
            checkpoint, _ = trainer._load_torch_snapshot(
                output / "best.pt", map_location="cpu", required_mode=0o444
            )
            replacement = dict(checkpoint)
            state = {
                name: value.clone() if isinstance(value, torch.Tensor) else value
                for name, value in checkpoint["model_state"].items()
            }
            tensor = next(
                value
                for value in state.values()
                if isinstance(value, torch.Tensor)
                and value.numel()
                and value.dtype.is_floating_point
            )
            tensor.reshape(-1)[0] += 1.0
            replacement["model_state"] = state
            trainer.atomic_torch_save(output / "best.pt", replacement)
        return result

    monkeypatch.setattr(trainer, "predict_experiment_positions", adversarial_predict)
    with pytest.raises(RuntimeError, match="SHA-256 drifted"):
        trainer.train(
            args,
            admitted_binding={
                "phase": "discovery",
                "context": args.expected_admitted_context_json,
            },
        )
    assert prediction_calls == 2
    assert not (output / "checkpoint_selection_lock.json").exists()


def test_chunk_round_padding_forward_loss_state_and_gradient_equivalence(
    tmp_path: Path,
) -> None:
    cache, proposer = _synthetic_cache(tmp_path)
    experiment = trainer.load_experiment(cache, proposer, outer_fold=3, seed=7)
    train_positions, _, _ = trainer.split_positions(experiment.metadata, 3)
    scaler = trainer.fit_outer_train_standardizer(experiment, train_positions)
    sessions = list(
        trainer.iter_identity_balanced_sessions(
            experiment.metadata, train_positions, seed=0, epoch=0, shuffle=False
        )
    )
    chunks = [sessions[0], sessions[1][:2]]
    lanes = [
        trainer._batch_from_experiment(
            experiment,
            scaler,
            chunk,
            torch.device("cpu"),
            session_offset=0,
            warmup_windows=0,
            radar_subset=None,
            include_targets=True,
        )
        for chunk in chunks
    ]
    padded = trainer._pad_session_lane_batches(lanes)
    assert padded["sequence_mask"].tolist() == [[True, True, True], [True, True, False]]
    assert not bool(padded["candidate_mask"][1, 2].any())
    assert not bool(padded["joint_radar_mask"][1, 2].any())
    assert not bool(padded["anchor_available"][1, 2])

    torch.manual_seed(811)
    batch_model = trainer.build_model("H2_full", torch.device("cpu"))
    batch_model.eval()
    single_model = trainer.build_model("H2_full", torch.device("cpu"))
    single_model.load_state_dict(batch_model.state_dict())
    single_model.eval()
    row_weights = torch.ones(len(experiment.metadata))
    factor_weights = torch.ones(4)
    denominator = torch.tensor(float(sum(map(len, chunks))))

    batch_output = trainer.forward_model(batch_model, padded, state=None)
    batch_loss, _ = trainer.compute_multitask_loss(
        batch_output,
        padded,
        row_weights,
        factor_weights,
        variant="H2_full",
        normalization_denominator=denominator,
        regularization_fraction=padded["lane_lengths"].float() / denominator,
    )
    batch_loss.backward()

    single_loss = torch.zeros(())
    single_outputs = []
    for lane, chunk in zip(lanes, chunks, strict=True):
        output = trainer.forward_model(single_model, lane, state=None)
        single_outputs.append(output)
        loss, _ = trainer.compute_multitask_loss(
            output,
            lane,
            row_weights,
            factor_weights,
            variant="H2_full",
            normalization_denominator=denominator,
            regularization_fraction=len(chunk) / float(denominator),
        )
        single_loss = single_loss + loss
    single_loss.backward()

    assert batch_loss.item() == pytest.approx(single_loss.item(), rel=2.0e-5, abs=2.0e-5)
    for name in (
        "candidate_mean_bpm",
        "source_rr_bpm",
        "factor_probabilities",
        "spike_sequence",
    ):
        for lane, chunk in enumerate(chunks):
            assert torch.allclose(
                batch_output[name][lane, : len(chunk)],
                single_outputs[lane][name][0],
                rtol=2.0e-5,
                atol=2.0e-5,
                equal_nan=True,
            )
    for batch_layer, lane_layers in zip(
        batch_output["state"], zip(*(output["state"] for output in single_outputs)), strict=True
    ):
        for state_index in range(2):
            expected = torch.cat(
                [layer[state_index] for layer in lane_layers], dim=0
            )
            assert torch.allclose(
                batch_layer[state_index], expected, rtol=2.0e-5, atol=2.0e-5
            )
    for batch_parameter, single_parameter in zip(
        batch_model.parameters(), single_model.parameters(), strict=True
    ):
        if batch_parameter.grad is None or single_parameter.grad is None:
            assert batch_parameter.grad is single_parameter.grad is None
        else:
            assert torch.allclose(
                batch_parameter.grad,
                single_parameter.grad,
                rtol=2.0e-4,
                atol=2.0e-5,
            )

    # Mutating every padded source value cannot affect any real lane output or
    # the preserved recurrent state of the shorter lane.
    mutated = {name: value.clone() for name, value in padded.items()}
    mutated["node_features"][1, 2] = 1234.0
    mutated["candidate_rr"][1, 2] = 44.0
    mutated["anchor_rr"][1, 2] = 44.0
    mutated["classical_rr"][1, 2] = 44.0
    mutated_output = trainer.forward_model(batch_model, mutated, state=None)
    for name in ("source_rr_bpm", "factor_probabilities", "spike_sequence"):
        assert torch.allclose(
            batch_output[name][:, :2],
            mutated_output[name][:, :2],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
    for original_layer, mutated_layer in zip(
        batch_output["state"], mutated_output["state"], strict=True
    ):
        for original, changed in zip(original_layer, mutated_layer, strict=True):
            assert torch.equal(original, changed)


def test_per_session_cvar_and_valid_length_spike_weighting_are_padding_inert() -> None:
    errors = torch.tensor(
        [
            [10.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [9.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
        ]
    )
    active = torch.tensor(
        [
            [True, True, True, True, True, True, True, True, True],
            [True, False, False, False, False, False, False, False, False],
        ]
    )
    cvar = trainer._per_session_cvar20(errors, active, torch.tensor(10.0))
    # Lane 0 contributes its top two and lane 1 its top one.  Pooling lanes
    # would incorrectly contribute only two values.
    assert cvar.item() == pytest.approx((10.0 + 8.0 + 9.0) / 10.0)
    pooled = torch.topk(errors[active], 2).values.sum() / 10.0
    assert cvar.item() != pytest.approx(pooled.item())

    rates = torch.tensor([[0.30, 0.30], [0.00, 0.00], [0.95, 0.95]])
    weighted = trainer._valid_length_spike_penalty(
        rates, torch.tensor([0.75, 0.25, 0.0])
    )
    expected = trainer._valid_length_spike_penalty(rates[:1], 0.75)
    expected = expected + trainer._valid_length_spike_penalty(rates[1:2], 0.25)
    assert weighted.item() == pytest.approx(expected.item())


def test_batched_target_free_prediction_matches_batch_one_and_reduces_forwards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, proposer = _synthetic_cache(tmp_path)
    experiment = trainer.load_experiment(cache, proposer, outer_fold=3, seed=7)
    train_positions, _, _ = trainer.split_positions(experiment.metadata, 3)
    scaler = trainer.fit_outer_train_standardizer(experiment, train_positions)
    raw_positions = np.asarray([0, 1, 2, 3, 4], np.int64)
    arrays = trainer.InferenceArrays(
        cache_index=np.asarray([100, 101, 102, 200, 201], np.int64),
        node_features=np.asarray(experiment.node_features[raw_positions], np.float32),
        candidate_rr=np.asarray(experiment.candidate_rr[raw_positions], np.float32),
        candidate_mask=np.asarray(experiment.candidate_mask[raw_positions], bool),
        joint_radar_mask=np.asarray(experiment.joint_radar_mask[raw_positions], bool),
        anchor_rr=np.asarray(experiment.anchor_rr[raw_positions], np.float32),
        anchor_std=np.asarray(experiment.anchor_std[raw_positions], np.float32),
        anchor_available=np.asarray(experiment.anchor_available[raw_positions], bool),
        classical_rr=experiment.metadata.iloc[raw_positions]["classical_rr_bpm"].to_numpy(
            np.float32, copy=True
        ),
        session_reset=np.asarray([True, False, False, True, False]),
    )
    torch.manual_seed(91)
    model = trainer.build_model("H0_no_factor", torch.device("cpu"))
    model.eval()
    real_forward = trainer.forward_model
    calls = 0

    def counted_forward(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return real_forward(*args, **kwargs)

    monkeypatch.setattr(trainer, "forward_model", counted_forward)
    batch_one = trainer.predict_inference_arrays(
        model,
        arrays,
        scaler,
        torch.device("cpu"),
        amp=False,
        chunk_windows=2,
        batch_sessions=1,
    )
    batch_one_calls = calls
    calls = 0
    batched = trainer.predict_inference_arrays(
        model,
        arrays,
        scaler,
        torch.device("cpu"),
        amp=False,
        chunk_windows=2,
        batch_sessions=4,
    )
    assert batch_one_calls == 3
    assert calls == 2
    for name in trainer.PredictionBundle.__dataclass_fields__:
        assert np.allclose(
            getattr(batch_one, name),
            getattr(batched, name),
            rtol=2.0e-5,
            atol=2.0e-5,
            equal_nan=True,
        )


def test_scientific_signature_excludes_only_v8_orchestration_fields() -> None:
    base = {
        "output_directory": "/tmp/discovery",
        "campaign_phase_label": "discovery",
        "promotion_authorization_path": None,
        "release_mode": "raw_anchor",
        "resume_flag": False,
        "model": {"dropout": 0.05},
        "optimizer": {"learning_rate": 3.0e-4},
        "nested": {"release_mode": "scientific_nested_value_is_retained"},
    }
    promoted = {
        **base,
        "output_directory": "/tmp/promotion",
        "campaign_phase_label": "promotion",
        "promotion_authorization_path": "/tmp/auth.json",
        "release_mode": "hard_source_argmax",
        "resume_flag": True,
    }
    first = trainer.canonical_scientific_signature(base)
    second = trainer.canonical_scientific_signature(promoted)
    assert first == second
    assert first["nested"]["release_mode"] == "scientific_nested_value_is_retained"
    assert trainer.scientific_signature_sha256(first) == trainer.scientific_signature_sha256(
        dict(reversed(list(second.items())))
    )
    changed = {**second, "model": {"dropout": 0.10}}
    assert trainer.scientific_signature_sha256(first) != trainer.scientific_signature_sha256(
        changed
    )
    with pytest.raises(ValueError, match="lacks orchestration"):
        trainer.canonical_scientific_signature({"model": {}})
    with pytest.raises(ValueError, match="contains orchestration"):
        trainer.scientific_signature_sha256({**first, "resume_flag": False})

    cache_changed = json.loads(json.dumps(first))
    cache_changed["input_bindings"] = {
        "verified_cache_inputs": {
            "manifest": {"sha256": "1" * 64, "bytes": 10},
            "outputs": {"node_features": {"sha256": "2" * 64, "bytes": 20}},
        }
    }
    cache_drifted = json.loads(json.dumps(cache_changed))
    cache_drifted["input_bindings"]["verified_cache_inputs"]["outputs"][
        "node_features"
    ]["sha256"] = "3" * 64
    assert trainer.scientific_signature_sha256(
        cache_changed
    ) != trainer.scientific_signature_sha256(cache_drifted)


def test_efficiency_benchmark_parser_and_direct_launch_hook_fail_closed(
    tmp_path: Path,
) -> None:
    args = trainer.parse_args(
        [
            "--mode", "efficiency_benchmark",
            "--cache", str(tmp_path / "cache"),
            "--proposer-stack", str(tmp_path / "proposer.npz"),
            "--output-dir", str(tmp_path / "out"),
            "--target-sealed-capability-receipt", str(tmp_path / "capability.json"),
            "--expected-admitted-context-json", json.dumps(
                {
                    "campaign_revision": "V8R4",
                    "infrastructure_revision": "V8R4A",
                    "authorization_generation": "CONTEXT1",
                    "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
                    "outer_fold": 3,
                    "seed": 20260828,
                    "variant": "H0_no_factor",
                },
                sort_keys=True,
            ),
            "--outer-fold", "3",
            "--seed", "20260828",
            "--variant", "H0_no_factor",
            "--epochs", "2",
            "--device", "cpu",
        ]
    )
    assert args.mode == "efficiency_benchmark"
    invocation = "ab" * 32
    assert trainer._benchmark_invocation_sha256(
        {
            "classification": "verified_v8_gpu_admitted_child_lifecycle",
            "phase": "efficiency_benchmark",
            "invocation_sha256": invocation,
        }
    ) == invocation
    with pytest.raises(
        RuntimeError,
        match="V8R4A pretrain authorization is not issued|inherited descriptor is absent",
    ):
        trainer._admitted_child_binding_for_cli(phase="efficiency_benchmark")


def test_admitted_cli_scope_exactly_binds_train_predict_and_benchmark_units() -> None:
    train_args = trainer.argparse.Namespace(
        outer_fold=4,
        seed=20260829,
        variant="H2_full",
        resume=True,
        release_mode=None,
    )
    train_binding = {
        "phase": "discovery",
        "context": {
            "outer_fold": 4,
            "seed": 20260829,
            "variant": "H2_full",
            "execution_number": 2,
            "resume": True,
        },
    }
    trainer._validate_admitted_cli_scope(
        train_args,
        train_binding,
        phase="discovery",
        expected_context=train_binding["context"],
    )
    with pytest.raises(RuntimeError, match="unit identity drifted"):
        trainer._validate_admitted_cli_scope(
            train_args,
            {
                **train_binding,
                "context": {**train_binding["context"], "seed": 20260828},
            },
            phase="discovery",
            expected_context={**train_binding["context"], "seed": 20260828},
        )
    with pytest.raises(RuntimeError, match="resume scope drifted"):
        trainer._validate_admitted_cli_scope(
            trainer.argparse.Namespace(**{**vars(train_args), "resume": False}),
            train_binding,
            phase="discovery",
            expected_context=train_binding["context"],
        )

    predict_args = trainer.argparse.Namespace(
        outer_fold=0,
        seed=20260830,
        variant="H1_factor",
        resume=False,
        release_mode="hard_source_argmax",
    )
    predict_binding = {
        "phase": "promotion_prediction",
        "context": {
            "outer_fold": 0,
            "seed": 20260830,
            "variant": "H1_factor",
            "release_mode": "hard_source_argmax",
            "attempt_number": 1,
        },
    }
    trainer._validate_admitted_cli_scope(
        predict_args,
        predict_binding,
        phase="promotion_prediction",
        expected_context=predict_binding["context"],
    )
    with pytest.raises(RuntimeError, match="unit identity drifted"):
        trainer._validate_admitted_cli_scope(
            trainer.argparse.Namespace(
                **{**vars(predict_args), "release_mode": "raw_anchor"}
            ),
            predict_binding,
            phase="promotion_prediction",
            expected_context=predict_binding["context"],
        )

    benchmark_args = trainer.argparse.Namespace(
        outer_fold=3,
        seed=20260828,
        variant="H0_no_factor",
        resume=False,
        release_mode=None,
    )
    trainer._validate_admitted_cli_scope(
        benchmark_args,
        {
            "phase": "efficiency_benchmark",
            "context": {
                "campaign_revision": "V8R4",
                "infrastructure_revision": "V8R4A",
                "authorization_generation": "CONTEXT1",
                "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
                "outer_fold": 3,
                "seed": 20260828,
                "variant": "H0_no_factor",
            },
        },
        phase="efficiency_benchmark",
        expected_context={
            "campaign_revision": "V8R4",
            "infrastructure_revision": "V8R4A",
            "authorization_generation": "CONTEXT1",
            "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
            "outer_fold": 3,
            "seed": 20260828,
            "variant": "H0_no_factor",
        },
    )
    fixed_benchmark_context = {
        "campaign_revision": "V8R4",
        "infrastructure_revision": "V8R4A",
        "authorization_generation": "CONTEXT1",
        "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
        "outer_fold": 3,
        "seed": 20260828,
        "variant": "H0_no_factor",
    }
    for mutated in (
        {key: value for key, value in fixed_benchmark_context.items() if key != "authorization_generation"},
        {**fixed_benchmark_context, "unexpected": True},
        {**fixed_benchmark_context, "authorization_generation": "ROOTBIND1"},
    ):
        with pytest.raises(RuntimeError, match="unit identity drifted"):
            trainer._validate_admitted_cli_scope(
                benchmark_args,
                {"phase": "efficiency_benchmark", "context": mutated},
                phase="efficiency_benchmark",
                expected_context=mutated,
            )


def test_trainer_bridges_explicit_capability_phase_and_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    class FakeAuthorization:
        @staticmethod
        def validate_pretrain_target_scoped_admitted_child(
            root: Path,
            capability: Path,
            binding: object,
            **kwargs: object,
        ) -> dict[str, object]:
            observed.update(
                {
                    "root": root,
                    "capability": capability,
                    "binding": binding,
                    **kwargs,
                }
            )
            return {
                "valid": True,
                "training_authorized": True,
                "commercial_claim_authorized": False,
            }

    monkeypatch.setattr(trainer, "_authorization_module", lambda: FakeAuthorization)
    binding = {"phase": "discovery", "context": {"outer_fold": 3}}
    capability = tmp_path / "capability.json"
    result = trainer.validate_pretrain_authorization(
        binding,
        target_sealed_capability_receipt=capability,
        expected_phase="discovery",
        expected_context={"outer_fold": 3},
        expected_outer_fold=3,
    )

    assert result["valid"] is True
    assert observed["capability"] == capability
    assert observed["expected_phase"] == "discovery"
    assert observed["expected_context"] == {"outer_fold": 3}
    assert observed["expected_outer_fold"] == 3


@pytest.mark.parametrize(
    ("entry_name", "execution_phase", "context"),
    (
        (
            "train",
            "discovery",
            {
                "outer_fold": 3,
                "seed": 7,
                "variant": "H0_no_factor",
                "execution_number": 0,
                "resume": False,
            },
        ),
        (
            "predict_target_free",
            "promotion_prediction",
            {
                "outer_fold": 3,
                "seed": 7,
                "variant": "H0_no_factor",
                "release_mode": "hard_source_argmax",
                "attempt_number": 0,
            },
        ),
    ),
)
def test_imported_real_entries_reject_forged_pretrain_before_side_effects(
    entry_name: str,
    execution_phase: str,
    context: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / entry_name
    capability = tmp_path / "capability.json"
    args = SimpleNamespace(
        campaign_phase="discovery",
        outer_fold=3,
        seed=7,
        variant="H0_no_factor",
        release_mode="hard_source_argmax",
        resume=False,
        promotion_authorization=None,
        target_sealed_capability_receipt=capability,
        expected_admitted_context_json=context,
        output_dir=output,
        cache=tmp_path / "must_not_stat_cache",
        proposer_stack=tmp_path / "must_not_stat_proposer",
        predict_input=tmp_path / "must_not_open_input.npz",
        checkpoint=tmp_path / "must_not_open_checkpoint.pt",
        scaler=tmp_path / "must_not_open_scaler.json",
        device="cuda",
    )
    binding = {"phase": execution_phase, "context": context}
    fresh = _authorized()
    forged = dict(fresh)
    # Strict JSON comparison must not inherit Python's True == 1 alias.
    forged["training_authorized"] = 1
    observed: dict[str, object] = {}

    def fresh_validation(*call_args: object, **call_kwargs: object) -> dict[str, object]:
        observed["args"] = call_args
        observed["kwargs"] = call_kwargs
        return dict(fresh)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("entry crossed its authorization boundary")

    monkeypatch.setattr(trainer, "validate_pretrain_authorization", fresh_validation)
    monkeypatch.setattr(trainer, "validate_phase_authorization", forbidden)
    monkeypatch.setattr(trainer, "load_experiment", forbidden)
    monkeypatch.setattr(trainer, "load_sanitized_inference_input", forbidden)
    monkeypatch.setattr(trainer, "seed_everything", forbidden)
    monkeypatch.setattr(trainer, "build_model", forbidden)
    monkeypatch.setattr(trainer.torch.cuda, "is_available", forbidden)

    entry = getattr(trainer, entry_name)
    with pytest.raises(RuntimeError, match="differs from fresh entry validation"):
        entry(args, pretrain=forged, admitted_binding=binding)

    assert observed["args"] == (binding,)
    assert observed["kwargs"] == {
        "target_sealed_capability_receipt": capability.resolve(),
        "expected_phase": execution_phase,
        "expected_context": context,
        "expected_outer_fold": 3,
    }
    assert not output.exists()


@pytest.mark.parametrize(
    ("entry_name", "execution_phase"),
    (("train", "discovery"), ("predict_target_free", "promotion_prediction")),
)
def test_imported_real_entries_reject_admitted_context_drift_before_validation(
    entry_name: str,
    execution_phase: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = {
        "outer_fold": 3,
        "seed": 7,
        "variant": "H0_no_factor",
        **(
            {"execution_number": 0, "resume": False}
            if entry_name == "train"
            else {
                "release_mode": "hard_source_argmax",
                "attempt_number": 0,
            }
        ),
    }
    output = tmp_path / entry_name
    args = SimpleNamespace(
        campaign_phase="discovery",
        outer_fold=3,
        seed=7,
        variant="H0_no_factor",
        release_mode="hard_source_argmax",
        resume=False,
        promotion_authorization=None,
        target_sealed_capability_receipt=tmp_path / "capability.json",
        expected_admitted_context_json=context,
        output_dir=output,
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("authorization bytes opened after context drift")

    monkeypatch.setattr(trainer, "validate_pretrain_authorization", forbidden)
    bad_binding = {
        "phase": execution_phase,
        "context": {**context, "seed": 8},
    }
    entry = getattr(trainer, entry_name)
    with pytest.raises(RuntimeError, match="lifecycle scope is invalid"):
        entry(args, pretrain=_authorized(), admitted_binding=bad_binding)
    assert not output.exists()


def test_expected_admitted_context_parser_rejects_duplicate_keys() -> None:
    with pytest.raises(trainer.argparse.ArgumentTypeError, match="duplicate"):
        trainer._parse_expected_admitted_context('{"seed":1,"seed":2}')


def test_efficiency_benchmark_returns_only_strict_target_free_timing_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, proposer = _synthetic_cache(tmp_path, seed=20260828, outer_fold=3)
    monkeypatch.setattr(trainer, "validate_pretrain_authorization", _authorized)
    expected_context = {
        "campaign_revision": "V8R4",
        "infrastructure_revision": "V8R4A",
        "authorization_generation": "CONTEXT1",
        "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
        "outer_fold": 3,
        "seed": 20260828,
        "variant": "H0_no_factor",
    }
    args = trainer.parse_args(
        [
            "--mode", "efficiency_benchmark",
            "--cache", str(cache),
            "--proposer-stack", str(proposer),
            "--output-dir", str(tmp_path / "must_remain_empty"),
            "--target-sealed-capability-receipt", str(tmp_path / "capability.json"),
            "--expected-admitted-context-json", json.dumps(expected_context, sort_keys=True),
            "--outer-fold", "3",
            "--seed", "20260828",
            "--variant", "H0_no_factor",
            "--epochs", "2",
            "--device", "cpu",
        ]
    )
    telemetry = trainer.run_efficiency_benchmark(
        args,
        admitted_binding={
            "classification": "verified_v8_gpu_admitted_child_lifecycle",
            "phase": "efficiency_benchmark",
            "context": expected_context,
            "invocation_sha256": "cd" * 32,
        },
    )
    assert set(telemetry) == {
        "invocation_sha256",
        "epochs_completed",
        "epochs",
        "optimizer_steps",
        "training_windows",
        "validation_windows",
        "peak_cuda_memory_bytes",
    }
    assert telemetry["epochs_completed"] == 2
    assert len(telemetry["epochs"]) == 2
    assert telemetry["optimizer_steps"] == 2
    assert telemetry["training_windows"] == 24
    assert telemetry["validation_windows"] == 6
    assert telemetry["peak_cuda_memory_bytes"] == 0
    assert not (tmp_path / "must_remain_empty").exists()
    forbidden = ("accuracy", "metric", "checkpoint", "selection", "target")
    serialized = json.dumps(telemetry, sort_keys=True).lower()
    assert not any(token in serialized for token in forbidden)
    for epoch in telemetry["epochs"]:
        assert set(epoch) == {
            "epoch",
            "warmup",
            "train_ns",
            "validation_ns",
            "total_ns",
            "optimizer_steps",
            "training_windows",
            "validation_windows",
        }
        assert epoch["train_ns"] > 0
        assert epoch["validation_ns"] > 0
        assert epoch["total_ns"] == epoch["train_ns"] + epoch["validation_ns"]
    assert [epoch["warmup"] for epoch in telemetry["epochs"]] == [True, False]


def test_imported_efficiency_rejects_forged_pretrain_before_cache_or_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = {
        "campaign_revision": "V8R4",
        "infrastructure_revision": "V8R4A",
        "authorization_generation": "CONTEXT1",
        "benchmark_id": "v8_hfr_2epoch_no_accuracy_metric_efficiency",
        "outer_fold": 3,
        "seed": 20260828,
        "variant": "H0_no_factor",
    }
    args = trainer.parse_args(
        [
            "--mode", "efficiency_benchmark",
            "--cache", str(tmp_path / "must_not_open_cache"),
            "--proposer-stack", str(tmp_path / "must_not_open_proposer"),
            "--output-dir", str(tmp_path / "must_not_exist"),
            "--target-sealed-capability-receipt", str(tmp_path / "capability.json"),
            "--expected-admitted-context-json", json.dumps(context, sort_keys=True),
            "--outer-fold", "3",
            "--seed", "20260828",
            "--variant", "H0_no_factor",
            "--epochs", "2",
            "--device", "cuda",
        ]
    )
    binding = {
        "classification": "verified_v8_gpu_admitted_child_lifecycle",
        "phase": "efficiency_benchmark",
        "context": context,
        "invocation_sha256": "ef" * 32,
    }
    fresh = _authorized()
    forged = dict(fresh)
    forged["training_authorized"] = 1
    observed: dict[str, object] = {}

    def fresh_validation(*call_args: object, **call_kwargs: object) -> dict[str, object]:
        observed["args"] = call_args
        observed["kwargs"] = call_kwargs
        return dict(fresh)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("benchmark crossed its authorization boundary")

    monkeypatch.setattr(trainer, "validate_pretrain_authorization", fresh_validation)
    monkeypatch.setattr(trainer, "load_experiment", forbidden)
    monkeypatch.setattr(trainer, "seed_everything", forbidden)
    monkeypatch.setattr(trainer.torch.cuda, "is_available", forbidden)

    with pytest.raises(RuntimeError, match="differs from fresh entry validation"):
        trainer.run_efficiency_benchmark(
            args, admitted_binding=binding, pretrain=forged
        )
    assert observed["args"] == (binding,)
    assert observed["kwargs"] == {
        "target_sealed_capability_receipt": (
            tmp_path / "capability.json"
        ).resolve(),
        "expected_phase": "efficiency_benchmark",
        "expected_context": context,
        "expected_outer_fold": 3,
    }
    assert not (tmp_path / "must_not_exist").exists()


# V8R4 authorization additions.  These live in the pre-authorized trainer
# regression file; the standalone staging boundary suite is only a review aid.
def test_v8r4_boundary_declaration_corrects_v8r3_attestation() -> None:
    boundary = trainer.DISCOVERY_LEAKAGE_BOUNDARY_DECLARATION
    assert boundary["campaign_revision"] == "V8R4"
    assert boundary["physical_input_partition"] == (
        "outer_excluded_training_validation_pack"
    )
    assert boundary["outer_prediction_pack_absent_during_discovery"] is True
    assert boundary["combined_target_bearing_cache_opened"] is False
    assert boundary["outer_test_identity_or_classical_context_materialized"] is False
    assert boundary["outer_test_feature_values_materialized_or_forwarded"] is False
    assert "outer_test_positions_constructed" not in boundary


def test_v8r4_audited_row_array_rejects_outer_and_whole_array_access() -> None:
    audit = trainer.RowAccessAudit(
        cache_index=np.asarray([1, 4], np.int64),
        fold=np.asarray([0, 1], np.int16),
        outer_fold=3,
    )
    wrapped = trainer.AuditedRowArray(np.arange(4).reshape(2, 2), audit, "node")
    assert np.array_equal(wrapped[np.asarray([0])], np.asarray([[0, 1]]))
    with pytest.raises(RuntimeError, match="whole-array"):
        np.asarray(wrapped)
    audit.fold[1] = 3
    with pytest.raises(RuntimeError, match="outer-test"):
        _ = wrapped[1]
    evidence = audit.snapshot()
    assert evidence["implicit_whole_array_conversions"] == 1
    assert evidence["outer_row_access_attempts"] == 1


def test_v8r4_validation_identity_is_fixed_unicode_pickle_free(tmp_path: Path) -> None:
    rows = 2
    one = np.ones(rows, np.float32)
    available = np.ones(rows, bool)
    bundle = trainer.PredictionBundle(
        cache_index=np.arange(rows, dtype=np.int64),
        raw_anchor_bpm=one,
        raw_anchor_available=available,
        hard_source_bpm=one,
        hard_source_available=available,
        fixed_confidence_switch_bpm=one,
        fixed_confidence_switch_available=available,
        selected_source_probability=one,
        selected_source_code=np.zeros(rows, np.int16),
        source_scale_bpm=one,
        quality=one,
        factor_probabilities=np.full((rows, 4), 0.25, np.float32),
        spike_rate=one,
    )
    metadata = pd.DataFrame(
        {
            "rr_bpm": [12.0, 13.0],
            "reference_valid": [True, True],
            "identity": ["person-a", "개체-b"],
        }
    )
    arrays, _ = trainer._validation_artifacts(
        bundle, SimpleNamespace(metadata=metadata), np.asarray([0, 1], np.int64)
    )
    assert arrays["identity"].dtype.kind == "U"
    output = tmp_path / "validation_predictions.npz"
    trainer.atomic_save_npz(output, arrays)
    with np.load(output, allow_pickle=False) as archive:
        assert archive["identity"].dtype.kind == "U"
        assert archive["identity"].tolist() == ["person-a", "개체-b"]


def test_v8r4a_atomic_outputs_are_born_and_committed_immutable(
    tmp_path: Path,
) -> None:
    observed_mode: list[int] = []
    direct = tmp_path / "history.json"

    def writer(descriptor: int) -> None:
        observed_mode.append(os.fstat(descriptor).st_mode & 0o777)
        os.write(descriptor, b"{}\n")

    trainer._atomic_publish_immutable(direct, writer)
    assert observed_mode == [0o444]
    assert direct.stat().st_mode & 0o777 == 0o444
    assert direct.stat().st_nlink == 1

    trainer.atomic_write_json(tmp_path / "run_manifest.json", {"ok": True})
    trainer.atomic_torch_save(tmp_path / "best.pt", {"value": torch.ones(2)})
    trainer.atomic_save_npz(
        tmp_path / "validation_predictions.npz",
        {"value": np.asarray([1.0], np.float32)},
    )
    for name in ("run_manifest.json", "best.pt", "validation_predictions.npz"):
        path = tmp_path / name
        assert path.stat().st_mode & 0o777 == 0o444
        assert path.stat().st_nlink == 1
    assert torch.load(tmp_path / "best.pt", weights_only=True)["value"].tolist() == [1.0, 1.0]
    with np.load(tmp_path / "validation_predictions.npz", allow_pickle=False) as archive:
        assert archive["value"].tolist() == [1.0]


def test_v8r4a_atomic_replace_changes_inode_but_never_mode(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    trainer.atomic_write_json(path, {"epoch": 1})
    first_inode = path.stat().st_ino
    trainer.atomic_write_json(path, {"epoch": 2})
    assert path.stat().st_ino != first_inode
    assert path.stat().st_mode & 0o777 == 0o444
    assert json.loads(path.read_text()) == {"epoch": 2}


def test_v8r4a_failed_atomic_writer_preserves_residue_for_quarantine(
    tmp_path: Path,
) -> None:
    def fail_after_write(descriptor: int) -> None:
        os.write(descriptor, b"partial")
        raise RuntimeError("injected kill boundary")

    with pytest.raises(RuntimeError, match="injected"):
        trainer._atomic_publish_immutable(tmp_path / "last.pt", fail_after_write)
    residues = list(tmp_path.glob(".last.pt.v8r4a-tmp-*"))
    assert len(residues) == 1
    residue = residues[0]
    assert residue.read_bytes() == b"partial"
    assert residue.stat().st_mode & 0o777 == 0o444
    assert residue.stat().st_nlink == 1
    with pytest.raises(RuntimeError, match="no files were deleted"):
        trainer.cleanup_stale_atomic_temporaries(tmp_path)
    assert residue.read_bytes() == b"partial"


def test_v8r4a_failed_atomic_writer_never_unlinks_swapped_path(
    tmp_path: Path,
) -> None:
    moved = tmp_path / "descriptor-owned-residue"
    observed_temporary: list[Path] = []

    def swap_path_then_fail(descriptor: int) -> None:
        os.write(descriptor, b"descriptor-owned")
        candidates = list(tmp_path.glob(".last.pt.v8r4a-tmp-*"))
        assert len(candidates) == 1
        temporary = candidates[0]
        observed_temporary.append(temporary)
        temporary.rename(moved)
        temporary.write_bytes(b"unrelated-sentinel")
        temporary.chmod(0o444)
        raise RuntimeError("injected path swap")

    with pytest.raises(RuntimeError, match="path swap"):
        trainer._atomic_publish_immutable(tmp_path / "last.pt", swap_path_then_fail)

    assert moved.read_bytes() == b"descriptor-owned"
    assert moved.stat().st_mode & 0o777 == 0o444
    assert len(observed_temporary) == 1
    assert observed_temporary[0].read_bytes() == b"unrelated-sentinel"


def test_atomic_npz_returns_committed_inode_binding_and_detects_replacement(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions.npz"
    first = trainer.atomic_save_npz(
        path, {"value": np.asarray([1], dtype=np.int64)}
    )
    assert first == trainer._verified_regular_file(path)[0]
    trainer.atomic_save_npz(path, {"value": np.asarray([2], dtype=np.int64)})
    with pytest.raises(RuntimeError, match="SHA-256 drifted"):
        trainer._assert_file_binding_current(first)


def test_v8r4a_stale_cleanup_never_claims_filename_only_ownership(
    tmp_path: Path,
) -> None:
    stale = tmp_path / (".last.pt.v8r4a-tmp-" + "a" * 32)
    stale.write_bytes(b"partial")
    stale.chmod(0o444)
    unknown = tmp_path / (".not-governed.bin.v8r4a-tmp-" + "b" * 32)
    unknown.write_bytes(b"keep")
    unknown.chmod(0o444)
    malformed = tmp_path / ".last.pt.v8r4a-tmp-short"
    malformed.write_bytes(b"keep")
    malformed.chmod(0o444)

    with pytest.raises(RuntimeError, match="no files were deleted"):
        trainer.cleanup_stale_atomic_temporaries(tmp_path)

    assert stale.exists()
    assert stale.read_bytes() == b"partial"
    assert unknown.exists()
    assert malformed.exists()


@pytest.mark.parametrize("unsafe", ["mutable", "hardlink", "symlink"])
def test_v8r4a_stale_cleanup_refuses_unsafe_matching_entry(
    tmp_path: Path, unsafe: str
) -> None:
    stale = tmp_path / (".best.pt.v8r4a-tmp-" + "c" * 32)
    if unsafe == "symlink":
        outside = tmp_path / "outside"
        outside.write_bytes(b"outside")
        stale.symlink_to(outside.name)
    else:
        stale.write_bytes(b"partial")
        stale.chmod(0o644 if unsafe == "mutable" else 0o444)
        if unsafe == "hardlink":
            os.link(stale, tmp_path / "alias")
    with pytest.raises(RuntimeError, match="requires quarantine"):
        trainer.cleanup_stale_atomic_temporaries(tmp_path)


def test_v8r4a_atomic_writer_refuses_unregistered_filename(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="unregistered"):
        trainer.atomic_write_json(tmp_path / "surprise.json", {})


def test_v8r4_resume_checkpoint_epoch_rejects_v8r3() -> None:
    payload = {
        "campaign_revision": "V8R4",
        "checkpoint_compatibility": "v8r4_nonouter_training_validation_pack_only",
        "run_signature_sha256": "0" * 64,
        "scientific_signature_sha256": "1" * 64,
        "reuse_context_sha256": "3" * 64,
        "scaler_sha256": "2" * 64,
        "epoch": 1,
        "stale": 0,
        "best_epoch": 1,
        "best_selection_key": (0,),
        "history": [{}],
        "model_state": {},
        "optimizer_state": {},
        "gradient_scaler_state": {},
        "python_rng_state": (),
        "numpy_rng_state": {
            "bit_generator": "MT19937",
            "keys": torch.zeros(624, dtype=torch.int64),
            "position": 0,
            "has_gauss": 0,
            "cached_gaussian": 0.0,
        },
        "torch_rng_state": torch.zeros(1, dtype=torch.uint8),
        "cuda_rng_state_all": [],
    }
    trainer.validate_v8r4_resume_checkpoint(payload)
    payload.pop("campaign_revision")
    with pytest.raises(RuntimeError, match="predates"):
        trainer.validate_v8r4_resume_checkpoint(payload)


def test_v8r4_csv_boundary_proves_nonouter_before_sensitive_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, proposer = _synthetic_cache(tmp_path)
    calls: list[object] = []
    original = trainer.pd.read_csv

    def recording_read_csv(*args: object, **kwargs: object) -> object:
        calls.append(kwargs.get("usecols"))
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer.pd, "read_csv", recording_read_csv)
    experiment = trainer.load_experiment(cache, proposer, outer_fold=3, seed=7)
    explicit = [value for value in calls if value is not None]
    assert explicit[0] == ["cache_index", "fold"]
    sensitive = next(
        index
        for index, columns in enumerate(explicit)
        if "identity" in columns or "rr_bpm" in columns
    )
    assert sensitive > 0
    assert 3 not in set(experiment.metadata["fold"])


def test_v8r4_outer_row_rejected_before_sensitive_csv_or_numpy_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, proposer = _synthetic_cache(tmp_path)
    metadata = pd.read_csv(cache / "metadata.csv")
    metadata.loc[0, "fold"] = 3
    metadata.to_csv(cache / "metadata.csv", index=False)
    _refresh_cache_output_binding(cache, "metadata")
    csv_calls: list[object] = []
    original_read_csv = trainer.pd.read_csv
    original_np_load = trainer.np.load
    feature_opens = 0

    def recording_read_csv(*args: object, **kwargs: object) -> object:
        csv_calls.append(kwargs.get("usecols"))
        return original_read_csv(*args, **kwargs)

    def recording_np_load(path: object, *args: object, **kwargs: object) -> object:
        nonlocal feature_opens
        if Path(path).name in {
            "node_features.npy",
            "candidate_bpm.npy",
            "candidate_mask.npy",
            "joint_radar_mask.npy",
        }:
            feature_opens += 1
        return original_np_load(path, *args, **kwargs)

    monkeypatch.setattr(trainer.pd, "read_csv", recording_read_csv)
    monkeypatch.setattr(trainer.np, "load", recording_np_load)
    with pytest.raises(RuntimeError, match="physical nonouter"):
        trainer.load_experiment(cache, proposer, outer_fold=3, seed=7)
    assert [value for value in csv_calls if value is not None] == [
        ["cache_index", "fold"]
    ]
    assert feature_opens == 0
