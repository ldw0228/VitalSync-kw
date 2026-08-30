from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_hcs_discovery_campaign.py"
SPEC = importlib.util.spec_from_file_location("run_hcs_discovery_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def _write_hashed_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    document = dict(value)
    document["content_sha256"] = RUN.canonical_content_sha256(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return document


def _write(path: Path, value: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return RUN.bind_file(path)


def _stub_source(kind: str, log: Path, *, fail_once: Path | None = None) -> str:
    prefix = f"""#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
import numpy as np

LOG = Path({str(log)!r})
with LOG.open('a', encoding='utf-8') as stream:
    stream.write({kind!r} + '\\n')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical(value):
    payload = dict(value); payload.pop('content_sha256', None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
"""
    if kind == "stack":
        return prefix + """
p=argparse.ArgumentParser(); p.add_argument('--discovery-index'); p.add_argument('--plan'); p.add_argument('--cache-dir'); p.add_argument('--outer-fold',type=int); p.add_argument('--seed',type=int); p.add_argument('--output'); a=p.parse_args()
index=json.loads(Path(a.discovery_index).read_text())
records=[r for r in index['records'] if r['outer_fold']==a.outer_fold and r['seed']==a.seed]
sources=[{'role':r['role'],'sha256':r['all_window_prediction']['sha256']} for r in records]
arrays={'cache_index':np.asarray([0],dtype=np.int64),'proposal_available':np.asarray([True]),'outer_fold':np.asarray(a.outer_fold,dtype=np.int16),'seed':np.asarray(a.seed,dtype=np.int64),'strict_nested':np.asarray(True),'outer_test_opened':np.asarray(False)}
provenance={'format_version':1,'classification':'retrospective_strict_nested_proposer_stack','strict_nested':True,'outer_test_opened':False,'outer_fold':a.outer_fold,'seed':a.seed,'source_units':sources}
d=hashlib.sha256()
for name in sorted(arrays):
    value=np.ascontiguousarray(arrays[name]); d.update(name.encode()+b'\\0'); d.update(value.dtype.str.encode()+b'\\0'); d.update(json.dumps(value.shape,separators=(',',':')).encode()); d.update(b'\\0'); d.update(value.tobytes())
d.update(json.dumps(provenance,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())
signature=d.hexdigest(); provenance['content_signature_sha256']=signature
arrays['content_signature_sha256']=np.asarray(signature); arrays['provenance_json']=np.asarray(json.dumps(provenance,sort_keys=True,separators=(',',':')))
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
with out.open('wb') as stream: np.savez_compressed(stream,**arrays)
"""
    if kind == "fallback":
        return prefix + """
p=argparse.ArgumentParser(); p.add_argument('--stack'); p.add_argument('--output'); a=p.parse_args(); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
out.write_text('cache_index,prediction_bpm,rr_std_bpm\\n0,20,1\\n')
with np.load(a.stack,allow_pickle=False) as z: outer=int(z['outer_fold'].item()); seed=int(z['seed'].item()); sig=str(z['content_signature_sha256'].item())
doc={'schema_version':1,'artifact_type':'strict_nested_fallback_oof','strict_nested':True,'outer_test_opened':False,'outer_fold':outer,'seed':seed,'source_stack':{'path':str(Path(a.stack).resolve()),'sha256':sha(a.stack),'content_signature_sha256':sig},'output_csv':{'path':str(out.resolve()),'sha256':sha(out),'columns':['cache_index','prediction_bpm','rr_std_bpm']}}
doc['content_sha256']=canonical(doc)
out.with_name(out.name+'.provenance.json').write_text(json.dumps(doc,sort_keys=True))
"""
    if kind == "cache":
        return prefix + """
p=argparse.ArgumentParser(); p.add_argument('--rf-cache'); p.add_argument('--svd-cache'); p.add_argument('--proposer'); p.add_argument('--fold-assignments'); p.add_argument('--output-dir'); p.add_argument('--batch-size'); p.add_argument('--merge-radius-bpm'); p.add_argument('--proposal-selection'); p.add_argument('--posterior-nms-suppression-bpm',default='1.25'); p.add_argument('--base-proposals'); p.add_argument('--svd-components',type=int); p.add_argument('--proposer-features',action='store_true'); a=p.parse_args()
out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); data=out/'data.bin'; data.write_bytes(b'cache')
i2=a.proposal_selection=='posterior-nms'
doc={'format_version':1,'complete':True,'build_signature_sha256':hashlib.sha256(str(out).encode()).hexdigest(),'row_count':1,'inputs':{'proposer':{'path':str(Path(a.proposer).resolve()),'sha256':sha(a.proposer)}},'candidate_policy':{'merge_radius_bpm':float(a.merge_radius_bpm),'proposal_selection':a.proposal_selection,'posterior_nms_suppression_bpm':float(a.posterior_nms_suppression_bpm),'base_source_policy':'explicit_expected_then_map_before_direct_modes' if i2 else 'none'},'evidence_policy':{'svd_components':a.svd_components,'proposer_posterior_feature_policy':'full_posterior_candidate_local_summaries_plus_exact_row_diagnostics' if a.proposer_features else 'disabled_backward_compatible_i1_schema'},'outputs':{'data':{'filename':'data.bin','bytes':data.stat().st_size,'sha256':sha(data)}}}
doc['content_sha256']=canonical(doc); (out/'manifest.json').write_text(json.dumps(doc,sort_keys=True))
"""
    assert kind == "trainer"
    fail = ""
    if fail_once is not None:
        fail = f"""
sentinel=Path({str(fail_once)!r})
if not sentinel.exists():
    sentinel.write_text('failed once'); out.mkdir(parents=True,exist_ok=True); (out/'partial.txt').write_text('preserve me'); raise SystemExit(7)
"""
    return prefix + """
p=argparse.ArgumentParser(); p.add_argument('--cache'); p.add_argument('--fallback-oof'); p.add_argument('--output-dir'); p.add_argument('--fold',type=int); p.add_argument('--seed',type=int); p.add_argument('--device'); p.add_argument('--deterministic',action='store_true'); p.add_argument('--preset'); p.add_argument('--epochs'); p.add_argument('--minimum-epochs'); p.add_argument('--patience'); p.add_argument('--learning-rate'); p.add_argument('--adaptive-iteration',type=int); p.add_argument('--maximum-coverage'); p.add_argument('--maximum-fpr'); p.add_argument('--minimum-precision'); p.add_argument('--discovery-only',action='store_true'); p.add_argument('--amp',action='store_true'); p.add_argument('--tail-weight'); p.add_argument('--cvar-weight'); p.add_argument('--warmup-windows'); p.add_argument('--gradient-accumulation-sessions'); a,extra=p.parse_known_args(); out=Path(a.output_dir)
""" + fail + """
out.mkdir(parents=True,exist_ok=True)
for name,value in [('best_checkpoint.pt',b'checkpoint'),('scaler.json',b'{}'),('fallback_policy.json',b'{}'),('run_manifest.json',b'{}')]: (out/name).write_bytes(value)
with (out/'validation_predictions.npz').open('wb') as stream: np.savez_compressed(stream,prediction=np.asarray([20.0]))
bad={'mae':2.0,'identity_macro_mae':2.0,'rmse':3.0,'within_2':0.5,'catastrophic_over_5':0.2,'tail_25_35_mae':5.0}
(out/'validation_metrics.json').write_text(json.dumps({'locked_final':bad},sort_keys=True))
cache_manifest=Path(a.cache)/'manifest.json'
lock={'schema_version':1,'outer_test_not_opened_before_this_lock':True,'outer_fold':a.fold,'seed':a.seed,'adaptive_iteration':a.adaptive_iteration,'cache_manifest_sha256':sha(cache_manifest),'fallback_oof_sha256':sha(a.fallback_oof),'checkpoint_sha256':sha(out/'best_checkpoint.pt'),'scaler_sha256':sha(out/'scaler.json'),'policy_sha256':sha(out/'fallback_policy.json'),'run_manifest_sha256':sha(out/'run_manifest.json')}
(out/'selection_lock.json').write_text(json.dumps(lock,sort_keys=True))
"""


def _fixture(tmp_path: Path, *, complete_units: int = 5, test_record: bool = False, fail_once: bool = False):
    manifests = tmp_path / "manifests"
    units: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    names = ["inner_pred_0", "inner_pred_1", "inner_pred_2", "inner_pred_5", "validation_pred_4"]
    for position, stem in enumerate(names):
        role = "hcs_validation" if stem.startswith("validation") else "hcs_train_oof"
        manifest_path = manifests / "outer_3" / f"{stem}.json"
        manifest = _write_hashed_json(
            manifest_path,
            {"schema_version": 1, "fold_id": 300 + position},
        )
        units.append(
            {
                "manifest": f"outer_3/{stem}.json",
                "manifest_content_sha256": manifest["content_sha256"],
                "prediction_fold": position,
                "role": role,
            }
        )
        if position < complete_units:
            checkpoint = _write(tmp_path / "sources" / stem / "checkpoint.pt", stem.encode())
            prediction = _write(tmp_path / "sources" / stem / "prediction.npz", (stem + "p").encode())
            records.append(
                {
                    "outer_fold": 3,
                    "seed": 11,
                    "role": role,
                    "manifest": str(manifest_path.resolve()),
                    "manifest_sha256": RUN.sha256_file(manifest_path),
                    "checkpoint": checkpoint,
                    "all_window_prediction": prediction,
                }
            )
    rf = tmp_path / "rf"; svd = tmp_path / "svd"
    rf_manifest = _write(rf / "manifest.json", b"{}")
    _write(svd / "manifest.json", b"{}")
    folds = _write(tmp_path / "folds.json", b"{}")
    plan = _write_hashed_json(
        manifests / "plan.json",
        {
            "schema_version": 1,
            "cache_manifest_sha256": rf_manifest["sha256"],
            "fold_assignments_sha256": folds["sha256"],
            "outer_folds": {
                "3": {
                    "outer_test_fold": 3,
                    "outer_validation_fold": 4,
                    "units": units,
                }
            },
        },
    )
    del plan
    if test_record:
        records.append(
            {
                "outer_fold": 3,
                "seed": 11,
                "role": "hcs_test_open_only_after_policy_lock",
                "manifest": str((manifests / "outer_3/test_pred_3.json").resolve()),
            }
        )
    index_path = tmp_path / "discovery_index.json"
    index_path.write_text(
        json.dumps(
            {"schema_version": 1, "outer_test_opened": False, "records": records},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "accuracy_targets_per_seed": {
                    "overall_mae_bpm_max": 1.0,
                    "identity_macro_mae_bpm_max": 1.0,
                    "rmse_bpm_max": 1.8,
                    "within_2_fraction_min": 0.9,
                    "over_5_fraction_max": 0.03,
                    "high_rr_25_35_mae_bpm_max": 2.0,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    log = tmp_path / "calls.log"
    stubs: dict[str, Path] = {}
    for kind in ("stack", "fallback", "cache", "trainer"):
        path = tmp_path / f"{kind}_stub.py"
        path.write_text(
            _stub_source(
                kind,
                log,
                fail_once=tmp_path / "trainer_failed_once.flag" if kind == "trainer" and fail_once else None,
            ),
            encoding="utf-8",
        )
        stubs[kind] = path
    argv = [
        "--discovery-index", str(index_path),
        "--plan", str(manifests / "plan.json"),
        "--contract", str(contract),
        "--artifact-root", str(tmp_path / "artifacts"),
        "--cache-root", str(tmp_path / "caches"),
        "--rf-cache", str(rf),
        "--svd-cache", str(svd),
        "--fold-assignments", str(tmp_path / "folds.json"),
        "--stack-builder", str(stubs["stack"]),
        "--fallback-builder", str(stubs["fallback"]),
        "--cache-builder", str(stubs["cache"]),
        "--trainer", str(stubs["trainer"]),
        "--python-executable", sys.executable,
        "--outer-folds", "3", "--seeds", "11", "--devices", "cpu",
        "--epochs", "1", "--minimum-epochs", "1", "--patience", "1",
    ]
    return argv, log


def test_dry_run_orders_dag_and_binds_iteration_settings(tmp_path: Path) -> None:
    argv, log = _fixture(tmp_path)
    result = RUN.run(RUN.parse_args([*argv, "--dry-run"]))
    stages = [record["stage"] for record in result["planned_commands"]]
    assert stages == [
        "build_strict_stack",
        "build_strict_fallback",
        "build_cache_i1_topk_merge050",
        "train_hcs_i1",
        "build_cache_i2r_posterior_nms125_svd12_merge050",
        "train_hcs_i2r",
        "train_hcs_i3",
    ]
    assert not log.exists()
    commands = {record["stage"]: record["argv"] for record in result["planned_commands"]}
    assert commands["build_cache_i1_topk_merge050"][-6:] == [
        "--proposal-selection", "topk", "--base-proposals", "none", "--svd-components", "6"
    ]
    i2 = commands["build_cache_i2r_posterior_nms125_svd12_merge050"]
    for token in ("posterior-nms", "1.25", "expected-map", "12", "--proposer-features"):
        assert token in i2
    i3 = commands["train_hcs_i3"]
    for token in ("--adaptive-iteration", "3", "--tail-weight", "2.0", "--cvar-weight", "0.15", "--discovery-only"):
        assert token in i3
    assert all("test_pred_" not in " ".join(command) for command in commands.values())


def test_failed_screens_continue_and_completed_dag_is_idempotent(tmp_path: Path) -> None:
    argv, log = _fixture(tmp_path)
    first = RUN.run(RUN.parse_args(argv))
    assert log.read_text(encoding="utf-8").splitlines() == [
        "stack", "fallback", "cache", "trainer", "cache", "trainer", "trainer"
    ]
    screens = first["groups"][0]["iterations"]
    assert [screens[tag]["screen"]["status"] for tag in RUN.ITERATION_ORDER] == [
        "failed", "failed", "failed"
    ]
    first_index = (tmp_path / "artifacts/campaign_index.json").read_bytes()
    second = RUN.run(RUN.parse_args(argv))
    assert second["content_sha256"] == first["content_sha256"]
    assert (tmp_path / "artifacts/campaign_index.json").read_bytes() == first_index
    assert len(log.read_text(encoding="utf-8").splitlines()) == 7


def test_outer_test_record_fails_closed_before_any_command(tmp_path: Path) -> None:
    argv, log = _fixture(tmp_path, test_record=True)
    with pytest.raises(RuntimeError, match="outer-test"):
        RUN.run(RUN.parse_args(argv))
    assert not log.exists()


def test_incomplete_five_unit_cover_waits_without_downstream_work(tmp_path: Path) -> None:
    argv, log = _fixture(tmp_path, complete_units=4)
    result = RUN.run(RUN.parse_args(argv))
    assert result["groups"][0]["discovery"]["status"] == "waiting_for_five_units"
    assert result["summary"]["waiting_groups"] == 1
    assert not log.exists()


def test_interrupted_trainer_preserves_partial_attempt_and_resumes_fresh(tmp_path: Path) -> None:
    argv, log = _fixture(tmp_path, fail_once=True)
    with pytest.raises(RUN.CommandError, match="status 7"):
        RUN.run(RUN.parse_args([*argv, "--iterations", "i1"]))
    partial = next((tmp_path / "artifacts").glob("outer_3/seed_11/training/i1_*/attempt_000/partial.txt"))
    assert partial.read_text(encoding="utf-8") == "preserve me"
    result = RUN.run(RUN.parse_args([*argv, "--iterations", "i1"]))
    output = Path(result["groups"][0]["iterations"]["i1"]["output"]["path"])
    assert output.name == "attempt_001"
    assert partial.read_text(encoding="utf-8") == "preserve me"
    assert log.read_text(encoding="utf-8").splitlines() == [
        "stack", "fallback", "cache", "trainer", "trainer"
    ]


def test_max_jobs_resumes_next_screen_on_stable_round_robin_device(tmp_path: Path) -> None:
    argv, log = _fixture(tmp_path)
    limited = [*argv, "--max-jobs", "1", "--devices", "cpu,cuda:7"]
    first = RUN.run(RUN.parse_args(limited))
    iterations = first["groups"][0]["iterations"]
    assert iterations["i1"]["status"] == "complete"
    assert iterations["i1"]["device"] == "cpu"
    assert iterations["i2r"]["status"] == "deferred_max_jobs"
    second = RUN.run(RUN.parse_args(limited))
    iterations = second["groups"][0]["iterations"]
    assert iterations["i2r"]["status"] == "complete"
    assert iterations["i2r"]["device"] == "cuda:7"
    assert iterations["i3"]["status"] == "deferred_max_jobs"
    third = RUN.run(RUN.parse_args(limited))
    iterations = third["groups"][0]["iterations"]
    assert iterations["i3"]["status"] == "complete"
    assert iterations["i3"]["device"] == "cpu"
    assert log.read_text(encoding="utf-8").splitlines().count("trainer") == 3
