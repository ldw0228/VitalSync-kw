from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/merge_nested_proposer_indexes.py"
SPEC = importlib.util.spec_from_file_location("merge_nested_proposer_indexes", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MERGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MERGE)

FIXED_SCRIPT = ROOT / "scripts/run_fixed_i3_pretest_campaign.py"
FIXED_SPEC = importlib.util.spec_from_file_location(
    "run_fixed_i3_pretest_campaign_for_merge_test", FIXED_SCRIPT
)
assert FIXED_SPEC is not None and FIXED_SPEC.loader is not None
FIXED = importlib.util.module_from_spec(FIXED_SPEC)
FIXED_SPEC.loader.exec_module(FIXED)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["content_sha256"] = MERGE.canonical_content_sha256(result)
    return result


def _artifact_binding(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "sha256": MERGE.sha256_file(path),
        "bytes": len(payload),
    }


def _manifest_names(outer: int) -> list[str]:
    return sorted(MERGE._manifest_name_for_outer(outer))


class MatrixFixture:
    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path
        self.manifest_root = tmp_path / "non_test_manifests"
        self.full_control = tmp_path / "full/control"
        self.retrain_control = tmp_path / "retrain/control"
        self.main_run = tmp_path / "runs/main"
        self.retrain_run = tmp_path / "runs/retrain"
        self.output = tmp_path / "merged/index.json"
        self.full_plan_path = self.full_control / "plan.json"
        self.main_index_path = self.full_control / "index.json"
        self.retrain_plan_path = self.retrain_control / "plan.json"
        self.retrain_index_path = self.retrain_control / "index.json"

        assignments = tmp_path / "fold_assignments.json"
        cache_manifest = tmp_path / "cache/manifest.json"
        _write_json(assignments, {"identity_to_fold": {}})
        _write_json(cache_manifest, {"sessions": []})
        self.assignments_binding = {
            "path": str(assignments.resolve()),
            "sha256": MERGE.sha256_file(assignments),
        }
        self.cache_binding = {
            "path": str(cache_manifest.resolve()),
            "sha256": MERGE.sha256_file(cache_manifest),
        }

        self.manifests: dict[tuple[int, str], tuple[Path, dict[str, Any]]] = {}
        for outer in MERGE.FOLDS:
            for name in _manifest_names(outer):
                role = (
                    "hcs_validation"
                    if name.startswith("validation_pred_")
                    else "hcs_train_oof"
                )
                document = _seal(
                    {
                        "schema_version": 1,
                        "outer_fold": outer,
                        "fold_id": outer * 100 + int(Path(name).stem.rsplit("_", 1)[1]),
                        "role": role,
                    }
                )
                path = self.manifest_root / f"outer_{outer}" / name
                _write_json(path, document)
                self.manifests[(outer, name)] = (path.resolve(), document)

        shared = {
            "schema_version": 1,
            "classification": MERGE.PLAN_CLASSIFICATION,
            "outer_test_opened": False,
            "outer_test_record_count": 0,
            "seeds": list(MERGE.SEEDS),
            "roles": sorted(MERGE.ROLES),
            "manifest_root": str(self.manifest_root.resolve()),
            "fold_assignments": self.assignments_binding,
            "cache_manifest": self.cache_binding,
            "source_bindings": {"sealed/source.py": {"sha256": "a" * 64}},
            "training_specification": {"epochs": 80, "model": "both"},
            "control_index_separate_from_run_root": True,
        }
        self.full_plan = _seal(
            {
                **shared,
                "outer_folds": list(MERGE.FOLDS),
                "requested_units": 90,
                "manifest_plan_content_sha256": "b" * 64,
                "reusable_run_root": str(self.main_run.resolve()),
                "units": self._units(MERGE.FOLDS, self.main_run),
            }
        )
        self.retrain_plan = _seal(
            {
                **shared,
                "outer_folds": sorted(MERGE.RETRAIN_FOLDS),
                "requested_units": 30,
                "manifest_plan_content_sha256": "c" * 64,
                "reusable_run_root": str(self.retrain_run.resolve()),
                "units": self._units(sorted(MERGE.RETRAIN_FOLDS), self.retrain_run),
            }
        )
        _write_json(self.full_plan_path, self.full_plan)
        _write_json(self.retrain_plan_path, self.retrain_plan)
        self.main_index = self._index(
            self.full_plan,
            self.full_plan_path,
            self.full_plan["units"],
        )
        self.retrain_index = self._index(
            self.retrain_plan,
            self.retrain_plan_path,
            self.retrain_plan["units"],
        )
        _write_json(self.main_index_path, self.main_index)
        _write_json(self.retrain_index_path, self.retrain_index)
        self._write_execution_evidence()

    @staticmethod
    def _runtime_seal_document(created_utc: str) -> dict[str, Any]:
        sources = []
        for path in (MERGE.CANONICAL_SUPERVISOR, MERGE.CANONICAL_RUNNER):
            stat = path.stat()
            sources.append(
                {
                    "path": str(path.resolve()),
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": MERGE.sha256_file(path),
                }
            )
        document: dict[str, Any] = {
            "schema_version": 1,
            "classification": "supplemental_runtime_input_byte_inventory",
            "post_launch_attestation": False,
            "attestation_phase": "prelaunch",
            "commercial_claim_authorized": False,
            "sources": sources,
            "input_trees": [],
            "campaign_bindings": [],
            "runtime": {"python": "test"},
            "created_utc": created_utc,
        }
        document["content_sha256"] = MERGE._runtime_seal_content_sha256(document)
        return document

    def _write_execution_evidence(self) -> None:
        execution_root = self.retrain_control.parent
        old_seals: list[tuple[Path, dict[str, Any]]] = []
        for position, name in enumerate(
            (
                "execution_prelaunch_runtime_input_seal.json",
                "execution_prelaunch_runtime_input_seal_v2.json",
            )
        ):
            path = execution_root / name
            document = self._runtime_seal_document(
                f"2026-08-28T00:00:0{position}+00:00"
            )
            _write_json(path, document)
            old_seals.append((path, document))
        selected_path = execution_root / "execution_prelaunch_runtime_input_seal_v3.json"
        selected = self._runtime_seal_document("2026-08-28T00:00:03+00:00")
        _write_json(selected_path, selected)
        supervisor = MERGE.CANONICAL_SUPERVISOR
        retrain_binding = {
            "path": str(self.retrain_index_path.resolve()),
            "sha256": MERGE.sha256_file(self.retrain_index_path),
            "bytes": self.retrain_index_path.stat().st_size,
            "content_sha256": self.retrain_index["content_sha256"],
        }
        seal_binding = {
            "path": str(selected_path.resolve()),
            "sha256": MERGE.sha256_file(selected_path),
            "bytes": selected_path.stat().st_size,
            "content_sha256": selected["content_sha256"],
            "verified_files": 2,
        }
        attestation: dict[str, Any] = {
            "schema_version": 1,
            "classification": "sealed_non_test_proposer_execution_attestation",
            "outer_test_opened": False,
            "outer_test_record_count": 0,
            "commercial_claim_authorized": False,
            "expected_units": 30,
            "completed_units": 30,
            "one_new_unit_per_invocation": True,
            "runtime_seal_verified_before_and_after_every_invocation": True,
            "invocations_this_resume": 30,
            "runtime_input_seal": seal_binding,
            "campaign_index": retrain_binding,
            "unit_command": [
                str(MERGE.CANONICAL_PYTHON_LAUNCHER.absolute()),
                str(MERGE.CANONICAL_RUNNER.resolve()),
                "--manifest-root",
                str(self.manifest_root.resolve()),
                "--control-root",
                str(self.retrain_control.resolve()),
                "--run-root",
                str(self.retrain_run.resolve()),
                "--outer-folds",
                "3,4",
                "--gpu-lock",
                str((MERGE.CANONICAL_GPU_ROOT / "gpu_admission.lock").resolve()),
                "--gpu-ledger",
                str((MERGE.CANONICAL_GPU_ROOT / "gpu_admission_ledger.jsonl").resolve()),
                "--max-new-units",
                "1",
            ],
            "supervisor": {
                "path": str(supervisor.resolve()),
                "sha256": MERGE.sha256_file(supervisor),
                "bytes": supervisor.stat().st_size,
            },
        }
        attestation["content_sha256"] = MERGE.canonical_content_sha256(attestation)
        self.execution_attestation_path = execution_root / "execution_attestation.json"
        _write_json(self.execution_attestation_path, attestation)
        self.execution_attestation = attestation
        self.selected_execution_seal_path = selected_path
        self.selected_execution_seal = selected
        note = {
            "schema_version": 1,
            "classification": "non_test_proposer_execution_runtime_seal_supersession",
            "created_utc": "2026-08-28T00:00:04+00:00",
            "outer_test_opened": False,
            "target_or_reference_accessed": False,
            "commercial_claim_authorized": False,
            "selected_runtime_seal": {
                "path": str(selected_path.resolve()),
                "sha256": MERGE.sha256_file(selected_path),
                "content_sha256": selected["content_sha256"],
                "verified_files": 2,
                "supervisor_sha256": MERGE.sha256_file(supervisor),
            },
            "superseded_runtime_seals": [
                {
                    "path": str(path.resolve()),
                    "sha256": MERGE.sha256_file(path),
                    "content_sha256": document["content_sha256"],
                    "reason": f"fixture superseded ABI {position}",
                }
                for position, (path, document) in enumerate(old_seals)
            ],
            "selected_execution_command": [
                "python",
                str(supervisor.resolve()),
                "--runtime-seal",
                str(selected_path.resolve()),
            ],
            "first_selected_seal_progress": {
                "completed_units": 1,
                "requested_units": 30,
                "state": "running",
            },
        }
        self.supersession_path = execution_root / "execution_runtime_seal_supersession.json"
        _write_json(self.supersession_path, note)

    def _units(self, folds: Any, run_root: Path) -> list[dict[str, Any]]:
        units: list[dict[str, Any]] = []
        for seed in MERGE.SEEDS:
            for outer in folds:
                for name in _manifest_names(outer):
                    manifest_path, manifest = self.manifests[(outer, name)]
                    stem = Path(name).stem
                    output_dir = run_root / f"seed_{seed}/outer_{outer}/{stem}"
                    units.append(
                        {
                            "unit_id": f"seed_{seed}/outer_{outer}/{stem}",
                            "seed": seed,
                            "outer_fold": outer,
                            "role": (
                                "hcs_validation"
                                if name.startswith("validation_pred_")
                                else "hcs_train_oof"
                            ),
                            "manifest": str(manifest_path),
                            "manifest_sha256": MERGE.sha256_file(manifest_path),
                            "manifest_content_sha256": manifest["content_sha256"],
                            "output_dir": str(output_dir.resolve()),
                            "checkpoint": str((output_dir / "snn_best.pt").resolve()),
                            "all_window_prediction": str(
                                (output_dir / "snn_prediction_all_windows.npz").resolve()
                            ),
                        }
                    )
        return units

    def _record(self, unit: Mapping[str, Any]) -> dict[str, Any]:
        checkpoint_path = Path(str(unit["checkpoint"]))
        prediction_path = Path(str(unit["all_window_prediction"]))
        checkpoint = _artifact_binding(
            checkpoint_path,
            f"checkpoint:{unit['unit_id']}:{checkpoint_path}".encode(),
        )
        prediction = _artifact_binding(
            prediction_path,
            f"prediction:{unit['unit_id']}:{prediction_path}".encode(),
        )
        return {
            "unit_id": unit["unit_id"],
            "seed": unit["seed"],
            "outer_fold": unit["outer_fold"],
            "role": unit["role"],
            "manifest": unit["manifest"],
            "manifest_sha256": unit["manifest_sha256"],
            "manifest_content_sha256": unit["manifest_content_sha256"],
            "output_dir": unit["output_dir"],
            "checkpoint": checkpoint,
            "all_window_prediction": prediction,
        }

    def _index(
        self,
        plan: Mapping[str, Any],
        plan_path: Path,
        units: list[dict[str, Any]],
    ) -> dict[str, Any]:
        records = [self._record(unit) for unit in units]
        return _seal(
            {
                "schema_version": 1,
                "classification": MERGE.INDEX_CLASSIFICATION,
                "campaign_plan_content_sha256": plan["content_sha256"],
                "campaign_plan": {
                    "path": str(plan_path.resolve()),
                    "sha256": MERGE.sha256_file(plan_path),
                    "content_sha256": plan["content_sha256"],
                },
                "manifest_root": plan["manifest_root"],
                "manifest_plan_content_sha256": plan[
                    "manifest_plan_content_sha256"
                ],
                "outer_test_opened": False,
                "outer_test_record_count": 0,
                "requested_units": len(records),
                "completed_units": len(records),
                "records": records,
            }
        )

    def fake_inspect(
        self, unit: Mapping[str, Any]
    ) -> tuple[str, dict[str, Any], None]:
        paths = {
            str(record["unit_id"]): record
            for record in (*self.main_index["records"], *self.retrain_index["records"])
            if record["checkpoint"]["path"] == str(unit["checkpoint"])
        }
        record = paths[str(unit["unit_id"])]
        # Rebind current bytes so post-index artifact tampering is detected.
        record = json.loads(json.dumps(record))
        for key in ("checkpoint", "all_window_prediction"):
            path = Path(record[key]["path"])
            record[key]["sha256"] = MERGE.sha256_file(path)
            record[key]["bytes"] = path.stat().st_size
        return "complete", record, None

    def merge(self, **overrides: Any) -> dict[str, Any]:
        arguments = {
            "full_plan_path": self.full_plan_path,
            "main_index_path": self.main_index_path,
            "retrain_plan_path": self.retrain_plan_path,
            "retrain_index_path": self.retrain_index_path,
            "output_path": self.output,
            "inspect_unit": self.fake_inspect,
        }
        arguments.update(overrides)
        return MERGE.merge_indexes(**arguments)


@pytest.fixture
def matrix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MatrixFixture:
    fixture = MatrixFixture(tmp_path)
    monkeypatch.setattr(MERGE.campaign, "verify_campaign_source_bindings", lambda _: None)
    return fixture


def test_merge_selects_exact_60_plus_30_cover_and_fixed_i3_accepts_it(
    matrix: MatrixFixture,
) -> None:
    result = matrix.merge()
    assert result["classification"] == MERGE.INDEX_CLASSIFICATION
    assert result["merge_classification"] == (
        "retrospective_current_source_uniform_90_unit_proposer_index"
    )
    assert result["campaign_plan_content_sha256"] == matrix.full_plan["content_sha256"]
    assert result["completed_units"] == 90
    assert len(result["records"]) == 90
    assert result["merge_provenance"]["source_indexes"]["main"][
        "selected_units"
    ] == 60
    assert result["merge_provenance"]["source_indexes"][
        "current_source_retrain_f34"
    ]["selected_units"] == 30
    for record in result["records"]:
        path = Path(record["checkpoint"]["path"])
        expected_root = (
            matrix.retrain_run
            if record["outer_fold"] in MERGE.RETRAIN_FOLDS
            else matrix.main_run
        )
        assert path.is_relative_to(expected_root.resolve())

    groups, provenance = FIXED.validate_non_test_plan_index(
        matrix.full_plan_path, matrix.output
    )
    assert len(groups) == 18
    assert provenance["index"]["completed_units"] == 90

    before = matrix.output.read_bytes()
    second = matrix.merge()
    assert matrix.output.read_bytes() == before
    assert second == result


def test_merge_rejects_incomplete_duplicate_and_test_records_before_opening(
    matrix: MatrixFixture,
) -> None:
    incomplete = json.loads(json.dumps(matrix.main_index))
    incomplete["records"].pop()
    incomplete["completed_units"] = 89
    incomplete = _seal({k: v for k, v in incomplete.items() if k != "content_sha256"})
    _write_json(matrix.main_index_path, incomplete)
    with pytest.raises(RuntimeError, match="incomplete|exactly"):
        matrix.merge()

    _write_json(matrix.main_index_path, matrix.main_index)
    duplicate = json.loads(json.dumps(matrix.main_index))
    duplicate["records"][-1] = duplicate["records"][0]
    duplicate = _seal({k: v for k, v in duplicate.items() if k != "content_sha256"})
    _write_json(matrix.main_index_path, duplicate)
    with pytest.raises(RuntimeError, match="duplicate|order/cover"):
        matrix.merge()

    _write_json(matrix.main_index_path, matrix.main_index)
    forbidden = matrix.root / "outer_0/test_pred_0.json"
    forbidden.parent.mkdir(parents=True, exist_ok=True)
    forbidden.write_text("deliberately not JSON", encoding="utf-8")
    test_record = json.loads(json.dumps(matrix.main_index))
    test_record["records"][0]["manifest"] = str(forbidden)
    test_record = _seal({k: v for k, v in test_record.items() if k != "content_sha256"})
    _write_json(matrix.main_index_path, test_record)
    with pytest.raises(RuntimeError, match="outer-test path is forbidden"):
        matrix.merge()


def test_merge_rejects_artifact_tampering_and_plan_mismatch(
    matrix: MatrixFixture,
) -> None:
    artifact = Path(matrix.main_index["records"][0]["checkpoint"]["path"])
    artifact.write_bytes(b"changed after the completed index")
    with pytest.raises(RuntimeError, match="validated artifacts"):
        matrix.merge()

    # Restore via fixture's deterministic record constructor, then introduce a
    # semantically sealed but incompatible retrain training specification.
    unit = matrix.full_plan["units"][0]
    original = matrix.main_index["records"][0]["checkpoint"]
    artifact.write_bytes(
        f"checkpoint:{unit['unit_id']}:{artifact}".encode()
    )
    assert MERGE.sha256_file(artifact) == original["sha256"]
    retrain = json.loads(json.dumps(matrix.retrain_plan))
    retrain["training_specification"]["epochs"] = 81
    retrain = _seal({k: v for k, v in retrain.items() if k != "content_sha256"})
    _write_json(matrix.retrain_plan_path, retrain)
    rebound = matrix._index(retrain, matrix.retrain_plan_path, retrain["units"])
    _write_json(matrix.retrain_index_path, rebound)
    with pytest.raises(RuntimeError, match="differs from full split authority"):
        matrix.merge()


def test_merge_binds_valid_runtime_seal_without_reopening_payloads(
    matrix: MatrixFixture,
) -> None:
    unavailable_payload = matrix.root / "must_not_be_opened/metadata.csv"
    seal = {
        "schema_version": 1,
        "classification": "supplemental_runtime_input_byte_inventory",
        "post_launch_attestation": True,
        "sources": [],
        "input_trees": [
            {
                "root": str(unavailable_payload.parent),
                "files": [
                    {
                        "path": str(unavailable_payload),
                        "sha256": "d" * 64,
                        "bytes": 123,
                    }
                ],
            }
        ],
        "campaign_bindings": [],
        "runtime": {"python": "test"},
        "created_utc": "2026-08-28T00:00:00+00:00",
    }
    seal["content_sha256"] = MERGE._runtime_seal_content_sha256(seal)
    seal_path = matrix.root / "runtime_input_seal.json"
    _write_json(seal_path, seal)
    result = matrix.merge(runtime_seals=[seal_path])
    binding = result["merge_provenance"]["runtime_seals"][0]
    assert binding["sha256"] == MERGE.sha256_file(seal_path)
    assert binding["content_sha256"] == seal["content_sha256"]
    assert not unavailable_payload.exists()


def test_merge_uses_receipt_selected_v3_seal_and_binds_complete_execution_chain(
    matrix: MatrixFixture,
) -> None:
    result = matrix.merge()
    provenance = result["merge_provenance"]
    assert provenance["runtime_seal_count"] == 1
    assert provenance["runtime_seals"][0]["path"] == str(
        matrix.selected_execution_seal_path.resolve()
    )
    assert provenance["retrain_execution_attestation"]["sha256"] == MERGE.sha256_file(
        matrix.execution_attestation_path
    )
    assert provenance["retrain_execution_attestation_required"] is True
    execution = provenance["retrain_execution"]
    assert execution["required"] is True
    assert execution["completion_evidence"]["completed_units"] == 30
    assert execution["authoritative_runtime_input_seal"][
        "payloads_rehashed_during_merge"
    ] is True
    assert execution["supervisor_live_rehashed"] is True
    assert execution["supersession_note"]["superseded_runtime_seal_count"] == 2


def test_merge_unconditionally_refuses_missing_final_attestation(
    matrix: MatrixFixture,
) -> None:
    matrix.execution_attestation_path.unlink()
    with pytest.raises(RuntimeError, match="requires.*30/30 execution attestation"):
        matrix.merge()


def test_merge_rejects_receipt_abi_drift_and_does_not_fall_back_to_old_seal(
    matrix: MatrixFixture,
) -> None:
    receipt = json.loads(json.dumps(matrix.execution_attestation))
    del receipt["runtime_input_seal"]["bytes"]
    receipt.pop("content_sha256")
    receipt["content_sha256"] = MERGE.canonical_content_sha256(receipt)
    _write_json(matrix.execution_attestation_path, receipt)
    with pytest.raises(RuntimeError, match="runtime seal.*binding|malformed"):
        matrix.merge()


def test_merge_rejects_retroactive_zero_invocation_attestation(
    matrix: MatrixFixture,
) -> None:
    receipt = json.loads(json.dumps(matrix.execution_attestation))
    receipt["invocations_this_resume"] = 0
    receipt.pop("content_sha256")
    receipt["content_sha256"] = MERGE.canonical_content_sha256(receipt)
    _write_json(matrix.execution_attestation_path, receipt)
    with pytest.raises(RuntimeError, match="attestation invariants are invalid"):
        matrix.merge()


def test_merge_rejects_noncanonical_supervisor_or_unit_command(
    matrix: MatrixFixture,
) -> None:
    receipt = json.loads(json.dumps(matrix.execution_attestation))
    alternate = matrix.root / "alternate_supervisor.py"
    alternate.write_text("# alternate\n", encoding="utf-8")
    receipt["supervisor"] = {
        "path": str(alternate.resolve()),
        "sha256": MERGE.sha256_file(alternate),
        "bytes": alternate.stat().st_size,
    }
    receipt.pop("content_sha256")
    receipt["content_sha256"] = MERGE.canonical_content_sha256(receipt)
    _write_json(matrix.execution_attestation_path, receipt)
    with pytest.raises(RuntimeError, match="receipt/canonical execution supervisor"):
        matrix.merge()

    matrix.execution_attestation_path.unlink()
    receipt = json.loads(json.dumps(matrix.execution_attestation))
    receipt["unit_command"][-1] = "2"
    receipt.pop("content_sha256")
    receipt["content_sha256"] = MERGE.canonical_content_sha256(receipt)
    _write_json(matrix.execution_attestation_path, receipt)
    with pytest.raises(RuntimeError, match="attestation invariants are invalid"):
        matrix.merge()


def test_merge_rejects_supersession_note_that_selects_old_seal(
    matrix: MatrixFixture,
) -> None:
    note = json.loads(matrix.supersession_path.read_text(encoding="utf-8"))
    old = note["superseded_runtime_seals"][0]
    note["selected_runtime_seal"].update(
        {
            "path": old["path"],
            "sha256": old["sha256"],
            "content_sha256": old["content_sha256"],
        }
    )
    _write_json(matrix.supersession_path, note)
    with pytest.raises(RuntimeError, match="supersession selected seal identity mismatch"):
        matrix.merge()


def test_merge_live_rehashes_receipt_selected_seal_payload(
    matrix: MatrixFixture,
) -> None:
    source = matrix.root / "sealed_source.py"
    source.write_text("version = 1\n", encoding="utf-8")
    stat = source.stat()
    selected = MatrixFixture._runtime_seal_document(
        "2026-08-28T00:00:03+00:00"
    )
    selected["sources"] = [
        {
            "path": str(source.resolve()),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": MERGE.sha256_file(source),
        }
    ]
    selected["content_sha256"] = MERGE._runtime_seal_content_sha256(selected)
    _write_json(matrix.selected_execution_seal_path, selected)
    receipt = json.loads(json.dumps(matrix.execution_attestation))
    receipt["runtime_input_seal"] = {
        "path": str(matrix.selected_execution_seal_path.resolve()),
        "sha256": MERGE.sha256_file(matrix.selected_execution_seal_path),
        "bytes": matrix.selected_execution_seal_path.stat().st_size,
        "content_sha256": selected["content_sha256"],
        "verified_files": 1,
    }
    receipt.pop("content_sha256")
    receipt["content_sha256"] = MERGE.canonical_content_sha256(receipt)
    _write_json(matrix.execution_attestation_path, receipt)
    # Remove the advisory note so this test isolates receipt-authoritative live
    # verification rather than requiring the note to be regenerated.
    matrix.supersession_path.unlink()
    source.write_text("version = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="runtime input changed after seal"):
        matrix.merge()


def test_merge_refuses_source_overwrite_and_different_existing_output(
    matrix: MatrixFixture,
) -> None:
    with pytest.raises(RuntimeError, match="must not overwrite"):
        matrix.merge(output_path=matrix.main_index_path)
    matrix.output.parent.mkdir(parents=True, exist_ok=True)
    matrix.output.write_text('{"unrelated":true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        matrix.merge()


def test_explicit_missing_or_tampered_runtime_seal_fails_closed(
    matrix: MatrixFixture,
) -> None:
    missing = matrix.root / "missing_runtime_seal.json"
    with pytest.raises(RuntimeError, match="explicit runtime input seal is missing"):
        matrix.merge(runtime_seals=[missing])

    bad = matrix.root / "bad_runtime_seal.json"
    _write_json(
        bad,
        {
            "schema_version": 1,
            "classification": "supplemental_runtime_input_byte_inventory",
            "content_sha256": "0" * 64,
        },
    )
    with pytest.raises(RuntimeError, match="content hash mismatch"):
        matrix.merge(runtime_seals=[bad])
