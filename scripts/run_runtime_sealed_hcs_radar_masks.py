#!/usr/bin/env python3
"""Run all 126 locked radar-mask units under the fixed runtime closure.

The existing radar-mask orchestrator remains the only inference executor.  It
is initialized target-free and then invoked serially with
``--max-new-units 1``.  Runtime, fixed-completion, and primary postlock guard
attestations are rehashed before and after every unit.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import run_locked_hcs_radar_mask_campaign as radar  # noqa: E402
import run_runtime_sealed_locked_hcs_oof as primary_guard  # noqa: E402


CAMPAIGN_ROOT = PROJECT_ROOT / "artifacts/campaigns/harmonic_candidate_set_snn_v2"
RUN_ROOT = PROJECT_ROOT / "artifacts/runs/harmonic_candidate_set_snn_v2"
DEFAULT_RUNTIME_SEAL = (
    CAMPAIGN_ROOT
    / "nested_proposer/current_source_merged/fixed_i3_pretest_runtime_seal.json"
)
DEFAULT_PRETEST_ROOT = RUN_ROOT / "hcs_fixed_i3_pretest"
DEFAULT_PRETEST_INDEX = DEFAULT_PRETEST_ROOT / "pretest_index.json"
DEFAULT_COMPLETION = DEFAULT_PRETEST_ROOT / "fixed_runtime_completion_attestation.json"
DEFAULT_PRIMARY_ROOT = RUN_ROOT / "hcs_locked_oof"
DEFAULT_POSTLOCK_GUARD = DEFAULT_PRIMARY_ROOT / "postlock_runtime_guard_attestation.json"
DEFAULT_OUTPUT_ROOT = RUN_ROOT / "hcs_locked_radar_masks"
DEFAULT_UNDERLYING = SCRIPT_ROOT / "run_locked_hcs_radar_mask_campaign.py"
DEFAULT_SAFE_HELPER = SCRIPT_ROOT / "build_locked_hcs_test_inputs.py"
DEFAULT_LOCKED_OOF_SOURCE = SCRIPT_ROOT / "run_locked_hcs_oof.py"
DEFAULT_ATTESTATION_NAME = "radar_mask_runtime_guard_attestation.json"
EXPECTED_UNITS = 126


class RadarRuntimeGuardError(RuntimeError):
    """A radar-mask runtime closure or bounded-progress invariant failed."""


def _target_absent(*roots: Path) -> None:
    for raw_root in roots:
        root = raw_root.expanduser().resolve()
        found = [
            root / name
            for name in primary_guard.FORBIDDEN_NAMES
            if (root / name).exists()
        ]
        if found:
            raise RadarRuntimeGuardError(
                f"target/canonical-receipt/evaluation artifact must be absent: {found[0]}"
            )


def verify_all_guards(
    *,
    runtime_input_seal: Path,
    completion_attestation: Path,
    pretest_index: Path,
    postlock_guard: Path,
) -> dict[str, Any]:
    try:
        closure = primary_guard.verify_closure(
            runtime_input_seal=runtime_input_seal,
            completion_attestation=completion_attestation,
            pretest_index=pretest_index,
        )
        postlock = primary_guard.verify_guard_attestation(
            postlock_guard,
            expected_runtime_seal=runtime_input_seal,
            expected_completion=completion_attestation,
            reverify_closure=True,
        )
    except Exception as exc:
        raise RadarRuntimeGuardError(str(exc)) from exc
    if (
        postlock["document"].get("runtime_input_seal")
        != closure["runtime_input_seal"]
        or postlock["document"].get("fixed_pretest_completion_attestation")
        != closure["completion_attestation"]
        or postlock["document"].get("pretest_index") != closure["pretest_index"]
    ):
        raise RadarRuntimeGuardError("primary postlock guard binds another runtime closure")
    return {**closure, "postlock_guard": postlock["binding"]}


def verify_radar_guard_attestation(
    path: Path,
    *,
    primary_output_root: Path,
    runtime_input_seal: Path,
    completion_attestation: Path,
    pretest_index: Path,
    postlock_guard: Path,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    document = primary_guard._json(resolved, "radar runtime guard attestation")  # noqa: SLF001
    if (
        document.get("schema_version") != 1
        or document.get("classification")
        != "locked_hcs_radar_mask_runtime_guard_attestation"
        or int(document.get("completed_units", -1)) != EXPECTED_UNITS
        or document.get("runtime_seal_verified_before_and_after_every_unit") is not True
        or document.get("postlock_guard_verified_before_and_after_every_unit") is not True
        or document.get("target_artifact_opened") is not False
        or document.get("gpu_execution_performed") is not False
        or document.get("commercial_claim_authorized") is not False
        or primary_guard.canonical_sha256(document) != document.get("content_sha256")
    ):
        raise RadarRuntimeGuardError("radar runtime guard attestation is invalid")
    guards = verify_all_guards(
        runtime_input_seal=runtime_input_seal,
        completion_attestation=completion_attestation,
        pretest_index=pretest_index,
        postlock_guard=postlock_guard,
    )
    if (
        document.get("runtime_input_seal") != guards["runtime_input_seal"]
        or document.get("fixed_pretest_completion_attestation")
        != guards["completion_attestation"]
        or document.get("postlock_runtime_guard_attestation") != guards["postlock_guard"]
    ):
        raise RadarRuntimeGuardError("radar runtime guard binds another closure")
    root = resolved.parent
    _target_absent(primary_output_root, root)
    plan_binding = primary_guard.completion.verify_binding(
        document.get("radar_mask_plan"),
        relative_to=root,
        label="radar runtime guard plan",
    )
    complete_binding = primary_guard.completion.verify_binding(
        document.get("complete_seal"),
        relative_to=root,
        label="radar runtime guard complete seal",
    )
    if Path(plan_binding["path"]) != (root / "control/plan.json").resolve():
        raise RadarRuntimeGuardError("radar guard plan path is non-canonical")
    if Path(complete_binding["path"]) != (root / "complete_seal.json").resolve():
        raise RadarRuntimeGuardError("radar guard complete-seal path is non-canonical")
    plan = _validated_plan(Path(plan_binding["path"]), root)
    seal = primary_guard._json(Path(complete_binding["path"]), "radar complete seal")  # noqa: SLF001
    if (
        seal.get("classification")
        != "locked_hcs_all_seven_radar_mask_predictions_sealed"
        or int(seal.get("unit_count", -1)) != EXPECTED_UNITS
        or seal.get("complete_matrix") is not True
        or seal.get("target_or_label_artifact_opened_before_seal") is not False
    ):
        raise RadarRuntimeGuardError("radar complete seal is invalid")
    receipts = document.get("unit_runtime_guard_receipts")
    if not isinstance(receipts, list) or len(receipts) != EXPECTED_UNITS:
        raise RadarRuntimeGuardError("radar guard does not bind 126 unit receipts")
    for position, (raw, unit) in enumerate(zip(receipts, plan["units"], strict=True), start=1):
        binding = primary_guard.completion.verify_binding(
            raw, relative_to=root, label=f"radar runtime receipt {position}"
        )
        _validate_receipt(
            Path(binding["path"]),
            position=position,
            unit=unit,
            output_root=root,
            guards=guards,
        )
    return {"document": document, "binding": primary_guard.bind_file(resolved)}


def _validated_plan(plan_path: Path, output_root: Path) -> dict[str, Any]:
    document = primary_guard._json(plan_path, "radar-mask plan")  # noqa: SLF001
    if (
        document.get("schema_version") != 1
        or document.get("classification")
        != "locked_hcs_seven_radar_mask_label_free_plan"
        or int(document.get("unit_count", -1)) != EXPECTED_UNITS
        or document.get("target_or_label_artifact_bound") is not False
        or document.get("execution", {}).get("device") != "cpu"
        or document.get("execution", {}).get("amp") is not False
        or document.get("execution", {}).get("shell") is not False
    ):
        raise RadarRuntimeGuardError("radar-mask plan is not the locked CPU target-free plan")
    units = document.get("units")
    if not isinstance(units, list) or len(units) != EXPECTED_UNITS:
        raise RadarRuntimeGuardError("radar-mask plan does not contain 126 units")
    observed: set[str] = set()
    for unit in units:
        if not isinstance(unit, Mapping):
            raise RadarRuntimeGuardError("radar-mask plan contains a non-object unit")
        unit_id = str(unit.get("unit_id", ""))
        if not unit_id or unit_id in observed:
            raise RadarRuntimeGuardError("radar-mask plan contains a duplicate unit")
        observed.add(unit_id)
        try:
            radar._validate_command_contract(unit, output_root.resolve())  # noqa: SLF001
        except Exception as exc:
            raise RadarRuntimeGuardError(f"invalid radar-mask command contract: {exc}") from exc
        for command in unit["commands"]:
            argv = [str(token) for token in command["argv"]]
            lowered = " ".join(argv).lower()
            if "cuda" in lowered or "run_gpu_admitted" in lowered or "--amp" in argv:
                raise RadarRuntimeGuardError("radar-mask unit attempts GPU execution")
    return document


def _unit_root(output_root: Path, unit: Mapping[str, Any]) -> Path:
    return (
        output_root
        / "units"
        / f"outer_{int(unit['outer_fold'])}_seed_{int(unit['seed'])}"
        / str(unit["radar_mask"])
    )


def _prefix_count(output_root: Path, units: Sequence[Mapping[str, Any]]) -> int:
    flags: list[bool] = []
    for unit in units:
        root = _unit_root(output_root, unit)
        required = (
            root / "receipt.json",
            root / "proposer_prediction.npz",
            root / "raw_source_prediction.npz",
            root / "sealed_label_free_predictions.npz",
        )
        exists = [path.is_file() for path in required]
        if any(exists) and not all(exists):
            raise RadarRuntimeGuardError(f"partial radar-mask unit payload: {unit['unit_id']}")
        flags.append(all(exists))
    prefix = 0
    while prefix < len(flags) and flags[prefix]:
        prefix += 1
    if any(flags[prefix:]):
        raise RadarRuntimeGuardError("radar-mask completed units are not a contiguous prefix")
    return prefix


def _receipt_paths(control: Path, position: int) -> tuple[Path, Path]:
    root = control / "runtime_guard_units"
    return root / f"unit_{position:03d}.pending.json", root / f"unit_{position:03d}.json"


def _unit_bindings(output_root: Path, unit: Mapping[str, Any]) -> dict[str, Any]:
    root = _unit_root(output_root, unit)
    return {
        "campaign_receipt": primary_guard.bind_file(root / "receipt.json"),
        "proposer_prediction": primary_guard.bind_file(root / "proposer_prediction.npz"),
        "raw_source_prediction": primary_guard.bind_file(root / "raw_source_prediction.npz"),
        "sealed_prediction": primary_guard.bind_file(
            root / "sealed_label_free_predictions.npz"
        ),
    }


def _validate_receipt(
    path: Path,
    *,
    position: int,
    unit: Mapping[str, Any],
    output_root: Path,
    guards: Mapping[str, Any],
) -> dict[str, Any]:
    document = primary_guard._json(path, f"radar runtime guard unit {position}")  # noqa: SLF001
    if (
        document.get("classification")
        != "locked_hcs_radar_mask_runtime_guard_unit_receipt"
        or int(document.get("position", -1)) != position
        or document.get("unit_id") != unit["unit_id"]
        or document.get("runtime_input_seal") != guards["runtime_input_seal"]
        or document.get("fixed_pretest_completion_attestation")
        != guards["completion_attestation"]
        or document.get("postlock_runtime_guard_attestation")
        != guards["postlock_guard"]
        or document.get("outputs") != _unit_bindings(output_root, unit)
        or document.get("runtime_verified_before") is not True
        or document.get("runtime_verified_after") is not True
        or document.get("target_artifact_opened") is not False
        or primary_guard.canonical_sha256(document) != document.get("content_sha256")
    ):
        raise RadarRuntimeGuardError(f"radar-mask runtime unit receipt is invalid: {path}")
    return document


def _command(
    *,
    python_executable: Path,
    underlying_source: Path,
    primary_plan: Path,
    primary_output_root: Path,
    output_root: Path,
    safe_helper: Path,
    locked_oof_source: Path,
    batch_size: int,
    dry_run: bool,
) -> list[str]:
    argv = [
        str(python_executable),
        str(underlying_source.resolve()),
        "--primary-plan",
        str(primary_plan.resolve()),
        "--primary-output-root",
        str(primary_output_root.resolve()),
        "--output-root",
        str(output_root.resolve()),
        "--python-executable",
        str(python_executable),
        "--safe-helper",
        str(safe_helper.resolve()),
        "--locked-oof-source",
        str(locked_oof_source.resolve()),
        "--batch-size",
        str(batch_size),
    ]
    if dry_run:
        argv.append("--dry-run")
    else:
        argv.extend(["--max-new-units", "1"])
    return argv


def run_supervisor(
    *,
    runtime_input_seal: Path,
    completion_attestation: Path,
    pretest_index: Path,
    postlock_guard: Path,
    primary_output_root: Path,
    output_root: Path,
    underlying_source: Path = DEFAULT_UNDERLYING,
    python_executable: Path = Path(sys.executable),
    safe_helper: Path = DEFAULT_SAFE_HELPER,
    locked_oof_source: Path = DEFAULT_LOCKED_OOF_SOURCE,
    batch_size: int = 128,
    max_new_units: int | None = None,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise RadarRuntimeGuardError("batch_size must be positive")
    if max_new_units is not None and int(max_new_units) < 0:
        raise RadarRuntimeGuardError("max_new_units cannot be negative")
    # Preserve the venv launcher rather than resolving its symlink to the base
    # interpreter, which may not have the project's dependencies installed.
    python_executable = Path(os.path.abspath(python_executable.expanduser()))
    if not python_executable.is_file() or not os.access(python_executable, os.X_OK):
        raise RadarRuntimeGuardError("python_executable must be an executable file")
    primary_root = primary_output_root.expanduser().resolve()
    root = output_root.expanduser().resolve()
    control = root / "control/runtime_guard"
    control.mkdir(parents=True, exist_ok=True)
    campaign_lock_stream = (control / "supervisor.lock").open("a+b")
    try:
        fcntl.flock(
            campaign_lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
        )
    except BlockingIOError as exc:
        campaign_lock_stream.close()
        raise RadarRuntimeGuardError(
            "another radar-mask runtime supervisor holds the serial campaign lock"
        ) from exc
    _target_absent(primary_root, root)
    guards = verify_all_guards(
        runtime_input_seal=runtime_input_seal,
        completion_attestation=completion_attestation,
        pretest_index=pretest_index,
        postlock_guard=postlock_guard,
    )
    executed_source = primary_guard.bind_file(underlying_source)
    imported_source = primary_guard.bind_file(Path(radar.__file__))
    if executed_source["sha256"] != imported_source["sha256"]:
        raise RadarRuntimeGuardError(
            "executed radar orchestrator differs from imported contract"
        )
    source_binding = primary_guard.bind_file(Path(__file__))
    python_binding = primary_guard.bind_file(python_executable)
    primary_plan = primary_root / "locked_oof_plan.json"
    plan_path = root / "control/plan.json"
    init_pending = control / "initialize.pending.json"
    init_receipt = control / "initialize.json"
    initialize = _command(
        python_executable=python_executable,
        underlying_source=underlying_source,
        primary_plan=primary_plan,
        primary_output_root=primary_root,
        output_root=root,
        safe_helper=safe_helper,
        locked_oof_source=locked_oof_source,
        batch_size=batch_size,
        dry_run=True,
    )
    if not init_receipt.is_file():
        if plan_path.exists() and not init_pending.exists():
            raise RadarRuntimeGuardError(
                "radar plan predates this runtime guard and cannot be retroactively attested"
            )
        if not init_pending.exists():
            pending = {
                "schema_version": 1,
                "classification": "locked_hcs_radar_mask_runtime_guard_initialize_pending",
                "runtime_input_seal": guards["runtime_input_seal"],
                "fixed_pretest_completion_attestation": guards["completion_attestation"],
                "postlock_runtime_guard_attestation": guards["postlock_guard"],
                "command": initialize,
                "runtime_verified_before": True,
                "target_artifact_opened": False,
            }
            pending["content_sha256"] = primary_guard.canonical_sha256(pending)
            primary_guard.immutable_json(init_pending, pending)
        if not plan_path.is_file():
            command_record = primary_guard._run_subprocess(  # noqa: SLF001
                initialize, cwd=PROJECT_ROOT
            )
        else:
            command_record = {"argv": initialize, "recovered_after_interruption": True}
        _target_absent(primary_root, root)
        after = verify_all_guards(
            runtime_input_seal=runtime_input_seal,
            completion_attestation=completion_attestation,
            pretest_index=pretest_index,
            postlock_guard=postlock_guard,
        )
        if after != guards:
            raise RadarRuntimeGuardError("runtime guards changed during radar initialization")
        plan = _validated_plan(plan_path, root)
        receipt = {
            "schema_version": 1,
            "classification": "locked_hcs_radar_mask_runtime_guard_initialize_receipt",
            "runtime_input_seal": guards["runtime_input_seal"],
            "fixed_pretest_completion_attestation": guards["completion_attestation"],
            "postlock_runtime_guard_attestation": guards["postlock_guard"],
            "plan": primary_guard.bind_file(plan_path),
            "command": command_record,
            "runtime_verified_before": True,
            "runtime_verified_after": True,
            "target_artifact_opened": False,
        }
        receipt["content_sha256"] = primary_guard.canonical_sha256(receipt)
        primary_guard.immutable_json(init_receipt, receipt)
    else:
        receipt = primary_guard._json(init_receipt, "radar initialization receipt")  # noqa: SLF001
        if (
            receipt.get("runtime_input_seal") != guards["runtime_input_seal"]
            or receipt.get("fixed_pretest_completion_attestation")
            != guards["completion_attestation"]
            or receipt.get("postlock_runtime_guard_attestation")
            != guards["postlock_guard"]
            or receipt.get("plan") != primary_guard.bind_file(plan_path)
            or primary_guard.canonical_sha256(receipt) != receipt.get("content_sha256")
        ):
            raise RadarRuntimeGuardError("radar initialization receipt is invalid")
        plan = _validated_plan(plan_path, root)

    units = plan["units"]
    prefix = _prefix_count(root, units)
    receipts: list[dict[str, Any]] = []
    for position in range(1, prefix + 1):
        pending_path, final_path = _receipt_paths(control, position)
        if final_path.is_file():
            receipts.append(
                _validate_receipt(
                    final_path,
                    position=position,
                    unit=units[position - 1],
                    output_root=root,
                    guards=guards,
                )
            )
        elif pending_path.is_file():
            pending = primary_guard._json(  # noqa: SLF001
                pending_path, f"pending radar unit {position}"
            )
            if (
                pending.get("classification")
                != "locked_hcs_radar_mask_runtime_guard_unit_pending"
                or pending.get("unit_id") != units[position - 1]["unit_id"]
                or pending.get("runtime_input_seal") != guards["runtime_input_seal"]
                or pending.get("fixed_pretest_completion_attestation")
                != guards["completion_attestation"]
                or pending.get("postlock_runtime_guard_attestation")
                != guards["postlock_guard"]
                or primary_guard.canonical_sha256(pending)
                != pending.get("content_sha256")
            ):
                raise RadarRuntimeGuardError(f"invalid pending radar unit: {position}")
            _target_absent(primary_root, root)
            after = verify_all_guards(
                runtime_input_seal=runtime_input_seal,
                completion_attestation=completion_attestation,
                pretest_index=pretest_index,
                postlock_guard=postlock_guard,
            )
            if after != guards:
                raise RadarRuntimeGuardError("runtime guards changed during radar recovery")
            final = {
                "schema_version": 1,
                "classification": "locked_hcs_radar_mask_runtime_guard_unit_receipt",
                "position": position,
                "unit_id": units[position - 1]["unit_id"],
                "runtime_input_seal": guards["runtime_input_seal"],
                "fixed_pretest_completion_attestation": guards[
                    "completion_attestation"
                ],
                "postlock_runtime_guard_attestation": guards["postlock_guard"],
                "outputs": _unit_bindings(root, units[position - 1]),
                "runtime_verified_before": True,
                "runtime_verified_after": True,
                "recovered_after_supervisor_interruption": True,
                "target_artifact_opened": False,
            }
            final["content_sha256"] = primary_guard.canonical_sha256(final)
            primary_guard.immutable_json(final_path, final)
            receipts.append(final)
        else:
            raise RadarRuntimeGuardError(
                f"completed radar unit lacks its runtime-guard receipt: {position}"
            )

    allowance = EXPECTED_UNITS if max_new_units is None else int(max_new_units)
    launched = 0
    while prefix < EXPECTED_UNITS and launched < allowance:
        position = prefix + 1
        unit = units[prefix]
        pending_path, receipt_path = _receipt_paths(control, position)
        _target_absent(primary_root, root)
        before = verify_all_guards(
            runtime_input_seal=runtime_input_seal,
            completion_attestation=completion_attestation,
            pretest_index=pretest_index,
            postlock_guard=postlock_guard,
        )
        if before != guards:
            raise RadarRuntimeGuardError("runtime guards changed before a radar unit")
        command = _command(
            python_executable=python_executable,
            underlying_source=underlying_source,
            primary_plan=primary_plan,
            primary_output_root=primary_root,
            output_root=root,
            safe_helper=safe_helper,
            locked_oof_source=locked_oof_source,
            batch_size=batch_size,
            dry_run=False,
        )
        pending = {
            "schema_version": 1,
            "classification": "locked_hcs_radar_mask_runtime_guard_unit_pending",
            "position": position,
            "unit_id": unit["unit_id"],
            "runtime_input_seal": guards["runtime_input_seal"],
            "fixed_pretest_completion_attestation": guards["completion_attestation"],
            "postlock_runtime_guard_attestation": guards["postlock_guard"],
            "max_new_units": 1,
            "command": command,
            "runtime_verified_before": True,
            "target_artifact_opened": False,
        }
        pending["content_sha256"] = primary_guard.canonical_sha256(pending)
        primary_guard.immutable_json(pending_path, pending)
        command_record = primary_guard._run_subprocess(command, cwd=PROJECT_ROOT)  # noqa: SLF001
        _target_absent(primary_root, root)
        after = verify_all_guards(
            runtime_input_seal=runtime_input_seal,
            completion_attestation=completion_attestation,
            pretest_index=pretest_index,
            postlock_guard=postlock_guard,
        )
        if after != guards:
            raise RadarRuntimeGuardError("runtime guards changed after a radar unit")
        new_prefix = _prefix_count(root, units)
        if new_prefix != position:
            raise RadarRuntimeGuardError(
                f"radar orchestrator did not make exactly one-unit progress: {prefix}->{new_prefix}"
            )
        final = {
            "schema_version": 1,
            "classification": "locked_hcs_radar_mask_runtime_guard_unit_receipt",
            "position": position,
            "unit_id": unit["unit_id"],
            "runtime_input_seal": guards["runtime_input_seal"],
            "fixed_pretest_completion_attestation": guards["completion_attestation"],
            "postlock_runtime_guard_attestation": guards["postlock_guard"],
            "outputs": _unit_bindings(root, unit),
            "command": command_record,
            "runtime_verified_before": True,
            "runtime_verified_after": True,
            "recovered_after_supervisor_interruption": False,
            "target_artifact_opened": False,
        }
        final["content_sha256"] = primary_guard.canonical_sha256(final)
        primary_guard.immutable_json(receipt_path, final)
        receipts.append(final)
        prefix = new_prefix
        launched += 1

    if prefix != EXPECTED_UNITS:
        return {
            "status": "locked_hcs_radar_mask_runtime_guard_incomplete",
            "completed_units": prefix,
            "new_units": launched,
            "expected_units": EXPECTED_UNITS,
            "target_artifact_opened": False,
            "gpu_execution_performed": False,
        }
    _target_absent(primary_root, root)
    final_guards = verify_all_guards(
        runtime_input_seal=runtime_input_seal,
        completion_attestation=completion_attestation,
        pretest_index=pretest_index,
        postlock_guard=postlock_guard,
    )
    if final_guards != guards:
        raise RadarRuntimeGuardError("runtime guards changed before radar complete seal")
    complete_seal = root / "complete_seal.json"
    seal = primary_guard._json(complete_seal, "radar complete seal")  # noqa: SLF001
    if (
        seal.get("classification")
        != "locked_hcs_all_seven_radar_mask_predictions_sealed"
        or int(seal.get("unit_count", -1)) != EXPECTED_UNITS
        or seal.get("complete_matrix") is not True
        or seal.get("target_or_label_artifact_opened_before_seal") is not False
    ):
        raise RadarRuntimeGuardError("radar complete seal is invalid")
    final = {
        "schema_version": 1,
        "classification": "locked_hcs_radar_mask_runtime_guard_attestation",
        "runtime_input_seal": guards["runtime_input_seal"],
        "fixed_pretest_completion_attestation": guards["completion_attestation"],
        "postlock_runtime_guard_attestation": guards["postlock_guard"],
        "radar_mask_plan": primary_guard.bind_file(plan_path),
        "complete_seal": primary_guard.bind_file(complete_seal),
        "completed_units": EXPECTED_UNITS,
        "unit_runtime_guard_receipts": [
            primary_guard.bind_file(_receipt_paths(control, position)[1])
            for position in range(1, EXPECTED_UNITS + 1)
        ],
        "runtime_seal_verified_before_and_after_every_unit": True,
        "postlock_guard_verified_before_and_after_every_unit": True,
        "bounded_forward_progress_one_unit_per_subprocess": True,
        "max_new_units_per_subprocess": 1,
        "target_artifact_opened": False,
        "gpu_execution_performed": False,
        "supervisor_source": source_binding,
        "underlying_source": executed_source,
        "python_executable": python_binding,
        "commercial_claim_authorized": False,
    }
    final["content_sha256"] = primary_guard.canonical_sha256(final)
    attestation = root / DEFAULT_ATTESTATION_NAME
    primary_guard.immutable_json(attestation, final)
    verified_attestation = verify_radar_guard_attestation(
        attestation,
        primary_output_root=primary_root,
        runtime_input_seal=runtime_input_seal,
        completion_attestation=completion_attestation,
        pretest_index=pretest_index,
        postlock_guard=postlock_guard,
    )
    if verified_attestation["document"] != final:
        raise RadarRuntimeGuardError("radar runtime guard attestation changed after write")
    return {
        "status": "locked_hcs_radar_mask_runtime_guard_complete",
        "completed_units": EXPECTED_UNITS,
        "new_units": launched,
        "attestation": primary_guard.bind_file(attestation),
        "complete_seal": primary_guard.bind_file(complete_seal),
        "target_artifact_opened": False,
        "gpu_execution_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-seal", type=Path, default=DEFAULT_RUNTIME_SEAL)
    parser.add_argument("--completion-attestation", type=Path, default=DEFAULT_COMPLETION)
    parser.add_argument("--pretest-index", type=Path, default=DEFAULT_PRETEST_INDEX)
    parser.add_argument("--postlock-guard", type=Path, default=DEFAULT_POSTLOCK_GUARD)
    parser.add_argument("--primary-output-root", type=Path, default=DEFAULT_PRIMARY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--underlying-source", type=Path, default=DEFAULT_UNDERLYING)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--safe-helper", type=Path, default=DEFAULT_SAFE_HELPER)
    parser.add_argument("--locked-oof-source", type=Path, default=DEFAULT_LOCKED_OOF_SOURCE)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-new-units", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_supervisor(
        runtime_input_seal=args.runtime_seal,
        completion_attestation=args.completion_attestation,
        pretest_index=args.pretest_index,
        postlock_guard=args.postlock_guard,
        primary_output_root=args.primary_output_root,
        output_root=args.output_root,
        underlying_source=args.underlying_source,
        python_executable=args.python_executable,
        safe_helper=args.safe_helper,
        locked_oof_source=args.locked_oof_source,
        batch_size=args.batch_size,
        max_new_units=args.max_new_units,
    )
    print(json.dumps(result, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
