#!/usr/bin/env python3
"""Run the locked 18-unit HCS OOF inference under a byte-closed guard.

The canonical runner interprets ``--max-units`` as a prefix limit.  This
supervisor therefore invokes it with exactly ``completed_prefix + 1`` and
requires one-unit forward progress.  The fixed-i3 runtime seal and completion
attestation are revalidated immediately before and after prepare, every unit,
and the final sealing call.  No target or evaluation argument is exposed.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import run_locked_hcs_oof as locked_oof  # noqa: E402
import seal_fixed_i3_pretest_completion as completion  # noqa: E402
import seal_runtime_inputs as runtime_seal  # noqa: E402


CAMPAIGN_ROOT = PROJECT_ROOT / "artifacts/campaigns/harmonic_candidate_set_snn_v2"
RUN_ROOT = PROJECT_ROOT / "artifacts/runs/harmonic_candidate_set_snn_v2"
DEFAULT_RUNTIME_SEAL = (
    CAMPAIGN_ROOT
    / "nested_proposer/current_source_merged/fixed_i3_pretest_runtime_seal.json"
)
DEFAULT_PRETEST_ROOT = RUN_ROOT / "hcs_fixed_i3_pretest"
DEFAULT_PRETEST_INDEX = DEFAULT_PRETEST_ROOT / "pretest_index.json"
DEFAULT_COMPLETION = DEFAULT_PRETEST_ROOT / "fixed_runtime_completion_attestation.json"
DEFAULT_OUTPUT_ROOT = RUN_ROOT / "hcs_locked_oof"
DEFAULT_TEST_MANIFEST_ROOT = (
    CAMPAIGN_ROOT / "nested_proposer/full_oof_test/manifests"
)
DEFAULT_RF_CACHE = PROJECT_ROOT / "artifacts/cache/rf32s"
DEFAULT_UNDERLYING = SCRIPT_ROOT / "run_locked_hcs_oof.py"
DEFAULT_ATTESTATION_NAME = "postlock_runtime_guard_attestation.json"
EXPECTED_UNITS = 18
FORBIDDEN_NAMES = completion.FORBIDDEN_NAMES


class RuntimeGuardError(RuntimeError):
    """A runtime-closure, progress, or target-free invariant failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    document = dict(value)
    document.pop("content_sha256", None)
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bind_file(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeGuardError(f"required file is absent: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _json(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeGuardError(f"invalid {label}: {resolved} ({exc})") from exc
    if not isinstance(value, dict):
        raise RuntimeGuardError(f"{label} must be an object: {resolved}")
    return value


def _payload(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    target = path.expanduser().resolve()
    payload = _payload(value)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != payload:
            raise RuntimeGuardError(f"immutable runtime-guard collision: {target}")
        target.chmod(0o444)
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _assert_target_absent(output_root: Path) -> None:
    root = output_root.expanduser().resolve()
    found = [root / name for name in FORBIDDEN_NAMES if (root / name).exists()]
    if found:
        raise RuntimeGuardError(
            f"target/canonical-receipt/evaluation artifact must be absent: {found[0]}"
        )


def verify_closure(
    *, runtime_input_seal: Path, completion_attestation: Path, pretest_index: Path
) -> dict[str, Any]:
    runtime_path = runtime_input_seal.expanduser().resolve()
    try:
        runtime_result = runtime_seal.verify(runtime_path)
        completed = completion.verify_completion_attestation(
            completion_attestation,
            expected_runtime_seal=runtime_path,
            expected_pretest_index=pretest_index.expanduser().resolve(),
            reverify_payload=True,
        )
    except (RuntimeError, completion.FixedCompletionError) as exc:
        raise RuntimeGuardError(str(exc)) from exc
    if (
        completed["document"].get("runtime_content_sha256")
        != runtime_result.get("content_sha256")
    ):
        raise RuntimeGuardError("completion and runtime seal identities differ")
    return {
        "runtime_input_seal": bind_file(runtime_path),
        "runtime_content_sha256": runtime_result["content_sha256"],
        "completion_attestation": completed["binding"],
        "completion_content_sha256": completed["document"]["content_sha256"],
        "pretest_index": bind_file(pretest_index),
    }


def verify_guard_attestation(
    path: Path,
    *,
    expected_runtime_seal: Path | None = None,
    expected_completion: Path | None = None,
    reverify_closure: bool = True,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    document = _json(resolved, "locked-HCS postlock runtime guard")
    if (
        document.get("schema_version") != 1
        or document.get("classification")
        != "locked_hcs_oof_runtime_guard_attestation"
        or int(document.get("completed_units", -1)) != EXPECTED_UNITS
        or document.get("runtime_seal_verified_before_and_after_every_unit") is not True
        or document.get("target_artifact_opened") is not False
        or document.get("gpu_execution_performed") is not False
        or canonical_sha256(document) != document.get("content_sha256")
    ):
        raise RuntimeGuardError("locked-HCS postlock runtime guard is invalid")
    runtime_binding = completion.verify_binding(
        document.get("runtime_input_seal"),
        relative_to=resolved.parent,
        label="guard runtime input seal",
    )
    completion_binding = completion.verify_binding(
        document.get("fixed_pretest_completion_attestation"),
        relative_to=resolved.parent,
        label="guard fixed completion",
    )
    pretest_index_binding = completion.verify_binding(
        document.get("pretest_index"),
        relative_to=resolved.parent,
        label="guard pretest index",
    )
    for name in (
        "locked_oof_plan",
        "pretest_lock",
        "predictions_seal",
        "supervisor_source",
        "underlying_source",
        "python_executable",
    ):
        completion.verify_binding(
            document.get(name), relative_to=resolved.parent, label=f"guard {name}"
        )
    receipts = document.get("unit_runtime_guard_receipts")
    if not isinstance(receipts, list) or len(receipts) != EXPECTED_UNITS:
        raise RuntimeGuardError("postlock guard does not bind 18 unit receipts")
    for position, raw in enumerate(receipts, start=1):
        bound = completion.verify_binding(
            raw, relative_to=resolved.parent, label=f"guard unit receipt {position}"
        )
        receipt = _json(Path(bound["path"]), f"guard unit receipt {position}")
        if (
            receipt.get("classification")
            != "locked_hcs_oof_runtime_guard_unit_receipt"
            or int(receipt.get("position", -1)) != position
            or receipt.get("runtime_verified_before") is not True
            or receipt.get("runtime_verified_after") is not True
            or receipt.get("target_artifact_opened") is not False
            or canonical_sha256(receipt) != receipt.get("content_sha256")
        ):
            raise RuntimeGuardError(f"postlock unit receipt is invalid: {position}")
    if expected_runtime_seal is not None and Path(runtime_binding["path"]) != expected_runtime_seal.resolve():
        raise RuntimeGuardError("postlock guard binds another runtime seal")
    if expected_completion is not None and Path(completion_binding["path"]) != expected_completion.resolve():
        raise RuntimeGuardError("postlock guard binds another fixed completion")
    if reverify_closure:
        verify_closure(
            runtime_input_seal=Path(runtime_binding["path"]),
            completion_attestation=Path(completion_binding["path"]),
            pretest_index=Path(pretest_index_binding["path"]),
        )
        root = resolved.parent
        _assert_target_absent(root)
        try:
            locked_oof._verify_predictions_seal(root)  # noqa: SLF001
        except Exception as exc:
            raise RuntimeGuardError(f"guard predictions seal failed live validation: {exc}") from exc
    return {"document": document, "binding": bind_file(resolved)}


def _run_subprocess(argv: Sequence[str], *, cwd: Path) -> dict[str, Any]:
    if not argv or any(not isinstance(token, str) or not token for token in argv):
        raise RuntimeGuardError("subprocess argv must be a non-empty string array")
    completed = subprocess.run(
        list(argv),
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    record = {
        "argv": list(argv),
        "cwd": str(cwd.resolve()),
        "returncode": int(completed.returncode),
        "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
    }
    if completed.returncode != 0:
        tail = completed.stderr[-2000:] or completed.stdout[-2000:]
        raise RuntimeGuardError(
            f"guarded subprocess failed ({completed.returncode}): {tail}"
        )
    return record


def _validated_plan(plan_path: Path, output_root: Path) -> dict[str, Any]:
    try:
        plan, resolved = locked_oof.load_plan(plan_path)
    except Exception as exc:
        raise RuntimeGuardError(f"locked OOF plan validation failed: {exc}") from exc
    for unit in plan["units"]:
        stages = unit.get("stages")
        if unit.get("no_action_fast_path") is not True or not isinstance(stages, list):
            raise RuntimeGuardError("postlock plan is not the frozen no-action fast path")
        if [stage.get("name") for stage in stages] != list(locked_oof.FAST_NO_ACTION_STAGES):
            raise RuntimeGuardError("postlock plan contains a non-fast-path stage")
        for stage in stages:
            argv = [str(token) for token in stage.get("argv", [])]
            lowered = " ".join(argv).lower()
            if "cuda" in lowered or "run_gpu_admitted" in lowered or "--amp" in argv:
                raise RuntimeGuardError("postlock plan attempts GPU execution")
            if "--device" in argv and argv[argv.index("--device") + 1] != "cpu":
                raise RuntimeGuardError("postlock proposer prediction is not CPU-only")
    if resolved != plan_path.resolve():
        raise RuntimeGuardError("locked OOF plan resolved unexpectedly")
    if Path(str(plan.get("pretest_index", {}).get("path", ""))).resolve() == Path():
        raise RuntimeGuardError("locked OOF plan lacks pretest-index binding")
    return plan


def _ordered_units(plan: Mapping[str, Any]) -> list[tuple[int, int]]:
    return sorted(
        [(int(unit["outer_fold"]), int(unit["seed"])) for unit in plan["units"]],
        key=lambda item: (item[1], item[0]),
    )


def _unit_paths(output_root: Path, key: tuple[int, int]) -> tuple[Path, Path]:
    fold, seed = key
    unit_root = output_root / "units" / f"outer_{fold}_seed_{seed}"
    return unit_root / "derived_inference_lock.json", unit_root / "sealed_label_free_predictions.npz"


def _prefix_count(output_root: Path, order: Sequence[tuple[int, int]]) -> int:
    flags: list[bool] = []
    for key in order:
        lock, prediction = _unit_paths(output_root, key)
        if lock.exists() != prediction.exists():
            raise RuntimeGuardError(f"partial locked-OOF unit payload: {key}")
        flags.append(lock.is_file() and prediction.is_file())
    prefix = 0
    while prefix < len(flags) and flags[prefix]:
        prefix += 1
    if any(flags[prefix:]):
        raise RuntimeGuardError("locked-OOF completed units are not a contiguous prefix")
    return prefix


def _receipt_paths(control: Path, position: int) -> tuple[Path, Path]:
    root = control / "runtime_guard_units"
    return root / f"unit_{position:03d}.pending.json", root / f"unit_{position:03d}.json"


def _validate_unit_receipt(
    receipt_path: Path,
    *,
    position: int,
    key: tuple[int, int],
    output_root: Path,
    closure: Mapping[str, Any],
) -> dict[str, Any]:
    document = _json(receipt_path, f"runtime guard unit {position}")
    lock_path, prediction_path = _unit_paths(output_root, key)
    if (
        document.get("classification") != "locked_hcs_oof_runtime_guard_unit_receipt"
        or int(document.get("position", -1)) != position
        or [int(document.get("outer_fold", -1)), int(document.get("seed", -1))]
        != [key[0], key[1]]
        or document.get("runtime_verified_before") is not True
        or document.get("runtime_verified_after") is not True
        or document.get("target_artifact_opened") is not False
        or document.get("runtime_input_seal") != closure["runtime_input_seal"]
        or document.get("fixed_pretest_completion_attestation")
        != closure["completion_attestation"]
        or document.get("derived_lock") != bind_file(lock_path)
        or document.get("prediction") != bind_file(prediction_path)
        or canonical_sha256(document) != document.get("content_sha256")
    ):
        raise RuntimeGuardError(f"runtime guard unit receipt is invalid: {receipt_path}")
    return document


def _ensure_prior_receipts(
    *,
    control: Path,
    prefix: int,
    order: Sequence[tuple[int, int]],
    output_root: Path,
    closure: Mapping[str, Any],
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for position in range(1, prefix + 1):
        pending, final = _receipt_paths(control, position)
        if final.is_file():
            receipts.append(
                _validate_unit_receipt(
                    final,
                    position=position,
                    key=order[position - 1],
                    output_root=output_root,
                    closure=closure,
                )
            )
        elif pending.is_file():
            # A durable pending receipt proves that closure verification
            # completed before the canonical unit command.  Recovery performs
            # the after-verification now and closes it below.
            pending_document = _json(pending, f"pending runtime guard unit {position}")
            if (
                pending_document.get("classification")
                != "locked_hcs_oof_runtime_guard_unit_pending"
                or int(pending_document.get("position", -1)) != position
                or pending_document.get("runtime_input_seal")
                != closure["runtime_input_seal"]
                or pending_document.get("fixed_pretest_completion_attestation")
                != closure["completion_attestation"]
                or canonical_sha256(pending_document)
                != pending_document.get("content_sha256")
            ):
                raise RuntimeGuardError(f"invalid pending unit receipt: {pending}")
            _assert_target_absent(output_root)
            post = verify_closure(
                runtime_input_seal=Path(closure["runtime_input_seal"]["path"]),
                completion_attestation=Path(closure["completion_attestation"]["path"]),
                pretest_index=Path(closure["pretest_index"]["path"]),
            )
            if post != closure:
                raise RuntimeGuardError("runtime closure changed during pending-unit recovery")
            lock_path, prediction_path = _unit_paths(output_root, order[position - 1])
            final_document = {
                "schema_version": 1,
                "classification": "locked_hcs_oof_runtime_guard_unit_receipt",
                "position": position,
                "outer_fold": order[position - 1][0],
                "seed": order[position - 1][1],
                "runtime_input_seal": closure["runtime_input_seal"],
                "fixed_pretest_completion_attestation": closure["completion_attestation"],
                "derived_lock": bind_file(lock_path),
                "prediction": bind_file(prediction_path),
                "runtime_verified_before": True,
                "runtime_verified_after": True,
                "recovered_after_supervisor_interruption": True,
                "target_artifact_opened": False,
            }
            final_document["content_sha256"] = canonical_sha256(final_document)
            immutable_json(final, final_document)
            receipts.append(final_document)
        else:
            raise RuntimeGuardError(
                f"completed locked-OOF unit lacks its runtime-guard receipt: {position}"
            )
    return receipts


def _prepare_command(
    *,
    python_executable: Path,
    underlying_source: Path,
    pretest_index: Path,
    test_manifest_root: Path,
    output_root: Path,
    plan_path: Path,
    rf_cache: Path,
    proposer_trainer: Path,
    safe_helper: Path,
    gpu_wrapper: Path,
) -> list[str]:
    return [
        str(python_executable),
        str(underlying_source.resolve()),
        "prepare",
        "--pretest-index",
        str(pretest_index.resolve()),
        "--test-manifest-root",
        str(test_manifest_root.resolve()),
        "--output-root",
        str(output_root.resolve()),
        "--plan-output",
        str(plan_path.resolve()),
        "--rf-cache",
        str(rf_cache.resolve()),
        "--python-executable",
        str(python_executable),
        "--proposer-trainer",
        str(proposer_trainer.resolve()),
        "--safe-helper",
        str(safe_helper.resolve()),
        "--gpu-wrapper",
        str(gpu_wrapper.resolve()),
        "--train-device",
        "cpu",
        "--prediction-device",
        "cpu",
    ]


def run_supervisor(
    *,
    runtime_input_seal: Path,
    completion_attestation: Path,
    pretest_index: Path,
    test_manifest_root: Path,
    output_root: Path,
    underlying_source: Path = DEFAULT_UNDERLYING,
    python_executable: Path = Path(sys.executable),
    rf_cache: Path = DEFAULT_RF_CACHE,
    proposer_trainer: Path = SCRIPT_ROOT / "train.py",
    safe_helper: Path = SCRIPT_ROOT / "build_locked_hcs_test_inputs.py",
    gpu_wrapper: Path = SCRIPT_ROOT / "run_gpu_admitted.py",
    max_new_units: int | None = None,
) -> dict[str, Any]:
    if max_new_units is not None and int(max_new_units) < 0:
        raise RuntimeGuardError("max_new_units cannot be negative")
    # Do not resolve a virtual-environment launcher symlink: resolving it can
    # select the base interpreter and silently lose the venv site-packages.
    python_executable = Path(os.path.abspath(python_executable.expanduser()))
    if not python_executable.is_file() or not os.access(python_executable, os.X_OK):
        raise RuntimeGuardError("python_executable must be an executable file")
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
        raise RuntimeGuardError(
            "another locked-OOF runtime supervisor holds the serial campaign lock"
        ) from exc
    _assert_target_absent(root)
    closure = verify_closure(
        runtime_input_seal=runtime_input_seal,
        completion_attestation=completion_attestation,
        pretest_index=pretest_index,
    )
    source_binding = bind_file(underlying_source)
    imported_binding = bind_file(Path(locked_oof.__file__))
    if source_binding["sha256"] != imported_binding["sha256"]:
        raise RuntimeGuardError("executed locked-OOF source differs from imported contract")
    python_binding = bind_file(python_executable)
    supervisor_binding = bind_file(Path(__file__))
    plan_path = root / "locked_oof_plan.json"
    prepare_pending = control / "prepare.pending.json"
    prepare_receipt = control / "prepare.json"
    if not prepare_receipt.is_file():
        if plan_path.exists() and not prepare_pending.exists():
            raise RuntimeGuardError(
                "locked OOF plan predates this runtime guard and cannot be retroactively attested"
            )
        prepare_command = _prepare_command(
            python_executable=python_executable,
            underlying_source=underlying_source,
            pretest_index=pretest_index,
            test_manifest_root=test_manifest_root,
            output_root=root,
            plan_path=plan_path,
            rf_cache=rf_cache,
            proposer_trainer=proposer_trainer,
            safe_helper=safe_helper,
            gpu_wrapper=gpu_wrapper,
        )
        if not prepare_pending.exists():
            pending = {
                "schema_version": 1,
                "classification": "locked_hcs_oof_runtime_guard_prepare_pending",
                "runtime_input_seal": closure["runtime_input_seal"],
                "fixed_pretest_completion_attestation": closure["completion_attestation"],
                "command": prepare_command,
                "runtime_verified_before": True,
                "target_artifact_opened": False,
            }
            pending["content_sha256"] = canonical_sha256(pending)
            immutable_json(prepare_pending, pending)
        if not plan_path.is_file():
            command_record = _run_subprocess(prepare_command, cwd=PROJECT_ROOT)
        else:
            command_record = {"argv": prepare_command, "recovered_after_interruption": True}
        _assert_target_absent(root)
        after = verify_closure(
            runtime_input_seal=runtime_input_seal,
            completion_attestation=completion_attestation,
            pretest_index=pretest_index,
        )
        if after != closure:
            raise RuntimeGuardError("runtime closure changed during locked-OOF prepare")
        plan = _validated_plan(plan_path, root)
        recorded_index = completion.verify_binding(
            plan.get("pretest_index"), relative_to=plan_path.parent, label="plan pretest index"
        )
        if recorded_index != closure["pretest_index"]:
            raise RuntimeGuardError("locked OOF plan binds another pretest index")
        receipt = {
            "schema_version": 1,
            "classification": "locked_hcs_oof_runtime_guard_prepare_receipt",
            "runtime_input_seal": closure["runtime_input_seal"],
            "fixed_pretest_completion_attestation": closure["completion_attestation"],
            "plan": bind_file(plan_path),
            "command": command_record,
            "runtime_verified_before": True,
            "runtime_verified_after": True,
            "target_artifact_opened": False,
        }
        receipt["content_sha256"] = canonical_sha256(receipt)
        immutable_json(prepare_receipt, receipt)
    else:
        receipt = _json(prepare_receipt, "locked-OOF prepare guard receipt")
        if (
            receipt.get("runtime_input_seal") != closure["runtime_input_seal"]
            or receipt.get("fixed_pretest_completion_attestation")
            != closure["completion_attestation"]
            or receipt.get("plan") != bind_file(plan_path)
            or canonical_sha256(receipt) != receipt.get("content_sha256")
        ):
            raise RuntimeGuardError("locked-OOF prepare guard receipt is invalid")
        plan = _validated_plan(plan_path, root)

    order = _ordered_units(plan)
    if len(order) != EXPECTED_UNITS:
        raise RuntimeGuardError("locked OOF plan does not contain 18 units")
    prefix = _prefix_count(root, order)
    receipts = _ensure_prior_receipts(
        control=control,
        prefix=prefix,
        order=order,
        output_root=root,
        closure=closure,
    )
    allowance = EXPECTED_UNITS if max_new_units is None else int(max_new_units)
    launched = 0
    while prefix < EXPECTED_UNITS and launched < allowance:
        position = prefix + 1
        key = order[prefix]
        pending_path, receipt_path = _receipt_paths(control, position)
        _assert_target_absent(root)
        before = verify_closure(
            runtime_input_seal=runtime_input_seal,
            completion_attestation=completion_attestation,
            pretest_index=pretest_index,
        )
        if before != closure:
            raise RuntimeGuardError("runtime closure changed before a locked-OOF unit")
        infer_command = [
            str(python_executable),
            str(underlying_source.resolve()),
            "infer",
            "--plan",
            str(plan_path.resolve()),
            "--output-root",
            str(root),
            "--max-units",
            str(position),
        ]
        pending = {
            "schema_version": 1,
            "classification": "locked_hcs_oof_runtime_guard_unit_pending",
            "position": position,
            "outer_fold": key[0],
            "seed": key[1],
            "runtime_input_seal": closure["runtime_input_seal"],
            "fixed_pretest_completion_attestation": closure["completion_attestation"],
            "canonical_prefix_limit": position,
            "command": infer_command,
            "runtime_verified_before": True,
            "target_artifact_opened": False,
        }
        pending["content_sha256"] = canonical_sha256(pending)
        immutable_json(pending_path, pending)
        command_record = _run_subprocess(infer_command, cwd=PROJECT_ROOT)
        _assert_target_absent(root)
        after = verify_closure(
            runtime_input_seal=runtime_input_seal,
            completion_attestation=completion_attestation,
            pretest_index=pretest_index,
        )
        if after != closure:
            raise RuntimeGuardError("runtime closure changed after a locked-OOF unit")
        new_prefix = _prefix_count(root, order)
        if new_prefix != position:
            raise RuntimeGuardError(
                f"canonical infer did not make exactly one-unit progress: {prefix}->{new_prefix}"
            )
        lock_path, prediction_path = _unit_paths(root, key)
        receipt_document = {
            "schema_version": 1,
            "classification": "locked_hcs_oof_runtime_guard_unit_receipt",
            "position": position,
            "outer_fold": key[0],
            "seed": key[1],
            "runtime_input_seal": closure["runtime_input_seal"],
            "fixed_pretest_completion_attestation": closure["completion_attestation"],
            "derived_lock": bind_file(lock_path),
            "prediction": bind_file(prediction_path),
            "command": command_record,
            "runtime_verified_before": True,
            "runtime_verified_after": True,
            "recovered_after_supervisor_interruption": False,
            "target_artifact_opened": False,
        }
        receipt_document["content_sha256"] = canonical_sha256(receipt_document)
        immutable_json(receipt_path, receipt_document)
        receipts.append(receipt_document)
        prefix = new_prefix
        launched += 1

    if prefix != EXPECTED_UNITS:
        return {
            "status": "locked_hcs_oof_runtime_guard_incomplete",
            "completed_units": prefix,
            "new_units": launched,
            "expected_units": EXPECTED_UNITS,
            "target_artifact_opened": False,
            "gpu_execution_performed": False,
        }
    _assert_target_absent(root)
    final_closure = verify_closure(
        runtime_input_seal=runtime_input_seal,
        completion_attestation=completion_attestation,
        pretest_index=pretest_index,
    )
    if final_closure != closure:
        raise RuntimeGuardError("runtime closure changed before final predictions seal")
    try:
        locked_oof._verify_predictions_seal(root)  # noqa: SLF001
    except Exception as exc:
        raise RuntimeGuardError(f"final locked-OOF predictions seal is invalid: {exc}") from exc
    pretest_lock = root / "pretest_lock.json"
    predictions_seal = root / "predictions_seal.json"
    final = {
        "schema_version": 1,
        "classification": "locked_hcs_oof_runtime_guard_attestation",
        "runtime_input_seal": closure["runtime_input_seal"],
        "fixed_pretest_completion_attestation": closure["completion_attestation"],
        "pretest_index": closure["pretest_index"],
        "locked_oof_plan": bind_file(plan_path),
        "pretest_lock": bind_file(pretest_lock),
        "predictions_seal": bind_file(predictions_seal),
        "completed_units": EXPECTED_UNITS,
        "unit_runtime_guard_receipts": [
            bind_file(_receipt_paths(control, position)[1])
            for position in range(1, EXPECTED_UNITS + 1)
        ],
        "runtime_seal_verified_before_and_after_every_unit": True,
        "runtime_seal_verified_immediately_before_final_predictions_seal": True,
        "target_artifact_opened": False,
        "gpu_execution_performed": False,
        "bounded_forward_progress_one_unit_per_subprocess": True,
        "canonical_infer_prefix_limit_used": True,
        "supervisor_source": supervisor_binding,
        "underlying_source": source_binding,
        "python_executable": python_binding,
        "commercial_claim_authorized": False,
    }
    final["content_sha256"] = canonical_sha256(final)
    attestation_path = root / DEFAULT_ATTESTATION_NAME
    immutable_json(attestation_path, final)
    verify_guard_attestation(
        attestation_path,
        expected_runtime_seal=runtime_input_seal,
        expected_completion=completion_attestation,
        reverify_closure=True,
    )
    return {
        "status": "locked_hcs_oof_runtime_guard_complete",
        "completed_units": EXPECTED_UNITS,
        "new_units": launched,
        "attestation": bind_file(attestation_path),
        "predictions_seal": bind_file(predictions_seal),
        "target_artifact_opened": False,
        "gpu_execution_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-seal", type=Path, default=DEFAULT_RUNTIME_SEAL)
    parser.add_argument("--completion-attestation", type=Path, default=DEFAULT_COMPLETION)
    parser.add_argument("--pretest-index", type=Path, default=DEFAULT_PRETEST_INDEX)
    parser.add_argument("--test-manifest-root", type=Path, default=DEFAULT_TEST_MANIFEST_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--underlying-source", type=Path, default=DEFAULT_UNDERLYING)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--rf-cache", type=Path, default=DEFAULT_RF_CACHE)
    parser.add_argument("--proposer-trainer", type=Path, default=SCRIPT_ROOT / "train.py")
    parser.add_argument(
        "--safe-helper", type=Path, default=SCRIPT_ROOT / "build_locked_hcs_test_inputs.py"
    )
    parser.add_argument(
        "--gpu-wrapper", type=Path, default=SCRIPT_ROOT / "run_gpu_admitted.py"
    )
    parser.add_argument("--max-new-units", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_supervisor(
        runtime_input_seal=args.runtime_seal,
        completion_attestation=args.completion_attestation,
        pretest_index=args.pretest_index,
        test_manifest_root=args.test_manifest_root,
        output_root=args.output_root,
        underlying_source=args.underlying_source,
        python_executable=args.python_executable,
        rf_cache=args.rf_cache,
        proposer_trainer=args.proposer_trainer,
        safe_helper=args.safe_helper,
        gpu_wrapper=args.gpu_wrapper,
        max_new_units=args.max_new_units,
    )
    print(json.dumps(result, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
