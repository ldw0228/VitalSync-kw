#!/usr/bin/env python3
"""Record the fresh-process CUDA peak-stat initialization compatibility check."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, Mapping, Sequence

import torch


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def content_document(value: Mapping[str, Any]) -> dict[str, Any]:
    document = dict(value)
    document.pop("content_sha256", None)
    document["content_sha256"] = canonical_sha256(document)
    return document


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise RuntimeError(f"diagnostic output already exists: {destination}") from exc
        destination.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)


def diagnose(device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    available = bool(torch.cuda.is_available())
    if not available:
        raise RuntimeError("CUDA is unavailable")
    baseline: dict[str, Any]
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception as exc:  # the exception class/message are the evidence
        baseline = {
            "success": False,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }
    else:
        baseline = {"success": True, "exception_type": None, "exception_message": None}

    fixed: dict[str, Any]
    try:
        torch.cuda.init()
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    except Exception as exc:
        fixed = {
            "success": False,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }
    else:
        fixed = {"success": True, "exception_type": None, "exception_message": None}

    return content_document(
        {
            "schema_version": 1,
            "classification": "target_free_cuda_peak_stats_initialization_diagnostic",
            "requested_device": str(device),
            "cuda_available": available,
            "device_count": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "python": platform.python_version(),
            "baseline_reset_after_availability_check": baseline,
            "reset_after_explicit_init_and_set_device": fixed,
            "target_or_reference_artifact_opened": False,
            "target_or_reference_value_consulted": False,
            "commercial_claim_authorized": False,
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = diagnose(args.device)
    atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True, ensure_ascii=False), flush=True)
    return 0 if report["reset_after_explicit_init_and_set_device"]["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
