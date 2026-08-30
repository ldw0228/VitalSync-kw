#!/usr/bin/env python3
"""Additive deterministic-CPU protocol wrapper for streaming campaign v4."""

from __future__ import annotations

from pathlib import Path

import verify_harmonic_set_deployment_v4 as verifier_v4
import run_locked_hcs_streaming_campaign as _base


_ORIGINAL_RUNTIME_SOURCES = _base._runtime_sources
_ORIGINAL_DEFAULT_SPEC = _base.default_freeze_spec


def _runtime_sources_v4():
    sources = _ORIGINAL_RUNTIME_SOURCES()
    sources["underlying_orchestrator"] = sources["orchestrator"]
    sources["orchestrator"] = _base.bind_file(Path(__file__))
    return sources


def _default_freeze_spec_v4():
    document = _ORIGINAL_DEFAULT_SPEC()
    document.pop("content_sha256", None)
    verification = document["verification"]
    verification["atol"] = 5.0e-6
    verification["cpu_intraop_threads"] = 1
    verification["cpu_interop_threads"] = 1
    document["protocol_revision"] = "v4_deterministic_single_thread_cpu_parity"
    return _base._content_document(document)


_base.VERIFIER = verifier_v4
_base.torch = verifier_v4.torch
_base._runtime_sources = _runtime_sources_v4
_base.default_freeze_spec = _default_freeze_spec_v4


if __name__ == "__main__":
    raise SystemExit(_base.main())
