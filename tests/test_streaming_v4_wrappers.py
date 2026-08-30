from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_locked_hcs_streaming_campaign_v4.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("streaming_v4", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v4_spec_binds_additive_sources_and_deterministic_cpu_policy() -> None:
    document = MODULE._base.default_freeze_spec()
    assert document["protocol_revision"] == "v4_deterministic_single_thread_cpu_parity"
    assert document["verification"]["atol"] == 5.0e-6
    assert document["verification"]["rtol"] == 2.0e-6
    assert document["verification"]["cpu_intraop_threads"] == 1
    assert document["verification"]["cpu_interop_threads"] == 1
    sources = document["runtime_sources"]
    assert Path(sources["orchestrator"]["path"]).name == SCRIPT.name
    assert Path(sources["unit_verifier"]["path"]).name == (
        "verify_harmonic_set_deployment_v4.py"
    )
    assert Path(sources["underlying_orchestrator"]["path"]).name == (
        "run_locked_hcs_streaming_campaign.py"
    )
