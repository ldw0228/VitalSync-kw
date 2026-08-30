from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/benchmark_locked_proposer_deployment_v3.py"
)
SPEC = importlib.util.spec_from_file_location(
    "benchmark_locked_proposer_deployment_v3", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def test_cuda_context_is_initialized_before_peak_reset(monkeypatch) -> None:
    calls: list[tuple[str, object | None]] = []

    class FakeCuda:
        @staticmethod
        def init() -> None:
            calls.append(("init", None))

        @staticmethod
        def set_device(device: object) -> None:
            calls.append(("set_device", device))

        @staticmethod
        def empty_cache() -> None:
            calls.append(("empty_cache", None))

        @staticmethod
        def reset_peak_memory_stats(device: object) -> None:
            calls.append(("reset_peak_memory_stats", device))

    fake_torch = SimpleNamespace(cuda=FakeCuda())
    monkeypatch.setattr(RUN, "torch", fake_torch)
    device = object()
    RUN._initialize_cuda_measurement(device)
    assert calls == [
        ("init", None),
        ("set_device", device),
        ("empty_cache", None),
        ("reset_peak_memory_stats", device),
    ]
