#!/usr/bin/env python3
"""Deterministic-CPU launcher for the locked HCS deployment verifier.

The underlying verifier is intentionally unchanged.  Fixed single-thread CPU
execution prevents BLAS row-batch kernel selection from perturbing graph
features across chronological chunk sizes, which can otherwise flip spiking
states and discrete candidate routing despite identical inputs.
"""

from __future__ import annotations

import torch

import verify_harmonic_set_deployment as _base  # noqa: E402


def configure_deterministic_cpu_runtime() -> None:
    """Pin the CLI process without mutating a process that merely imports us."""

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("failed to enforce the locked single-thread CPU runtime")


def __getattr__(name: str):
    return getattr(_base, name)


def main() -> int:
    configure_deterministic_cpu_runtime()
    return int(_base.main())


if __name__ == "__main__":
    raise SystemExit(main())
