#!/usr/bin/env python3
"""Deterministic-CPU launcher for the locked HCS deployment verifier.

The underlying verifier is intentionally unchanged.  Fixed single-thread CPU
execution prevents BLAS row-batch kernel selection from perturbing graph
features across chronological chunk sizes, which can otherwise flip spiking
states and discrete candidate routing despite identical inputs.
"""

from __future__ import annotations

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import verify_harmonic_set_deployment as _base  # noqa: E402


def __getattr__(name: str):
    return getattr(_base, name)


if __name__ == "__main__":
    raise SystemExit(_base.main())
