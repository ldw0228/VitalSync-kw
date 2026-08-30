from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_locked_hcs_test_inputs.py"
SPEC = importlib.util.spec_from_file_location("build_locked_hcs_test_inputs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


def test_all_seven_nonempty_radar_masks_are_predeclared() -> None:
    assert set(HELPER.RADAR_MASKS) == {
        "radars_123",
        "radars_12",
        "radars_13",
        "radars_23",
        "radar_1",
        "radar_2",
        "radar_3",
    }
    assert len(set(HELPER.RADAR_MASKS.values())) == 7
    assert all(any(pattern) for pattern in HELPER.RADAR_MASKS.values())


def test_masked_loader_intersects_physical_availability_without_touching_rows() -> None:
    original = {
        "index": torch.tensor([7, 8]),
        "radar_mask": torch.tensor(
            [[True, True, False], [True, False, True]], dtype=torch.bool
        ),
        "map": torch.ones(2, 3, 2, 2),
    }
    batches = list(
        HELPER._masked_proposer_loader([original], HELPER.RADAR_MASKS["radars_13"])
    )
    assert len(batches) == 1
    assert torch.equal(batches[0]["index"], original["index"])
    assert torch.equal(
        batches[0]["radar_mask"],
        torch.tensor([[True, False, False], [True, False, True]]),
    )
    assert torch.equal(
        original["radar_mask"],
        torch.tensor([[True, True, False], [True, False, True]]),
    )


def test_masked_loader_rejects_wrong_radar_topology() -> None:
    with pytest.raises(RuntimeError, match="topology"):
        list(
            HELPER._masked_proposer_loader(
                [{"radar_mask": torch.ones(2, 2, dtype=torch.bool)}],
                HELPER.RADAR_MASKS["radars_123"],
            )
        )


def test_cli_defaults_to_full_mask_and_rejects_unknown_mask() -> None:
    parser = HELPER.build_parser()
    args = parser.parse_args(
        [
            "proposer-predict",
            "--cache-dir",
            "cache",
            "--checkpoint",
            "checkpoint.pt",
            "--test-manifest",
            "manifest.json",
            "--output",
            "prediction.npz",
        ]
    )
    assert args.radar_mask == "radars_123"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "proposer-predict",
                "--cache-dir",
                "cache",
                "--checkpoint",
                "checkpoint.pt",
                "--test-manifest",
                "manifest.json",
                "--output",
                "prediction.npz",
                "--radar-mask",
                "none",
            ]
        )
