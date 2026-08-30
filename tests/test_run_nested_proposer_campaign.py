from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_nested_proposer_campaign.py"
SPEC = importlib.util.spec_from_file_location("run_nested_proposer_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def test_discovery_manifest_router_never_returns_outer_test(tmp_path: Path) -> None:
    for outer in (3, 4):
        directory = tmp_path / f"outer_{outer}"
        directory.mkdir()
        for prediction in range(4):
            (directory / f"inner_pred_{prediction}.json").write_text(
                json.dumps({"fold_id": 100 * outer + prediction}), encoding="utf-8"
            )
        (directory / "validation_pred_5.json").write_text(
            json.dumps({"fold_id": 100 * outer + 50}), encoding="utf-8"
        )
        (directory / f"test_pred_{outer}.json").write_text(
            json.dumps({"fold_id": 100 * outer + 60}), encoding="utf-8"
        )
    selected = RUN.discovery_manifests(tmp_path, [3, 4])
    assert len(selected) == 10
    assert all(not path.name.startswith("test_pred_") for path in selected)


def test_discovery_router_fails_on_incomplete_topology(tmp_path: Path) -> None:
    directory = tmp_path / "outer_3"
    directory.mkdir()
    (directory / "inner_pred_0.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="four inner"):
        RUN.discovery_manifests(tmp_path, [3])

