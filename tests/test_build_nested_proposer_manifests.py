from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_nested_proposer_manifests.py"
SPEC = importlib.util.spec_from_file_location("build_nested_proposer_manifests", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_build_plan_matches_declared_nested_discovery_topology(tmp_path: Path) -> None:
    assignments = tmp_path / "folds.json"
    cache = tmp_path / "cache.json"
    _write_json(
        assignments,
        {"identity_to_fold": {f"I{fold}": fold for fold in range(6)}},
    )
    _write_json(cache, {"sessions": []})
    records, summary = BUILD.build_plan(
        assignments_path=assignments,
        cache_manifest=cache,
        outer_folds=[3, 4],
    )
    assert len(records) == 12
    by_name = {str(path): manifest for path, manifest in records}
    first = by_name["outer_3/inner_pred_0.json"]
    assert first["fold_id"] == 300
    assert first["identities"] == {
        "train": ["I2", "I5"],
        "validation": ["I1"],
        "prediction": ["I0"],
        "excluded": ["I3", "I4"],
        "scaler": ["I2", "I5"],
    }
    validation = by_name["outer_4/validation_pred_5.json"]
    assert validation["fold_id"] == 450
    assert validation["identities"]["train"] == ["I1", "I2", "I3"]
    assert validation["identities"]["validation"] == ["I0"]
    assert validation["identities"]["prediction"] == ["I5"]
    assert validation["identities"]["excluded"] == ["I4"]
    test = by_name["outer_3/test_pred_3.json"]
    assert test["fold_id"] == 360
    assert test["identities"]["train"] == ["I0", "I1", "I2", "I5"]
    assert test["identities"]["validation"] == ["I4"]
    assert test["identities"]["prediction"] == ["I3"]
    assert test["identities"]["excluded"] == []
    assert summary["outer_folds"]["3"]["hcs_training_folds"] == [0, 1, 2, 5]
    for manifest in by_name.values():
        assert manifest["content_sha256"] == BUILD.canonical_content_sha256(manifest)
        assert manifest["fold_assignments"]["sha256"] == hashlib.sha256(
            assignments.read_bytes()
        ).hexdigest()
