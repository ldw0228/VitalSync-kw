from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_svd_features.py"
_SPEC = importlib.util.spec_from_file_location("build_svd_features_provenance", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_SVD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SVD
_SPEC.loader.exec_module(_SVD)


def _task() -> dict[str, object]:
    return {
        "pipeline_sha256": "1" * 64,
        "canonical_source_fingerprint": "source",
        "canonical_session_manifest_sha256": "2" * 64,
        "canonical_acquisition_session_manifest_sha256": "3" * 64,
        "canonical_acquisition_binding": {
            "schema_version": "snn_rr.feature_cache_acquisition.v1",
            "acquisition_session_manifest_sha256": "3" * 64,
            "mapping_sha256": "4" * 64,
        },
        "selected_rows_sha256": "5" * 64,
        "valid_only": False,
        "components": 12,
        "nfft": 4096,
        "n_iter": 2,
        "variant_names": ["raw", "centered"],
    }


def test_svd_session_signature_binds_canonical_manifest_and_acquisition_hashes() -> None:
    baseline = _SVD._session_signature(_task())

    changed_manifest = _task()
    changed_manifest["canonical_session_manifest_sha256"] = "a" * 64
    assert _SVD._session_signature(changed_manifest) != baseline

    changed_mapping = _task()
    changed_mapping["canonical_acquisition_binding"] = dict(
        changed_mapping["canonical_acquisition_binding"]
    )
    changed_mapping["canonical_acquisition_binding"]["mapping_sha256"] = "b" * 64
    assert _SVD._session_signature(changed_mapping) != baseline

    changed_acquisition = _task()
    changed_acquisition["canonical_acquisition_session_manifest_sha256"] = "d" * 64
    assert _SVD._session_signature(changed_acquisition) != baseline


def test_svd_acquisition_binding_is_exact_detached_json() -> None:
    manifest = {
        "acquisition_contract": {
            "schema_version": "snn_rr.feature_cache_acquisition.v1",
            "mapping_sha256": "c" * 64,
        }
    }
    binding = _SVD._canonical_acquisition_binding(manifest)
    assert binding == manifest["acquisition_contract"]
    assert binding is not manifest["acquisition_contract"]
    assert _SVD._canonical_acquisition_binding({}) is None
