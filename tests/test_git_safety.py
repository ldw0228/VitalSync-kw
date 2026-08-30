from __future__ import annotations

from scripts.check_git_safety import path_reasons


def test_git_safety_allows_reviewed_core_files() -> None:
    safe_paths = (
        "AGENTS.md",
        "src/snn_rr/data.py",
        "scripts/train.py",
        "tests/test_data.py",
        "configs/default.yaml",
        "restore/bootstrap_env.sh",
        "artifacts/SNN_PROJECT_DEVELOPMENT_PROGRESS_2026-08-30.md",
        "artifacts/SNN_PROJECT_DEVELOPMENT_PROGRESS_2026-08-31.md",
        "artifacts/COMMERCIAL_SNN_GOAL_V3_2026-08-31.md",
    )

    for path in safe_paths:
        assert path_reasons(path) == []


def test_git_safety_rejects_private_and_generated_paths() -> None:
    unsafe_paths = (
        "HAI_EXPERIMENT/S02_RJS/radar_1.dat",
        "HAI_EXPERIMENT-20260827T035530Z-1-001.zip",
        "HAI_Experiment_UWB_GuideLine.pptx",
        "sync_tool_S02.m",
        "artifacts/cache/rf32s/maps.npy",
        "artifacts/runs/model/snn_best.pt",
        "artifacts/acquisition/session_manifest.json",
        "artifacts/final_report.json",
        "restore/SnnProject_RESTORE_INDEX_2026-08-30.md",
    )

    for path in unsafe_paths:
        assert path_reasons(path)


def test_git_safety_rejects_secrets_and_unreviewed_files() -> None:
    assert path_reasons(".env")
    assert path_reasons("credentials-production.json")
    assert path_reasons("new_unreviewed_root_file.txt")
    assert path_reasons("artifacts/new_report.md")
