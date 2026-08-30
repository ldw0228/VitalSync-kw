#!/usr/bin/env python3
"""Fail closed when staged or tracked files cross the Git safety boundary."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import PurePosixPath


MAX_FILE_BYTES = 10 * 1024 * 1024

SAFE_ROOT_FILES = {
    ".gitattributes",
    ".gitignore",
    "AGENTS.md",
    "GIT_REPOSITORY_POLICY.md",
    "README.md",
    "REPORT.md",
    "RESTORE_GUIDE.md",
    "pyproject.toml",
}

SAFE_SOURCE_PREFIXES = (
    ".githooks/",
    "configs/",
    "scripts/",
    "src/snn_rr/",
    "tests/",
)

SAFE_RESTORE_FILES = {
    "restore/bootstrap_env.sh",
    "restore/build_integrity_manifest.py",
    "restore/requirements-linux-cu130.txt",
    "restore/requirements-top-level.txt",
    "restore/verify_restore.sh",
}

SAFE_ARTIFACTS = {
    "artifacts/COMMERCIAL_GOAL_AUDIT.md",
    "artifacts/COMMERCIAL_SNN_CONTINUOUS_EXECUTION_PLAN_V4.md",
    "artifacts/COMMERCIAL_SNN_CONTINUOUS_EXECUTION_PLAN_V4_AMENDMENT_01.md",
    "artifacts/COMMERCIAL_SNN_GOAL_V2.md",
    "artifacts/COMMERCIAL_SNN_MASTER_EXECUTION_PLAN_V3.md",
    "artifacts/COMMERCIAL_SNN_PROGRESS_V2.md",
    "artifacts/SNN_PROJECT_COMPACT_TECH_STACK_2026-08-30.md",
    "artifacts/SNN_PROJECT_DEVELOPMENT_PROGRESS_2026-08-30.md",
    "artifacts/SNN_PROJECT_DEVELOPMENT_PROGRESS_2026-08-31.md",
    "artifacts/SNN_PROJECT_TECHNICAL_STATUS_REPORT_2026-08-30.md",
    "artifacts/SNN_PROJECT_WORKSTREAM_TECHNOLOGY_BLUEPRINT_2026-08-30.md",
    "artifacts/SNN_TRAINING_METHODOLOGY_RECOMMENDATION_2026-08-30.md",
}

BLOCKED_SUFFIXES = {
    ".acq",
    ".arrow",
    ".bin",
    ".ckpt",
    ".csv",
    ".dat",
    ".feather",
    ".gz",
    ".h5",
    ".hdf5",
    ".joblib",
    ".jpeg",
    ".jpg",
    ".jsonl",
    ".log",
    ".mat",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pdf",
    ".pickle",
    ".pkl",
    ".png",
    ".pptx",
    ".pt",
    ".pth",
    ".rar",
    ".tar",
    ".tgz",
    ".tif",
    ".tiff",
    ".tsv",
    ".wav",
    ".xls",
    ".xlsx",
    ".zip",
    ".zst",
}

SECRET_PATTERNS = (
    ("private key", re.compile(rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----")),
    ("GitHub token", re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("AWS access key", re.compile(rb"AKIA[0-9A-Z]{16}")),
    ("Google API key", re.compile(rb"AIza[0-9A-Za-z_-]{30,}")),
)


def git_bytes(*args: str) -> bytes:
    return subprocess.check_output(("git", *args), stderr=subprocess.STDOUT)


def listed_paths(mode: str) -> list[str]:
    if mode == "staged":
        raw = git_bytes(
            "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"
        )
    else:
        raw = git_bytes("ls-files", "-z")
    return sorted(part.decode("utf-8", "strict") for part in raw.split(b"\0") if part)


def index_blob(path: str) -> bytes:
    return git_bytes("show", f":{path}")


def index_mode(path: str) -> str | None:
    output = git_bytes("ls-files", "--stage", "--", path).decode("utf-8", "strict")
    return output.split(maxsplit=1)[0] if output else None


def path_reasons(path: str) -> list[str]:
    reasons: list[str] = []
    normalized = PurePosixPath(path).as_posix()
    name_lower = PurePosixPath(normalized).name.lower()

    allowed = (
        normalized in SAFE_ROOT_FILES
        or normalized in SAFE_RESTORE_FILES
        or normalized in SAFE_ARTIFACTS
        or normalized.startswith(SAFE_SOURCE_PREFIXES)
    )
    if not allowed:
        reasons.append("path is outside the reviewed allowlist")

    suffixes = {suffix.lower() for suffix in PurePosixPath(normalized).suffixes}
    if suffixes & BLOCKED_SUFFIXES:
        reasons.append("blocked data/model/archive suffix")
    if "/__pycache__/" in f"/{normalized}/" or name_lower.endswith((".pyc", ".pyo")):
        reasons.append("Python cache")
    if name_lower == ".env" or name_lower.startswith(".env."):
        reasons.append("environment secret file")
    if name_lower.endswith((".pem", ".key")):
        reasons.append("key material")
    if "credential" in name_lower or "secret" in name_lower:
        reasons.append("credential/secret filename")
    return reasons


def audit(mode: str) -> int:
    try:
        paths = listed_paths(mode)
    except subprocess.CalledProcessError as exc:
        print(exc.output.decode("utf-8", "replace"), file=sys.stderr)
        return 2

    failures: list[str] = []
    total_bytes = 0
    for path in paths:
        reasons = path_reasons(path)
        try:
            mode_bits = index_mode(path)
            blob = index_blob(path)
        except subprocess.CalledProcessError as exc:
            failures.append(f"{path}: cannot inspect staged blob: {exc.output!r}")
            continue

        total_bytes += len(blob)
        if mode_bits == "120000":
            reasons.append("symlink is not allowed")
        if len(blob) > MAX_FILE_BYTES:
            reasons.append(f"file exceeds {MAX_FILE_BYTES} bytes")
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(blob):
                reasons.append(f"possible {label}")
        if reasons:
            failures.append(f"{path}: {', '.join(sorted(set(reasons)))}")

    if failures:
        print("Git safety check FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(
        f"Git safety check passed: {len(paths)} {mode} files, "
        f"{total_bytes / (1024 * 1024):.2f} MiB"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--staged", action="store_true", help="audit staged files")
    group.add_argument("--all-tracked", action="store_true", help="audit all tracked files")
    args = parser.parse_args()
    return audit("staged" if args.staged else "tracked")


if __name__ == "__main__":
    raise SystemExit(main())
