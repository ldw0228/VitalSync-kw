#!/usr/bin/env python3
"""Hash-bind a runtime source closure and immutable input trees.

This is a supplemental provenance seal for long campaigns whose primary plan
does not enumerate every imported module or every payload behind a cache
manifest.  ``snapshot`` writes one immutable inventory.  ``verify`` hashes the
same paths again and fails if any path, byte count, or SHA-256 differs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = (
    PROJECT_ROOT / "scripts/__init__.py",
    PROJECT_ROOT / "scripts/run_full_nested_proposer_campaign.py",
    PROJECT_ROOT / "scripts/build_nested_proposer_manifests.py",
    PROJECT_ROOT / "scripts/train.py",
    PROJECT_ROOT / "scripts/predict_custom_split_all_windows.py",
    PROJECT_ROOT / "scripts/predict_all_windows.py",
    PROJECT_ROOT / "scripts/run_gpu_admitted.py",
    PROJECT_ROOT / "src/snn_rr/__init__.py",
    PROJECT_ROOT / "src/snn_rr/cache.py",
    PROJECT_ROOT / "src/snn_rr/data.py",
    PROJECT_ROOT / "src/snn_rr/metrics.py",
    PROJECT_ROOT / "src/snn_rr/models.py",
    PROJECT_ROOT / "src/snn_rr/split_authority.py",
    PROJECT_ROOT / "configs/default.yaml",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    payload.pop("created_utc", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"runtime input is not a regular file: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(resolved),
    }


def _tree_files(root: Path) -> Iterable[Path]:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise RuntimeError(f"runtime input tree is not a directory: {resolved}")
    for path in sorted(resolved.rglob("*"), key=lambda item: str(item)):
        if path.is_symlink():
            raise RuntimeError(f"runtime input tree contains a symlink: {path}")
        if path.is_file():
            yield path


def inventory(
    *,
    sources: Sequence[Path],
    trees: Sequence[Path],
    bindings: Sequence[Path],
    post_launch_attestation: bool = True,
) -> dict[str, Any]:
    source_paths = [path.expanduser().resolve() for path in sources]
    if len(source_paths) != len(set(source_paths)):
        raise RuntimeError("runtime source list contains duplicates")
    tree_roots = [path.expanduser().resolve() for path in trees]
    tree_records: list[dict[str, Any]] = []
    for root in tree_roots:
        files = [_binding(path) for path in _tree_files(root)]
        if not files:
            raise RuntimeError(f"runtime input tree is empty: {root}")
        tree_records.append(
            {
                "root": str(root),
                "file_count": len(files),
                "total_bytes": sum(int(item["bytes"]) for item in files),
                "files": files,
            }
        )
    result: dict[str, Any] = {
        "schema_version": 1,
        "classification": "supplemental_runtime_input_byte_inventory",
        "attestation_phase": (
            "post_launch" if post_launch_attestation else "prelaunch"
        ),
        "post_launch_attestation": bool(post_launch_attestation),
        "commercial_claim_authorized": False,
        "sources": [_binding(path) for path in source_paths],
        "input_trees": tree_records,
        "campaign_bindings": [_binding(path) for path in bindings],
        "runtime": {
            "python": sys.version,
            "executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    result["content_sha256"] = canonical_sha256(result)
    return result


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if target.exists():
        if target.read_bytes() != payload:
            raise RuntimeError(f"refusing to replace immutable runtime seal: {target}")
        return
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)


def verify(path: Path) -> dict[str, Any]:
    target = path.expanduser().resolve()
    try:
        sealed = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid runtime input seal: {target} ({exc})") from exc
    if not isinstance(sealed, dict) or canonical_sha256(sealed) != sealed.get(
        "content_sha256"
    ):
        raise RuntimeError("runtime input seal content hash mismatch")
    expected: list[Mapping[str, Any]] = []
    expected.extend(sealed.get("sources", []))
    expected.extend(sealed.get("campaign_bindings", []))
    for tree in sealed.get("input_trees", []):
        if not isinstance(tree, Mapping):
            raise RuntimeError("runtime input seal tree entry is invalid")
        expected.extend(tree.get("files", []))
    for item in expected:
        if not isinstance(item, Mapping):
            raise RuntimeError("runtime input seal binding is invalid")
        observed = _binding(Path(str(item.get("path", ""))))
        for field in ("bytes", "mtime_ns", "sha256"):
            if observed[field] != item.get(field):
                raise RuntimeError(
                    f"runtime input changed after seal: {observed['path']} ({field})"
                )
    for tree in sealed.get("input_trees", []):
        observed_paths = {
            str(path.expanduser().resolve())
            for path in _tree_files(Path(str(tree["root"])))
        }
        expected_paths = {str(item["path"]) for item in tree["files"]}
        if observed_paths != expected_paths:
            raise RuntimeError(f"runtime input tree membership changed: {tree['root']}")
    return {
        "status": "verified",
        "seal": str(target),
        "content_sha256": sealed["content_sha256"],
        "verified_files": len(expected),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot.add_argument("--source", action="append", type=Path)
    snapshot.add_argument("--tree", action="append", type=Path, default=[])
    snapshot.add_argument("--bind", action="append", type=Path, default=[])
    snapshot.add_argument(
        "--phase",
        choices=("prelaunch", "post_launch"),
        default="post_launch",
        help="Record whether the inventory was captured before or after launch.",
    )
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--seal", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "snapshot":
        document = inventory(
            sources=args.source or DEFAULT_SOURCES,
            trees=args.tree,
            bindings=args.bind,
            post_launch_attestation=args.phase == "post_launch",
        )
        atomic_json(args.output, document)
        result = {
            "status": "sealed",
            "output": str(args.output.expanduser().resolve()),
            "content_sha256": document["content_sha256"],
            "files": len(document["sources"])
            + len(document["campaign_bindings"])
            + sum(len(tree["files"]) for tree in document["input_trees"]),
        }
    else:
        result = verify(args.seal)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
