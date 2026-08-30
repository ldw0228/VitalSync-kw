#!/usr/bin/env python3
"""Build deterministic SHA-256 and path/size/mode manifests for a restore set."""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import tempfile
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)


def _relative_to_root(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"path escapes root: {path}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--include", action="append", required=True)
    parser.add_argument("--content-manifest", type=Path, required=True)
    parser.add_argument("--sha256-list", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"root is not a directory: {root}")

    content_output = args.content_manifest
    checksum_output = args.sha256_list
    if not content_output.is_absolute():
        content_output = root / content_output
    if not checksum_output.is_absolute():
        checksum_output = root / checksum_output

    excluded = {
        _relative_to_root(content_output, root),
        _relative_to_root(checksum_output, root),
    }
    files: set[Path] = set()
    for raw_include in args.include:
        include = (root / raw_include).resolve()
        relative_include = _relative_to_root(include, root)
        include = root / relative_include
        if include.is_symlink():
            raise SystemExit(f"symbolic links are not supported: {relative_include}")
        if include.is_file():
            files.add(relative_include)
        elif include.is_dir():
            for candidate in include.rglob("*"):
                if candidate.is_symlink():
                    raise SystemExit(
                        f"symbolic links are not supported: {candidate.relative_to(root)}"
                    )
                if candidate.is_file():
                    files.add(candidate.relative_to(root))
        else:
            raise SystemExit(f"include does not exist: {relative_include}")

    files.difference_update(excluded)
    rows: list[tuple[str, int, str, str]] = []
    for relative in sorted(files, key=lambda item: item.as_posix()):
        path = root / relative
        file_stat = path.stat()
        rows.append(
            (
                relative.as_posix(),
                file_stat.st_size,
                f"{stat.S_IMODE(file_stat.st_mode):04o}",
                _sha256(path),
            )
        )

    content_lines = ["path\tbytes\tmode\tsha256"]
    checksum_lines: list[str] = []
    for relative, size, mode, digest in rows:
        content_lines.append(f"{relative}\t{size}\t{mode}\t{digest}")
        checksum_lines.append(f"{digest}  {relative}")

    _atomic_write(content_output, "\n".join(content_lines) + "\n")
    _atomic_write(checksum_output, "\n".join(checksum_lines) + "\n")
    print(f"manifested {len(rows)} files under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
