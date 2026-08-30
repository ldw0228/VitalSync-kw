#!/usr/bin/env python3
"""Audit HAI_EXPERIMENT without modifying the source recordings."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys
from typing import Any, Iterable


# Permit ``python scripts/audit_dataset.py`` before an editable installation.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from snn_rr.data import audit_manifest, build_dataset_manifest  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only QC for XeThru radar and BIOPAC MAT recordings: files, "
            "record shape, frame counters, NaN/Inf, synchronization, and clipping."
        )
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default=str(_REPOSITORY_ROOT / "HAI_EXPERIMENT"),
        help="HAI_EXPERIMENT directory (default: repository HAI_EXPERIMENT)",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "csv"),
        default="text",
        help="report serialization (default: text summary)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the report here instead of stdout",
    )
    parser.add_argument(
        "--radar-amplitude-limit",
        type=float,
        default=0.1,
        help="absolute radar sample threshold for isolated corruption (default: 0.1)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="include OK subjects in the text summary",
    )
    parser.add_argument(
        "--fail-on",
        choices=("never", "error", "warning"),
        default="never",
        help="optional CI exit policy (default: never)",
    )
    return parser


def _problem_details(subject: dict[str, Any]) -> list[str]:
    details: list[str] = []
    if subject["missing_radars"]:
        details.append("missing radar " + ",".join(map(str, subject["missing_radars"])))
    for radar_id, radar in subject["radars"].items():
        if radar["amplitude_outlier_count"]:
            first = radar.get("first_amplitude_outlier") or {}
            details.append(
                f"radar{radar_id} amplitude outliers={radar['amplitude_outlier_count']} "
                f"max={radar['max_abs']:.6g}"
                + (
                    f" (sequence={first.get('frame_sequence')}, bin={first.get('bin_index')})"
                    if first
                    else ""
                )
            )
        if radar["counter_gap_count"]:
            details.append(f"radar{radar_id} counter gaps={radar['counter_gap_count']}")
        if radar["nan_count"] or radar["inf_count"]:
            details.append(
                f"radar{radar_id} NaN={radar['nan_count']} Inf={radar['inf_count']}"
            )
        if radar["record_size_remainder_bytes"]:
            details.append(
                f"radar{radar_id} trailing bytes={radar['record_size_remainder_bytes']}"
            )
    biopac = subject.get("biopac")
    if biopac:
        if biopac.get("rsp_clipped_count"):
            details.append(
                f"RSP clipped={biopac['rsp_clipped_count']} "
                f"({biopac['rsp_clipped_fraction']:.3%})"
            )
        if biopac.get("ecg_clipped_count"):
            details.append(
                f"ECG clipped={biopac['ecg_clipped_count']} "
                f"({biopac['ecg_clipped_fraction']:.3%})"
            )
        if biopac.get("error"):
            details.append("BIOPAC: " + biopac["error"])
    sync = subject.get("sync")
    if sync:
        details.extend(sync.get("warnings", []))
    if not details:
        details.extend(subject.get("warnings", []))
    return details


def render_text(report: dict[str, Any], *, show_all: bool = False) -> str:
    summary = report["summary"]
    counts = summary["status_counts"]
    lines = [
        f"Dataset: {report['dataset_root']}",
        (
            f"Subjects: {summary['subject_count']} total, "
            f"{summary['usable_subject_count']} usable | "
            f"ok={counts['ok']} warning={counts['warning']} "
            f"missing={counts['missing']} error={counts['error']}"
        ),
        (
            "Checks: missing files, 740-byte shape, header/bin count, frame counter, "
            "NaN/Inf, v13 timing, radar/BIOPAC synchronization, amplitude/clipping"
        ),
    ]
    for subject in report["subjects"]:
        if not show_all and subject["status"] == "ok":
            continue
        details = _problem_details(subject)
        lines.append(
            f"{subject['subject_id']}: {subject['status'].upper()}"
            + (" | " + "; ".join(details) if details else "")
        )
    return "\n".join(lines) + "\n"


_CSV_COLUMNS = (
    "subject_id",
    "subject_status",
    "usable",
    "modality",
    "radar_id",
    "modality_status",
    "frame_count",
    "shape",
    "counter_gap_count",
    "nan_count",
    "inf_count",
    "max_abs",
    "amplitude_outlier_count",
    "clipped_count",
    "clipped_fraction",
    "start_datetime",
    "duration_seconds",
    "sync_start_offset_seconds",
    "sync_end_offset_seconds",
    "warnings",
)


def _csv_rows(report: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for subject in report["subjects"]:
        base = {
            "subject_id": subject["subject_id"],
            "subject_status": subject["status"],
            "usable": subject["usable"],
            "sync_start_offset_seconds": (
                subject["sync"].get("radar_to_biopac_start_seconds")
                if subject.get("sync")
                else None
            ),
            "sync_end_offset_seconds": (
                subject["sync"].get("radar_end_minus_biopac_end_seconds")
                if subject.get("sync")
                else None
            ),
        }
        emitted = False
        for radar_id, radar in subject["radars"].items():
            emitted = True
            yield {
                **base,
                "modality": "radar",
                "radar_id": radar_id,
                "modality_status": radar["status"],
                "frame_count": radar["frame_count"],
                "counter_gap_count": radar["counter_gap_count"],
                "nan_count": radar["nan_count"],
                "inf_count": radar["inf_count"],
                "max_abs": radar["max_abs"],
                "amplitude_outlier_count": radar["amplitude_outlier_count"],
                "warnings": "; ".join(radar["warnings"]),
            }
        biopac = subject.get("biopac")
        if biopac:
            emitted = True
            yield {
                **base,
                "modality": "biopac_rsp",
                "modality_status": biopac["status"],
                "frame_count": biopac.get("shape", [None])[0],
                "shape": "x".join(map(str, biopac.get("shape", []))),
                "nan_count": biopac.get("rsp_nan_or_inf_count"),
                "clipped_count": biopac.get("rsp_clipped_count"),
                "clipped_fraction": biopac.get("rsp_clipped_fraction"),
                "start_datetime": biopac.get("start_datetime"),
                "duration_seconds": biopac.get("duration_seconds"),
                "warnings": "; ".join(biopac.get("warnings", [])),
            }
            yield {
                **base,
                "modality": "biopac_ecg",
                "modality_status": biopac["status"],
                "frame_count": biopac.get("shape", [None])[0],
                "shape": "x".join(map(str, biopac.get("shape", []))),
                "nan_count": biopac.get("ecg_nan_or_inf_count"),
                "clipped_count": biopac.get("ecg_clipped_count"),
                "clipped_fraction": biopac.get("ecg_clipped_fraction"),
                "start_datetime": biopac.get("start_datetime"),
                "duration_seconds": biopac.get("duration_seconds"),
                "warnings": "; ".join(biopac.get("warnings", [])),
            }
        if not emitted:
            yield {
                **base,
                "modality": "subject",
                "modality_status": "missing",
                "warnings": "; ".join(subject.get("warnings", [])),
            }


def render_csv(report: dict[str, Any]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in _csv_rows(report):
        writer.writerow(row)
    return output.getvalue()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_dataset_manifest(args.dataset_root)
    report = audit_manifest(
        manifest, radar_amplitude_limit=args.radar_amplitude_limit
    )
    if args.format == "json":
        serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    elif args.format == "csv":
        serialized = render_csv(report)
    else:
        serialized = render_text(report, show_all=args.all)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        sys.stdout.write(serialized)

    counts = report["summary"]["status_counts"]
    if args.fail_on == "error" and counts["error"]:
        return 2
    if args.fail_on == "warning" and (
        counts["error"] or counts["warning"] or counts["missing"]
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
