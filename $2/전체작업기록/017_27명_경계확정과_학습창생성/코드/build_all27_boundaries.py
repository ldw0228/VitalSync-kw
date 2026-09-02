from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "$2"
OUT = WORKSPACE / "outputs" / "breathing_v3"
MANIFEST = OUT / "subject_protocol_manifest.csv"

sys.path.insert(0, str(WORKSPACE / "SNN_v2"))
from prepare_labels import detect_breath_resume, detect_peak_time, subject_features  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows: list[dict] = []
    for record in read_csv(MANIFEST):
        if record["usability"] != "usable_both":
            continue
        subject = record["subject"]
        m01, m02, m03, m04 = (float(record[key]) for key in ("m01_s", "m02_s", "m03_s", "m04_s"))

        # Same frozen protocol rule used for the original 10 development subjects.
        s1_pad = min(15.0, max(0.0, (m02 - m01 - 180.0) / 2.0))
        s1_content = m01 + s1_pad
        turn1_expected = s1_content + 60.0
        turn2_expected = s1_content + 120.0
        motion = subject_features(subject)
        turn1 = detect_peak_time(motion, turn1_expected, 9.0)
        turn2 = detect_peak_time(motion, turn2_expected, 9.0)
        turn_basis = "radar_motion_peak_near_protocol_time"
        if turn2 <= turn1 + 35.0:
            turn1, turn2 = turn1_expected, turn2_expected
            turn_basis = "protocol_fallback_due_to_invalid_peak_order"

        s2_pad = min(18.0, max(0.0, (m04 - m03 - 360.0) / 2.0))
        s2_content = m03 + s2_pad
        apnea_start = s2_content + 120.0
        apnea_resume, resume_basis = detect_breath_resume(subject, apnea_start, apnea_start + 30.0)
        apnea_resume = min(max(float(apnea_resume), apnea_start + 25.0), apnea_start + 30.0)

        rows.append({
            "subject": subject,
            "m01_s": m01,
            "m02_s": m02,
            "s01_duration_s": m02 - m01,
            "s01_content_start_s": s1_content,
            "turn1_s": turn1,
            "turn2_s": turn2,
            "turn_basis": turn_basis,
            "m03_s": m03,
            "m04_s": m04,
            "s02_duration_s": m04 - m03,
            "s02_content_start_s": s2_content,
            "apnea_start_s": apnea_start,
            "apnea_resume_s": apnea_resume,
            "apnea_duration_s": apnea_resume - apnea_start,
            "resume_basis": resume_basis,
            "marker_basis": record["decision_basis"],
            "label_basis": "protocol_timing+radar_turn_motion+aligned_BIOPAC_resume",
        })
        print(f"[boundary] {subject}: turn={turn1:.1f}/{turn2:.1f}s, apnea={apnea_resume-apnea_start:.1f}s", flush=True)

    path = OUT / "all27_boundary_decisions.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "subject_count": len(rows),
        "subjects": [row["subject"] for row in rows],
        "turn_peak_count": sum(row["turn_basis"] == "radar_motion_peak_near_protocol_time" for row in rows),
        "turn_fallback_count": sum(row["turn_basis"] != "radar_motion_peak_near_protocol_time" for row in rows),
        "apnea_rsp_detected_count": sum(str(row["resume_basis"]).startswith("RSP") for row in rows),
        "apnea_protocol_fallback_count": sum(not str(row["resume_basis"]).startswith("RSP") for row in rows),
        "excluded": {
            "S01_CMS": "ambiguous S01/S02 marker boundary",
            "S22_KJH": "missing M01",
            "S24_KHJ": "missing UWB raw data",
        },
    }
    (OUT / "all27_boundary_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
