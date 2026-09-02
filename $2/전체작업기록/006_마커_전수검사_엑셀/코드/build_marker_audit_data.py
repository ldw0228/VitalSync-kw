import json
from pathlib import Path

import numpy as np

import sys
HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
ROOT = WORKSPACE.parent
sys.path.insert(0, str(WORKSPACE))

from matv5_reader import loadmat
from sync_subject import detect_markers


DATASET = ROOT / "HAI_EXPERIMENT"
MATLAB_JSON = WORKSPACE / "revalidation" / "matlab_raw" / "matlab_results.json"
NORMALIZATION_JSON = WORKSPACE / "sync_results" / "normalization_records.json"
FINAL_JSON = WORKSPACE / "sync_results" / "final_sync_records.json"
OUTPUT_JSON = HERE / "marker_audit_data.json"

GRID = [9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5]
MATCH_TOL = 2.0
ALT_TOL = 15.0
AMBIGUITY_TOL = 5.0

PAIR_RULES_22 = [
    ("1번 호흡", 150, 240, "PPT 3분"),
    ("2번 호흡", 240, 450, "PPT 표기 4분30초 / 세부 합계 약 6분"),
    ("3번 픽업-1", 5, 90, "총시간 미기재"),
    ("3번 픽업-2", 5, 90, "총시간 미기재"),
    ("4번 낙상-즉시", 5, 30, "즥시 일어나기"),
    ("4번 낙상-30초 유지", 25, 75, "30초 유지+이동"),
    ("4번 낙상-천천히", 25, 90, "천천히 눕기+30초 유지"),
    ("4번 물건 집기", 5, 40, "S04부터 추가된 것으로 판단"),
    ("5번 코스", 130, 230, "16칸×10초=약 160초+이동"),
    ("6번 왕복", 5, 70, "PPT 1분 미만"),
    ("7번 자유", 5, 70, "PPT 1분 미만"),
]
PAIR_RULES_20 = PAIR_RULES_22[:7] + PAIR_RULES_22[8:]


def expected_count(number):
    return 20 if number <= 3 else 22


def subject_number(subject):
    return int(subject[1:3])


def load_rsp(subject):
    files = sorted((DATASET / subject / "BIOPAC").rglob("*.mat"))
    if not files:
        return None, None
    data = loadmat(files[0])
    rsp = np.asarray(data["data"][:, 0], dtype=np.float64)
    isi = float(np.asarray(data.get("isi", [[4]])).reshape(-1)[0])
    return rsp, 1000.0 / isi


def choose_threshold(counts, expected):
    enough = [t for t in GRID if counts[str(t)] >= expected]
    if enough:
        return max(enough)
    return min(GRID, key=lambda t: (abs(counts[str(t)] - expected), -t))


def normalized(values):
    values = np.asarray(values, dtype=float)
    span = values[-1] - values[0]
    return (values - values[0]) / span if span > 0 else np.zeros_like(values)


def select_template(candidates, template):
    c = np.asarray(candidates, dtype=float)
    count = len(template)
    if len(c) < count:
        return []
    if len(c) == count:
        return c.tolist()
    best_cost = float("inf")
    best = None
    n = len(c)
    for first in range(0, n - count + 1):
        for last in range(first + count - 1, n):
            span = c[last] - c[first]
            if span <= 0:
                continue
            pred = c[first] + template * span
            dp = np.full((count, n), np.inf)
            prev = np.full((count, n), -1, dtype=int)
            dp[0, first] = 0.0
            for k in range(1, count - 1):
                j_min = first + k
                j_max = last - (count - 1 - k)
                for j in range(j_min, j_max + 1):
                    prior_indices = np.arange(first + k - 1, j)
                    if prior_indices.size == 0:
                        continue
                    expected_gap = pred[k] - pred[k - 1]
                    scale = max(0.015 * span, 0.5 * expected_gap, 1.0)
                    costs = dp[k - 1, prior_indices] + ((c[j] - c[prior_indices] - expected_gap) / scale) ** 2
                    pos = int(np.argmin(costs))
                    p = int(prior_indices[pos])
                    dp[k, j] = costs[pos] + ((c[j] - pred[k]) / span) ** 2
                    prev[k, j] = p
            prior_indices = np.arange(first + count - 2, last)
            if prior_indices.size == 0:
                continue
            pos = int(np.argmin(dp[count - 2, prior_indices]))
            p = int(prior_indices[pos])
            cost = float(dp[count - 2, p])
            if cost < best_cost:
                indices = [last, p]
                for k in range(count - 2, 0, -1):
                    indices.append(int(prev[k, indices[-1]]))
                indices.reverse()
                best = c[indices]
                best_cost = cost
    return [] if best is None else best.tolist()


def greedy_match(candidates, selected, tol=MATCH_TOL):
    available = set(range(len(candidates)))
    matches = {}
    unmatched = []
    for s_idx, value in enumerate(selected):
        choices = sorted(available, key=lambda i: abs(candidates[i] - value))
        if choices and abs(candidates[choices[0]] - value) <= tol:
            c_idx = choices[0]
            matches[c_idx] = s_idx
            available.remove(c_idx)
        else:
            unmatched.append({"selected_index": s_idx + 1, "selected_s": value})
    return matches, unmatched


matlab = json.loads(MATLAB_JSON.read_text(encoding="utf-8"))
matlab_by_subject = {row["subject"]: row for row in matlab}
normalization = json.loads(NORMALIZATION_JSON.read_text(encoding="utf-8"))
norm_by_number = {row["number"]: row for row in normalization}
final = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
final_by_number = {row["number"]: row for row in final}

# Existing values retained in the workbook. S03 has three missing boundaries.
existing_selected = {
    2: [25, 217, 245, 611, 708, 723, 740, 754, 842, 896, 896, 944, 946, 1000, 1020, 1184, 1232, 1259, 1286, 1295],
    3: [23, 210, 276, 646, None, 736, None, None, 934, 945, 971, 1010, 1028, 1072, 1168, 1358, 1412, 1433, 1466, 1478],
}

# Build a robust 22-boundary timing template from completed GPT rows.
complete_22 = [
    row["markers_radar_s"] for row in final
    if row.get("markers_radar_s") and len(row["markers_radar_s"]) == 22
]
template22 = np.median(np.vstack([normalized(row) for row in complete_22]), axis=0)
template20 = np.delete(template22, [14, 15])

subjects = sorted((p.name for p in DATASET.iterdir() if p.is_dir() and p.name.startswith("S")))
records = []
candidate_rows = []
boundary_rows = []
pair_rows = []

for subject in subjects:
    number = subject_number(subject)
    expected = expected_count(number)
    raw_row = matlab_by_subject.get(subject, {})
    raw_offset = raw_row.get("offset_s") if raw_row.get("status") == "OK" else None
    raw_bio = raw_row.get("marker_biopac_s", []) if raw_row.get("status") == "OK" else []
    raw_radar = raw_row.get("marker_radar_s", []) if raw_row.get("status") == "OK" else []

    rsp, fs = load_rsp(subject)
    detections = {}
    counts = {}
    if rsp is not None:
        for threshold in GRID:
            vals = detect_markers(rsp, fs, threshold).tolist()
            detections[str(threshold)] = vals
            counts[str(threshold)] = len(vals)
    else:
        for threshold in GRID:
            detections[str(threshold)] = []
            counts[str(threshold)] = 0

    norm = norm_by_number.get(number, {})
    audit_threshold = norm.get("threshold")
    if audit_threshold is None:
        audit_threshold = choose_threshold(counts, expected)
    audit_threshold = float(audit_threshold)

    final_row = final_by_number.get(number, {})
    final_offset = norm.get("offset_s")
    if final_offset is None:
        final_offset = raw_offset

    candidates_bio = detections.get(str(audit_threshold), [])
    candidates_radar = [v - final_offset for v in candidates_bio] if final_offset is not None else []
    if not raw_bio and detections.get("8.5"):
        raw_bio = detections["8.5"]

    selected_source = ""
    selected_slots = []
    selected_flat = []
    if number in existing_selected:
        selected_source = "기존 엑셀값"
        selected_slots = existing_selected[number]
        selected_flat = [float(v) for v in selected_slots if v is not None]
    elif final_row.get("markers_radar_s"):
        selected_source = "GPT 최종 선택"
        selected_flat = [float(v) for v in final_row["markers_radar_s"]]
        selected_slots = selected_flat[:]

    template = template20 if expected == 20 else template22
    recommended = []
    if not selected_flat and candidates_radar:
        recommended = select_template(candidates_radar, template)

    display_target = selected_flat if selected_flat else recommended
    matches, unmatched = greedy_match(candidates_radar, selected_flat) if selected_flat else ({}, [])
    rec_matches, _ = greedy_match(candidates_radar, recommended) if recommended else ({}, [])
    raw_matches, raw_unmatched = greedy_match(raw_radar, selected_flat) if selected_flat and raw_radar else ({}, [])

    statuses = []
    for idx, value in enumerate(candidates_radar):
        if idx in matches:
            status = "선택"
            boundary_index = matches[idx] + 1
        elif idx in rec_matches:
            status = "추천후보"
            boundary_index = rec_matches[idx] + 1
        elif display_target and min(abs(value - v) for v in display_target) <= ALT_TOL:
            status = "대체후보"
            boundary_index = int(np.argmin([abs(value - v) for v in display_target])) + 1
        else:
            status = "미선택"
            boundary_index = None
        statuses.append(status)
        candidate_rows.append({
            "number": number,
            "subject": subject,
            "threshold": audit_threshold,
            "candidate_index": idx + 1,
            "biopac_s": candidates_bio[idx],
            "radar_s": value if final_offset is not None else None,
            "status": status,
            "boundary_index": boundary_index,
        })

    boundary_target = selected_slots if selected_slots else recommended
    if boundary_target:
        padded = list(boundary_target) + [None] * max(0, expected - len(boundary_target))
        for b_idx in range(expected):
            target = padded[b_idx] if b_idx < len(padded) else None
            nearby = []
            if candidates_radar and target is not None:
                nearby = sorted(candidates_radar, key=lambda v: abs(v - target))[:3]
                nearby = [v for v in nearby if abs(v - target) <= ALT_TOL]
            boundary_rows.append({
                "number": number,
                "subject": subject,
                "expected_count": expected,
                "boundary_index": b_idx + 1,
                "selected_or_recommended_s": target,
                "source": selected_source if selected_flat else "시간패턴 추천(미확정)",
                "candidate_1_s": nearby[0] if len(nearby) > 0 else None,
                "candidate_2_s": nearby[1] if len(nearby) > 1 else None,
                "candidate_3_s": nearby[2] if len(nearby) > 2 else None,
                "ambiguous": sum(abs(v - target) <= AMBIGUITY_TOL for v in nearby) > 1 if target is not None else False,
            })

    rules = PAIR_RULES_20 if expected == 20 else PAIR_RULES_22
    if boundary_target:
        padded = list(boundary_target) + [None] * max(0, expected - len(boundary_target))
        for p_idx, (label, low, high, basis) in enumerate(rules):
            start = padded[p_idx * 2] if p_idx * 2 < len(padded) else None
            end = padded[p_idx * 2 + 1] if p_idx * 2 + 1 < len(padded) else None
            duration = end - start if start is not None and end is not None else None
            if duration is None:
                duration_status = "경계 누락"
            elif low <= duration <= high:
                duration_status = "시간 적합"
            else:
                duration_status = "시간 확인 필요"
            pair_rows.append({
                "number": number,
                "subject": subject,
                "pair_index": p_idx + 1,
                "scenario": label,
                "start_s": start,
                "end_s": end,
                "duration_s": duration,
                "min_s": low,
                "max_s": high,
                "duration_status": duration_status,
                "basis": basis,
                "source": selected_source if selected_flat else "시간패턴 추천(미확정)",
            })

    ambiguous_count = sum(1 for row in boundary_rows if row["number"] == number and row["ambiguous"])
    time_issues = sum(1 for row in pair_rows if row["number"] == number and row["duration_status"] != "시간 적합")
    if number == 24:
        overall = "원본 누락"
    elif not selected_flat:
        overall = "후보군 검토 필요"
    elif len(selected_flat) != expected:
        overall = "선택 마커 수 불일치"
    elif unmatched:
        overall = "수동값/원시 불일치"
    elif time_issues:
        overall = "시간 구간 확인 필요"
    elif ambiguous_count:
        overall = "대체 후보 있음"
    else:
        overall = "자동 점검 통과"

    raw_statuses = []
    for idx, value in enumerate(raw_radar):
        if idx in raw_matches:
            raw_statuses.append("선택 일치")
        elif selected_flat and min(abs(value - v) for v in selected_flat) <= ALT_TOL:
            raw_statuses.append("근접 후보")
        else:
            raw_statuses.append("미선택")
    if not raw_radar and raw_bio:
        raw_statuses = ["미선택"] * len(raw_bio)

    records.append({
        "number": number,
        "subject": subject,
        "expected_count": expected,
        "protocol": "초기 20개(낙상 3구간)" if expected == 20 else "변경 22개(낙상 물건집기 포함)",
        "raw8_count": len(raw_bio),
        "raw8_biopac": raw_bio,
        "raw8_radar": raw_radar,
        "raw8_statuses": raw_statuses,
        "raw_offset_s": raw_offset,
        "audit_threshold": audit_threshold,
        "candidate_count": len(candidates_bio),
        "candidate_biopac": candidates_bio,
        "candidate_radar": candidates_radar,
        "candidate_statuses": statuses,
        "selected_count": len(selected_flat),
        "selected_source": selected_source or "없음",
        "selected_markers": selected_flat,
        "selected_slots": selected_slots,
        "recommended_count": len(recommended),
        "recommended_markers": recommended,
        "matched_candidate_count": len(matches),
        "unmatched_selected_count": len(unmatched),
        "raw8_matched_count": len(raw_matches),
        "raw8_unmatched_selected_count": len(raw_unmatched),
        "ambiguous_boundary_count": ambiguous_count,
        "time_issue_count": time_issues,
        "overall_status": overall,
        "note": final_row.get("note", ""),
        "threshold_counts": counts,
    })

payload = {
    "rules": {
        "match_tolerance_s": MATCH_TOL,
        "alternative_window_s": ALT_TOL,
        "ambiguity_window_s": AMBIGUITY_TOL,
        "expected": "S01-S03=20, S04-S30=22",
        "pair_rules_20": PAIR_RULES_20,
        "pair_rules_22": PAIR_RULES_22,
    },
    "records": records,
    "candidate_rows": candidate_rows,
    "boundary_rows": boundary_rows,
    "pair_rows": pair_rows,
}
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

print(json.dumps({
    "output": str(OUTPUT_JSON),
    "subjects": len(records),
    "candidate_rows": len(candidate_rows),
    "boundary_rows": len(boundary_rows),
    "pair_rows": len(pair_rows),
    "status_counts": {status: sum(r["overall_status"] == status for r in records) for status in sorted(set(r["overall_status"] for r in records))},
    "threshold_summary": [{"subject": r["subject"], "threshold": r["audit_threshold"], "candidate_count": r["candidate_count"], "expected": r["expected_count"]} for r in records],
}, ensure_ascii=False, indent=2))
