import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "$2" / "audit_build" / "marker_audit_data.json"
OUT = ROOT / "$2" / "outputs" / "snn_v2"


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    boundary = {r["subject"]: r for r in rows(OUT / "boundary_decisions.csv")}
    marker_rows = []
    candidate_rows = []
    for r in audit["records"]:
        selected = r.get("selected_slots") or []
        m = [selected[i] if i < len(selected) else None for i in range(4)]
        s1_d = m[1] - m[0] if m[0] is not None and m[1] is not None else None
        s2_d = m[3] - m[2] if m[2] is not None and m[3] is not None else None
        if s2_d is not None:
            pad = min(18.0, max(0.0, (s2_d - 360.0) / 2.0))
            content = m[2] + pad
            squat_lo, squat_hi = content + 210, content + 300
        else:
            content = squat_lo = squat_hi = None
        candidates = list(r.get("candidate_radar") or [])
        internal = [v for v in candidates if m[2] is not None and m[3] is not None and m[2] + 2 < v < m[3] - 2]
        squat = [v for v in internal if squat_lo <= v <= squat_hi] if squat_lo is not None else []
        marker_rows.append({
            "번호": r["number"], "참가자": r["subject"], "예상 마커수": r["expected_count"],
            "선택 마커수": r["selected_count"], "전체 판정": r["overall_status"], "선택 출처": r["selected_source"],
            "M01 1번 시작(s)": m[0], "M02 1번 종료(s)": m[1], "1번 길이(s)": s1_d,
            "M03 2번 시작(s)": m[2], "M04 2번 종료(s)": m[3], "2번 길이(s)": s2_d,
            "2번 내부 후보수": len(internal), "운동구간 후보수": len(squat),
            "운동 오검출 점검": "후보 존재-파형 확인" if squat else ("후보 없음" if s2_d is not None else "경계 미확정"),
            "학습 사용": "A등급 사용" if r["subject"] in boundary else "초기 학습 제외",
            "비고": r.get("note", ""),
        })
        for value in internal:
            rel = value - content if content is not None else None
            if rel is None:
                phase = "미정"
            elif rel < 60: phase = "평소 호흡"
            elif rel < 120: phase = "느린 호흡"
            elif rel < 150: phase = "숨 참기(25~30초)"
            elif rel < 210: phase = "숨 참은 뒤 호흡"
            elif rel < 225: phase = "운동 준비"
            elif rel < 285: phase = "스쿼트"
            elif rel < 300: phase = "의자 착석"
            else: phase = "운동 후 호흡"
            candidate_rows.append({
                "참가자": r["subject"], "후보 시각(s)": value, "화면 시작 추정 대비(s)": rel,
                "예상 단계": phase, "M03/M04 선택 여부": "선택" if any(abs(value - x) <= 2 for x in m[2:4] if x is not None) else "미선택",
                "자동 분류": "운동 오검출 유력" if phase in ("운동 준비", "스쿼트", "의자 착석") else "중간 고값-마커 아님",
                "판단 근거": "2번 내부에는 별도 시나리오 마커가 없으며 화면 지시에 따라 연속 진행",
                "최종 상태": "검토 필요" if r["subject"] not in boundary else "학습 경계에서 제외",
            })
    data = {
        "marker_rows": marker_rows,
        "candidate_rows": candidate_rows,
        "boundary_rows": rows(OUT / "boundary_decisions.csv"),
        "window_rows": rows(OUT / "label_manifest.csv"),
        "screening_rows": rows(OUT / "encoding_decoding_screening.csv"),
        "loso_rows": rows(OUT / "loso_metrics.csv"),
        "summary": json.loads((OUT / "model_comparison_summary.json").read_text(encoding="utf-8")),
    }
    (OUT / "workbook_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
