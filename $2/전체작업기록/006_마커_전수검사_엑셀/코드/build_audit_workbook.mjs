import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SOURCE = path.resolve(HERE, "../outputs/sync_revalidation/Dataset_issue_$2_MATLAB_revalidated.xlsx");
const DATA = path.resolve(HERE, "marker_audit_data.json");
const OUTDIR = path.resolve(HERE, "../outputs/sync_audit");
const OUTPUT = path.join(OUTDIR, "Dataset_issue_$2_전수검사_통합표.xlsx");

const BLUE = "#1F4E78";
const MID_BLUE = "#4472C4";
const LIGHT_BLUE = "#D9EAF7";
const GREEN = "#C6EFCE";
const YELLOW = "#FFF2CC";
const RED = "#F4CCCC";
const GREY = "#E7E6E6";
const WHITE = "#FFFFFF";
const TEXT = "#1F2937";
const BORDER = "#B4C6E7";

const data = JSON.parse(await fs.readFile(DATA, "utf8"));
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(SOURCE));

function colName(index) {
  let n = index;
  let s = "";
  while (n > 0) {
    n -= 1;
    s = String.fromCharCode(65 + (n % 26)) + s;
    n = Math.floor(n / 26);
  }
  return s;
}

function titleBand(sheet, title, subtitle, endCol) {
  sheet.mergeCells(`A1:${endCol}1`);
  sheet.getRange("A1").values = [[title]];
  sheet.getRange(`A1:${endCol}1`).format = {
    fill: BLUE,
    font: { bold: true, color: WHITE, size: 18 },
    verticalAlignment: "center",
  };
  sheet.getRange("A1").format.rowHeight = 30;
  sheet.mergeCells(`A2:${endCol}2`);
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange(`A2:${endCol}2`).format = {
    fill: "#F3F6FA",
    font: { italic: true, color: "#5B6573", size: 10 },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.getRange("A2").format.rowHeight = 30;
  sheet.showGridLines = false;
}

function styleHeader(range) {
  range.format = {
    fill: MID_BLUE,
    font: { bold: true, color: WHITE },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: BORDER },
  };
}

function applyStatusFill(cell, status) {
  let fill = GREY;
  if (status === "자동 점검 통과" || status === "선택" || status === "선택 일치" || status === "시간 적합") fill = GREEN;
  else if (status === "대체 후보 있음" || status === "대체후보" || status === "시간 구간 확인 필요" || status === "시간 확인 필요") fill = YELLOW;
  else if (status === "추천후보" || status === "후보군 검토 필요") fill = LIGHT_BLUE;
  else if (status === "선택 마커 수 불일치" || status === "수동값/원시 불일치" || status === "원본 누락" || status === "경계 누락") fill = RED;
  cell.format.fill = fill;
}

// 1) 전수검사 요약
{
  const sheet = workbook.worksheets.add("전수검사 요약");
  titleBand(sheet, "UWB–BIOPAC 마커 전수검사", "원본 시트는 보존하고, S01~S03=20개 / S04 이후=22개 규칙과 PPT 시간 범위로 raw·후보·최종 선택을 대조", "Q");
  sheet.getRange("A4:B7").values = [["검사 대상", null], ["예상 개수 미만 RAW", null], ["후보군/불일치 검토", null], ["자동 점검 통과", null]];
  sheet.getRange("A4:A7").format = { fill: LIGHT_BLUE, font: { bold: true }, borders: { preset: "all", style: "thin", color: BORDER } };
  sheet.getRange("B4:B7").formulas = [["=COUNTA(B10:B39)"], ["=COUNTIF(F10:F39,\"<0\")"], ["=COUNTIF(P10:P39,\"<>자동 점검 통과\")"], ["=COUNTIF(P10:P39,\"자동 점검 통과\")"]];
  sheet.getRange("B4:B7").format = { fill: WHITE, font: { bold: true, color: BLUE }, horizontalAlignment: "right", borders: { preset: "all", style: "thin", color: BORDER }, numberFormat: "0" };

  const headers = ["번호", "참가자", "프로토콜 기준", "기대 마커", "RAW 8.5V 수", "RAW-기대", "RAW offset(s)", "후보 임계값(V)", "후보 수", "현재 선택 수", "선택 출처", "후보 일치 수", "RAW 일치 수", "대체 후보 경계", "시간 이슈", "종합 판정", "특이사항"];
  sheet.getRange("A9:Q9").values = [headers];
  styleHeader(sheet.getRange("A9:Q9"));
  const rows = data.records.map((r, i) => [
    r.number, r.subject, r.protocol, r.expected_count, r.raw8_count, null, r.raw_offset_s,
    r.audit_threshold, r.candidate_count, r.selected_count, r.selected_source,
    r.matched_candidate_count, r.raw8_matched_count, r.ambiguous_boundary_count,
    r.time_issue_count, r.overall_status, r.note,
  ]);
  sheet.getRange(`A10:Q${9 + rows.length}`).values = rows;
  sheet.getRange(`F10:F${9 + rows.length}`).formulas = rows.map((_, i) => [`=E${10 + i}-D${10 + i}`]);
  sheet.getRange(`A10:Q${9 + rows.length}`).format = {
    font: { color: TEXT, size: 10 },
    verticalAlignment: "center",
    borders: { insideHorizontal: { style: "thin", color: "#D9E2F3" } },
  };
  sheet.getRange(`D10:J${9 + rows.length}`).format.numberFormat = "0.0";
  sheet.getRange(`A10:A${9 + rows.length}`).format.numberFormat = "0";
  sheet.getRange(`D10:F${9 + rows.length}`).format.numberFormat = "0";
  sheet.getRange(`H10:H${9 + rows.length}`).format.numberFormat = "0.0";
  sheet.getRange(`Q10:Q${9 + rows.length}`).format.wrapText = true;
  data.records.forEach((r, i) => applyStatusFill(sheet.getRange(`P${10 + i}`), r.overall_status));
  sheet.freezePanes.freezeRows(9);
  sheet.freezePanes.freezeColumns(2);
  const widths = [7, 14, 25, 11, 12, 11, 13, 14, 10, 12, 18, 13, 12, 14, 10, 20, 52];
  widths.forEach((w, i) => { sheet.getRange(`${colName(i + 1)}:${colName(i + 1)}`).format.columnWidth = w; });
  sheet.getRange("9:9").format.rowHeight = 32;
  sheet.getRange("10:39").format.rowHeight = 24;
}

// 2) RAW 8.5V matrix: raw BIOPAC values, one marker per cell.
{
  const maxMarkers = Math.max(...data.records.map((r) => r.raw8_biopac.length));
  const endCol = colName(6 + maxMarkers);
  const sheet = workbook.worksheets.add("RAW 8.5V");
  titleBand(sheet, "RAW 8.5V 원시 마커", "각 셀은 BIOPAC 원시시간의 마커 1개입니다. 초록=현재 선택값과 일치, 노랑=근접 후보, 회색=미선택. 레이더 시간은 BIOPAC 시간-offset입니다.", endCol);
  sheet.getRange("A4:F4").values = [["번호", "참가자", "기대 수", "RAW 수", "offset(s)", "판정"]];
  const markerHeaders = Array.from({ length: maxMarkers }, (_, i) => `RAW ${String(i + 1).padStart(2, "0")} (s)`);
  sheet.getRange(`G4:${endCol}4`).values = [markerHeaders];
  styleHeader(sheet.getRange(`A4:${endCol}4`));
  const rows = data.records.map((r) => [r.number, r.subject, r.expected_count, r.raw8_count, r.raw_offset_s, r.overall_status, ...r.raw8_biopac, ...Array(maxMarkers - r.raw8_biopac.length).fill(null)]);
  sheet.getRange(`A5:${endCol}${4 + rows.length}`).values = rows;
  sheet.getRange(`D5:E${4 + rows.length}`).format.numberFormat = "0.0";
  sheet.getRange(`G5:${endCol}${4 + rows.length}`).format.numberFormat = "0.0";
  sheet.getRange(`A5:${endCol}${4 + rows.length}`).format = {
    borders: { insideHorizontal: { style: "thin", color: "#E2E8F0" } },
    verticalAlignment: "center",
  };
  data.records.forEach((r, rowIdx) => {
    applyStatusFill(sheet.getRange(`F${5 + rowIdx}`), r.overall_status);
    r.raw8_statuses.forEach((status, markerIdx) => applyStatusFill(sheet.getRange(`${colName(7 + markerIdx)}${5 + rowIdx}`), status));
  });
  sheet.freezePanes.freezeRows(4);
  sheet.freezePanes.freezeColumns(6);
  [7, 14, 10, 10, 11, 20].forEach((w, i) => { sheet.getRange(`${colName(i + 1)}:${colName(i + 1)}`).format.columnWidth = w; });
  for (let i = 0; i < maxMarkers; i++) sheet.getRange(`${colName(7 + i)}:${colName(7 + i)}`).format.columnWidth = 11;
}

// 3) Candidate matrix at participant-specific audit threshold.
{
  const maxCandidates = Math.max(...data.records.map((r) => Math.max(r.candidate_radar.length, r.candidate_biopac.length)));
  const endCol = colName(8 + maxCandidates);
  const sheet = workbook.worksheets.add("후보 마커");
  titleBand(sheet, "참가자별 후보 마커", "초록=현재 선택, 파랑=시간패턴 추천(아직 미확정), 노랑=같은 경계의 대체 후보, 회색=미선택. 레이더가 없으면 BIOPAC 시간만 표시합니다.", endCol);
  sheet.getRange("A4:H4").values = [["번호", "참가자", "기대 수", "임계값(V)", "후보 수", "시간축", "선택 수", "판정"]];
  sheet.getRange(`I4:${endCol}4`).values = [Array.from({ length: maxCandidates }, (_, i) => `후보 ${String(i + 1).padStart(2, "0")} (s)`)];
  styleHeader(sheet.getRange(`A4:${endCol}4`));
  const rows = data.records.map((r) => {
    const values = r.candidate_radar.length ? r.candidate_radar : r.candidate_biopac;
    const axis = r.candidate_radar.length ? "레이더" : "BIOPAC only";
    return [r.number, r.subject, r.expected_count, r.audit_threshold, r.candidate_count, axis, r.selected_count || r.recommended_count, r.overall_status, ...values, ...Array(maxCandidates - values.length).fill(null)];
  });
  sheet.getRange(`A5:${endCol}${4 + rows.length}`).values = rows;
  sheet.getRange(`D5:D${4 + rows.length}`).format.numberFormat = "0.0";
  sheet.getRange(`I5:${endCol}${4 + rows.length}`).format.numberFormat = "0.0";
  sheet.getRange(`A5:${endCol}${4 + rows.length}`).format = { borders: { insideHorizontal: { style: "thin", color: "#E2E8F0" } }, verticalAlignment: "center" };
  data.records.forEach((r, rowIdx) => {
    applyStatusFill(sheet.getRange(`H${5 + rowIdx}`), r.overall_status);
    r.candidate_statuses.forEach((status, markerIdx) => applyStatusFill(sheet.getRange(`${colName(9 + markerIdx)}${5 + rowIdx}`), status));
  });
  sheet.freezePanes.freezeRows(4);
  sheet.freezePanes.freezeColumns(8);
  [7, 14, 10, 12, 10, 14, 10, 20].forEach((w, i) => { sheet.getRange(`${colName(i + 1)}:${colName(i + 1)}`).format.columnWidth = w; });
  for (let i = 0; i < maxCandidates; i++) sheet.getRange(`${colName(9 + i)}:${colName(9 + i)}`).format.columnWidth = 11;
}

function boundaryLabel(expected, idx) {
  const labels22 = ["1 시작", "1 종료", "2 시작", "2 종료", "3-1 시작", "3-1 종료", "3-2 시작", "3-2 종료", "4-1 시작", "4-1 종료", "4-2 시작", "4-2 종료", "4-3 시작", "4-3 종료", "4-4 시작", "4-4 종료", "5 시작", "5 종료", "6 시작", "6 종료", "7 시작", "7 종료"];
  const labels20 = labels22.slice(0, 14).concat(labels22.slice(16));
  return (expected === 20 ? labels20 : labels22)[idx - 1];
}

function markerLabels(expected) {
  const labels22 = ["1 시작", "1 종료", "2 시작", "2 종료", "3-1 시작", "3-1 종료", "3-2 시작", "3-2 종료", "4-1 시작", "4-1 종료", "4-2 시작", "4-2 종료", "4-3 시작", "4-3 종료", "4-4 시작", "4-4 종료", "5 시작", "5 종료", "6 시작", "6 종료", "7 시작", "7 종료"];
  return expected === 20 ? labels22.slice(0, 14).concat(labels22.slice(16)) : labels22;
}

function scenarioHeaderFill(label) {
  if (label.startsWith("1 ")) return "#D9EAF7";
  if (label.startsWith("2 ")) return "#E4DFEC";
  if (label.startsWith("3-")) return "#FCE4D6";
  if (label.startsWith("4-")) return "#F4CCCC";
  if (label.startsWith("5 ")) return "#E2F0D9";
  if (label.startsWith("6 ")) return "#DDEBF7";
  return "#E7E6E6";
}

function scenarioSelectedFill(label) {
  if (label.startsWith("1 ")) return "#BDD7EE";
  if (label.startsWith("2 ")) return "#D9E1F2";
  if (label.startsWith("3-")) return "#FCE4D6";
  if (label.startsWith("4-")) return "#F4CCCC";
  if (label.startsWith("5 ")) return "#C6E0B4";
  if (label.startsWith("6 ")) return "#B7DEE8";
  return "#FFE699";
}

function nearIndex(values, target, tolerance, used = null) {
  let best = -1;
  let bestDiff = Infinity;
  values.forEach((value, idx) => {
    if (used?.has(idx)) return;
    const diff = Math.abs(value - target);
    if (diff <= tolerance && diff < bestDiff) {
      best = idx;
      bestDiff = diff;
    }
  });
  return best;
}

function buildMarkerEvents(record) {
  const radarAxis = record.candidate_radar.length > 0 || record.raw8_radar.length > 0;
  const candidateValues = radarAxis ? record.candidate_radar : record.candidate_biopac;
  const rawValues = radarAxis ? record.raw8_radar : record.raw8_biopac;
  const targetValues = record.selected_count > 0
    ? (record.selected_slots ?? record.selected_markers)
    : record.recommended_markers;
  const events = candidateValues.map((value) => ({ candidate: value, raw: null, targets: [], alternatives: [] }));

  const usedForRaw = new Set();
  rawValues.forEach((value) => {
    const candidates = events.map((event) => event.candidate ?? event.raw ?? value);
    const idx = nearIndex(candidates, value, 4.0, usedForRaw);
    if (idx >= 0 && events[idx].candidate != null) {
      events[idx].raw = value;
      usedForRaw.add(idx);
    } else {
      events.push({ candidate: null, raw: value, targets: [], alternatives: [] });
    }
  });

  targetValues.forEach((value, markerIdx) => {
    if (value == null) return;
    const representatives = events.map((event) => event.candidate ?? event.raw);
    let idx = nearIndex(representatives, value, 2.0);
    if (idx < 0) {
      idx = events.findIndex((event) => event.targets.some((target) => Math.abs(target.value - value) <= 0.01));
    }
    if (idx < 0) {
      events.push({ candidate: null, raw: null, targets: [], alternatives: [] });
      idx = events.length - 1;
    }
    events[idx].targets.push({ markerIdx, value });
  });

  const subjectBoundaries = data.boundary_rows.filter((row) => row.number === record.number);
  subjectBoundaries.forEach((boundary) => {
    [boundary.candidate_2_s, boundary.candidate_3_s].forEach((value) => {
      if (value == null) return;
      const representatives = events.map((event) => event.candidate ?? event.raw ?? event.targets[0]?.value);
      const idx = nearIndex(representatives, value, 2.0);
      if (idx >= 0) events[idx].alternatives.push({ boundaryIndex: boundary.boundary_index, value, selected: boundary.selected_or_recommended_s });
    });
  });

  events.forEach((event) => {
    event.sortValue = event.candidate ?? event.raw ?? event.targets[0]?.value ?? 0;
  });
  events.sort((a, b) => a.sortValue - b.sortValue);
  return { events, radarAxis, targetValues };
}

// 4) Selected/recommended marker-number map, split by protocol version.
{
  const sheet = workbook.worksheets.add("선택 마커 번호표");
  titleBand(sheet, "선택 마커 번호표", "M01, M02… 번호를 시나리오 시작·종료 경계에 직접 연결했습니다. 초록=현재 선택, 파랑=시간패턴 추천(미확정), 빈칸=선택 불가/경계 누락.", "Z");

  function writeSection(startRow, expected, records, sectionTitle) {
    const endCol = colName(4 + expected);
    sheet.mergeCells(`A${startRow}:${endCol}${startRow}`);
    sheet.getRange(`A${startRow}`).values = [[sectionTitle]];
    sheet.getRange(`A${startRow}:${endCol}${startRow}`).format = { fill: BLUE, font: { bold: true, color: WHITE, size: 12 }, verticalAlignment: "center" };
    sheet.getRange(`A${startRow}`).format.rowHeight = 24;
    const ids = Array.from({ length: expected }, (_, i) => `M${String(i + 1).padStart(2, "0")}`);
    const labels = markerLabels(expected);
    sheet.getRange(`A${startRow + 1}:D${startRow + 2}`).values = [["번호", "참가자", "값 출처", "판정"], [null, null, null, null]];
    sheet.getRange(`E${startRow + 1}:${endCol}${startRow + 1}`).values = [ids];
    sheet.getRange(`E${startRow + 2}:${endCol}${startRow + 2}`).values = [labels];
    styleHeader(sheet.getRange(`A${startRow + 1}:D${startRow + 2}`));
    sheet.getRange(`E${startRow + 1}:${endCol}${startRow + 1}`).format = { font: { bold: true, color: BLUE }, horizontalAlignment: "center", borders: { preset: "all", style: "thin", color: BORDER } };
    labels.forEach((label, i) => {
      const col = colName(5 + i);
      sheet.getRange(`${col}${startRow + 2}`).format = { fill: scenarioHeaderFill(label), font: { bold: true, color: TEXT }, horizontalAlignment: "center", wrapText: true, borders: { preset: "all", style: "thin", color: BORDER } };
    });
    const bodyRows = records.map((r) => {
      const hasSelected = r.selected_count > 0;
      const values = hasSelected ? (r.selected_slots ?? r.selected_markers) : r.recommended_markers;
      const padded = [...values, ...Array(Math.max(0, expected - values.length)).fill(null)].slice(0, expected);
      const source = hasSelected ? r.selected_source : (r.recommended_count ? "시간패턴 추천(미확정)" : "없음");
      return [r.number, r.subject, source, r.overall_status, ...padded];
    });
    const bodyStart = startRow + 3;
    const bodyEnd = bodyStart + bodyRows.length - 1;
    sheet.getRange(`A${bodyStart}:${endCol}${bodyEnd}`).values = bodyRows;
    sheet.getRange(`E${bodyStart}:${endCol}${bodyEnd}`).format.numberFormat = "0.0";
    sheet.getRange(`A${bodyStart}:${endCol}${bodyEnd}`).format = { borders: { insideHorizontal: { style: "thin", color: "#E2E8F0" } }, verticalAlignment: "center" };
    records.forEach((r, idx) => {
      const row = bodyStart + idx;
      applyStatusFill(sheet.getRange(`D${row}`), r.overall_status);
      const values = r.selected_count > 0 ? (r.selected_slots ?? r.selected_markers) : r.recommended_markers;
      values.forEach((value, markerIdx) => {
        if (value == null) return;
        const cell = sheet.getRange(`${colName(5 + markerIdx)}${row}`);
        cell.format.fill = r.selected_count > 0 ? GREEN : LIGHT_BLUE;
        cell.format.font = { bold: true, color: TEXT, italic: r.selected_count === 0 };
      });
    });
    return bodyEnd;
  }

  const early = data.records.filter((r) => r.number <= 3);
  const late = data.records.filter((r) => r.number >= 4);
  const earlyEnd = writeSection(4, 20, early, "초기 프로토콜 — S01~S03 / 20개 마커");
  writeSection(earlyEnd + 3, 22, late, "변경 프로토콜 — S04~S30 / 22개 마커");
  sheet.freezePanes.freezeColumns(4);
  [7, 14, 24, 20].forEach((w, i) => { sheet.getRange(`${colName(i + 1)}:${colName(i + 1)}`).format.columnWidth = w; });
  for (let i = 0; i < 22; i++) sheet.getRange(`${colName(5 + i)}:${colName(5 + i)}`).format.columnWidth = 11;
}

// 5) Combined raw/candidate/selection view with scenario colors and hover comments.
{
  const prepared = data.records.map((record) => ({ record, ...buildMarkerEvents(record) }));
  const maxEvents = Math.max(...prepared.map((item) => item.events.length));
  const endCol = colName(7 + maxEvents);
  const sheet = workbook.worksheets.add("전체 마커 선택표");
  titleBand(
    sheet,
    "전체 원시·후보·선택 마커 통합표",
    "참가자마다 RAW 8.5V → 전수 후보 → 선택값 → M번호/시나리오 순서입니다. 미선택 검출값은 회색, 선택값은 시나리오별 색으로 표시하며 파랑·빨강 글씨 셀에는 마우스를 올리면 검토 메모가 보입니다.",
    "Z",
  );
  workbook.comments.setSelf({ displayName: "User" });

  const legendLabels = ["1번 호흡", "2번 호흡", "3번 픽업", "4번 낙상", "5번 코스", "6번 왕복", "7번 자유"];
  sheet.getRange("A4").values = [["시나리오 색"]];
  sheet.getRange("A4").format = { fill: BLUE, font: { bold: true, color: WHITE }, horizontalAlignment: "center" };
  legendLabels.forEach((label, idx) => {
    const col = colName(2 + idx);
    sheet.getRange(`${col}4`).values = [[label]];
    sheet.getRange(`${col}4`).format = {
      fill: scenarioSelectedFill(`${idx + 1} `),
      font: { bold: true, color: "#000000" },
      horizontalAlignment: "center",
      borders: { preset: "all", style: "thin", color: BORDER },
    };
  });
  sheet.getRange("J4").values = [["회색"]];
  sheet.getRange("K4").values = [["미선택 검출값"]];
  sheet.getRange("J4:K4").format = { fill: GREY, font: { color: "#000000" }, horizontalAlignment: "center", borders: { preset: "all", style: "thin", color: BORDER } };
  sheet.getRange("A5").values = [["글자 색"]];
  sheet.getRange("A5").format = { fill: BLUE, font: { bold: true, color: WHITE }, horizontalAlignment: "center" };
  sheet.getRange("B5:C5").values = [["검정", "예상 오차범위 내"]];
  sheet.getRange("D5:E5").values = [["파랑", "대체 후보/RAW 미일치"]];
  sheet.getRange("F5:G5").values = [["빨강", "시간 이탈·미확정·수동 불일치"]];
  sheet.getRange("B5:C5").format = { font: { color: "#000000" }, borders: { preset: "all", style: "thin", color: BORDER } };
  sheet.getRange("D5:E5").format = { font: { color: "#0070C0", bold: true }, borders: { preset: "all", style: "thin", color: BORDER } };
  sheet.getRange("F5:G5").format = { font: { color: "#C00000", bold: true }, borders: { preset: "all", style: "thin", color: BORDER } };

  const headers = ["번호", "참가자", "행 종류", "시간축", "기대 수", "선택/추천 수", "판정", ...Array.from({ length: maxEvents }, (_, idx) => `E${String(idx + 1).padStart(2, "0")}`)];
  sheet.getRange(`A7:${endCol}7`).values = [headers];
  styleHeader(sheet.getRange(`A7:${endCol}7`));

  const allRows = [];
  prepared.forEach(({ record, events, radarAxis }) => {
    const selectedCount = record.selected_count || record.recommended_count;
    const axis = radarAxis ? "레이더(s)" : "BIOPAC(s)";
    const pad = (values) => [...values, ...Array(maxEvents - values.length).fill(null)];
    const rawRow = events.map((event) => event.raw);
    const candidateRow = events.map((event) => event.candidate);
    const selectedRow = events.map((event) => {
      if (!event.targets.length) return null;
      const unique = [...new Set(event.targets.map((target) => Number(target.value.toFixed(3))))];
      return unique.length === 1 ? unique[0] : unique.join(" / ");
    });
    const labelRow = events.map((event) => event.targets.map((target) => {
      const markerId = `M${String(target.markerIdx + 1).padStart(2, "0")}`;
      return `${markerId} ${boundaryLabel(record.expected_count, target.markerIdx + 1)}`;
    }).join(" / ") || null);
    allRows.push(
      [record.number, record.subject, "RAW 8.5V", axis, record.expected_count, selectedCount, record.overall_status, ...pad(rawRow)],
      [null, null, "전수 후보", axis, null, null, null, ...pad(candidateRow)],
      [null, null, record.selected_count ? "선택값" : "추천값(미확정)", axis, null, null, null, ...pad(selectedRow)],
      [null, null, "M번호 / 시나리오", axis, null, null, null, ...pad(labelRow)],
    );
  });
  const bodyStart = 8;
  const bodyEnd = bodyStart + allRows.length - 1;
  sheet.getRange(`A${bodyStart}:${endCol}${bodyEnd}`).values = allRows;
  sheet.getRange(`H${bodyStart}:${endCol}${bodyEnd}`).format = {
    fill: GREY,
    font: { color: "#000000", size: 9 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#D9D9D9" },
    numberFormat: "0.0",
  };
  sheet.getRange(`A${bodyStart}:G${bodyEnd}`).format = {
    font: { color: TEXT, size: 9 },
    verticalAlignment: "center",
    wrapText: true,
    borders: { insideHorizontal: { style: "thin", color: "#E2E8F0" } },
  };

  const boundaryLookup = new Map(data.boundary_rows.map((row) => [`${row.number}:${row.boundary_index}`, row]));
  const pairLookup = new Map(data.pair_rows.map((row) => [`${row.number}:${row.pair_index}`, row]));

  prepared.forEach(({ record, events }, participantIdx) => {
    const rawRow = bodyStart + participantIdx * 4;
    const candidateRow = rawRow + 1;
    const selectedRow = rawRow + 2;
    const labelRow = rawRow + 3;
    applyStatusFill(sheet.getRange(`G${rawRow}`), record.overall_status);
    sheet.getRange(`A${rawRow}:G${rawRow}`).format.font = { bold: true, color: TEXT, size: 9 };
    sheet.getRange(`A${labelRow}:${endCol}${labelRow}`).format.borders = { bottom: { style: "medium", color: BORDER } };

    events.forEach((event, eventIdx) => {
      const col = colName(8 + eventIdx);
      if (event.targets.length) {
        const firstLabel = boundaryLabel(record.expected_count, event.targets[0].markerIdx + 1);
        const fill = scenarioSelectedFill(firstLabel);
        sheet.getRange(`${col}${rawRow}:${col}${labelRow}`).format.fill = fill;

        const reasons = new Set();
        let fontColor = "#000000";
        if (record.selected_count === 0) reasons.add("시간패턴 추천값으로 아직 확정되지 않음");
        if (event.raw == null && event.candidate == null) reasons.add("선택값이 RAW 8.5V 및 전수 후보와 ±2초 이내에서 일치하지 않음");
        else if (event.raw == null) reasons.add("전수 후보에는 있으나 RAW 8.5V에는 일치값이 없음");

        const detailLines = [];
        event.targets.forEach((target) => {
          const boundaryIndex = target.markerIdx + 1;
          const boundary = boundaryLookup.get(`${record.number}:${boundaryIndex}`);
          const pair = pairLookup.get(`${record.number}:${Math.floor(target.markerIdx / 2) + 1}`);
          if (boundary?.ambiguous) reasons.add("같은 경계에 가까운 대체 후보가 둘 이상 있음");
          if (pair && pair.duration_status !== "시간 적합") reasons.add(`${pair.scenario} 지속시간이 예상 범위를 벗어나거나 경계가 누락됨`);
          const candidates = boundary
            ? [boundary.candidate_1_s, boundary.candidate_2_s, boundary.candidate_3_s].filter((value) => value != null).map((value) => Number(value).toFixed(1)).join(", ")
            : "없음";
          detailLines.push(
            `${`M${String(boundaryIndex).padStart(2, "0")}`} ${boundaryLabel(record.expected_count, boundaryIndex)}: ${Number(target.value).toFixed(1)} s`,
            `가까운 후보: ${candidates || "없음"}`,
            pair ? `${pair.scenario}: ${pair.duration_s == null ? "계산 불가" : `${Number(pair.duration_s).toFixed(1)} s`} / 예상 ${pair.min_s}~${pair.max_s} s / ${pair.duration_status}` : "시간 검증 자료 없음",
          );
        });

        if (record.selected_count === 0 || [...reasons].some((reason) => reason.includes("벗어나") || reason.includes("일치하지") || reason.includes("확정되지"))) fontColor = "#C00000";
        else if (reasons.size > 0) fontColor = "#0070C0";
        sheet.getRange(`${col}${rawRow}:${col}${labelRow}`).format.font = { color: fontColor, bold: true, italic: record.selected_count === 0, size: 9 };

        if (reasons.size > 0) {
          const note = [
            `검토 구분: ${[...reasons].join(" / ")}`,
            `참가자: ${record.subject}`,
            `값 출처: ${record.selected_count ? record.selected_source : "시간패턴 추천(미확정)"}`,
            `RAW 8.5V 일치: ${event.raw == null ? "없음" : `${Number(event.raw).toFixed(1)} s`}`,
            `전수 후보 일치: ${event.candidate == null ? "없음" : `${Number(event.candidate).toFixed(1)} s`}`,
            ...detailLines,
            record.note ? `참가자 특이사항: ${record.note}` : null,
          ].filter(Boolean).join("\n");
          workbook.comments.addThread({ cell: sheet.getRange(`${col}${selectedRow}`) }, note);
        }
      }

      if (event.alternatives.length) {
        const targetCellRow = event.candidate != null ? candidateRow : rawRow;
        sheet.getRange(`${col}${targetCellRow}`).format.font = { color: "#0070C0", bold: true, size: 9 };
        const note = event.alternatives.map((alternative) => {
          const markerId = `M${String(alternative.boundaryIndex).padStart(2, "0")}`;
          return `${markerId} ${boundaryLabel(record.expected_count, alternative.boundaryIndex)}의 대체 후보 ${Number(alternative.value).toFixed(1)} s (현재 선택/추천 ${Number(alternative.selected).toFixed(1)} s)`;
        }).join("\n");
        workbook.comments.addThread({ cell: sheet.getRange(`${col}${targetCellRow}`) }, note);
      }
    });
  });

  sheet.freezePanes.freezeRows(7);
  sheet.freezePanes.freezeColumns(7);
  [7, 14, 18, 12, 9, 13, 22].forEach((width, idx) => { sheet.getRange(`${colName(idx + 1)}:${colName(idx + 1)}`).format.columnWidth = width; });
  for (let idx = 0; idx < maxEvents; idx++) sheet.getRange(`${colName(8 + idx)}:${colName(8 + idx)}`).format.columnWidth = 13;
  sheet.getRange("4:5").format.rowHeight = 22;
  sheet.getRange("7:7").format.rowHeight = 28;
  for (let row = bodyStart; row <= bodyEnd; row += 4) {
    sheet.getRange(`${row}:${row + 2}`).format.rowHeight = 20;
    sheet.getRange(`${row + 3}:${row + 3}`).format.rowHeight = 34;
  }
}

// 6) One row per expected boundary with top candidate group.
{
  const sheet = workbook.worksheets.add("경계별 후보군");
  titleBand(sheet, "경계별 후보군", "선택값 또는 시간패턴 추천값을 기준으로 ±15초 내 가까운 후보를 최대 3개 표시합니다. 후보가 둘 이상이면 ‘애매’로 표시합니다.", "L");
  const headers = ["번호", "참가자", "기대 수", "마커 번호", "경계 의미", "선택/추천(s)", "출처", "후보1(s)", "후보2(s)", "후보3(s)", "애매 여부", "확인 메모"];
  sheet.getRange("A4:L4").values = [headers];
  styleHeader(sheet.getRange("A4:L4"));
  const rows = data.boundary_rows.map((r) => [r.number, r.subject, r.expected_count, `M${String(r.boundary_index).padStart(2, "0")}`, boundaryLabel(r.expected_count, r.boundary_index), r.selected_or_recommended_s, r.source, r.candidate_1_s, r.candidate_2_s, r.candidate_3_s, r.ambiguous ? "애매" : "단일/없음", ""]);
  sheet.getRange(`A5:L${4 + rows.length}`).values = rows;
  sheet.getRange(`F5:J${4 + rows.length}`).format.numberFormat = "0.0";
  sheet.getRange(`A5:L${4 + rows.length}`).format = { borders: { insideHorizontal: { style: "thin", color: "#E2E8F0" } }, verticalAlignment: "center" };
  data.boundary_rows.forEach((r, i) => {
    const row = 5 + i;
    if (r.source.includes("추천")) sheet.getRange(`F${row}`).format.fill = LIGHT_BLUE;
    else sheet.getRange(`F${row}`).format.fill = GREEN;
    if (r.ambiguous) sheet.getRange(`K${row}`).format.fill = YELLOW;
    if (r.candidate_2_s != null) sheet.getRange(`I${row}`).format.fill = YELLOW;
    if (r.candidate_3_s != null) sheet.getRange(`J${row}`).format.fill = YELLOW;
  });
  sheet.freezePanes.freezeRows(4);
  sheet.freezePanes.freezeColumns(2);
  [7, 14, 10, 11, 18, 14, 24, 12, 12, 12, 12, 30].forEach((w, i) => { sheet.getRange(`${colName(i + 1)}:${colName(i + 1)}`).format.columnWidth = w; });
}

// 7) Pair-duration validation using formulas.
{
  const sheet = workbook.worksheets.add("시간 검증");
  titleBand(sheet, "시나리오 시간 검증", "선택된 마커 쌍 또는 미확정 추천 쌍의 지속시간을 PPT 기준 범위와 비교합니다. 시간 적합만으로 실제 마커임을 확정할 수는 없습니다.", "L");
  const headers = ["번호", "참가자", "쌍 번호", "시나리오", "시작(s)", "종료(s)", "지속시간(s)", "최소(s)", "최대(s)", "시간 판정", "PPT/판정 근거", "값 출처"];
  sheet.getRange("A4:L4").values = [headers];
  styleHeader(sheet.getRange("A4:L4"));
  const rows = data.pair_rows.map((r) => [r.number, r.subject, r.pair_index, r.scenario, r.start_s, r.end_s, null, r.min_s, r.max_s, null, r.basis, r.source]);
  sheet.getRange(`A5:L${4 + rows.length}`).values = rows;
  sheet.getRange(`G5:G${4 + rows.length}`).formulas = rows.map((_, i) => [`=IF(OR(E${5 + i}="",F${5 + i}=""),"",F${5 + i}-E${5 + i})`]);
  sheet.getRange(`J5:J${4 + rows.length}`).formulas = rows.map((_, i) => [`=IF(G${5 + i}="","경계 누락",IF(AND(G${5 + i}>=H${5 + i},G${5 + i}<=I${5 + i}),"시간 적합","시간 확인 필요"))`]);
  sheet.getRange(`E5:I${4 + rows.length}`).format.numberFormat = "0.0";
  sheet.getRange(`A5:L${4 + rows.length}`).format = { borders: { insideHorizontal: { style: "thin", color: "#E2E8F0" } }, verticalAlignment: "center" };
  data.pair_rows.forEach((r, i) => applyStatusFill(sheet.getRange(`J${5 + i}`), r.duration_status));
  sheet.freezePanes.freezeRows(4);
  sheet.freezePanes.freezeColumns(2);
  [7, 14, 10, 23, 12, 12, 14, 10, 10, 17, 36, 24].forEach((w, i) => { sheet.getRange(`${colName(i + 1)}:${colName(i + 1)}`).format.columnWidth = w; });
}

// 8) Audit rules and legend.
{
  const sheet = workbook.worksheets.add("판정 기준");
  titleBand(sheet, "전수검사 판정 기준 및 범례", "원시 데이터, 후보 데이터, 최종 선택을 분리해 재현성과 수동 검토 가능성을 높이기 위한 기록", "F");
  const rows = [
    ["구분", "값/규칙", "설명", "색상", "원본/근거", "주의"],
    ["프로토콜", "S01~S03: 20개", "낙상 3구간", "", "PPT Fall 3가지 상황", "실험 특이사항 우선"],
    ["프로토콜", "S04~S30: 22개", "낙상 물건 집기 추가", "", "Dataset_issue S04 메모", "S04부터 유지로 기록"],
    ["원시 검출", "8.5V", "MATLAB 고정 임계값, 4초 이내 병합", "", "sync_validate_batch.m", "실제 누름 횟수와 동일하지 않을 수 있음"],
    ["후보 검출", "참가자별 임계값", "기존 GPT 후보 선정에 사용한 임계값", "", "normalization_records.json", "RAW 8.5V와 별도 층"],
    ["마커 일치", "±2.0초", "현재 선택값과 후보값을 같은 이벤트로 연결", "초록", "전수검사 규칙", "정확한 파형 확인 필요"],
    ["대체 후보", "±15초", "같은 경계에서 검토할 가까운 후보", "노랑", "전수검사 규칙", "후보2·3은 자동 확정 아님"],
    ["시간패턴 추천", "20/22 경계 템플릿", "선택값이 없는 행에만 추천", "파랑", "완성된 참가자의 중앙 시간패턴", "반드시 MATLAB/영상으로 확인"],
    ["미선택", "후보로만 존재", "현재 선택 또는 추천에 연결되지 않음", "회색", "후보 검출 결과", "오검출이라고 단정하지 않음"],
    ["시간 검증", "1번 호흡", "150~240초", "", "PPT 총 3분", "준비·종료 동작 포함 가능"],
    ["시간 검증", "2번 호흡", "240~450초", "", "PPT 표기 4분30초/세부 합계 약 6분", "PPT 내부 불일치"],
    ["시간 검증", "픽업 각 코스", "5~90초", "", "총시간 미기재", "느슨한 확인 범위"],
    ["시간 검증", "낙상 즉시", "5~30초", "", "즉시 일어나기", "이동 포함"],
    ["시간 검증", "낙상 유지", "25~75초", "", "30초 유지", "이동 포함"],
    ["시간 검증", "낙상 천천히", "25~90초", "", "천천히 눕기+30초", "이동 포함"],
    ["시간 검증", "낙상 물건 집기", "5~40초", "", "S04부터 추가", "총시간 미기재"],
    ["시간 검증", "5번 코스", "130~230초", "", "16칸×10초", "이동 포함"],
    ["시간 검증", "6·7번", "5~70초", "", "각 1분 미만", "시작·종료 동작 포함"],
    ["출처", "MATLAB RAW", "$2/revalidation/matlab_raw/matlab_results.json", "", "로컬 파일", ""],
    ["출처", "최종 선택", "$2/sync_results/final_sync_records.json", "", "로컬 파일", ""],
    ["출처", "실험 가이드", "HAI_Experiment_UWB_GuideLine.pptx", "", "로컬 파일", ""],
  ];
  sheet.getRange(`A4:F${3 + rows.length}`).values = rows;
  styleHeader(sheet.getRange("A4:F4"));
  sheet.getRange(`A5:F${3 + rows.length}`).format = { wrapText: true, verticalAlignment: "top", borders: { insideHorizontal: { style: "thin", color: "#D9E2F3" } } };
  const colorMap = { 6: GREEN, 7: YELLOW, 8: LIGHT_BLUE, 9: GREY };
  for (const [row, fill] of Object.entries(colorMap)) sheet.getRange(`D${Number(row) + 3}`).format.fill = fill;
  [16, 24, 42, 12, 32, 38].forEach((w, i) => { sheet.getRange(`${colName(i + 1)}:${colName(i + 1)}`).format.columnWidth = w; });
  sheet.freezePanes.freezeRows(4);
}

// Compact verification before export.
const summaryInspect = await workbook.inspect({ kind: "table", range: "전수검사 요약!A1:Q16", include: "values,formulas", tableMaxRows: 16, tableMaxCols: 17, maxChars: 10000 });
console.log(summaryInspect.ndjson);
const combinedInspect = await workbook.inspect({ kind: "table", range: "전체 마커 선택표!A1:Z19", include: "values,formulas", tableMaxRows: 19, tableMaxCols: 26, maxChars: 12000 });
console.log(combinedInspect.ndjson);
const errorInspect = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: "formula error scan" });
console.log(errorInspect.ndjson);

await fs.mkdir(OUTDIR, { recursive: true });
const previews = [
  ["Sheet1", "A1:Q12", "01_original.png"],
  ["MATLAB 재검증", "A1:J18", "02_matlab.png"],
  ["전수검사 요약", "A1:Q18", "03_summary.png"],
  ["RAW 8.5V", "A1:T16", "04_raw.png"],
  ["후보 마커", "A1:T16", "05_candidates.png"],
  ["선택 마커 번호표", "A1:Z22", "06_marker_map.png"],
  ["전체 마커 선택표", "A1:Z27", "07_combined.png"],
  ["경계별 후보군", "A1:L24", "08_boundaries.png"],
  ["시간 검증", "A1:L24", "09_time.png"],
  ["판정 기준", "A1:F24", "10_rules.png"],
];
for (const [sheetName, range, fileName] of previews) {
  const blob = await workbook.render({ sheetName, range, scale: 1.2, format: "png" });
  await fs.writeFile(path.join(HERE, fileName), new Uint8Array(await blob.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(OUTPUT);
console.log(JSON.stringify({ output: OUTPUT, sheets: workbook.worksheets.items.map((s) => s.name) }));
