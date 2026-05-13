from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import correlate, correlation_lags


def zscore(x: np.ndarray, axis: int | None = None, eps: float = 1e-8) -> np.ndarray:
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    return (x - mean) / (std + eps)


def best_field_bin(sync: object, fields: list[str]) -> dict[str, object]:
    fs_uwb = float(np.asarray(getattr(sync, "Fs_uwb")).squeeze())
    fs_biopac = float(np.asarray(getattr(sync, "Fs_biopac")).squeeze())
    ref = np.asarray(getattr(sync, "biopac_resp"), dtype=np.float64).squeeze()
    t_ref = np.arange(ref.size) / fs_biopac
    best: dict[str, object] | None = None

    for field in fields:
        if not hasattr(sync, field):
            continue
        radar = np.asarray(getattr(sync, field), dtype=np.float64)
        if radar.ndim != 2 or radar.shape[1] < 50:
            continue
        t_uwb = np.arange(radar.shape[1]) / fs_uwb
        ref_interp = np.interp(t_uwb, t_ref, ref)
        corr = np.mean(zscore(radar, axis=1) * zscore(ref_interp).reshape(1, -1), axis=1)
        bin_index = int(np.argmax(np.abs(corr)))
        row = {
            "field": field,
            "bin": bin_index,
            "corr": float(corr[bin_index]),
            "abs_corr": float(abs(corr[bin_index])),
            "duration_sec": float(min(radar.shape[1] / fs_uwb, ref.size / fs_biopac)),
            "uwb_samples": int(radar.shape[1]),
            "ref_samples": int(ref.size),
            "fs_uwb": fs_uwb,
            "fs_biopac": fs_biopac,
        }
        if best is None or float(row["abs_corr"]) > float(best["abs_corr"]):
            best = row

    if best is None:
        raise ValueError("No valid radar field found")
    return best


def lag_check(sync: object, field: str, bin_index: int, max_lag_sec: float) -> dict[str, float]:
    fs_uwb = float(np.asarray(getattr(sync, "Fs_uwb")).squeeze())
    fs_biopac = float(np.asarray(getattr(sync, "Fs_biopac")).squeeze())
    ref = np.asarray(getattr(sync, "biopac_resp"), dtype=np.float64).squeeze()
    radar = np.asarray(getattr(sync, field), dtype=np.float64)[bin_index]

    t_ref = np.arange(ref.size) / fs_biopac
    ref_interp = np.interp(np.arange(radar.size) / fs_uwb, t_ref, ref)
    x = zscore(radar)
    y = zscore(ref_interp)

    corr_full = correlate(x, y, mode="full")
    lags = correlation_lags(x.size, y.size, mode="full")
    max_lag = int(round(max_lag_sec * fs_uwb))
    mask = (lags >= -max_lag) & (lags <= max_lag)
    idx = int(np.argmax(np.abs(corr_full[mask])))
    lag_samples = int(lags[mask][idx])
    lag_corr = float(corr_full[mask][idx] / x.size)
    return {
        "best_lag_samples": lag_samples,
        "best_lag_sec": float(lag_samples / fs_uwb),
        "lag_corr": lag_corr,
        "lag_abs_corr": abs(lag_corr),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Folder containing UWB_Biopac_SyncData.mat files")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-lag-sec", type=float, default=30.0)
    parser.add_argument("--top-lag-check", type=int, default=10)
    parser.add_argument("--fields", nargs="+", default=["com_row", "tv_row", "com_col", "tv_col"])
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)
    mat_files = sorted(root.rglob("UWB_Biopac_SyncData.mat"), key=lambda p: str(p))

    rows: list[dict[str, object]] = []
    for mat_path in mat_files:
        rel = str(mat_path.relative_to(root))
        try:
            mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            sync = mat["UWB_Biopac_SyncData"]
            row = best_field_bin(sync, args.fields)
            row["file"] = rel
            rows.append(row)
        except Exception as exc:
            rows.append({"file": rel, "error": str(exc)})

    valid = [r for r in rows if "abs_corr" in r]
    valid.sort(key=lambda r: float(r["abs_corr"]), reverse=True)

    for row in valid[: args.top_lag_check]:
        mat = loadmat(root / str(row["file"]), squeeze_me=True, struct_as_record=False)
        sync = mat["UWB_Biopac_SyncData"]
        row.update(lag_check(sync, str(row["field"]), int(row["bin"]), args.max_lag_sec))

    json_path = out_dir / "syncdata_quality_scan.json"
    csv_path = out_dir / "syncdata_quality_scan.csv"
    json_path.write_text(json.dumps(valid, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = sorted({key for row in valid for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(valid)

    print(f"scanned files: {len(mat_files)}")
    print("top results:")
    for row in valid[:10]:
        print(json.dumps(row, ensure_ascii=False))
    print(f"saved json: {json_path}")
    print(f"saved csv: {csv_path}")


if __name__ == "__main__":
    main()
