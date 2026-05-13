from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import butter, detrend, sosfiltfilt


def zscore_axis(x: np.ndarray, axis: int, eps: float = 1e-8) -> np.ndarray:
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    return (x - mean) / (std + eps)


def row_normalize(data: np.ndarray) -> np.ndarray:
    """Match ROW_DATA_VECTOR_NORMALIZATION.m for range x time matrices."""
    return zscore_axis(data, axis=1)


def column_normalize(data: np.ndarray) -> np.ndarray:
    """Match COLUMN_DATA_VECTOR_NORMALIZATION.m for range x time matrices."""
    return zscore_axis(data, axis=0)


def lowpass(x: np.ndarray, cutoff_hz: float, fs_hz: float, order: int = 4) -> np.ndarray:
    sos = butter(order, cutoff_hz, btype="lowpass", fs=fs_hz, output="sos")
    return sosfiltfilt(sos, x)


def build_syncdata(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    sample = root / "sample_condition"
    raw_file = sample / "DeltaR=10cm_Angle=0_Band1_Supine_Trial1.mat"
    filtered_ref_file = sample / "FilteredBreath_Ref_DeltaR=10cm_Angle=0__Band1_Supine_Trial1.mat"
    radar_breath_file = sample / "FilteredBreathRadar_DeltaR=10cm_Angle=0_Band1_Supine_Trial1.mat"

    range_res_m = 0.0039
    time_res_radar = 0.0533
    time_res_ref = 0.325
    fs_uwb = 1.0 / time_res_radar
    fs_reference = 1.0 / time_res_ref
    sample_n = 256
    pre_range_index_matlab = 100
    pre_idx = pre_range_index_matlab - 1

    raw = loadmat(raw_file)["bScan"].astype(np.float64)
    filtered_ref = loadmat(filtered_ref_file)["Ref"].squeeze().astype(np.float64)
    provided_breath = loadmat(radar_breath_file)["BreathSignalRadar"].squeeze().astype(np.float64)

    # AAL DataCode.m style: BS = detrend(bScan, 1); BS = BS';
    background_subtracted = detrend(raw, axis=0, type="linear").T

    range_slice = background_subtracted[pre_idx:sample_n, :]
    range_index_relative_1_based = np.argmax(range_slice, axis=0) + 1
    estimated_range_index_matlab = int(np.fix(np.mean(range_index_relative_1_based)) + pre_range_index_matlab)
    estimated_range_index = estimated_range_index_matlab - 1
    estimated_range_m = estimated_range_index * range_res_m

    radar_breath_raw = background_subtracted[estimated_range_index, :]
    radar_breath_filtered = lowpass(radar_breath_raw, cutoff_hz=1.5, fs_hz=fs_uwb)
    breath_reproduction_corr = float(np.corrcoef(radar_breath_filtered, provided_breath)[0, 1])

    uwb_row = row_normalize(background_subtracted)
    uwb_col = column_normalize(background_subtracted)

    sync_data = {
        # Field names intentionally mirror the graduate-student MATLAB output.
        "Fs_uwb": np.array([[fs_uwb]], dtype=np.float64),
        "Fs_biopac": np.array([[fs_reference]], dtype=np.float64),
        "biopac_resp": filtered_ref.reshape(-1, 1),
        "radar_resp": radar_breath_filtered.reshape(-1, 1),
        "Fs_radar_resp": np.array([[fs_uwb]], dtype=np.float64),
        # AAL has one radar, so com_* is the real radar and tv_* is a compatibility alias.
        "com_row": uwb_row,
        "com_col": uwb_col,
        "tv_row": uwb_row.copy(),
        "tv_col": uwb_col.copy(),
    }

    metadata = {
        "source_dataset": "Radar Human Breathing Dataset for AAL and Search and Rescue",
        "source_role": "AAL reference signal is lidar/filtered reference, mapped to biopac_resp for compatibility.",
        "condition": "DeltaR=10cm, Angle=0, Band1, Supine, Trial1",
        "single_radar_alias": "com_* contains AAL radar; tv_* duplicates com_* for UWB_Biopac_SyncData compatibility.",
        "raw_bscan_shape": list(raw.shape),
        "sync_matrix_shape_range_by_time": list(uwb_row.shape),
        "reference_shape": list(filtered_ref.shape),
        "radar_resp_shape": list(radar_breath_filtered.shape),
        "fs_uwb_hz": fs_uwb,
        "fs_reference_hz": fs_reference,
        "fs_radar_resp_hz": fs_uwb,
        "selected_range_index_0_based": estimated_range_index,
        "estimated_range_m": estimated_range_m,
        "radar_breath_reproduction_corr": breath_reproduction_corr,
    }
    return sync_data, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"C:\Users\hai\Desktop\uwb_aal_raw")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "syncdata_like"
    out_dir.mkdir(parents=True, exist_ok=True)

    sync_data, metadata = build_syncdata(root)
    mat_path = out_dir / "AAL_UWB_Biopac_SyncData_like.mat"
    json_path = out_dir / "AAL_UWB_Biopac_SyncData_like_summary.json"

    savemat(
        mat_path,
        {
            "UWB_Biopac_SyncData": sync_data,
            "AAL_Metadata": metadata,
        },
    )
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    print(f"saved mat: {mat_path}")
    print(f"saved summary: {json_path}")


if __name__ == "__main__":
    main()
