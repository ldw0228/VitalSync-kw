from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, detrend, sosfiltfilt


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=r"C:\Users\hai\Desktop\uwb_aal_raw",
        help="Local folder containing sample_condition/",
    )
    args = parser.parse_args()

    root = Path(args.root)
    sample = root / "sample_condition"
    out = root / "outputs"
    out.mkdir(exist_ok=True)

    raw_file = sample / "DeltaR=10cm_Angle=0_Band1_Supine_Trial1.mat"
    cal_file = sample / "Calibration_Band1_DeltaR=10cm.mat"
    bs_file = sample / "BS_DeltaR=10cm_Angle=0_Band1_Supine_Trial1.mat"
    radar_breath_file = sample / "FilteredBreathRadar_DeltaR=10cm_Angle=0_Band1_Supine_Trial1.mat"
    ref_file = sample / "Ref_DeltaR=10cm_Angle=0__Band1_Supine_Trial1.mat"
    filtered_ref_file = sample / "FilteredBreath_Ref_DeltaR=10cm_Angle=0__Band1_Supine_Trial1.mat"

    sample_n = 256
    range_res = 0.0039
    time_res_radar = 0.0533
    fs_radar = 1 / time_res_radar
    time_res_ref = 0.325
    fs_ref = 1 / time_res_ref
    pre_range_index_matlab = 100
    pre_idx = pre_range_index_matlab - 1

    raw = loadmat(raw_file)["bScan"].astype(np.float64)
    cal = loadmat(cal_file)["bScan"].astype(np.float64)
    provided_bs = loadmat(bs_file)["BS"].astype(np.float64)
    provided_breath = loadmat(radar_breath_file)["BreathSignalRadar"].squeeze().astype(np.float64)
    ref = loadmat(ref_file)["Ref"].squeeze().astype(np.float64)
    filtered_ref = loadmat(filtered_ref_file)["Ref"].squeeze().astype(np.float64)

    # Equivalent to MATLAB: BS=detrend(bScan,1); BS=BS';
    bs = detrend(raw, axis=0, type="linear").T
    bs_corr = float(np.corrcoef(bs.reshape(-1), provided_bs.reshape(-1))[0, 1])

    # Match MATLAB 1-based indexing:
    # [~,RandeIndex]=max(BS(PreRangeIndx:SampleN,:));
    # EstimatedRangeIndex=fix(mean(RandeIndex))+PreRangeIndx;
    range_slice = bs[pre_idx:sample_n, :]
    range_index_relative_1_based = np.argmax(range_slice, axis=0) + 1
    estimated_range_index_matlab = int(np.fix(np.mean(range_index_relative_1_based)) + pre_range_index_matlab)
    estimated_range_index = estimated_range_index_matlab - 1
    estimated_range_m = estimated_range_index * range_res

    breath_raw = bs[estimated_range_index, :]
    sos = butter(4, 1.5, btype="lowpass", fs=fs_radar, output="sos")
    breath_filtered = sosfiltfilt(sos, breath_raw)
    breath_corr = float(np.corrcoef(breath_filtered, provided_breath)[0, 1])

    t_radar = np.arange(raw.shape[0]) * time_res_radar
    t_ref = np.arange(ref.shape[0]) * time_res_ref
    ref_interp = np.interp(t_radar, t_ref, filtered_ref)
    ref_corr_preview = float(np.corrcoef(zscore(breath_filtered), zscore(ref_interp))[0, 1])

    summary = {
        "dataset": "Radar Human Breathing Dataset for AAL and Search and Rescue",
        "condition": "DeltaR=10cm, Angle=0, Band1, Supine, Trial1",
        "raw_bscan_shape": list(raw.shape),
        "calibration_bscan_shape": list(cal.shape),
        "background_subtracted_shape": list(bs.shape),
        "provided_background_subtracted_shape": list(provided_bs.shape),
        "bs_reproduction_corr": bs_corr,
        "estimated_range_index_0_based": estimated_range_index,
        "estimated_range_m": estimated_range_m,
        "radar_fs_hz": fs_radar,
        "ref_fs_hz": fs_ref,
        "breath_signal_reproduction_corr": breath_corr,
        "radar_vs_reference_interp_corr_preview": ref_corr_preview,
    }
    (out / "aal_sample_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.imshow(raw.T, aspect="auto", origin="lower", cmap="gray")
    plt.axhline(estimated_range_index, color="r", linewidth=1)
    plt.title("Raw radar bScan (range x slow-time)")
    plt.ylabel("Range bin")

    plt.subplot(3, 1, 2)
    plt.imshow(bs, aspect="auto", origin="lower", cmap="gray")
    plt.axhline(estimated_range_index, color="r", linewidth=1)
    plt.title("Background-subtracted radar (linear detrend)")
    plt.ylabel("Range bin")

    plt.subplot(3, 1, 3)
    plt.plot(t_radar, zscore(breath_filtered), label="Radar breath signal (z)")
    plt.plot(t_radar, zscore(ref_interp), label="Lidar reference interp (z)", alpha=0.8)
    plt.title("Radar breath signal vs reference preview")
    plt.xlabel("Time (s)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out / "aal_raw_to_breath_preview.png", dpi=160)

    print(json.dumps(summary, indent=2))
    print(f"saved plot: {out / 'aal_raw_to_breath_preview.png'}")


if __name__ == "__main__":
    main()

