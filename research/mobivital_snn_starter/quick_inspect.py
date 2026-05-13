from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mobivital_dataset import best_bin_by_corr, load_mobivital_csv
from spike_encoding import delta_spike_encode, zscore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to a MobiVital CSV file")
    parser.add_argument("--out", default=None, help="Optional plot output path")
    parser.add_argument("--threshold-scale", type=float, default=0.75)
    args = parser.parse_args()

    arr = load_mobivital_csv(args.csv)
    best_bin, corr = best_bin_by_corr(arr.magnitude, arr.respiration)
    signal = zscore(arr.magnitude[:, best_bin])
    target = zscore(arr.respiration)
    spikes = delta_spike_encode(
        signal,
        threshold_scale=args.threshold_scale,
        bipolar=True,
    )
    spike_rate = float(spikes.sum() / spikes.size)

    print(f"shape: rows={arr.magnitude.shape[0]}, bins={arr.magnitude.shape[1]}")
    print(f"best magnitude bin: {best_bin + 1} (1-based)")
    print(f"distance estimate: {0.30 + best_bin * 0.0514:.4f} m")
    print(f"corr with respiration: {corr[best_bin]:.4f}")
    print(f"delta spike rate: {spike_rate:.4f}")

    t = np.arange(len(signal)) / arr.fs
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, signal, label="UWB magnitude z-score")
    plt.plot(t, target, label="respiration z-score", alpha=0.8)
    plt.legend(loc="upper right")
    plt.title("UWB magnitude vs respiration")

    plt.subplot(3, 1, 2)
    plt.imshow(arr.magnitude.T, aspect="auto", origin="lower")
    plt.axhline(best_bin, color="r", linewidth=1)
    plt.title("UWB magnitude map")
    plt.ylabel("range bin")

    plt.subplot(3, 1, 3)
    plt.eventplot(
        [
            np.where(spikes[0] > 0)[0] / arr.fs,
            np.where(spikes[1] > 0)[0] / arr.fs,
        ],
        colors=["tab:blue", "tab:orange"],
        lineoffsets=[1, 0],
    )
    plt.yticks([1, 0], ["positive", "negative"])
    plt.xlabel("time (s)")
    plt.title("Delta spikes")
    plt.tight_layout()

    out = args.out
    if out is None:
        out = str(Path(args.csv).with_name("quick_inspect.png"))
    plt.savefig(out, dpi=160)
    print(f"saved plot: {out}")


if __name__ == "__main__":
    main()

