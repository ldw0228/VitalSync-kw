from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spike_encoding import delta_spike_encode, zscore

try:
    import torch
    from torch.utils.data import Dataset
except OSError:
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


@dataclass
class MobiVitalArrays:
    imu: np.ndarray
    uwb_i: np.ndarray
    uwb_q: np.ndarray
    magnitude: np.ndarray
    phase: np.ndarray
    respiration: np.ndarray
    pulse: np.ndarray
    fs: int = 50


def load_mobivital_csv(csv_path: str | Path) -> MobiVitalArrays:
    data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
    imu = data[:, 0:6]
    uwb_i = data[:, 12:132]
    uwb_q = data[:, 132:252]
    magnitude = np.sqrt(uwb_i**2 + uwb_q**2)
    phase = np.unwrap(np.arctan2(uwb_q, uwb_i), axis=0)
    respiration = data[:, 252]
    pulse = data[:, 253]
    return MobiVitalArrays(
        imu=imu,
        uwb_i=uwb_i,
        uwb_q=uwb_q,
        magnitude=magnitude,
        phase=phase,
        respiration=respiration,
        pulse=pulse,
    )


def best_bin_by_corr(feature: np.ndarray, target: np.ndarray) -> tuple[int, np.ndarray]:
    """Return 0-based best range-bin index by absolute correlation."""
    feature_z = zscore(feature, axis=0)
    target_z = zscore(target, axis=0).reshape(-1, 1)
    corr = np.mean(feature_z * target_z, axis=0)
    return int(np.argmax(np.abs(corr))), corr


class MobiVitalWindowDataset(Dataset):
    """Windowed UWB dataset for baseline and SNN experiments.

    Output:
        x: tensor shaped [channels, time]
        y: tensor shaped [1, time], z-scored respiration waveform
    """

    def __init__(
        self,
        csv_paths: list[str | Path],
        window_sec: float = 10.0,
        stride_sec: float = 2.0,
        feature: str = "magnitude",
        bin_index: int | None = None,
        bin_radius: int = 0,
        encode: str = "delta",
        threshold_scale: float = 0.75,
    ) -> None:
        self.fs = 50
        self.window = int(window_sec * self.fs)
        self.stride = int(stride_sec * self.fs)
        self.feature = feature
        self.bin_index = bin_index
        self.bin_radius = bin_radius
        self.encode = encode
        self.threshold_scale = threshold_scale
        self.items: list[tuple[np.ndarray, np.ndarray]] = []

        for csv_path in csv_paths:
            arr = load_mobivital_csv(csv_path)
            feat = getattr(arr, feature)
            target = zscore(arr.respiration).astype(np.float32)
            use_bin = bin_index
            if use_bin is None:
                use_bin, _ = best_bin_by_corr(feat, target)
            lo = max(0, use_bin - bin_radius)
            hi = min(feat.shape[1], use_bin + bin_radius + 1)
            selected = zscore(feat[:, lo:hi], axis=0).astype(np.float32)

            for start in range(0, selected.shape[0] - self.window + 1, self.stride):
                end = start + self.window
                # Shape feature channels as [bins, time].
                x_win = selected[start:end].T
                y_win = target[start:end]
                if encode == "delta":
                    x_encoded = delta_spike_encode(
                        x_win,
                        threshold_scale=threshold_scale,
                        bipolar=True,
                    )
                    x_encoded = x_encoded.reshape(-1, x_encoded.shape[-1])
                elif encode == "none":
                    x_encoded = x_win
                else:
                    raise ValueError(f"Unknown encode mode: {encode}")
                self.items.append((x_encoded.astype(np.float32), y_win[None, :]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if torch is None:
            raise RuntimeError("PyTorch is required for tensor output.")
        x, y = self.items[idx]
        return torch.from_numpy(x), torch.from_numpy(y)
