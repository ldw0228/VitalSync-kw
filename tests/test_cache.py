import numpy as np
import pandas as pd
import pytest

from snn_rr.cache import (
    append_causal_history_features,
    causal_history_features,
)


def _history_metadata() -> pd.DataFrame:
    # Deliberately interleave sessions and scramble chronology.  Reference
    # columns are included to verify they have no influence on the result.
    return pd.DataFrame(
        {
            "session_id": ["A", "B", "A", "A", "B", "A"],
            "window_number": [2, 1, 0, 4, 0, 1],
            "classical_rr_bpm": [12.0, 101.0, 10.0, 14.0, 100.0, 11.0],
            "classical_confidence": [0.3, 0.2, 0.1, 0.5, 0.1, 0.2],
            "radar_peak_spread_bpm": [2.0, 11.0, 0.0, 4.0, 10.0, 1.0],
            "rr_bpm": [999.0, 999.0, 999.0, 999.0, 999.0, 999.0],
            "reference_valid": [True, False, True, False, True, False],
        }
    )


def test_causal_history_uses_exact_prior_windows_within_session():
    metadata = _history_metadata()
    features, names = causal_history_features(
        metadata, lags=(1, 2, 4), rolling_windows=(4,)
    )
    column = {name: index for index, name in enumerate(names)}

    # A/window 2 sees A/window 1 and A/window 0 despite the scrambled rows.
    row = 0
    assert features[row, column["history_lag_1_classical_rr_bpm"]] == 11.0
    assert features[row, column["history_lag_1_available"]] == 1.0
    assert features[row, column["history_lag_2_classical_rr_bpm"]] == 10.0
    assert features[row, column["history_lag_2_available"]] == 1.0
    assert features[row, column["history_lag_4_available"]] == 0.0

    # A/window 4 has no exact lag-1 row, but has exact lag-2 and lag-4 rows.
    row = 3
    assert features[row, column["history_lag_1_available"]] == 0.0
    assert features[row, column["history_lag_2_classical_rr_bpm"]] == 12.0
    assert features[row, column["history_lag_4_classical_rr_bpm"]] == 10.0
    assert features[row, column["history_roll_4_rr_median_bpm"]] == 11.0
    assert features[row, column["history_roll_4_rr_mad_bpm"]] == 1.0
    assert features[row, column["history_roll_4_rr_trend_bpm_per_window"]] == 1.0
    assert features[row, column["history_roll_4_available_fraction"]] == 0.75
    assert features[row, column["history_roll_4_sufficient"]] == 1.0

    # B/window 1 sees B/window 0, never a preceding row from session A.
    row = 1
    assert features[row, column["history_lag_1_classical_rr_bpm"]] == 100.0
    assert features[row, column["history_lag_1_radar_peak_spread_bpm"]] == 10.0


def test_causal_history_is_reference_label_independent_and_strictly_past():
    metadata = _history_metadata()
    first, names = causal_history_features(metadata)

    changed_labels = metadata.copy()
    changed_labels["rr_bpm"] = np.arange(len(metadata)) * -1234.0
    changed_labels["reference_valid"] = ~changed_labels["reference_valid"]
    second, second_names = causal_history_features(changed_labels)
    np.testing.assert_array_equal(first, second)
    assert names == second_names

    # Changing A/window 4 (a future row) cannot alter A/window 2 history.
    changed_future = metadata.copy()
    changed_future.loc[3, "classical_rr_bpm"] = 55.0
    third, _ = causal_history_features(changed_future)
    np.testing.assert_array_equal(first[0], third[0])


def test_append_causal_history_shape_dtype_and_validation():
    metadata = _history_metadata()
    aux = np.ones((len(metadata), 3), dtype=np.float16)
    augmented, names = append_causal_history_features(aux, metadata)
    assert augmented.shape == (len(metadata), 3 + len(names))
    assert augmented.dtype == np.float32
    np.testing.assert_array_equal(augmented[:, :3], np.ones((len(metadata), 3)))

    with pytest.raises(ValueError, match="one row"):
        append_causal_history_features(aux[:-1], metadata)
    with pytest.raises(ValueError, match="duplicate"):
        duplicate = pd.concat([metadata, metadata.iloc[[0]]], ignore_index=True)
        causal_history_features(duplicate)
    with pytest.raises(ValueError, match="positive"):
        causal_history_features(metadata, lags=(0,))

