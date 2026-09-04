import numpy as np
import numpy.testing as npt
import pytest
from scipy.signal import butter, filtfilt, savgol_filter

from gait_analyzer.operator import Operator


# ----------------------------------------------------------------------------------------
# Operator.moving_average
# ----------------------------------------------------------------------------------------


def test_moving_average_constant_signal_unchanged():
    x = np.ones(20) * 3.5
    x_averaged = Operator.moving_average(x, window_size=5)
    npt.assert_allclose(x_averaged, x)


def test_moving_average_matches_manual_computation_interior():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    x_averaged = Operator.moving_average(x, window_size=3)
    # Interior points: mean of the point and its immediate neighbors
    for i in range(1, len(x) - 1):
        expected = np.mean(x[i - 1 : i + 2])
        npt.assert_allclose(x_averaged[i], expected)


def test_moving_average_edge_windows_are_truncated():
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    x_averaged = Operator.moving_average(x, window_size=3)
    # First point: mean of x[0:2] (no data before index 0)
    npt.assert_allclose(x_averaged[0], np.mean(x[0:2]))
    # Last point: mean of x[-2:]
    npt.assert_allclose(x_averaged[-1], np.mean(x[-2:]))


def test_moving_average_flattens_column_vector():
    x_flat = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    x_column = x_flat.reshape(-1, 1)
    npt.assert_allclose(Operator.moving_average(x_column, window_size=3), Operator.moving_average(x_flat, window_size=3))


def test_moving_average_even_window_size_raises():
    x = np.arange(20.0)
    with pytest.raises(ValueError, match="window_size must be an odd number"):
        Operator.moving_average(x, window_size=4)


def test_moving_average_2d_non_column_raises():
    x = np.zeros((4, 3))
    with pytest.raises(ValueError, match="x must be a vector"):
        Operator.moving_average(x, window_size=3)


def test_moving_average_window_too_large_raises():
    x = np.arange(6.0)
    with pytest.raises(ValueError, match="window_size must be smaller than half"):
        Operator.moving_average(x, window_size=5)


# ----------------------------------------------------------------------------------------
# Operator.apply_filtfilt
# ----------------------------------------------------------------------------------------


def test_apply_filtfilt_matches_scipy_directly_when_no_nans():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(2, 200))
    sampling_rate = 100
    order = 2
    cutoff_freq = 10

    filtered = Operator.apply_filtfilt(data, order=order, sampling_rate=sampling_rate, cutoff_freq=cutoff_freq)

    nyquist = 0.5 * sampling_rate
    b, a = butter(order, cutoff_freq / nyquist, btype="low", analog=False)
    expected = np.array([filtfilt(b, a, row) for row in data])

    npt.assert_allclose(filtered, expected)


def test_apply_filtfilt_preserves_constant_signal():
    data = np.ones((1, 100)) * 5.0
    filtered = Operator.apply_filtfilt(data, order=2, sampling_rate=100, cutoff_freq=10)
    npt.assert_allclose(filtered, data, rtol=1e-6)


def test_apply_filtfilt_leaves_nan_columns_as_nan():
    data = np.ones((1, 50))
    data[0, 10:15] = np.nan
    filtered = Operator.apply_filtfilt(data, order=2, sampling_rate=100, cutoff_freq=10)
    assert np.all(np.isnan(filtered[0, 10:15]))
    assert not np.any(np.isnan(np.delete(filtered[0, :], np.arange(10, 15))))


# ----------------------------------------------------------------------------------------
# Operator.apply_savgol
# ----------------------------------------------------------------------------------------


def test_apply_savgol_matches_scipy_directly_when_no_nans():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(3, 101))

    filtered = Operator.apply_savgol(data, window_length=11, polyorder=2)

    expected = np.array([savgol_filter(row, window_length=11, polyorder=2) for row in data])
    npt.assert_allclose(filtered, expected)


def test_apply_savgol_recovers_polynomial_signal_exactly():
    # A savgol filter with polyorder >= the underlying polynomial degree should reproduce
    # the signal exactly (away from the boundary effects, which are handled by scipy itself).
    t = np.linspace(0, 1, 101)
    data = (2.0 + 3.0 * t - 4.0 * t**2).reshape(1, -1)
    filtered = Operator.apply_savgol(data, window_length=11, polyorder=2)
    npt.assert_allclose(filtered, data, atol=1e-8)


def test_apply_savgol_leaves_nan_columns_as_nan():
    data = np.linspace(0, 1, 60).reshape(1, -1)
    data[0, 20:25] = np.nan
    filtered = Operator.apply_savgol(data, window_length=7, polyorder=2)
    assert np.all(np.isnan(filtered[0, 20:25]))


# ----------------------------------------------------------------------------------------
# Operator.from_marker_frame_to_analog_frame / from_analog_frame_to_marker_frame
# ----------------------------------------------------------------------------------------


def test_from_marker_frame_to_analog_frame_int():
    analogs_time_vector = np.linspace(0, 1, 100)
    markers_time_vector = np.linspace(0, 1, 10)
    # ratio is 10, so marker frame i corresponds to analog frame 10*i
    assert Operator.from_marker_frame_to_analog_frame(analogs_time_vector, markers_time_vector, 3) == 30


def test_from_marker_frame_to_analog_frame_list():
    analogs_time_vector = np.linspace(0, 1, 100)
    markers_time_vector = np.linspace(0, 1, 10)
    result = Operator.from_marker_frame_to_analog_frame(analogs_time_vector, markers_time_vector, [0, 2, 5])
    assert result == [0, 20, 50]


def test_from_marker_frame_to_analog_frame_ndarray():
    analogs_time_vector = np.linspace(0, 1, 100)
    markers_time_vector = np.linspace(0, 1, 10)
    result = Operator.from_marker_frame_to_analog_frame(analogs_time_vector, markers_time_vector, np.array([1, 4, 9]))
    npt.assert_array_equal(result, np.array([10, 40, 90]))


def test_from_marker_frame_to_analog_frame_invalid_type_raises():
    analogs_time_vector = np.linspace(0, 1, 100)
    markers_time_vector = np.linspace(0, 1, 10)
    with pytest.raises(ValueError, match="marker_idx must be"):
        Operator.from_marker_frame_to_analog_frame(analogs_time_vector, markers_time_vector, "not_an_int")


def test_from_marker_frame_to_analog_frame_2d_ndarray_raises():
    analogs_time_vector = np.linspace(0, 1, 100)
    markers_time_vector = np.linspace(0, 1, 10)
    with pytest.raises(ValueError, match="1D numpy array"):
        Operator.from_marker_frame_to_analog_frame(analogs_time_vector, markers_time_vector, np.zeros((2, 2)))


def test_from_analog_frame_to_marker_frame_int():
    analogs_time_vector = np.linspace(0, 1, 100)
    markers_time_vector = np.linspace(0, 1, 10)
    assert Operator.from_analog_frame_to_marker_frame(analogs_time_vector, markers_time_vector, 30) == 3


def test_from_analog_frame_to_marker_frame_list():
    analogs_time_vector = np.linspace(0, 1, 100)
    markers_time_vector = np.linspace(0, 1, 10)
    result = Operator.from_analog_frame_to_marker_frame(analogs_time_vector, markers_time_vector, [0, 20, 55])
    assert result == [0, 2, 6]  # round(55/10) == 6


def test_from_analog_frame_to_marker_frame_ndarray():
    analogs_time_vector = np.linspace(0, 1, 100)
    markers_time_vector = np.linspace(0, 1, 10)
    result = Operator.from_analog_frame_to_marker_frame(analogs_time_vector, markers_time_vector, np.array([10, 40, 90]))
    npt.assert_array_equal(result, np.array([1, 4, 9]))


def test_from_analog_frame_to_marker_frame_invalid_type_raises():
    analogs_time_vector = np.linspace(0, 1, 100)
    markers_time_vector = np.linspace(0, 1, 10)
    with pytest.raises(ValueError, match="analog_idx must be"):
        Operator.from_analog_frame_to_marker_frame(analogs_time_vector, markers_time_vector, "nope")


def test_frame_conversion_round_trip():
    analogs_time_vector = np.linspace(0, 1, 200)
    markers_time_vector = np.linspace(0, 1, 20)
    marker_indices = np.array([0, 1, 5, 10, 19])
    analog_indices = Operator.from_marker_frame_to_analog_frame(analogs_time_vector, markers_time_vector, marker_indices)
    back_to_marker = Operator.from_analog_frame_to_marker_frame(analogs_time_vector, markers_time_vector, analog_indices)
    npt.assert_array_equal(back_to_marker, marker_indices)
