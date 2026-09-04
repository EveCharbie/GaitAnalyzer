import numpy as np
import numpy.testing as npt
import pytest

from gait_analyzer.plots.plot_utils import (
    LegToPlot,
    PlotType,
    DimentionsToPlot,
    EventIndexType,
    get_unit_conversion_factor,
    get_unit_names,
    split_cycles,
    split_cycle,
    mean_cycles,
)


# ----------------------------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------------------------


def test_leg_to_plot_values():
    assert LegToPlot.LEFT.value == "L"
    assert LegToPlot.RIGHT.value == "R"
    assert LegToPlot.BOTH.value == "both"
    assert LegToPlot.DOMINANT.value == "dominant"


def test_plot_type_values():
    assert PlotType.Q.value == "q_filtered"
    assert PlotType.GRF.value == "f_ext_sorted_filtered"
    assert PlotType.EMG.value == "normalized_emg"


def test_dimensions_to_plot_values():
    assert DimentionsToPlot.BIDIMENTIONAL.value == "2D"
    assert DimentionsToPlot.TRIDIMENTIONAL.value == "3D"


def test_event_index_type_values():
    assert EventIndexType.MARKERS.value == "markers"
    assert EventIndexType.ANALOGS.value == "analogs"


# ----------------------------------------------------------------------------------------
# get_unit_conversion_factor / get_unit_names
# ----------------------------------------------------------------------------------------


def test_get_unit_conversion_factor_q_is_rad_to_deg():
    npt.assert_allclose(get_unit_conversion_factor(PlotType.Q, subject_mass=None), 180 / np.pi)


def test_get_unit_conversion_factor_tau_divides_by_mass():
    npt.assert_allclose(get_unit_conversion_factor(PlotType.TAU, subject_mass=50.0), 1 / 50.0)


def test_get_unit_conversion_factor_grf_is_array_of_nine():
    factor = get_unit_conversion_factor(PlotType.GRF, subject_mass=70.0)
    assert isinstance(factor, np.ndarray)
    assert factor.shape == (9,)
    npt.assert_allclose(factor[:6], 1.0)
    npt.assert_allclose(factor[6:], 1 / (70.0 * 9.8066499999999994))


def test_get_unit_conversion_factor_invalid_type_raises():
    with pytest.raises(ValueError, match="plot_type must be a PlotType"):
        get_unit_conversion_factor("not_a_plot_type", subject_mass=70.0)


def test_get_unit_names_scalar_and_list():
    assert get_unit_names(PlotType.Q) == r"[$^\circ$]"
    grf_units = get_unit_names(PlotType.GRF)
    assert isinstance(grf_units, list)
    assert len(grf_units) == 9


def test_get_unit_names_invalid_type_raises():
    with pytest.raises(ValueError, match="plot_type must be a PlotType"):
        get_unit_names("not_a_plot_type")


# ----------------------------------------------------------------------------------------
# split_cycles
# ----------------------------------------------------------------------------------------


def test_split_cycles_basic_scalar_unit_conversion():
    # 2 data dimensions, 10 frames, events at 0, 3, 7, 10
    data = np.arange(20.0).reshape(2, 10)
    cycles = split_cycles(data, event_idx=[0, 3, 7, 10], plot_type=PlotType.ANGULAR_MOMENTUM, subject_mass=None)
    assert len(cycles) == 3
    npt.assert_allclose(cycles[0], data[:, 0:3])
    npt.assert_allclose(cycles[1], data[:, 3:7])
    npt.assert_allclose(cycles[2], data[:, 7:10])


def test_split_cycles_applies_array_unit_conversion_per_row():
    data = np.ones((9, 6))
    cycles = split_cycles(data, event_idx=[0, 3, 6], plot_type=PlotType.GRF, subject_mass=10.0)
    expected_factor = get_unit_conversion_factor(PlotType.GRF, subject_mass=10.0)
    for cycle in cycles:
        for i_row in range(9):
            npt.assert_allclose(cycle[i_row, :], expected_factor[i_row])


def test_split_cycles_applies_scalar_unit_conversion():
    data = np.ones((3, 10)) * 2.0
    cycles = split_cycles(data, event_idx=[0, 5, 10], plot_type=PlotType.Q, subject_mass=None)
    for cycle in cycles:
        npt.assert_allclose(cycle, 2.0 * (180 / np.pi))


def test_split_cycles_non_ndarray_input_raises():
    with pytest.raises(ValueError, match="data must be a numpy array"):
        split_cycles([[1, 2], [3, 4]], event_idx=[0, 1], plot_type=PlotType.Q, subject_mass=None)


def test_split_cycles_wrong_ndim_raises():
    with pytest.raises(ValueError, match="2D numpy array"):
        split_cycles(np.zeros((2, 2, 2)), event_idx=[0, 1], plot_type=PlotType.Q, subject_mass=None)


def test_split_cycles_empty_data_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        split_cycles(np.zeros((0, 5)), event_idx=[0, 1], plot_type=PlotType.Q, subject_mass=None)


def test_split_cycles_event_out_of_bounds_raises():
    data = np.zeros((2, 5))
    with pytest.raises(RuntimeError, match="too short"):
        split_cycles(data, event_idx=[0, 10], plot_type=PlotType.Q, subject_mass=None)


# ----------------------------------------------------------------------------------------
# split_cycle
# ----------------------------------------------------------------------------------------


def test_split_cycle_extracts_single_slice():
    data = np.arange(30.0).reshape(3, 10)
    cycles = split_cycle(data, cycle_start=2, cycle_end=6, plot_type=PlotType.ANGULAR_MOMENTUM, subject_mass=None)
    assert len(cycles) == 1
    npt.assert_allclose(cycles[0], data[:, 2:6])


def test_split_cycle_out_of_bounds_raises():
    data = np.zeros((2, 5))
    with pytest.raises(RuntimeError, match="too short"):
        split_cycle(data, cycle_start=0, cycle_end=8, plot_type=PlotType.Q, subject_mass=None)


def test_split_cycle_wrong_ndim_raises():
    with pytest.raises(ValueError, match="2D numpy array"):
        split_cycle(np.zeros((2, 2, 2)), cycle_start=0, cycle_end=1, plot_type=PlotType.Q, subject_mass=None)


# ----------------------------------------------------------------------------------------
# mean_cycles
# ----------------------------------------------------------------------------------------


def test_mean_cycles_constant_cycles_returns_constant_mean_and_zero_std():
    # Two identical constant-valued cycles of different lengths (interpolation doesn't
    # change a constant signal)
    cycle1 = np.ones((2, 10)) * 3.0
    cycle2 = np.ones((2, 15)) * 3.0
    mean_data, std_data = mean_cycles([cycle1, cycle2], index_to_keep=None, nb_frames_interp=21)
    npt.assert_allclose(mean_data, 3.0, atol=1e-10)
    npt.assert_allclose(std_data, 0.0, atol=1e-10)


def test_mean_cycles_averages_two_linear_ramps():
    # cycle A ramps 0 -> 1, cycle B ramps 0 -> 3 over the normalized cycle; mean should ramp 0 -> 2
    nb_frames_interp = 11
    x = np.linspace(0, 1, nb_frames_interp)
    cycle_a = np.linspace(0, 1, 20).reshape(1, -1)
    cycle_b = np.linspace(0, 3, 20).reshape(1, -1)
    mean_data, std_data = mean_cycles([cycle_a, cycle_b], index_to_keep=None, nb_frames_interp=nb_frames_interp)
    expected_mean = (np.linspace(0, 1, nb_frames_interp) + np.linspace(0, 3, nb_frames_interp)) / 2
    npt.assert_allclose(mean_data[0, :], expected_mean, atol=1e-8)
    expected_std = np.std(np.vstack([np.linspace(0, 1, nb_frames_interp), np.linspace(0, 3, nb_frames_interp)]), axis=0)
    npt.assert_allclose(std_data[0, :], expected_std, atol=1e-8)


def test_mean_cycles_index_to_keep_selects_dimensions():
    cycle = np.vstack([np.zeros(10), np.ones(10) * 5.0, np.ones(10) * 9.0])
    mean_data, _ = mean_cycles([cycle], index_to_keep=[1, 2], nb_frames_interp=5)
    assert mean_data.shape == (2, 5)
    npt.assert_allclose(mean_data[0, :], 5.0)
    npt.assert_allclose(mean_data[1, :], 9.0)


def test_mean_cycles_empty_list_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        mean_cycles([], index_to_keep=None, nb_frames_interp=10)


def test_mean_cycles_inconsistent_dimensions_raises():
    cycle_a = np.zeros((2, 10))
    cycle_b = np.zeros((3, 10))
    with pytest.raises(ValueError, match="inconsistant"):
        mean_cycles([cycle_a, cycle_b], index_to_keep=None, nb_frames_interp=5)
