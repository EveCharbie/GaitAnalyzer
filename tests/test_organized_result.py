import numpy as np
import numpy.testing as npt
import pytest

from gait_analyzer.statistical_analysis.organized_result import OrganizedResult, ResultObject
from gait_analyzer.plots.plot_utils import LegToPlot, PlotType, EventIndexType


# ----------------------------------------------------------------------------------------
# ResultObject.add
# ----------------------------------------------------------------------------------------


def test_result_object_add_none_data_is_ignored():
    result = ResultObject(groups_to_compare=None, nb_frames_interp=101)
    result.add(data=None, subject_name="P01", condition_name="cond_a")
    assert result.data == {}


def test_result_object_add_non_list_raises():
    result = ResultObject(groups_to_compare=None, nb_frames_interp=101)
    with pytest.raises(ValueError, match="data must be a list"):
        result.add(data=np.zeros((2, 5)), subject_name="P01", condition_name="cond_a")


def test_result_object_add_non_ndarray_elements_raises():
    result = ResultObject(groups_to_compare=None, nb_frames_interp=101)
    with pytest.raises(ValueError, match="must be numpy arrays"):
        result.add(data=[[1, 2, 3]], subject_name="P01", condition_name="cond_a")


def test_result_object_add_defaults_group_to_all():
    result = ResultObject(groups_to_compare=None, nb_frames_interp=101)
    cycle = np.ones((2, 10))
    result.add(data=[cycle], subject_name="P01", condition_name="cond_a")
    assert list(result.data.keys()) == ["all"]
    assert list(result.data["all"].keys()) == ["cond_a"]
    assert list(result.data["all"]["cond_a"].keys()) == ["P01"]
    npt.assert_allclose(result.data["all"]["cond_a"]["P01"][0], cycle)


def test_result_object_add_accumulates_multiple_calls():
    result = ResultObject(groups_to_compare={"g1": ["P01"]}, nb_frames_interp=101)
    cycle1 = np.ones((2, 5))
    cycle2 = np.ones((2, 5)) * 2
    result.add(data=[cycle1], subject_name="P01", condition_name="cond_a", group_name="g1")
    result.add(data=[cycle2], subject_name="P01", condition_name="cond_a", group_name="g1")
    stored = result.data["g1"]["cond_a"]["P01"]
    assert len(stored) == 2
    npt.assert_allclose(stored[0], cycle1)
    npt.assert_allclose(stored[1], cycle2)


# ----------------------------------------------------------------------------------------
# ResultObject.mean_per_subject / mean_per_group
# ----------------------------------------------------------------------------------------


def test_mean_per_subject_matches_manual_mean_cycles():
    result = ResultObject(groups_to_compare=None, nb_frames_interp=11)
    cycle_a = np.linspace(0, 1, 10).reshape(1, -1)
    cycle_b = np.linspace(0, 3, 20).reshape(1, -1)
    result.add(data=[cycle_a, cycle_b], subject_name="P01", condition_name="cond_a")

    subject_mean, subject_std = result.mean_per_subject()

    from gait_analyzer.plots.plot_utils import mean_cycles

    expected_mean, expected_std = mean_cycles([cycle_a, cycle_b], index_to_keep=None, nb_frames_interp=11)
    npt.assert_allclose(subject_mean["all"]["cond_a"]["P01"], expected_mean)
    npt.assert_allclose(subject_std["all"]["cond_a"]["P01"], expected_std)
    assert result.mean_data_per_subject is subject_mean


def test_mean_per_group_averages_across_subjects():
    result = ResultObject(groups_to_compare={"g1": ["P01", "P02"]}, nb_frames_interp=5)
    result.add(data=[np.ones((1, 5)) * 2.0], subject_name="P01", condition_name="cond_a", group_name="g1")
    result.add(data=[np.ones((1, 5)) * 4.0], subject_name="P02", condition_name="cond_a", group_name="g1")

    subject_mean, _ = result.mean_per_subject()
    group_mean, group_std = result.mean_per_group(subject_mean)

    npt.assert_allclose(group_mean["g1"]["cond_a"], 3.0)
    npt.assert_allclose(group_std["g1"]["cond_a"], 1.0)


# ----------------------------------------------------------------------------------------
# OrganizedResult helper methods (constructed via __new__ to avoid touching the filesystem
# in OrganizedResult.__init__, which scans a result_folder for pickled results)
# ----------------------------------------------------------------------------------------


def _make_organized_result(plot_type, leg_to_plot=LegToPlot.RIGHT, cycles_to_analyze=None):
    organized_result = OrganizedResult.__new__(OrganizedResult)
    organized_result.plot_type = plot_type
    organized_result.leg_to_plot = leg_to_plot
    if plot_type in [PlotType.GRF, PlotType.EMG]:
        organized_result.event_index_type = EventIndexType.ANALOGS
    elif plot_type in [PlotType.MUSCLE_FORCES]:
        organized_result.event_index_type = EventIndexType.NONE
    else:
        organized_result.event_index_type = EventIndexType.MARKERS
    return organized_result


def test_get_data_to_split_selects_grf_side():
    organized_result = _make_organized_result(PlotType.GRF, leg_to_plot=LegToPlot.LEFT)
    grf = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(float)
    data = {"f_ext_sorted_filtered": grf}
    result = organized_result.get_data_to_split(data)
    npt.assert_allclose(result, grf[0, :, :])

    organized_result_right = _make_organized_result(PlotType.GRF, leg_to_plot=LegToPlot.RIGHT)
    result_right = organized_result_right.get_data_to_split(data)
    npt.assert_allclose(result_right, grf[1, :, :])


def test_get_data_to_split_both_legs_raises_not_implemented():
    organized_result = _make_organized_result(PlotType.GRF, leg_to_plot=LegToPlot.BOTH)
    with pytest.raises(NotImplementedError, match="both legs"):
        organized_result.get_data_to_split({"f_ext_sorted_filtered": np.zeros((2, 3, 4))})


def test_get_data_to_split_non_grf_returns_raw_data():
    organized_result = _make_organized_result(PlotType.Q)
    q_data = np.ones((5, 100))
    result = organized_result.get_data_to_split({"q_filtered": q_data})
    npt.assert_allclose(result, q_data)


def test_get_event_index_analogs_returns_event_unchanged():
    organized_result = _make_organized_result(PlotType.GRF)
    event = [10, 20, 30]
    result = organized_result.get_event_index(event, None, np.linspace(0, 1, 100), np.linspace(0, 1, 10))
    assert result == event


def test_get_event_index_markers_converts_and_shifts_to_zero():
    organized_result = _make_organized_result(PlotType.Q)
    analog_time_vector = np.linspace(0, 1, 100)
    markers_time_vector = np.linspace(0, 1, 10)
    # analog frames -> marker frames [2, 4, 6]; with cycles_to_analyze=None the default
    # slice is [0:-1], dropping the last event, then results are shifted to start at 0.
    event = [20, 40, 60]
    result = organized_result.get_event_index(event, None, analog_time_vector, markers_time_vector)
    npt.assert_array_equal(result, [0, 2])


def test_get_event_index_markers_respects_cycles_to_analyze_range():
    organized_result = _make_organized_result(PlotType.Q)
    analog_time_vector = np.linspace(0, 1, 100)
    markers_time_vector = np.linspace(0, 1, 10)
    event = [10, 20, 30, 40, 50]  # -> marker frames [1, 2, 3, 4, 5]
    result = organized_result.get_event_index(event, range(1, 4), analog_time_vector, markers_time_vector)
    # start_cycle=1, end_cycle=4 -> marker frames [2, 3, 4] shifted to start at 0
    npt.assert_array_equal(result, [0, 1, 2])


def test_get_event_index_invalid_type_raises():
    organized_result = OrganizedResult.__new__(OrganizedResult)
    organized_result.event_index_type = "invalid"
    with pytest.raises(RuntimeError, match="EventIndexType.ANALOGS or EventIndexType.MARKERS"):
        organized_result.get_event_index([1], None, np.linspace(0, 1, 10), np.linspace(0, 1, 5))
