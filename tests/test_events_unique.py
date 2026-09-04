import numpy as np
import numpy.testing as npt
import pytest
import pickle

from gait_analyzer.events.unique_events import UniqueEvents
from gait_analyzer.operator import Operator


def _make_bare_unique_events(make_experimental_data, tmp_path, nb_platforms: int, nb_analog_frames: int):
    events = UniqueEvents.__new__(UniqueEvents)
    events.minimal_vertical_force_threshold = 15
    events.experimental_data = make_experimental_data(result_folder=str(tmp_path))
    events.experimental_data.platform_corners = [None] * nb_platforms
    events.experimental_data.nb_analog_frames = nb_analog_frames
    events.is_loaded_events = False
    events.type = "unique"
    events.events = [{"heel_touch": [], "toes_off": []} for _ in range(nb_platforms)]
    return events


def test_constructor_requires_experimental_data_instance():
    with pytest.raises(ValueError, match="must be an instance of ExperimentalData"):
        UniqueEvents(experimental_data="not_experimental_data", skip_if_existing=False)


def test_constructor_requires_bool_skip_if_existing(make_experimental_data, tmp_path):
    experimental_data = make_experimental_data(result_folder=str(tmp_path))
    experimental_data.platform_corners = []
    with pytest.raises(ValueError, match="skip_if_existing must be a boolean"):
        UniqueEvents(experimental_data=experimental_data, skip_if_existing="no")


def test_detect_heel_touch_finds_first_rising_edge_per_platform(make_experimental_data, tmp_path):
    nb_analog_frames = 100
    events = _make_bare_unique_events(make_experimental_data, tmp_path, nb_platforms=2, nb_analog_frames=nb_analog_frames)
    f_ext_sorted = np.zeros((2, 9, nb_analog_frames))
    f_ext_sorted[0, 8, 20:60] = 100.0  # platform 0: single stance block [20, 60)
    f_ext_sorted[1, 8, 40:80] = 100.0  # platform 1: single stance block [40, 80)
    events.experimental_data.f_ext_sorted = f_ext_sorted

    events.detect_heel_touch()

    expected_platform0 = np.where(
        np.diff((np.abs(Operator.moving_average(f_ext_sorted[0, 8, :], 21)) > 15).astype(int)) == 1
    )[0] + 1
    expected_platform1 = np.where(
        np.diff((np.abs(Operator.moving_average(f_ext_sorted[1, 8, :], 21)) > 15).astype(int)) == 1
    )[0] + 1
    npt.assert_array_equal(events.events[0]["heel_touch"], expected_platform0)
    npt.assert_array_equal(events.events[1]["heel_touch"], expected_platform1)


def test_detect_toes_off_finds_falling_edge_per_platform(make_experimental_data, tmp_path):
    nb_analog_frames = 100
    events = _make_bare_unique_events(make_experimental_data, tmp_path, nb_platforms=1, nb_analog_frames=nb_analog_frames)
    f_ext_sorted = np.zeros((1, 9, nb_analog_frames))
    f_ext_sorted[0, 8, 20:60] = 100.0
    events.experimental_data.f_ext_sorted = f_ext_sorted

    events.detect_toes_off()

    expected = np.where(
        np.diff((np.abs(Operator.moving_average(f_ext_sorted[0, 8, :], 21)) > 15).astype(int)) == -1
    )[0]
    npt.assert_array_equal(events.events[0]["toes_off"], expected)


def test_get_frame_range_returns_all_marker_frames(make_experimental_data, tmp_path):
    events = _make_bare_unique_events(make_experimental_data, tmp_path, nb_platforms=1, nb_analog_frames=10)
    events.experimental_data.markers_time_vector = np.linspace(0, 1, 37)
    result = events.get_frame_range(cycles_to_analyze=None)
    npt.assert_array_equal(result, np.arange(37))


def test_get_frame_range_raises_when_cycles_to_analyze_given(make_experimental_data, tmp_path):
    events = _make_bare_unique_events(make_experimental_data, tmp_path, nb_platforms=1, nb_analog_frames=10)
    events.experimental_data.markers_time_vector = np.linspace(0, 1, 10)
    with pytest.raises(NotImplementedError, match="All frames should be analyzed"):
        events.get_frame_range(cycles_to_analyze=range(0, 3))


def test_save_and_reload_round_trip(make_experimental_data, tmp_path):
    events = _make_bare_unique_events(make_experimental_data, tmp_path, nb_platforms=2, nb_analog_frames=10)
    events.events[0]["heel_touch"] = [3, 4]
    events.events[1]["toes_off"] = [7]

    events.save_events()

    result_path = events.get_result_file_full_path()
    with open(result_path, "rb") as f:
        saved = pickle.load(f)
    assert list(saved["events"][0]["heel_touch"]) == [3, 4]

    reloaded = UniqueEvents.__new__(UniqueEvents)
    reloaded.experimental_data = events.experimental_data
    assert reloaded.check_if_existing() is True
    assert list(reloaded.events[0]["heel_touch"]) == [3, 4]
    assert list(reloaded.events[1]["toes_off"]) == [7]


def test_inputs_and_outputs(make_experimental_data, tmp_path):
    events = _make_bare_unique_events(make_experimental_data, tmp_path, nb_platforms=1, nb_analog_frames=5)
    assert events.inputs() == {"experimental_data": events.experimental_data}
    outputs = events.outputs()
    assert set(outputs.keys()) == {"events", "is_loaded_events"}
    assert outputs["events"] is events.events
