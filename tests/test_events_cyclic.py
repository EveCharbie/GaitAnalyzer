import numpy as np
import numpy.testing as npt
import pytest
import pickle

from gait_analyzer.events.cyclic_events import CyclicEvents
from gait_analyzer.subject import Side
from gait_analyzer.operator import Operator


def _make_bare_events(make_experimental_data, tmp_path, nb_analog_frames: int = 100):
    """
    A real CyclicEvents instance with its (heavy) __init__ bypassed, populated with only the
    bare attributes each method under test reads/writes -- mirroring exactly what __init__
    sets up before calling find_event_timestamps().
    """
    events = CyclicEvents.__new__(CyclicEvents)
    events.minimal_vertical_force_threshold = 50
    events.minimal_forward_force_threshold = 5
    events.heel_velocity_threshold = 0.05
    events.experimental_data = make_experimental_data(result_folder=str(tmp_path))
    events.experimental_data.nb_analog_frames = nb_analog_frames
    events.right_leg_index = 1
    events.left_leg_index = 0
    events.is_loaded_events = False
    events.type = "cyclic"
    events.events = {
        "right_leg_heel_touch": [],
        "right_leg_toes_touch": [],
        "right_leg_heel_off": [],
        "right_leg_toes_off": [],
        "left_leg_heel_touch": [],
        "left_leg_toes_touch": [],
        "left_leg_heel_off": [],
        "left_leg_toes_off": [],
    }
    events.phases_right_leg = {
        "flat_foot": np.zeros((nb_analog_frames,)),
        "toes_only": np.zeros((nb_analog_frames,)),
        "swing": np.zeros((nb_analog_frames,)),
        "heel_only": np.zeros((nb_analog_frames,)),
    }
    events.phases_left_leg = {
        "flat_foot": np.zeros((nb_analog_frames,)),
        "toes_only": np.zeros((nb_analog_frames,)),
        "swing": np.zeros((nb_analog_frames,)),
        "heel_only": np.zeros((nb_analog_frames,)),
    }
    events.phases = {
        "heelR_toesR": np.zeros((nb_analog_frames,)),
        "toesR": np.zeros((nb_analog_frames,)),
        "toesR_heelL": np.zeros((nb_analog_frames,)),
        "toesR_heelL_toesL": np.zeros((nb_analog_frames,)),
        "heelL_toesL": np.zeros((nb_analog_frames,)),
        "toesL": np.zeros((nb_analog_frames,)),
        "toesL_heelR": np.zeros((nb_analog_frames,)),
        "toesL_heelR_toesR": np.zeros((nb_analog_frames,)),
    }
    return events


# ----------------------------------------------------------------------------------------
# Constructor validation
# ----------------------------------------------------------------------------------------


def test_constructor_requires_experimental_data_instance(make_experimental_data, tmp_path):
    with pytest.raises(ValueError, match="must be an instance of ExperimentalData"):
        CyclicEvents(
            experimental_data="not_experimental_data",
            force_plate_sides=[Side.LEFT, Side.RIGHT],
            skip_if_existing=False,
            plot_phases_flag=False,
        )


def test_constructor_requires_two_force_plate_sides(make_experimental_data, tmp_path):
    experimental_data = make_experimental_data(result_folder=str(tmp_path), nb_analog_frames=10)
    with pytest.raises(NotImplementedError, match="only supports two force plates"):
        CyclicEvents(
            experimental_data=experimental_data,
            force_plate_sides=[Side.LEFT],
            skip_if_existing=False,
            plot_phases_flag=False,
        )


def test_constructor_requires_side_elements(make_experimental_data, tmp_path):
    experimental_data = make_experimental_data(result_folder=str(tmp_path), nb_analog_frames=10)
    with pytest.raises(ValueError, match="must be Side"):
        CyclicEvents(
            experimental_data=experimental_data,
            force_plate_sides=["left", "right"],
            skip_if_existing=False,
            plot_phases_flag=False,
        )


def test_constructor_requires_bool_flags(make_experimental_data, tmp_path):
    experimental_data = make_experimental_data(result_folder=str(tmp_path), nb_analog_frames=10)
    with pytest.raises(ValueError, match="skip_if_existing must be a boolean"):
        CyclicEvents(
            experimental_data=experimental_data,
            force_plate_sides=[Side.LEFT, Side.RIGHT],
            skip_if_existing="no",
            plot_phases_flag=False,
        )


# ----------------------------------------------------------------------------------------
# detect_swing_phases_temporary
# ----------------------------------------------------------------------------------------


def test_detect_swing_phases_temporary_matches_thresholded_moving_average(make_experimental_data, tmp_path):
    nb_analog_frames = 100
    events = _make_bare_events(make_experimental_data, tmp_path, nb_analog_frames)
    rng = np.random.default_rng(1)
    f_ext_sorted = np.zeros((2, 9, nb_analog_frames))
    f_ext_sorted[0, 8, :] = rng.uniform(0, 100, nb_analog_frames)  # left vertical GRF
    f_ext_sorted[1, 8, :] = rng.uniform(0, 100, nb_analog_frames)  # right vertical GRF
    events.experimental_data.f_ext_sorted = f_ext_sorted

    events.detect_swing_phases_temporary(show_debug_plot_flag=False)

    expected_left = np.abs(Operator.moving_average(f_ext_sorted[0, 8, :], 21)) < 50
    expected_right = np.abs(Operator.moving_average(f_ext_sorted[1, 8, :], 21)) < 50
    npt.assert_array_equal(events.phases_left_leg["swing"], expected_left)
    npt.assert_array_equal(events.phases_right_leg["swing"], expected_right)


# ----------------------------------------------------------------------------------------
# detect_toes_off
# ----------------------------------------------------------------------------------------


def test_detect_toes_off_returns_start_of_each_swing_block(make_experimental_data, tmp_path):
    nb_analog_frames = 100
    events = _make_bare_events(make_experimental_data, tmp_path, nb_analog_frames)
    events.phases_left_leg["swing"][10:16] = 1  # block [10..15]
    events.phases_left_leg["swing"][50:58] = 1  # block [50..57]
    events.phases_right_leg["swing"][30:34] = 1  # block [30..33]

    events.detect_toes_off()

    assert events.events["left_leg_toes_off"] == [10, 50]
    assert events.events["right_leg_toes_off"] == [30]


def test_detect_toes_off_skips_blocks_touching_the_trial_boundaries(make_experimental_data, tmp_path):
    nb_analog_frames = 100
    events = _make_bare_events(make_experimental_data, tmp_path, nb_analog_frames)
    events.phases_left_leg["swing"][0:5] = 1  # starts at frame 0 -> skipped
    events.phases_left_leg["swing"][20:25] = 1  # kept
    # np.array_split on an all-zero swing array still yields one (empty) block, which
    # detect_toes_off then indexes into unconditionally -- give the right leg a normal,
    # non-empty swing block so this test only exercises the left-leg boundary behavior.
    events.phases_right_leg["swing"][40:45] = 1

    events.detect_toes_off()

    assert events.events["left_leg_toes_off"] == [20]
    assert events.events["right_leg_toes_off"] == [40]


# ----------------------------------------------------------------------------------------
# detect_toes_touch / detect_heel_touch (using a constant GRF so the moving-average window
# leaves the signal completely unchanged, making the expected index exact and hand-derivable)
# ----------------------------------------------------------------------------------------


def test_detect_toes_touch_with_constant_grf_picks_end_of_swing(make_experimental_data, tmp_path):
    nb_analog_frames = 100
    events = _make_bare_events(make_experimental_data, tmp_path, nb_analog_frames)
    events.phases_left_leg["swing"][10:16] = 1  # ends at frame 15
    events.phases_left_leg["swing"][50:58] = 1  # ends at frame 57 (last block)
    # np.array_split on an all-zero swing array still yields one (empty) block, which
    # detect_toes_touch then indexes into unconditionally -- give the right leg a normal,
    # non-empty swing block so this test only exercises the left-leg behavior.
    events.phases_right_leg["swing"][70:75] = 1
    f_ext_sorted = np.zeros((2, 9, nb_analog_frames))
    f_ext_sorted[0, 8, :] = 100.0  # constant vertical GRF -> moving_average leaves it unchanged
    f_ext_sorted[1, 8, :] = 80.0
    events.experimental_data.f_ext_sorted = f_ext_sorted

    events.detect_toes_touch()

    # argmax over a constant array returns the first index (0), so toes_touch == end_swing_idx
    assert events.events["left_leg_toes_touch"] == [15, 57]
    assert events.events["right_leg_toes_touch"] == [74]


def test_detect_heel_touch_with_constant_grf_above_threshold(make_experimental_data, tmp_path):
    nb_analog_frames = 100
    events = _make_bare_events(make_experimental_data, tmp_path, nb_analog_frames)
    events.phases_left_leg["swing"][10:16] = 1  # ends at frame 15
    events.phases_left_leg["swing"][50:58] = 1  # ends at frame 57
    # np.array_split on an all-zero swing array still yields one (empty) block, which
    # detect_heel_touch then indexes into unconditionally -- give the right leg a normal,
    # non-empty swing block so this test only exercises the left-leg behavior.
    events.phases_right_leg["swing"][70:76] = 1  # ends at frame 75
    f_ext_sorted = np.zeros((2, 9, nb_analog_frames))
    f_ext_sorted[:, 7, :] = 10.0  # constant antero-posterior GRF, always above the 5N threshold
    events.experimental_data.f_ext_sorted = f_ext_sorted

    events.detect_heel_touch(show_debug_plot_flag=False)

    # Since the threshold is already exceeded at idx = swing_phase[-1] - 5, the search loop
    # never advances idx, so heel_touch = swing_phase[-1] - 3 (see detect_heel_touch's
    # `idx -= 1` followed by `(swing_phase[-1] + idx) // 2`).
    assert events.events["left_leg_heel_touch"] == [12, 54]
    assert events.events["right_leg_heel_touch"] == [72]


# ----------------------------------------------------------------------------------------
# detect_leg_phases_between_events / detect_phases_both_legs (pure boolean/index algebra,
# tested independently of the (heuristic) event-detection methods above)
# ----------------------------------------------------------------------------------------


def test_detect_leg_phases_between_events_fills_the_correct_window(make_experimental_data, tmp_path):
    nb_analog_frames = 30
    events = _make_bare_events(make_experimental_data, tmp_path, nb_analog_frames)
    events.events["left_leg_toes_off"] = [5, 20]
    events.events["left_leg_heel_touch"] = [10, 25]
    events.events["right_leg_toes_off"] = [3]
    events.events["right_leg_heel_touch"] = [8]

    events.detect_leg_phases_between_events("swing", "toes_off", "heel_touch")

    expected_left = np.zeros(nb_analog_frames)
    expected_left[5:11] = 1
    expected_left[20:26] = 1
    npt.assert_array_equal(events.phases_left_leg["swing"], expected_left)

    expected_right = np.zeros(nb_analog_frames)
    expected_right[3:9] = 1
    npt.assert_array_equal(events.phases_right_leg["swing"], expected_right)


def test_detect_leg_phases_between_events_ignores_unmatched_opening_event(make_experimental_data, tmp_path):
    nb_analog_frames = 20
    events = _make_bare_events(make_experimental_data, tmp_path, nb_analog_frames)
    # No closing event after the last opening event at all -> that window is simply never filled
    events.events["left_leg_toes_off"] = [15]
    events.events["left_leg_heel_touch"] = []

    events.detect_leg_phases_between_events("swing", "toes_off", "heel_touch")

    npt.assert_array_equal(events.phases_left_leg["swing"], np.zeros(nb_analog_frames))


def test_detect_phases_both_legs_is_logical_and(make_experimental_data, tmp_path):
    nb_analog_frames = 10
    events = _make_bare_events(make_experimental_data, tmp_path, nb_analog_frames)
    # detect_phases_both_legs(phase_name, left_leg_phase_name, right_leg_phase_name) reads
    # left_leg_phase_name from phases_left_leg and right_leg_phase_name from phases_right_leg.
    events.phases_left_leg["swing"][3:8] = 1
    events.phases_right_leg["flat_foot"][:5] = 1

    events.detect_phases_both_legs("heelR_toesR", "swing", "flat_foot")

    expected = np.zeros(nb_analog_frames, dtype=bool)
    expected[3:5] = True
    npt.assert_array_equal(events.phases["heelR_toesR"], expected)


# ----------------------------------------------------------------------------------------
# get_frame_range
# ----------------------------------------------------------------------------------------


def test_get_frame_range_with_no_cycles_to_analyze_uses_full_range(make_experimental_data, tmp_path):
    events = _make_bare_events(make_experimental_data, tmp_path, nb_analog_frames=200)
    events.events["right_leg_heel_touch"] = [10, 20, 30, 40, 50]
    events.experimental_data.analogs_time_vector = np.linspace(0, 1, 200)
    events.experimental_data.markers_time_vector = np.linspace(0, 1, 20)  # ratio 10

    frame_range, padded_frame_range = events.get_frame_range(cycles_to_analyze=None)

    heel_touches_marker_frames = Operator.from_analog_frame_to_marker_frame(
        events.experimental_data.analogs_time_vector, events.experimental_data.markers_time_vector, [10, 20, 30, 40, 50]
    )
    assert frame_range == range(heel_touches_marker_frames[0], heel_touches_marker_frames[-1])
    assert padded_frame_range == frame_range  # start_cycle=0 -> not padded; end_cycle=-1 -> not padded


def test_get_frame_range_with_explicit_cycles_pads_the_range(make_experimental_data, tmp_path):
    events = _make_bare_events(make_experimental_data, tmp_path, nb_analog_frames=1000)
    events.events["right_leg_heel_touch"] = list(range(0, 200, 10))  # 20 heel touches
    events.experimental_data.analogs_time_vector = np.linspace(0, 1, 1000)
    events.experimental_data.markers_time_vector = np.linspace(0, 1, 100)  # ratio 10

    frame_range, padded_frame_range = events.get_frame_range(cycles_to_analyze=range(8, 12))

    heel_touches_marker_frames = Operator.from_analog_frame_to_marker_frame(
        events.experimental_data.analogs_time_vector, events.experimental_data.markers_time_vector, events.events["right_leg_heel_touch"]
    )
    assert frame_range == range(heel_touches_marker_frames[8], heel_touches_marker_frames[12])
    # padded by 5 cycles on each side since start_cycle(8) > 5 and end_cycle(12) < len-5
    assert padded_frame_range == range(heel_touches_marker_frames[3], heel_touches_marker_frames[17])


# ----------------------------------------------------------------------------------------
# save_events / check_if_existing / inputs / outputs
# ----------------------------------------------------------------------------------------


def test_save_and_reload_round_trip(make_experimental_data, tmp_path):
    events = _make_bare_events(make_experimental_data, tmp_path, nb_analog_frames=10)
    events.events["left_leg_toes_off"] = [1, 2, 3]
    events.phases_left_leg["swing"][:5] = 1

    events.save_events()

    result_path = events.get_result_file_full_path()
    with open(result_path, "rb") as f:
        saved = pickle.load(f)
    assert saved["events"]["left_leg_toes_off"] == [1, 2, 3]

    reloaded = CyclicEvents.__new__(CyclicEvents)
    reloaded.experimental_data = events.experimental_data
    assert reloaded.check_if_existing() is True
    assert reloaded.events["left_leg_toes_off"] == [1, 2, 3]
    npt.assert_array_equal(reloaded.phases_left_leg["swing"], events.phases_left_leg["swing"])


def test_inputs_and_outputs(make_experimental_data, tmp_path):
    events = _make_bare_events(make_experimental_data, tmp_path, nb_analog_frames=5)
    assert events.inputs() == {"experimental_data": events.experimental_data}
    outputs = events.outputs()
    assert set(outputs.keys()) == {"events", "phases_left_leg", "phases_right_leg", "phases", "is_loaded_events"}
    assert outputs["events"] is events.events
