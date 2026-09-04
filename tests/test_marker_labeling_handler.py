import numpy as np
import numpy.testing as npt
import pytest

from gait_analyzer.utils.marker_labeling_handler import MarkerLabelingHandler


def _make_handler(write_synthetic_point_c3d):
    marker_names = ["MA", "MB", "MC"]
    nb_frames = 10
    positions = np.zeros((3, 3, nb_frames))
    for i_marker in range(3):
        positions[0, i_marker, :] = i_marker  # distinct x per marker (in meters)
        positions[2, i_marker, :] = np.linspace(0, 1, nb_frames)
    c3d_path = write_synthetic_point_c3d(marker_names, positions)
    return MarkerLabelingHandler(c3d_path), positions


def test_marker_labeling_handler_reads_markers_and_names(write_synthetic_point_c3d):
    handler, positions = _make_handler(write_synthetic_point_c3d)
    assert handler.marker_names == ["MA", "MB", "MC"]
    assert handler.markers.shape[1] == 3
    assert handler.markers.shape[2] == 10
    # Positions are stored back in mm by the c3d format; compare after converting
    npt.assert_allclose(handler.markers[:3, :, :] / 1000.0, positions, atol=1e-5)


def test_invert_marker_labeling_swaps_markers_in_frame_range(write_synthetic_point_c3d):
    handler, positions = _make_handler(write_synthetic_point_c3d)
    original = handler.markers.copy()

    handler.invert_marker_labeling(["MA", "MB"], frame_start=2, frame_end=5)

    idx_a = handler.marker_names.index("MA")
    idx_b = handler.marker_names.index("MB")

    # Inside the frame range, MA and MB should be swapped
    npt.assert_allclose(handler.markers[:, idx_a, 2:6], original[:, idx_b, 2:6])
    npt.assert_allclose(handler.markers[:, idx_b, 2:6], original[:, idx_a, 2:6])
    # Outside the frame range, nothing changes
    npt.assert_allclose(handler.markers[:, idx_a, :2], original[:, idx_a, :2])
    npt.assert_allclose(handler.markers[:, idx_b, :2], original[:, idx_b, :2])
    npt.assert_allclose(handler.markers[:, idx_a, 6:], original[:, idx_a, 6:])
    # MC (untouched marker) is unaffected everywhere
    idx_c = handler.marker_names.index("MC")
    npt.assert_allclose(handler.markers[:, idx_c, :], original[:, idx_c, :])


def test_invert_marker_labeling_requires_two_names(write_synthetic_point_c3d):
    handler, _ = _make_handler(write_synthetic_point_c3d)
    with pytest.raises(ValueError, match="exactly two marker names"):
        handler.invert_marker_labeling(["MA"], frame_start=0, frame_end=1)


def test_invert_marker_labeling_requires_list_type(write_synthetic_point_c3d):
    handler, _ = _make_handler(write_synthetic_point_c3d)
    with pytest.raises(TypeError, match="list of two marker names"):
        handler.invert_marker_labeling("MA,MB", frame_start=0, frame_end=1)


def test_invert_marker_labeling_invalid_frame_range_raises(write_synthetic_point_c3d):
    handler, _ = _make_handler(write_synthetic_point_c3d)
    with pytest.raises(ValueError, match="Invalid frame range"):
        handler.invert_marker_labeling(["MA", "MB"], frame_start=5, frame_end=2)
    with pytest.raises(ValueError, match="Invalid frame range"):
        handler.invert_marker_labeling(["MA", "MB"], frame_start=0, frame_end=1000)


def test_remove_label_sets_nan_in_frame_range_only(write_synthetic_point_c3d):
    handler, positions = _make_handler(write_synthetic_point_c3d)
    handler.remove_label("MB", frame_start=3, frame_end=6)

    idx_b = handler.marker_names.index("MB")
    assert np.all(np.isnan(handler.markers[:3, idx_b, 3:7]))
    assert not np.any(np.isnan(handler.markers[:3, idx_b, :3]))
    assert not np.any(np.isnan(handler.markers[:3, idx_b, 7:]))
    # Other markers are untouched
    idx_a = handler.marker_names.index("MA")
    assert not np.any(np.isnan(handler.markers[:3, idx_a, :]))


def test_remove_label_requires_string_name(write_synthetic_point_c3d):
    handler, _ = _make_handler(write_synthetic_point_c3d)
    with pytest.raises(TypeError, match="name of the marker"):
        handler.remove_label(123, frame_start=0, frame_end=1)


def test_remove_label_invalid_frame_range_raises(write_synthetic_point_c3d):
    handler, _ = _make_handler(write_synthetic_point_c3d)
    with pytest.raises(ValueError, match="Invalid frame range"):
        handler.remove_label("MA", frame_start=-1, frame_end=2)
