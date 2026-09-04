"""
Shared pytest fixtures for the GaitAnalyzer test suite.

Philosophy
----------
This codebase is a scientific pipeline built on top of heavy, file-based dependencies
(biorbd models scaled from OpenSim, c3d motion capture trials, biobuddy scaling). Rather
than mocking these dependencies away, the fixtures below build small but *real* instances
of them:
  - a tiny but valid biorbd model (written as a real .bioMod file and loaded with
    biorbd.Model), instead of a mock model,
  - real ExperimentalData/other gait_analyzer instances, built by bypassing their (heavy,
    file-dependent) __init__ with __new__ and then filling in the handful of numpy
    attributes the method under test actually reads. This keeps tests using the real
    classes defined in gait_analyzer (isinstance checks downstream still pass) while
    remaining fast and hermetic.
"""

import numpy as np
import pytest
import biorbd
import c3d as py_c3d

from gait_analyzer.experimental_data import ExperimentalData
from gait_analyzer.subject import Subject


SYNTHETIC_BIOMOD = """
version 4

gravity 0 0 -9.81

segment pelvis
    translations xyz
    rotations xyz
    mass 10.0
    com 0 0 0
    inertia
        1.0 0.0 0.0
        0.0 1.0 0.0
        0.0 0.0 1.0
endsegment

segment segment2
    parent pelvis
    RTinMatrix 0
    RT 0 0 0 xyz 0 0 0.5
    rotations x
    mass 2.0
    com 0 0 -0.25
    inertia
        0.05 0.0 0.0
        0.0 0.05 0.0
        0.0 0.0 0.01
endsegment

segment calcn_l
    parent pelvis
    RTinMatrix 0
    RT 0 0 0 xyz 0.1 0 -1.0
    mass 1.0
    com 0 0 0
    inertia
        0.001 0.0 0.0
        0.0 0.001 0.0
        0.0 0.0 0.001
endsegment

segment calcn_r
    parent pelvis
    RTinMatrix 0
    RT 0 0 0 xyz -0.1 0 -1.0
    mass 1.0
    com 0 0 0
    inertia
        0.001 0.0 0.0
        0.0 0.001 0.0
        0.0 0.0 0.001
endsegment
"""


@pytest.fixture
def synthetic_model_path(tmp_path) -> str:
    """
    A small but valid .bioMod file: a free-flyer 'pelvis' (mass 10), a 'segment2' that
    rotates about local x through a point 0.5 m above the pelvis origin (mass 2), and two
    fixed feet segments 'calcn_l'/'calcn_r' placed symmetrically about x=0 (mass 1 each,
    matching the segment names InverseDynamicsPerformer attaches external forces to).
    """
    model_path = tmp_path / "synthetic_model.bioMod"
    model_path.write_text(SYNTHETIC_BIOMOD)
    return str(model_path)


@pytest.fixture
def synthetic_biorbd_model(synthetic_model_path) -> biorbd.Model:
    return biorbd.Model(synthetic_model_path)


@pytest.fixture
def make_experimental_data(tmp_path):
    """
    Factory fixture returning a real (not mocked) ExperimentalData instance with its
    __init__ bypassed (since __init__ requires an actual c3d file, a ModelCreator, and
    biorbd model markers). Only the attributes needed by the method(s) under test are
    filled in; everything else is left as a clearly-invalid sentinel so that a test
    relying on an unset attribute fails loudly instead of silently passing.

    Usage: `make_experimental_data(f_ext_sorted_filtered=..., analogs_time_vector=..., ...)`
    """

    def _make(**overrides) -> ExperimentalData:
        exp_data = ExperimentalData.__new__(ExperimentalData)
        exp_data.force_threshold = 15
        exp_data.c3d_full_file_path = str(tmp_path / "dummy_trial.c3d")
        exp_data.result_folder = str(tmp_path)
        exp_data.model_creator = None
        exp_data.markers_to_ignore = []
        exp_data.analogs_to_ignore = []
        exp_data.c3d = None
        exp_data.model_marker_names = None
        exp_data.marker_sampling_frequency = None
        exp_data.markers_dt = None
        exp_data.marker_units = None
        exp_data.nb_marker_frames = None
        exp_data.markers_sorted = None
        exp_data.analogs_sampling_frequency = None
        exp_data.normalized_emg = None
        exp_data.analog_names = None
        exp_data.platform_corners = None
        exp_data.analogs_dt = None
        exp_data.nb_analog_frames = None
        exp_data.f_ext_sorted = None
        exp_data.f_ext_sorted_filtered = None
        exp_data.markers_time_vector = None
        exp_data.analogs_time_vector = None
        for key, value in overrides.items():
            setattr(exp_data, key, value)
        return exp_data

    return _make


@pytest.fixture
def real_subject() -> Subject:
    return Subject(subject_name="test_subject", subject_mass=70.0, subject_height=1.75)


@pytest.fixture
def write_synthetic_point_c3d(tmp_path):
    """
    Factory fixture that writes a real, minimal point-only c3d file (using the independent
    `c3d` package, since writing fresh point c3d files through this environment's installed
    ezc3d build is broken -- see the ezc3d.c3d().write() TypeError on POINT:UNITS for a
    pristine c3d object) and returns the path. The file is then read back with ezc3d in the
    tests below, exactly like the gait_analyzer code under test does.

    `marker_positions` must be a (3, nb_markers, nb_frames) array in meters; it is converted
    to millimeters on write since that is the unit gait_analyzer/ezc3d expect by default.
    """

    def _write(marker_names: list[str], marker_positions: np.ndarray, point_rate: float = 100.0, file_name: str = "synthetic.c3d") -> str:
        assert marker_positions.shape[0] == 3
        assert marker_positions.shape[1] == len(marker_names)
        nb_frames = marker_positions.shape[2]

        writer = py_c3d.Writer(point_rate=point_rate, analog_rate=0.0, point_units="mm")
        writer.set_point_labels(marker_names)

        frames = []
        for i_frame in range(nb_frames):
            points = np.zeros((len(marker_names), 5), dtype=np.float32)
            points[:, :3] = (marker_positions[:, :, i_frame] * 1000.0).T  # m -> mm
            points[:, 3] = 0.0  # residual: 0 = valid
            analog = np.zeros((0, 0))
            frames.append((points, analog))
        writer.add_frames(frames)

        out_path = tmp_path / file_name
        with open(out_path, "wb") as handle:
            writer.write(handle)
        return str(out_path)

    return _write
