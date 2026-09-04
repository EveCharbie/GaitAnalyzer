import numpy as np
import numpy.testing as npt
import pytest

from gait_analyzer.kinematics_reconstructor import KinematicsReconstructor, ReconstructionType
from gait_analyzer.model_creator import ModelCreator
from gait_analyzer.events.cyclic_events import CyclicEvents
from gait_analyzer.events.unique_events import UniqueEvents


def test_reconstruction_type_enum_values():
    assert ReconstructionType.ONLY_LM.value == "only_lm"
    assert ReconstructionType.LM.value == "lm"
    assert ReconstructionType.TRF.value == "trf"
    assert ReconstructionType.EKF.value == "ekf"
    assert ReconstructionType.LSQ.value == "lsq"


# ----------------------------------------------------------------------------------------
# Constructor validation (all of these raise before any heavy IK/model-loading work happens)
# ----------------------------------------------------------------------------------------


def test_constructor_requires_experimental_data_instance(make_experimental_data):
    events = CyclicEvents.__new__(CyclicEvents)
    model_creator = ModelCreator.__new__(ModelCreator)
    with pytest.raises(ValueError, match="must be an instance of ExperimentalData"):
        KinematicsReconstructor(
            experimental_data="not_experimental_data",
            model_creator=model_creator,
            events=events,
            cycles_to_analyze=None,
            reconstruction_type=None,
            skip_if_existing=False,
            animate_kinematics_flag=False,
            plot_kinematics_flag=False,
        )


def test_constructor_requires_model_creator_instance(make_experimental_data):
    experimental_data = make_experimental_data()
    events = CyclicEvents.__new__(CyclicEvents)
    with pytest.raises(ValueError, match="must be an instance of ModelCreator"):
        KinematicsReconstructor(
            experimental_data=experimental_data,
            model_creator="not_a_model_creator",
            events=events,
            cycles_to_analyze=None,
            reconstruction_type=None,
            skip_if_existing=False,
            animate_kinematics_flag=False,
            plot_kinematics_flag=False,
        )


def test_constructor_requires_cyclic_or_unique_events(make_experimental_data):
    experimental_data = make_experimental_data()
    model_creator = ModelCreator.__new__(ModelCreator)
    with pytest.raises(ValueError, match="must be an instance of CyclicEvents or UniqueEvents"):
        KinematicsReconstructor(
            experimental_data=experimental_data,
            model_creator=model_creator,
            events="not_events",
            cycles_to_analyze=None,
            reconstruction_type=None,
            skip_if_existing=False,
            animate_kinematics_flag=False,
            plot_kinematics_flag=False,
        )


def test_constructor_forbids_cycles_to_analyze_with_unique_events(make_experimental_data):
    experimental_data = make_experimental_data()
    model_creator = ModelCreator.__new__(ModelCreator)
    events = UniqueEvents.__new__(UniqueEvents)
    with pytest.raises(NotImplementedError, match="cycles_to_analyze must be None"):
        KinematicsReconstructor(
            experimental_data=experimental_data,
            model_creator=model_creator,
            events=events,
            cycles_to_analyze=range(0, 3),
            reconstruction_type=None,
            skip_if_existing=False,
            animate_kinematics_flag=False,
            plot_kinematics_flag=False,
        )


def test_constructor_rejects_invalid_reconstruction_type(make_experimental_data):
    experimental_data = make_experimental_data()
    model_creator = ModelCreator.__new__(ModelCreator)
    events = CyclicEvents.__new__(CyclicEvents)
    with pytest.raises(ValueError, match="reconstruction_type must be an instance of ReconstructionType"):
        KinematicsReconstructor(
            experimental_data=experimental_data,
            model_creator=model_creator,
            events=events,
            cycles_to_analyze=None,
            reconstruction_type=123,
            skip_if_existing=False,
            animate_kinematics_flag=False,
            plot_kinematics_flag=False,
        )


def test_constructor_rejects_list_with_non_reconstruction_type_elements(make_experimental_data):
    experimental_data = make_experimental_data()
    model_creator = ModelCreator.__new__(ModelCreator)
    events = CyclicEvents.__new__(CyclicEvents)
    with pytest.raises(ValueError, match="must be a list of ReconstructionType"):
        KinematicsReconstructor(
            experimental_data=experimental_data,
            model_creator=model_creator,
            events=events,
            cycles_to_analyze=None,
            reconstruction_type=[ReconstructionType.LM, "not_a_reconstruction_type"],
            skip_if_existing=False,
            animate_kinematics_flag=False,
            plot_kinematics_flag=False,
        )


# ----------------------------------------------------------------------------------------
# filter_kinematics: physically-motivated invariants on a hand-designed q(t) = t signal
# (pure numpy, no biorbd model is needed since filter_kinematics only reads self.q/self.t)
# ----------------------------------------------------------------------------------------


def _make_bare_reconstructor(q, t):
    reconstructor = KinematicsReconstructor.__new__(KinematicsReconstructor)
    reconstructor.q = q
    reconstructor.t = t
    return reconstructor


def test_filter_kinematics_recovers_constant_velocity_from_a_linear_ramp():
    nb_frames = 101
    t = np.linspace(0, 1, nb_frames)
    q = np.zeros((1, nb_frames))
    q[0, :] = t  # constant velocity of 1 unit/s

    reconstructor = _make_bare_reconstructor(q, t)
    q_filtered, qdot, qddot = reconstructor.filter_kinematics()

    assert q_filtered.shape == q.shape
    assert qdot.shape == q.shape
    assert qddot.shape == q.shape

    # Away from the trial boundaries (where savgol/filtfilt edge effects are largest), a pure
    # ramp should be reproduced almost exactly, with near-constant velocity and near-zero
    # acceleration.
    interior = slice(15, -15)
    npt.assert_allclose(q_filtered[0, interior], q[0, interior], atol=1e-3)
    npt.assert_allclose(qdot[0, interior], 1.0, atol=1e-2)
    npt.assert_allclose(qddot[0, interior], 0.0, atol=1e-1)


def test_filter_kinematics_zero_signal_stays_zero():
    nb_frames = 101
    t = np.linspace(0, 1, nb_frames)
    q = np.zeros((2, nb_frames))

    reconstructor = _make_bare_reconstructor(q, t)
    q_filtered, qdot, qddot = reconstructor.filter_kinematics()

    npt.assert_allclose(q_filtered, 0.0, atol=1e-10)
    npt.assert_allclose(qdot, 0.0, atol=1e-8)
    npt.assert_allclose(qddot, 0.0, atol=1e-6)
