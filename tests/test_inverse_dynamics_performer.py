import numpy as np
import numpy.testing as npt
import pytest
import pickle

from gait_analyzer.inverse_dynamics_performer import InverseDynamicsPerformer
from gait_analyzer.kinematics_reconstructor import KinematicsReconstructor


def _make_bare_kinematics_reconstructor(nb_q, nb_frames):
    reconstructor = KinematicsReconstructor.__new__(KinematicsReconstructor)
    reconstructor.q_filtered = np.zeros((nb_q, nb_frames))
    reconstructor.qdot = np.zeros((nb_q, nb_frames))
    reconstructor.qddot = np.zeros((nb_q, nb_frames))
    reconstructor.t = np.linspace(0, 1, nb_frames)
    return reconstructor


# ----------------------------------------------------------------------------------------
# Constructor validation (raise before any inverse-dynamics computation happens)
# ----------------------------------------------------------------------------------------


def test_constructor_requires_experimental_data_instance():
    kinematics_reconstructor = _make_bare_kinematics_reconstructor(nb_q=3, nb_frames=5)
    with pytest.raises(ValueError, match="must be an instance of ExperimentalData"):
        InverseDynamicsPerformer(
            experimental_data="not_experimental_data",
            kinematics_reconstructor=kinematics_reconstructor,
            skip_if_existing=False,
            reintegrate_flag=False,
            animate_dynamics_flag=False,
        )


def test_constructor_requires_kinematics_reconstructor_instance(make_experimental_data):
    experimental_data = make_experimental_data()
    with pytest.raises(ValueError, match="must be an instance of biorbd.Model"):
        InverseDynamicsPerformer(
            experimental_data=experimental_data,
            kinematics_reconstructor="not_a_kinematics_reconstructor",
            skip_if_existing=False,
            reintegrate_flag=False,
            animate_dynamics_flag=False,
        )


def test_constructor_requires_bool_flags(make_experimental_data):
    experimental_data = make_experimental_data()
    kinematics_reconstructor = _make_bare_kinematics_reconstructor(nb_q=3, nb_frames=5)
    with pytest.raises(ValueError, match="skip_if_existing must be a boolean"):
        InverseDynamicsPerformer(
            experimental_data=experimental_data,
            kinematics_reconstructor=kinematics_reconstructor,
            skip_if_existing="no",
            reintegrate_flag=False,
            animate_dynamics_flag=False,
        )


# ----------------------------------------------------------------------------------------
# get_f_ext_at_frame: delegates to ExperimentalData.get_f_ext_at_marker_frame per foot
# ----------------------------------------------------------------------------------------


def test_get_f_ext_at_frame_uses_correct_platform_and_component_slices(
    synthetic_biorbd_model, make_experimental_data
):
    nb_frames = 4
    f_ext_sorted_filtered = np.zeros((2, 9, nb_frames))
    f_ext_sorted_filtered[0, :, :] = np.arange(9).reshape(-1, 1)  # left/calcn_l: CoP=[0,1,2], M=[3,4,5], F=[6,7,8]
    f_ext_sorted_filtered[1, :, :] = np.arange(9, 18).reshape(-1, 1)  # right/calcn_r

    experimental_data = make_experimental_data(
        f_ext_sorted_filtered=f_ext_sorted_filtered,
        analogs_time_vector=np.linspace(0, 1, nb_frames),
        markers_time_vector=np.linspace(0, 1, nb_frames),
    )

    performer = InverseDynamicsPerformer.__new__(InverseDynamicsPerformer)
    performer.biorbd_model = synthetic_biorbd_model
    performer.experimental_data = experimental_data

    # Should not raise: calcn_l/calcn_r exist on the synthetic model (see conftest)
    performer.get_f_ext_at_frame(0)


# ----------------------------------------------------------------------------------------
# save/check_if_existing/inputs/outputs
# ----------------------------------------------------------------------------------------


def test_save_and_reload_round_trip(make_experimental_data, tmp_path):
    # NOTE: check_if_existing does `data["q_reintegrated"] != 0` and feeds the result
    # straight into an `if`, which raises ValueError ("truth value of an array with more
    # than one element is ambiguous") whenever q_reintegrated is an actual multi-element
    # array reloaded from disk -- a real, reproducible bug in
    # InverseDynamicsPerformer.check_if_existing whenever reintegrate_flag was previously
    # True. This test uses the q_reintegrated=0 sentinel (the "no reintegration" case,
    # see save_inverse_dynamics) to exercise the round trip without hitting that bug.
    performer = InverseDynamicsPerformer.__new__(InverseDynamicsPerformer)
    performer.experimental_data = make_experimental_data(result_folder=str(tmp_path))
    performer.tau = np.ones((3, 4))
    performer.q_reintegrated = 0
    performer.is_loaded_inverse_dynamics = False

    performer.save_inverse_dynamics()

    result_path = performer.get_result_file_full_path()
    with open(result_path, "rb") as f:
        saved = pickle.load(f)
    npt.assert_allclose(saved["tau"], performer.tau)

    reloaded = InverseDynamicsPerformer.__new__(InverseDynamicsPerformer)
    reloaded.experimental_data = performer.experimental_data
    assert reloaded.check_if_existing() is True
    npt.assert_allclose(reloaded.tau, performer.tau)
    assert reloaded.q_reintegrated is None


def test_save_inverse_dynamics_stores_zero_for_missing_reintegration(make_experimental_data, tmp_path):
    performer = InverseDynamicsPerformer.__new__(InverseDynamicsPerformer)
    performer.experimental_data = make_experimental_data(result_folder=str(tmp_path))
    performer.tau = np.zeros((2, 2))
    performer.q_reintegrated = None
    performer.is_loaded_inverse_dynamics = False

    performer.save_inverse_dynamics()

    assert performer.q_reintegrated == 0


def test_inputs_and_outputs(synthetic_biorbd_model, make_experimental_data):
    performer = InverseDynamicsPerformer.__new__(InverseDynamicsPerformer)
    performer.biorbd_model = synthetic_biorbd_model
    performer.experimental_data = make_experimental_data()
    performer.q_filtered = np.zeros((3, 2))
    performer.qdot = np.zeros((3, 2))
    performer.qddot = np.zeros((3, 2))
    performer.tau = np.zeros((3, 2))
    performer.q_reintegrated = 0
    performer.is_loaded_inverse_dynamics = False

    inputs = performer.inputs()
    assert inputs["biorbd_model"] is synthetic_biorbd_model
    outputs = performer.outputs()
    assert set(outputs.keys()) == {"tau", "q_reintegrated", "is_loaded_inverse_dynamics"}
