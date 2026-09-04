import numpy as np
import numpy.testing as npt
import pytest
import pickle

from gait_analyzer.biomechanics_quantities.mechanical_energy_calculator import (
    MechanicalEnergyCalculator,
)
from gait_analyzer.biomechanics_quantities.angular_momentum_calculator import (
    AngularMomentumCalculator,
)


def _segments_data_at_rest(synthetic_biorbd_model, nb_frames):
    """
    Build the segments_data dict (as produced by AngularMomentumCalculator) for the
    synthetic model at q=qdot=0: every segment is static, so COM is constant and COMdot=0.
    """
    calculator = AngularMomentumCalculator.__new__(AngularMomentumCalculator)
    calculator.model = synthetic_biorbd_model
    nb_q = synthetic_biorbd_model.nbQ()
    calculator.q = np.zeros((nb_q, nb_frames))
    calculator.qdot = np.zeros((nb_q, nb_frames))
    calculator.nb_frames = nb_frames
    segments_data, _, _ = calculator.calculate_angular_momentum_segment_and_total()
    return segments_data


def _make_calculator(synthetic_biorbd_model, make_experimental_data, real_subject, q, qdot, segments_data, tmp_path):
    calculator = MechanicalEnergyCalculator.__new__(MechanicalEnergyCalculator)
    calculator.model = synthetic_biorbd_model
    calculator.experimental_data = make_experimental_data(result_folder=str(tmp_path))
    calculator.q = q
    calculator.qdot = qdot
    calculator.subject_mass = real_subject.subject_mass
    calculator.subject_height = real_subject.subject_height
    calculator.gravity = synthetic_biorbd_model.getGravity().to_array()
    calculator.nb_frames = q.shape[1]
    calculator.segments_data = segments_data
    calculator.mechanical_energy = None
    calculator.mechanical_energy_normalized = None
    calculator.E_pot_vec = None
    calculator.E_kin_vec = None
    calculator.E_kin_global_vec = None
    calculator.E_pot_normalized = None
    calculator.E_kin_normalized = None
    return calculator


def test_mechanical_energy_at_rest_is_purely_potential(
    synthetic_biorbd_model, make_experimental_data, real_subject, tmp_path
):
    nb_frames = 3
    nb_q = synthetic_biorbd_model.nbQ()
    q = np.zeros((nb_q, nb_frames))
    qdot = np.zeros((nb_q, nb_frames))
    segments_data = _segments_data_at_rest(synthetic_biorbd_model, nb_frames)
    calculator = _make_calculator(synthetic_biorbd_model, make_experimental_data, real_subject, q, qdot, segments_data, tmp_path)

    calculator.compute_mechanical_energy()

    npt.assert_allclose(calculator.E_kin_vec, 0.0, atol=1e-10)
    npt.assert_allclose(calculator.E_kin_global_vec, 0.0, atol=1e-10)

    g = np.linalg.norm(calculator.gravity)
    com_z_at_rest = synthetic_biorbd_model.CoM(np.zeros(nb_q)).to_array()[2]
    expected_e_pot = real_subject.subject_mass * g * com_z_at_rest
    npt.assert_allclose(calculator.E_pot_vec, expected_e_pot)
    npt.assert_allclose(calculator.mechanical_energy, expected_e_pot, atol=1e-8)


def test_mechanical_energy_normalization_matches_formula(
    synthetic_biorbd_model, make_experimental_data, real_subject, tmp_path
):
    nb_frames = 2
    nb_q = synthetic_biorbd_model.nbQ()
    q = np.zeros((nb_q, nb_frames))
    qdot = np.zeros((nb_q, nb_frames))
    segments_data = _segments_data_at_rest(synthetic_biorbd_model, nb_frames)
    calculator = _make_calculator(synthetic_biorbd_model, make_experimental_data, real_subject, q, qdot, segments_data, tmp_path)

    calculator.compute_mechanical_energy()

    g = np.linalg.norm(calculator.gravity)
    normalization = real_subject.subject_mass * g * real_subject.subject_height
    npt.assert_allclose(calculator.mechanical_energy_normalized, calculator.mechanical_energy / normalization)
    npt.assert_allclose(calculator.E_pot_normalized, calculator.E_pot_vec / normalization)
    npt.assert_allclose(calculator.E_kin_normalized, calculator.E_kin_vec / normalization)


def test_mechanical_energy_raises_on_implausible_segment_velocity(
    synthetic_biorbd_model, make_experimental_data, real_subject, tmp_path
):
    nb_frames = 1
    nb_q = synthetic_biorbd_model.nbQ()
    q = np.zeros((nb_q, nb_frames))
    qdot = np.zeros((nb_q, nb_frames))
    qdot[6, 0] = 1000.0  # segment2's rotational dof spun unrealistically fast
    segments_data = _segments_data_at_rest(synthetic_biorbd_model, nb_frames)
    # Recompute segments_data consistently with this qdot (COMdot must reflect it for the
    # guard to trigger the way it does in production, where segments_data comes from the
    # matching AngularMomentumCalculator run).
    angmom = AngularMomentumCalculator.__new__(AngularMomentumCalculator)
    angmom.model = synthetic_biorbd_model
    angmom.q = q
    angmom.qdot = qdot
    angmom.nb_frames = nb_frames
    segments_data, _, _ = angmom.calculate_angular_momentum_segment_and_total()

    calculator = _make_calculator(synthetic_biorbd_model, make_experimental_data, real_subject, q, qdot, segments_data, tmp_path)

    with pytest.raises(RuntimeError, match="implausible for human gait"):
        calculator.compute_mechanical_energy()


def test_save_and_reload_round_trip(synthetic_biorbd_model, make_experimental_data, real_subject, tmp_path):
    nb_frames = 3
    nb_q = synthetic_biorbd_model.nbQ()
    q = np.zeros((nb_q, nb_frames))
    qdot = np.zeros((nb_q, nb_frames))
    segments_data = _segments_data_at_rest(synthetic_biorbd_model, nb_frames)
    calculator = _make_calculator(synthetic_biorbd_model, make_experimental_data, real_subject, q, qdot, segments_data, tmp_path)
    calculator.compute_mechanical_energy()
    calculator.save()

    result_path = calculator.result_file_path()
    with open(result_path, "rb") as f:
        saved = pickle.load(f)
    npt.assert_allclose(saved["Mechanical_energy"], calculator.mechanical_energy)

    outputs = calculator.outputs()
    assert set(outputs.keys()) == {
        "mechanical_energy",
        "mechanical_energy_normalized",
        "mechanical_energy_potential",
        "mechanical_energy_pot_norm",
        "mechanical_energy_kinetic",
        "mechanical_energy_kin_norm",
        "mechanical_energy_kinetic_com",
    }

    reloaded = MechanicalEnergyCalculator.__new__(MechanicalEnergyCalculator)
    reloaded.experimental_data = calculator.experimental_data
    reloaded.mechanical_energy = None
    assert reloaded.check_if_existing() is True
    npt.assert_allclose(reloaded.mechanical_energy, calculator.mechanical_energy)


def test_outputs_is_empty_dict_before_computation(synthetic_biorbd_model, make_experimental_data, real_subject, tmp_path):
    calculator = MechanicalEnergyCalculator.__new__(MechanicalEnergyCalculator)
    calculator.mechanical_energy = None
    assert calculator.outputs() == {}
