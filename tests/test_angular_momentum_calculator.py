import numpy as np
import numpy.testing as npt
import pickle

from gait_analyzer.biomechanics_quantities.angular_momentum_calculator import (
    AngularMomentumCalculator,
)


def _make_calculator(synthetic_biorbd_model, make_experimental_data, real_subject, q, qdot, tmp_path):
    calculator = AngularMomentumCalculator.__new__(AngularMomentumCalculator)
    calculator.model = synthetic_biorbd_model
    calculator.experimental_data = make_experimental_data(result_folder=str(tmp_path))
    calculator.q = q
    calculator.qdot = qdot
    calculator.subject_mass = real_subject.subject_mass
    calculator.subject_height = real_subject.subject_height
    calculator.gravity = synthetic_biorbd_model.getGravity().to_array()
    calculator.nb_frames = q.shape[1]
    calculator.dof_names = [m.to_string() for m in synthetic_biorbd_model.nameDof()]
    calculator.total_angular_momentum = None
    calculator.H_segments = None
    calculator.H_total = None
    calculator.segments_data = None
    calculator.total_angular_momentum_norm = None
    return calculator


def test_angular_momentum_is_zero_at_rest(synthetic_biorbd_model, make_experimental_data, real_subject, tmp_path):
    nb_q = synthetic_biorbd_model.nbQ()
    nb_frames = 3
    q = np.zeros((nb_q, nb_frames))
    qdot = np.zeros((nb_q, nb_frames))
    calculator = _make_calculator(synthetic_biorbd_model, make_experimental_data, real_subject, q, qdot, tmp_path)

    segments_data, h_segments, h_total = calculator.calculate_angular_momentum_segment_and_total()

    npt.assert_allclose(h_total, 0.0, atol=1e-10)
    for seg_name in h_segments:
        npt.assert_allclose(h_segments[seg_name], 0.0, atol=1e-10)
    assert set(segments_data.keys()) == {"pelvis", "segment2", "calcn_l", "calcn_r"}
    npt.assert_allclose(segments_data["pelvis"]["Masse"], 10.0)
    npt.assert_allclose(segments_data["segment2"]["Masse"], 2.0)


def test_angular_momentum_symmetric_rotation_has_no_y_or_z_component(
    synthetic_biorbd_model, make_experimental_data, real_subject, tmp_path
):
    # segment2 rotates about local x (through a point on the model's z axis); calcn_l/calcn_r
    # are placed symmetrically about x=0 and stay static. This configuration is symmetric
    # under y -> -y reflection combined with the rotation axis, so the total angular momentum
    # about the global CoM must be purely along x.
    nb_q = synthetic_biorbd_model.nbQ()
    nb_frames = 5
    q = np.zeros((nb_q, nb_frames))
    qdot = np.zeros((nb_q, nb_frames))
    qdot[6, :] = 1.0  # segment2's rotational dof (see conftest.SYNTHETIC_BIOMOD)
    calculator = _make_calculator(synthetic_biorbd_model, make_experimental_data, real_subject, q, qdot, tmp_path)

    _, _, h_total = calculator.calculate_angular_momentum_segment_and_total()

    npt.assert_allclose(h_total[1, :], 0.0, atol=1e-10)
    npt.assert_allclose(h_total[2, :], 0.0, atol=1e-10)
    assert np.all(np.abs(h_total[0, :]) > 1e-6)  # non-trivial x component


def test_angular_momentum_matches_independent_recombination(
    synthetic_biorbd_model, make_experimental_data, real_subject, tmp_path
):
    """
    Recompute H_total frame-by-frame from calculate_angular_momentum_segment_and_total's own
    per-segment outputs (segments_data, H_segments) using the textbook definition
    H_total = sum_segments(H_seg), which is exactly what calculate_angular_momentum_segment_and_total
    is supposed to produce -- this guards against H_total silently drifting from H_segments.
    """
    nb_q = synthetic_biorbd_model.nbQ()
    nb_frames = 4
    q = np.zeros((nb_q, nb_frames))
    qdot = np.zeros((nb_q, nb_frames))
    qdot[6, :] = 0.7
    calculator = _make_calculator(synthetic_biorbd_model, make_experimental_data, real_subject, q, qdot, tmp_path)

    segments_data, h_segments, h_total = calculator.calculate_angular_momentum_segment_and_total()

    recombined = np.zeros_like(h_total)
    for seg_name in h_segments:
        recombined += h_segments[seg_name]
    npt.assert_allclose(h_total, recombined, atol=1e-12)


def test_normalize_total_angular_momentum_matches_formula(
    synthetic_biorbd_model, make_experimental_data, real_subject, tmp_path
):
    nb_q = synthetic_biorbd_model.nbQ()
    nb_frames = 3
    q = np.zeros((nb_q, nb_frames))
    qdot = np.zeros((nb_q, nb_frames))
    calculator = _make_calculator(synthetic_biorbd_model, make_experimental_data, real_subject, q, qdot, tmp_path)
    calculator.H_total = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    calculator.normalize_total_angular_momentum()

    g_norm = np.linalg.norm(calculator.gravity)
    expected_factor = real_subject.subject_mass * real_subject.subject_height * np.sqrt(g_norm * real_subject.subject_height)
    npt.assert_allclose(calculator.total_angular_momentum_norm, calculator.H_total / expected_factor)


def test_save_and_reload_round_trip(synthetic_biorbd_model, make_experimental_data, real_subject, tmp_path):
    nb_q = synthetic_biorbd_model.nbQ()
    nb_frames = 3
    q = np.zeros((nb_q, nb_frames))
    qdot = np.zeros((nb_q, nb_frames))
    calculator = _make_calculator(synthetic_biorbd_model, make_experimental_data, real_subject, q, qdot, tmp_path)
    calculator.calculate_angular_momentum_segment_and_total()
    calculator.normalize_total_angular_momentum()
    calculator.save_angular_momentum()

    result_path = calculator.get_result_file_full_path()
    with open(result_path, "rb") as f:
        saved = pickle.load(f)
    npt.assert_allclose(saved["H_total"], calculator.H_total)
    npt.assert_allclose(saved["total_angular_momentum_norm"], calculator.total_angular_momentum_norm)

    reloaded = AngularMomentumCalculator.__new__(AngularMomentumCalculator)
    reloaded.experimental_data = calculator.experimental_data
    assert reloaded.check_if_existing() is True
    npt.assert_allclose(reloaded.H_total, calculator.H_total)
    npt.assert_allclose(reloaded.total_angular_momentum_norm, calculator.total_angular_momentum_norm)
