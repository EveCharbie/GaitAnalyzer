import numpy as np
import numpy.testing as npt
import pickle

from gait_analyzer.biomechanics_quantities.com_mma_distance_calculator import (
    ComMmaDistanceCalculator,
)
from gait_analyzer.biomechanics_quantities.angular_momentum_calculator import (
    AngularMomentumCalculator,
)


def _make_angular_momentum_stub(h_total: np.ndarray) -> AngularMomentumCalculator:
    calculator = AngularMomentumCalculator.__new__(AngularMomentumCalculator)
    calculator.H_total = h_total
    return calculator


def test_compute_com_mma_distance_matches_manual_cross_product(make_experimental_data, real_subject):
    nb_frames = 20
    t = np.linspace(0, 1, nb_frames)

    # H_total(t) = (0, 0, t) so that Hdot = (0, 0, 1) constant (np.gradient of a linear ramp)
    h_total = np.zeros((3, nb_frames))
    h_total[2, :] = t

    # Constant horizontal ground reaction force, well above the near-zero-force guard
    f_ext_sorted_filtered = np.zeros((2, 9, nb_frames))
    f_ext_sorted_filtered[0, 6:9, :] = np.array([10.0, 0.0, 0.0])[:, None]
    f_ext_sorted_filtered[1, 6:9, :] = np.array([0.0, 5.0, 0.0])[:, None]

    experimental_data = make_experimental_data(
        f_ext_sorted_filtered=f_ext_sorted_filtered,
        markers_time_vector=t,
    )
    angular_momentum = _make_angular_momentum_stub(h_total)
    q = np.zeros((3, nb_frames))

    calculator = ComMmaDistanceCalculator(
        angular_momentum_calculator=angular_momentum,
        experimental_data=experimental_data,
        subject=real_subject,
        q=q,
        skip_if_existing=False,
    )

    expected_hdot = np.gradient(h_total, t, axis=1)
    npt.assert_allclose(calculator.Hdot, expected_hdot)

    expected_f_resultant = np.tile(np.array([[10.0], [5.0], [0.0]]), (1, nb_frames))
    npt.assert_allclose(calculator.F_resultant, expected_f_resultant)

    for i in range(nb_frames):
        F = expected_f_resultant[:, i]
        Hdot = expected_hdot[:, i]
        expected_r_mma = np.cross(F, Hdot) / np.linalg.norm(F) ** 2
        npt.assert_allclose(calculator.r_MMA[:, i], expected_r_mma)
        npt.assert_allclose(calculator.dCoM_MMA_norm[i], np.linalg.norm(expected_r_mma))


def test_compute_com_mma_distance_is_nan_when_force_is_near_zero(make_experimental_data, real_subject):
    nb_frames = 5
    t = np.linspace(0, 1, nb_frames)
    h_total = np.zeros((3, nb_frames))
    h_total[0, :] = t  # arbitrary non-trivial Hdot

    f_ext_sorted_filtered = np.zeros((2, 9, nb_frames))
    # Resultant force norm^2 well below the 1e-6 threshold used in compute_com_mma_distance
    f_ext_sorted_filtered[0, 6:9, :] = 1e-5
    f_ext_sorted_filtered[1, 6:9, :] = 0.0

    experimental_data = make_experimental_data(f_ext_sorted_filtered=f_ext_sorted_filtered, markers_time_vector=t)
    angular_momentum = _make_angular_momentum_stub(h_total)
    q = np.zeros((3, nb_frames))

    calculator = ComMmaDistanceCalculator(
        angular_momentum_calculator=angular_momentum,
        experimental_data=experimental_data,
        subject=real_subject,
        q=q,
        skip_if_existing=False,
    )

    assert np.all(np.isnan(calculator.r_MMA))
    assert np.all(np.isnan(calculator.dCoM_MMA_norm))


def test_compute_resultant_force_sums_all_platforms(make_experimental_data, real_subject):
    nb_frames = 4
    h_total = np.zeros((3, nb_frames))
    f_ext_sorted_filtered = np.zeros((2, 9, nb_frames))
    f_ext_sorted_filtered[0, 6:9, :] = np.array([1.0, 2.0, 3.0])[:, None]
    f_ext_sorted_filtered[1, 6:9, :] = np.array([4.0, 5.0, 6.0])[:, None]

    experimental_data = make_experimental_data(
        f_ext_sorted_filtered=f_ext_sorted_filtered, markers_time_vector=np.linspace(0, 1, nb_frames)
    )
    angular_momentum = _make_angular_momentum_stub(h_total)

    calculator = ComMmaDistanceCalculator(
        angular_momentum_calculator=angular_momentum,
        experimental_data=experimental_data,
        subject=real_subject,
        q=np.zeros((3, nb_frames)),
        skip_if_existing=False,
    )
    expected = np.tile(np.array([[5.0], [7.0], [9.0]]), (1, nb_frames))
    npt.assert_allclose(calculator.F_resultant, expected)


def test_save_and_reload_round_trip(make_experimental_data, real_subject, tmp_path):
    nb_frames = 6
    t = np.linspace(0, 1, nb_frames)
    h_total = np.vstack([t, np.zeros(nb_frames), np.zeros(nb_frames)])
    f_ext_sorted_filtered = np.zeros((2, 9, nb_frames))
    f_ext_sorted_filtered[0, 6:9, :] = np.array([1.0, 0.0, 0.0])[:, None]

    experimental_data = make_experimental_data(
        f_ext_sorted_filtered=f_ext_sorted_filtered,
        markers_time_vector=t,
        result_folder=str(tmp_path),
    )
    angular_momentum = _make_angular_momentum_stub(h_total)

    calculator = ComMmaDistanceCalculator(
        angular_momentum_calculator=angular_momentum,
        experimental_data=experimental_data,
        subject=real_subject,
        q=np.zeros((3, nb_frames)),
        skip_if_existing=False,
    )

    result_path = calculator.get_result_file_full_path()
    with open(result_path, "rb") as f:
        saved = pickle.load(f)
    npt.assert_allclose(saved["dCoM_MMA_norm"], calculator.dCoM_MMA_norm)

    outputs = calculator.outputs()
    assert set(outputs.keys()) == {"Hdot", "F_resultant", "r_MMA", "dCoM_MMA_norm"}

    reloaded = ComMmaDistanceCalculator(
        angular_momentum_calculator=angular_momentum,
        experimental_data=experimental_data,
        subject=real_subject,
        q=np.zeros((3, nb_frames)),
        skip_if_existing=True,
    )
    assert reloaded.is_loaded_dcom_mma is True
    npt.assert_allclose(reloaded.dCoM_MMA_norm, calculator.dCoM_MMA_norm)
