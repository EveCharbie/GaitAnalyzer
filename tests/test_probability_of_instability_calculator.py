import numpy as np
import numpy.testing as npt
import pickle

from gait_analyzer.biomechanics_quantities.probability_of_instability_calculator import (
    ProbabilityOfInstability,
)
from gait_analyzer.biomechanics_quantities.margin_of_stability_calculator import (
    MarginOfStabilityCalculator,
)


def _make_margin_of_stability_stub(ml_mos: np.ndarray, ap_mos: np.ndarray) -> MarginOfStabilityCalculator:
    """
    A real MarginOfStabilityCalculator instance with its (heavy, c3d/biorbd-dependent)
    __init__ bypassed: ProbabilityOfInstability only ever reads .ML_MoS/.AP_MoS from it.
    """
    calculator = MarginOfStabilityCalculator.__new__(MarginOfStabilityCalculator)
    calculator.ML_MoS = ml_mos
    calculator.AP_MoS = ap_mos
    return calculator


def _make_ground_reaction_forces(nb_frames: int, foot1_stance_intervals, foot2_stance_intervals, amplitude=100.0):
    """
    Build a (2, 9, nb_frames) f_ext_sorted_filtered array with a vertical force (index 8,
    i.e. the 3rd component of the [6:9] force slice) that is `amplitude` during the given
    stance intervals and 0 otherwise, for each of the two platforms/feet.
    """
    f_ext = np.zeros((2, 9, nb_frames))
    for start, stop in foot1_stance_intervals:
        f_ext[0, 8, start:stop] = amplitude
    for start, stop in foot2_stance_intervals:
        f_ext[1, 8, start:stop] = amplitude
    return f_ext


def test_detect_heel_strikes_and_mos_per_step_and_poi_are_numerically_exact(make_experimental_data, real_subject):
    nb_frames = 100
    # foot1 (platform 0) stance: [10,30) and [50,70); foot2 (platform 1) stance: [30,50) and [70,90)
    f_ext_sorted_filtered = _make_ground_reaction_forces(
        nb_frames, foot1_stance_intervals=[(10, 30), (50, 70)], foot2_stance_intervals=[(30, 50), (70, 90)]
    )
    experimental_data = make_experimental_data(
        f_ext_sorted_filtered=f_ext_sorted_filtered,
        analogs_sampling_frequency=100,
        marker_sampling_frequency=100,  # factor = fs_grf / fs_mos = 1, so mos indices == analog indices
    )

    ml_mos = np.full(nb_frames, np.nan)
    ap_mos = np.full(nb_frames, np.nan)
    ml_mos[10:30], ap_mos[10:30] = 0.05, 0.10
    ml_mos[30:50], ap_mos[30:50] = -0.02, -0.01
    ml_mos[50:70], ap_mos[50:70] = 0.03, 0.08
    margin_of_stability = _make_margin_of_stability_stub(ml_mos, ap_mos)

    poi = ProbabilityOfInstability(
        marginofstability_calculator=margin_of_stability,
        experimental_data=experimental_data,
        subject=real_subject,
        skip_if_existing=False,
        heel_strike_threshold=20,
    )

    # Heel strikes: foot1 (marked 1) at 10 and 50, foot2 (marked 0) at 30 and 70, sorted by frame
    npt.assert_array_equal(poi.heel_strikes_all[:, 0], [10, 30, 50, 70])
    npt.assert_array_equal(poi.heel_strikes_all[:, 1], [1, 0, 1, 0])

    # 3 segments between the 4 heel strikes: [10:30], [30:50], [50:70]
    npt.assert_allclose(poi.mos_steps_ML, [0.05, -0.02, 0.03])
    npt.assert_allclose(poi.mos_steps_AP, [0.10, -0.01, 0.08])

    npt.assert_allclose(poi.mu_ML, np.mean([0.05, -0.02, 0.03]))
    npt.assert_allclose(poi.sigma_ML, np.std([0.05, -0.02, 0.03], ddof=1))
    npt.assert_allclose(poi.mu_AP, np.mean([0.10, -0.01, 0.08]))
    npt.assert_allclose(poi.sigma_AP, np.std([0.10, -0.01, 0.08], ddof=1))

    # PoI is the empirical percentage of steps with a negative MoS: 1 out of 3 steps for both
    npt.assert_allclose(poi.PoI_ML, 100 / 3)
    npt.assert_allclose(poi.PoI_AP, 100 / 3)


def test_poi_is_zero_when_no_step_has_negative_mos(make_experimental_data, real_subject):
    nb_frames = 60
    f_ext_sorted_filtered = _make_ground_reaction_forces(
        nb_frames, foot1_stance_intervals=[(5, 20), (35, 50)], foot2_stance_intervals=[(20, 35)]
    )
    experimental_data = make_experimental_data(
        f_ext_sorted_filtered=f_ext_sorted_filtered,
        analogs_sampling_frequency=60,
        marker_sampling_frequency=60,
    )
    ml_mos = np.full(nb_frames, 0.1)
    ap_mos = np.full(nb_frames, 0.2)
    margin_of_stability = _make_margin_of_stability_stub(ml_mos, ap_mos)

    poi = ProbabilityOfInstability(
        marginofstability_calculator=margin_of_stability,
        experimental_data=experimental_data,
        subject=real_subject,
        skip_if_existing=False,
    )
    npt.assert_allclose(poi.PoI_ML, 0.0)
    npt.assert_allclose(poi.PoI_AP, 0.0)


def test_poi_with_fewer_than_two_ml_steps_leaves_ml_as_nan_and_skips_ap(make_experimental_data, real_subject):
    # Only one ML step (2 heel strikes total on foot1, none on foot2) should short-circuit
    # compute_poi() before the AP branch is ever reached -- this documents the current
    # early-return behavior of ProbabilityOfInstability.compute_poi rather than asserting it
    # is the intended design.
    nb_frames = 40
    f_ext_sorted_filtered = _make_ground_reaction_forces(nb_frames, foot1_stance_intervals=[(5, 20)], foot2_stance_intervals=[])
    experimental_data = make_experimental_data(
        f_ext_sorted_filtered=f_ext_sorted_filtered,
        analogs_sampling_frequency=40,
        marker_sampling_frequency=40,
    )
    ml_mos = np.full(nb_frames, 0.1)
    ap_mos = np.full(nb_frames, 0.2)
    margin_of_stability = _make_margin_of_stability_stub(ml_mos, ap_mos)

    poi = ProbabilityOfInstability(
        marginofstability_calculator=margin_of_stability,
        experimental_data=experimental_data,
        subject=real_subject,
        skip_if_existing=False,
    )
    assert len(poi.mos_steps_ML) < 2
    assert np.isnan(poi.mu_ML) and np.isnan(poi.sigma_ML) and np.isnan(poi.PoI_ML)
    # AP was never reached
    assert poi.mu_AP is None
    assert poi.PoI_AP is None


def test_save_and_reload_round_trip(make_experimental_data, real_subject, tmp_path):
    nb_frames = 60
    f_ext_sorted_filtered = _make_ground_reaction_forces(
        nb_frames, foot1_stance_intervals=[(5, 20), (35, 50)], foot2_stance_intervals=[(20, 35)]
    )
    experimental_data = make_experimental_data(
        f_ext_sorted_filtered=f_ext_sorted_filtered,
        analogs_sampling_frequency=60,
        marker_sampling_frequency=60,
        result_folder=str(tmp_path),
    )
    ml_mos = np.full(nb_frames, 0.1)
    ap_mos = np.full(nb_frames, -0.2)
    margin_of_stability = _make_margin_of_stability_stub(ml_mos, ap_mos)

    poi = ProbabilityOfInstability(
        marginofstability_calculator=margin_of_stability,
        experimental_data=experimental_data,
        subject=real_subject,
        skip_if_existing=False,
    )

    result_path = poi.get_result_file_full_path()
    with open(result_path, "rb") as f:
        saved = pickle.load(f)
    npt.assert_allclose(saved["PoI_ML"], poi.PoI_ML)
    npt.assert_allclose(saved["PoI_AP"], poi.PoI_AP)

    outputs = poi.outputs()
    assert set(outputs.keys()) == {"PoI_ML", "PoI_AP"}
    npt.assert_allclose(outputs["PoI_ML"], poi.PoI_ML)
    npt.assert_allclose(outputs["PoI_AP"], poi.PoI_AP)

    reloaded = ProbabilityOfInstability(
        marginofstability_calculator=margin_of_stability,
        experimental_data=experimental_data,
        subject=real_subject,
        skip_if_existing=True,
    )
    assert reloaded.is_loaded_mos is True
    npt.assert_allclose(reloaded.PoI_ML, poi.PoI_ML)
    npt.assert_allclose(reloaded.PoI_AP, poi.PoI_AP)
