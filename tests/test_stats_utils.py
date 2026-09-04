import numpy as np
import numpy.testing as npt
import pytest

from gait_analyzer.statistical_analysis.stats_utils import QuantityToExtractType, StatsType


def test_quantity_to_extract_type_values():
    assert QuantityToExtractType.PEAK_TO_PEAK.value == "peak_to_peak"
    assert QuantityToExtractType.MEAN.value == "mean"
    assert QuantityToExtractType.MAX.value == "max"
    assert QuantityToExtractType.MIN.value == "min"


def test_paired_t_test_requires_quantity_to_extract_type():
    with pytest.raises(ValueError, match="quantity_to_extract must be of type QuantityToExtractType"):
        StatsType.PAIRED_T_TEST(quantity_to_extract="mean")


def test_paired_t_test_stores_quantity_to_extract():
    stats = StatsType.PAIRED_T_TEST(quantity_to_extract=QuantityToExtractType.MEAN)
    assert stats.value == "paired_t_test"
    assert stats.quantity_to_extract == QuantityToExtractType.MEAN


def _synthetic_data(nb_components=2, nb_frames=10):
    """
    Build a synthetic {"all": {condition: {subject: ndarray}}} structure like the one
    OrganizedResult.results.mean_data_per_subject produces, with two subjects and two
    conditions with known, hand-computable values.
    """
    rng = np.random.default_rng(123)
    data = {"all": {}}
    for condition in ["cond_a", "cond_b"]:
        data["all"][condition] = {}
        for subject in ["subj_1", "subj_2"]:
            data["all"][condition][subject] = rng.normal(loc=1.0, scale=0.1, size=(nb_components, nb_frames))
    return data


def test_get_data_frame_mean_matches_manual_nanmean():
    data = _synthetic_data(nb_components=2, nb_frames=8)
    stats = StatsType.PAIRED_T_TEST(quantity_to_extract=QuantityToExtractType.MEAN)
    data_df, metrics_names = stats.get_data_frame(data)

    assert metrics_names == ["mean_0", "mean_1"]
    assert set(data_df["condition"]) == {"cond_a", "cond_b"}
    assert set(data_df["subject"]) == {"subj_1", "subj_2"}
    assert len(data_df) == 4  # 2 conditions x 2 subjects

    for condition in ["cond_a", "cond_b"]:
        for subject in ["subj_1", "subj_2"]:
            row = data_df[(data_df["condition"] == condition) & (data_df["subject"] == subject)]
            expected = np.nanmean(data["all"][condition][subject], axis=1)
            npt.assert_allclose(row["mean_0"].values[0], expected[0])
            npt.assert_allclose(row["mean_1"].values[0], expected[1])


def test_get_data_frame_peak_to_peak_matches_manual_computation():
    data = _synthetic_data(nb_components=1, nb_frames=15)
    stats = StatsType.PAIRED_T_TEST(quantity_to_extract=QuantityToExtractType.PEAK_TO_PEAK)
    data_df, metrics_names = stats.get_data_frame(data)

    assert metrics_names == ["peak_to_peak_0"]
    for condition in ["cond_a", "cond_b"]:
        for subject in ["subj_1", "subj_2"]:
            row = data_df[(data_df["condition"] == condition) & (data_df["subject"] == subject)]
            values = data["all"][condition][subject]
            expected = np.nanmax(values, axis=1) - np.nanmin(values, axis=1)
            npt.assert_allclose(row["peak_to_peak_0"].values[0], expected[0])


def test_perform_stats_requires_all_key():
    data = {"group_1": {}}
    stats = StatsType.PAIRED_T_TEST(quantity_to_extract=QuantityToExtractType.MEAN)
    with pytest.raises(RuntimeError, match="can only be used to compare conditions, not groups"):
        stats.perform_stats(data)


def test_spm1d_not_implemented():
    with pytest.raises(NotImplementedError, match="SPM1D statistical analysis is not implemented"):
        StatsType.SPM1D()
