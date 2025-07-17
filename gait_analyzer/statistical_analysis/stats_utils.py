from enum import Enum
from typing import TypeAlias

from pingouin import pairwise_tests, print_table, plot_paired
from pandas import DataFrame
import numpy as np


class QuantityToExtractType(Enum):
    """
    Enum for the quantity to extract from the time series. This is used for the t-tests.
    """

    PEAK_TO_PEAK = "peak_to_peak"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    # RMS = "rms"


class StatsType:

    class PAIRED_T_TEST:
        def __init__(self, quantity_to_extract: QuantityToExtractType):
            self.value = "paired_t_test"
            if not isinstance(quantity_to_extract, QuantityToExtractType):
                raise ValueError("quantity_to_extract must be of type QuantityToExtractType")
            self.quantity_to_extract = quantity_to_extract

        def get_data_frame(self, data):
            metrics_name = self.quantity_to_extract.value
            data_df = None
            for condition in data["all"].keys():
                for subject in data["all"][condition].keys():

                    # Initialize here because we do not know in advance how many components the data has
                    if data_df is None:
                        nb_components = data["all"][condition][subject].shape[0]
                        metrics_names = [f"{metrics_name}_{i}" for i in range(nb_components)]
                        columns = ["group", "condition", "subject"] + metrics_names
                        data_df = DataFrame(columns)

                    # Get the right values based on the quantity to extract
                    if self.quantity_to_extract == QuantityToExtractType.PEAK_TO_PEAK:
                        values = np.nanmax(data["all"][condition][subject], axis=1) - np.nanmin(
                            data["all"][condition][subject], axis=1
                        )
                    elif self.quantity_to_extract == QuantityToExtractType.MEAN:
                        values = np.nanmean(data["all"][condition][subject], axis=1)
                    elif QuantityToExtractType.MIN:
                        values = np.nanmin(data["all"][condition][subject], axis=1)
                    elif QuantityToExtractType.MAX:
                        values = np.nanmax(data["all"][condition][subject], axis=1)
                    else:
                        raise NotImplementedError(f"Unsupported quantity to extract: {self.quantity_to_extract.value}")

                # Add the data to the dataframe
                data_df = data_df.append(
                    {
                        "group": "all",
                        "condition": condition,
                        "subject": subject,
                        **{f"{metrics_name}_{i}": values[i] for i in range(nb_components)},
                    },
                    ignore_index=True,
                )

            return data_df, metrics_names

        def perform_stats(self, data):
            """
            We assume that the data is normally distributed. If you'd like to check normality, please notify the developers.
            """
            if "all" not in data.keys():
                raise RuntimeError("PAIRED_T_TEST can only be used to compare conditions, not groups.")

            data_df, metrics_names = self.get_data_frame(data)
            for metrics_name in metrics_names:
                posthoc = pairwise_tests(
                    data=data_df,
                    dv=metrics_name,
                    within="conditions",
                    subject="subject",
                    parametric=True,
                    effsize="hedges",
                )
                print_table(posthoc, floatfmt=".3f")

        def plot_stats(self, data) -> None:
            data_df, metrics_names = self.get_data_frame(data)
            for metrics_name in metrics_names:
                ax = plot_paired(data=data_df, dv=metrics_name, within="conditions", subject="subject")
                ax.show()

    class SPM1D:
        def __init__(self):
            self.value = "spm1d"
            raise NotImplementedError(
                "SPM1D statistical analysis is not implemented yet. If you need it, please open an issue on GitHub."
            )


Stats: TypeAlias = StatsType.PAIRED_T_TEST | StatsType.SPM1D
