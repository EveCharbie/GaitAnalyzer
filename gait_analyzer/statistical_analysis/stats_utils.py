from enum import Enum
from typing import TypeAlias

from pingouin import pairwise_tests, print_table, plot_paired
from pandas import DataFrame, concat
import numpy as np
import matplotlib.pyplot as plt


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

                    if data_df is None:
                        # Initialize here because we do not know in advance how many components the data has
                        nb_components = data["all"][condition][subject].shape[0]
                        metrics_names = [f"{metrics_name}_{i}" for i in range(nb_components)]
                        data_df = DataFrame([
                            {
                                "group": "all",
                                "condition": condition,
                                "subject": subject,
                                **{f"{metrics_name}_{i}": values[i] for i in range(nb_components)},
                            }]
                        )
                    else:
                        # Add the data to the dataframe
                        new_row = DataFrame([
                            {
                                "group": "all",
                                "condition": condition,
                                "subject": subject,
                                **{f"{metrics_name}_{i}": values[i] for i in range(nb_components)},
                            }]
                        )

                        data_df = concat([data_df, new_row], ignore_index=True)
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
                    within="condition",
                    subject="subject",
                    parametric=True,
                    effsize="hedges",
                )
                print_table(posthoc, floatfmt=".3f")

        def plot_stats(self, data, save_plot_name: str = None, order: list[str] = None) -> None:
            data_df, metrics_names = self.get_data_frame(data)
            nb_metric = len(metrics_names)
            fig, axs = plt.subplots(nb_metric, 1, figsize=(5, 5))
            min_value = data_df[metrics_names].min().min()
            max_value = data_df[metrics_names].max().max()
            for i_metric, metrics_name in enumerate(metrics_names):
                plot_paired(data=data_df,
                            dv=metrics_name,
                            within="condition",
                            subject="subject",
                            ax=axs[i_metric],
                            order=order)
                axs[i_metric].set_ylim((min_value - 0.1 * max_value, 1.1 * max_value))
            if save_plot_name:
                plt.savefig(save_plot_name, dpi=300, bbox_inches="tight")
            plt.show()

    class SPM1D:
        def __init__(self):
            self.value = "spm1d"
            raise NotImplementedError(
                "SPM1D statistical analysis is not implemented yet. If you need it, please open an issue on GitHub."
            )


Stats: TypeAlias = StatsType.PAIRED_T_TEST | StatsType.SPM1D
