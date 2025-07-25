import os
import pickle
import numpy as np

from gait_analyzer.statistical_analysis.organized_result import OrganizedResult
from gait_analyzer.statistical_analysis.stats_utils import StatsType, Stats


class StatsPerformer:
    def __init__(
        self,
        organized_result: OrganizedResult,
        stats_type: Stats,
    ):
        # Checks
        if not isinstance(organized_result, OrganizedResult):
            raise ValueError("organized_result must be a OrganizedResult")
        if not isinstance(stats_type, (StatsType.PAIRED_T_TEST, StatsType.SPM1D)):
            raise ValueError("stats_type must be a StatsType")

        # Initial attributes
        self.organized_result = organized_result
        self.stats_type = stats_type

    def perform_stats(self):
        self.stats_type.perform_stats(self.organized_result.results.mean_data_per_subject)

    def plot_stats(self, save_plot_name: str = None, order: list[str] = None):
        return self.stats_type.plot_stats(self.organized_result.results.mean_data_per_subject, save_plot_name, order)
