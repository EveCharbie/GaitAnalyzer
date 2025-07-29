from gait_analyzer.plots.plot_abstract import PlotAbstract
from gait_analyzer.plots.plot_utils import LegToPlot, PlotType
from gait_analyzer.statistical_analysis.organized_result import OrganizedResult


class PlotLegData(PlotAbstract):
    def __init__(
        self,
        organized_result: OrganizedResult,
    ):
        # Checks
        if not isinstance(organized_result, OrganizedResult):
            raise ValueError("organized_result must be OrganizedResult type")

        # Initialize the parent class (PlotAbstract)
        super().__init__(organized_result=organized_result)

        # Prepare the plot
        self.get_plot_indices_and_labels()

    def get_plot_indices_and_labels(self):
        # TODO: Charbie -> This section could be generalized to handle the DoFs specific to the model used.

        # Establish the plot indices and labels
        if self.organized_result.plot_type == PlotType.GRF:
            plot_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            plot_labels = ["CoPx", "CoPy", "CoPz", "Mx", "My", "Mz", "Fx", "Fy", "Fz"]
        elif self.organized_result.plot_type == PlotType.EMG:
            plot_idx = None
            plot_labels = None
        elif self.organized_result.plot_type == PlotType.MUSCLE_FORCES:
            plot_idx = None
            if self.organized_result.leg_to_plot == LegToPlot.RIGHT:
                plot_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            elif self.organized_result.leg_to_plot == LegToPlot.LEFT:
                plot_idx = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            elif self.organized_result.leg_to_plot == LegToPlot.BOTH:
                plot_idx = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
            else:
                raise ValueError(
                    f"leg_to_plot {self.organized_result.leg_to_plot} not recognized. It must be a in LegToPlot.RIGHT, LegToPlot.LEFT, LegToPlot.BOTH, or LegToPlot.DOMINANT."
                )
            plot_labels = ['semiten',
                             'bifemlh',
                             'sar',
                             'tfl',
                             'med_gas',
                             'lat_gas',
                             'soleus',
                             'tib_post',
                             'tib_ant',
                             'per_long',]
        else:
            if self.organized_result.leg_to_plot == LegToPlot.RIGHT:
                plot_idx = [20, 3, 6, 9, 10]
            elif self.organized_result.leg_to_plot == LegToPlot.LEFT:
                plot_idx = [20, 3, 13, 16, 17]
            elif self.organized_result.leg_to_plot == LegToPlot.BOTH:
                plot_idx = [[20, 3, 6, 9, 10], [20, 3, 13, 16, 17]]
            else:
                raise ValueError(
                    f"leg_to_plot {self.organized_result.leg_to_plot} not recognized. It must be a in LegToPlot.RIGHT, LegToPlot.LEFT, LegToPlot.BOTH, or LegToPlot.DOMINANT."
                )
            plot_labels = ["Torso", "Pelvis", "Hip", "Knee", "Ankle"]

        # Establish plot specific parameters
        if self.organized_result.leg_to_plot in [LegToPlot.RIGHT, LegToPlot.LEFT]:
            n_cols = 1
            fig_width = 5
        else:
            n_cols = 2
            fig_width = 10

        # Store the output
        self.plot_idx = plot_idx
        self.plot_labels = plot_labels
        self.n_cols = n_cols
        self.fig_width = fig_width
