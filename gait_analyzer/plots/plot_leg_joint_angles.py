
from gait_analyzer.plots.plot_abstract import PlotAbstract, EventIndexType
from gait_analyzer.plots.plot_utils import LegToPlot, PlotType


class PlotLegData(PlotAbstract):
    def __init__(
        self, result_folder: str, leg_to_plot: LegToPlot, plot_type: PlotType, conditions_to_compare: list[str], unique_event_to_split: list[callable] = None
    ):
        # Checks
        if not isinstance(plot_type, PlotType):
            raise ValueError("plot_type must be PlotType type")

        # Initial attributes
        self.plot_type = plot_type
        self.event_index_type = EventIndexType.ANALOGS if self.plot_type == PlotType.GRF else EventIndexType.MARKERS

        # Initialize the parent class (PlotAbstract)
        super(PlotLegData, self).__init__(result_folder, leg_to_plot, conditions_to_compare, self.get_data_to_split, unique_event_to_split)

        # Prepare the plot
        self.get_plot_indices_and_labels()

    def get_data_to_split(self, data):
        if self.plot_type == PlotType.GRF:
            if self.leg_to_plot == LegToPlot.LEFT:
                data_to_split = data[self.plot_type.value][0, :, :].squeeze()
            elif self.leg_to_plot == LegToPlot.RIGHT:
                data_to_split = data[self.plot_type.value][1, :, :].squeeze()
            else:
                raise NotImplementedError(
                    "Plotting GRF on both legs is not implemented yet. If you encounter this error, please notify the developers."
                )
        else:
            data_to_split = data[self.plot_type.value]
        return data_to_split

    def get_plot_indices_and_labels(self):

        # Establish the plot indices and labels
        if self.plot_type == PlotType.GRF:
            plot_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            plot_labels = ["CoPx", "CoPy", "CoPz", "Mx", "My", "Mz", "Fx", "Fy", "Fz"]
        else:
            if self.leg_to_plot == LegToPlot.RIGHT:
                plot_idx = [20, 3, 6, 9, 10]
            elif self.leg_to_plot == LegToPlot.LEFT:
                plot_idx = [20, 3, 13, 16, 17]
            elif self.leg_to_plot == LegToPlot.BOTH:
                plot_idx = [[20, 3, 6, 9, 10], [20, 3, 13, 16, 17]]
            else:
                raise ValueError(
                    f"leg_to_plot {self.leg_to_plot} not recoginzed. It must be a in LegToPlot.RIGHT, LegToPlot.LEFT, LegToPlot.BOTH, or LegToPlot.DOMINANT."
                )
            plot_labels = ["Torso", "Pelvis", "Hip", "Knee", "Ankle"]

        # Establish plot specific parameters
        if self.leg_to_plot in [LegToPlot.RIGHT, LegToPlot.LEFT]:
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
