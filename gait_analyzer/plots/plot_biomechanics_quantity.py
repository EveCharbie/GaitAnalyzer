from gait_analyzer.plots.plot_abstract import PlotAbstract
from gait_analyzer.plots.plot_utils import PlotType
from gait_analyzer.statistical_analysis.organized_result import OrganizedResult


class PlotBiomechanicsQuantity(PlotAbstract):
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

        # Establish the plot indices and labels
        if self.organized_result.plot_type == PlotType.ANGULAR_MOMENTUM:
            plot_idx = [0, 1, 2]
            plot_labels = ["Hx", "Hy", "Hz"]
        else:
            raise RuntimeError(f"PlotType not recognized {self.organized_result.plot_type}")

        # For now, we consider that all biomechanics quantities are 3D
        n_cols = 1
        fig_width = 5

        # Store the output
        self.plot_idx = plot_idx
        self.plot_labels = plot_labels
        self.n_cols = n_cols
        self.fig_width = fig_width
