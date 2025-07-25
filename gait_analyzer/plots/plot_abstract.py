import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

from gait_analyzer.plots.plot_utils import get_unit_names
from gait_analyzer.statistical_analysis.organized_result import OrganizedResult


class PlotAbstract:
    def __init__(
        self,
        organized_result: OrganizedResult,
    ):
        # Checks
        if not isinstance(organized_result, OrganizedResult):
            raise ValueError("organized_result must be a OrganizedResult")

        # Initial attributes
        self.organized_result = organized_result
        self.data = organized_result.results.data

        # Extended attributes
        self.plot_idx = None
        self.plot_labels = None
        self.n_cols = None
        self.fig_width = None
        self.fig = None

    def draw_plot(self):
        # TODO: Charbie -> combine plots in one figure (Q and Power for example side by side)
        if self.plot_idx is not None and self.n_cols is not None:
            n_rows = len(self.plot_idx) // self.n_cols
        else:
            first_key = list(self.data.keys())[0]
            n_rows = self.data[first_key][0].shape[0]
            self.plot_idx = list(range(n_rows))
            self.n_cols = 1
        fig, axs = plt.subplots(n_rows, self.n_cols, figsize=(self.fig_width, 10))
        n_data_to_plot = len(self.data)
        normalized_time = np.linspace(0, 100, self.organized_result.nb_frames_interp)

        # Plot the data
        unit_str = get_unit_names(self.organized_result.plot_type)
        lines_list = []
        labels_list = []
        subject_mean, subject_std = self.organized_result.results.mean_per_subject()
        group_mean, group_std = self.organized_result.results.mean_per_group(subject_mean=subject_mean)
        for group_name in group_mean.keys():
            color_index = 0
            nb_condition = len(group_mean[group_name].keys())
            colors = [colormaps["magma"](i / nb_condition) for i in range(nb_condition)]
            for i_condition, condition_name in enumerate(group_mean[group_name].keys()):
                mean_data = group_mean[group_name][condition_name][self.plot_idx, :]
                std_data = group_std[group_name][condition_name][self.plot_idx, :]
                label = condition_name if group_name == "all" else group_name
                for i_ax, ax in enumerate(axs):
                    ax.fill_between(
                        normalized_time,
                        mean_data[i_ax, :] - std_data[i_ax, :],
                        mean_data[i_ax, :] + std_data[i_ax, :],
                        color=colors[color_index],
                        alpha=0.3,
                    )
                    if i_ax == 0:
                        lines_list += ax.plot(
                            normalized_time, mean_data[i_ax, :], label=label, color=colors[color_index]
                        )
                        labels_list += [label]
                    else:
                        ax.plot(normalized_time, mean_data[i_ax, :], label=label, color=colors[color_index])
                    this_unit_str = unit_str if isinstance(unit_str, str) else unit_str[i_ax]
                    if self.plot_labels is not None:
                        ax.set_ylabel(f"{self.plot_labels[i_ax]} " + this_unit_str)
                    else:
                        ax.set_ylabel(f"Data {i_ax} " + this_unit_str)
                color_index += 1
                axs[-1].set_xlabel("Normalized time [%]")

        axs[0].legend(lines_list, labels_list, bbox_to_anchor=(0.5, 1.6), loc="upper center")
        fig.subplots_adjust(top=0.9)
        fig.tight_layout()
        fig.savefig(f"plot_conditions_{self.organized_result.plot_type.value}.png")
        self.fig = fig

    def save(self, file_name: str):
        self.fig.savefig(file_name)

    def show(self):
        self.fig.show()
