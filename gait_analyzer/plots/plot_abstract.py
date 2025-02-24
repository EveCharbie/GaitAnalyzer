import os
from enum import Enum
import pickle
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colormaps

from gait_analyzer.operator import Operator
from gait_analyzer.plots.plot_utils import split_cycles, mean_cycles, LegToPlot, get_unit_names


class EventIndexType(Enum):
    MARKERS = "markers"
    ANALOGS = "analogs"


class PlotAbstract:
    def __init__(
        self, result_folder: str, leg_to_plot: LegToPlot, conditions_to_compare: list[str], get_data_to_split: callable
    ):
        # Checks
        if not isinstance(result_folder, str):
            raise ValueError("result_folder must be a string")
        if not os.path.isdir(result_folder):
            raise ValueError(f"The result_folder specified {result_folder} does not exist.")
        if not isinstance(leg_to_plot, LegToPlot):
            raise ValueError("leg_to_plot must be LegToPlot type")
        if not isinstance(conditions_to_compare, list):
            raise ValueError("conditions_to_compare must be a list")
        if not all(isinstance(cond, str) for cond in conditions_to_compare):
            raise ValueError("conditions_to_compare must be a list of strings")
        if not callable(get_data_to_split):
            raise ValueError("get_data_to_split must be a callable")

        # Initial attributes
        self.result_folder = result_folder
        self.conditions_to_compare = conditions_to_compare
        self.leg_to_plot = leg_to_plot
        self.get_data_to_split = get_data_to_split

        # Extended attributes
        self.cycles_data = None
        self.plot_idx = None
        self.plot_labels = None
        self.n_cols = None
        self.fig_width = None
        self.fig = None
        self.event_index_type = None

        # Prepare the plot
        self.prepare_cycles()

    def get_event_index(self, event, cycles_to_analyze, analog_time_vector, markers_time_vector):
        if self.event_index_type == EventIndexType.ANALOGS:
            event_index = event
        else:
            event_idx_markers = Operator.from_analog_frame_to_marker_frame(
                analog_time_vector,
                markers_time_vector,
                event,
            )
            events_idx_q = np.array(event_idx_markers)[cycles_to_analyze.start : cycles_to_analyze.stop]
            events_idx_q -= events_idx_q[0]
            event_index = list(events_idx_q)
        return event_index

    def get_splitted_cycles(self, current_file: str, partial_output_file_name: str):
        this_cycles_data = None
        condition_name = None
        if current_file.endswith("results.pkl"):
            with open(current_file, "rb") as f:
                data = pickle.load(f)
            subject_name = data["subject_name"]
            subject_mass = data["subject_mass"]
            condition_name = partial_output_file_name.replace(subject_name, "").replace("_results.pkl", "")
            if self.leg_to_plot == LegToPlot.DOMINANT:
                raise NotImplementedError(
                    "Plotting the dominant leg is not implemented yet. If you encounter this error, please notify the developers."
                )

            if condition_name in self.conditions_to_compare:
                event_index = self.get_event_index(
                    event=data["events"]["right_leg_heel_touch"],
                    cycles_to_analyze=data["cycles_to_analyze"],
                    analog_time_vector=data["analogs_time_vector"],
                    markers_time_vector=data["markers_time_vector"],
                )
                data_to_split = self.get_data_to_split(data)
                this_cycles_data = split_cycles(
                    data_to_split, event_index, plot_type=self.plot_type, subject_mass=subject_mass
                )
        return this_cycles_data, condition_name

    def prepare_cycles(self):
        """
        This function prepares the data to plot.
        """

        # TODO: ThomasAout/FloEthv -> please decide if you want to compare mean of all participants
        cycles_data = {cond: [] for cond in self.conditions_to_compare}
        # Load the treated data to plot
        for result_file in os.listdir(self.result_folder):
            if os.path.isdir(os.path.join(self.result_folder, result_file)):
                for file_in_sub_folder in os.listdir(os.path.join(self.result_folder, result_file)):
                    file_in_sub_folder = os.path.join(self.result_folder, result_file, file_in_sub_folder)
                    partial_output_file_name = file_in_sub_folder.replace(f"{self.result_folder}/{result_file}/", "")
                    this_cycles_data, condition_name = self.get_splitted_cycles(
                        current_file=file_in_sub_folder, partial_output_file_name=partial_output_file_name
                    )
                    if this_cycles_data is not None:
                        cycles_data[condition_name] += this_cycles_data
            else:
                if result_file.endswith("results.pkl"):
                    this_cycles_data, condition_name = self.get_splitted_cycles(
                        current_file=result_file, partial_output_file_name=result_file
                    )
                    if this_cycles_data is not None:
                        cycles_data[condition_name] += this_cycles_data

        self.cycles_data = cycles_data

    def draw_plot(self):
        # TODO: Charbie -> combine plots in one figure (Q and Power for example side by side)
        n_rows = len(self.plot_idx) // self.n_cols
        fig, axs = plt.subplots(n_rows, self.n_cols, figsize=(self.fig_width, 10))
        n_data_to_plot = len(self.cycles_data)
        colors = [colormaps["magma"](i / n_data_to_plot) for i in range(n_data_to_plot)]
        nb_frames_interp = 101
        normalized_time = np.linspace(0, 100, nb_frames_interp)

        # Store the mean ans std for further analysis
        all_mean_data = np.zeros((n_data_to_plot, len(self.plot_idx), nb_frames_interp))
        all_std_data = np.zeros((n_data_to_plot, len(self.plot_idx), nb_frames_interp))

        # Plot the data
        unit_str = get_unit_names(self.plot_type)
        lines_list = []
        labels_list = []
        for i_condition, key in enumerate(self.cycles_data):
            cycles = self.cycles_data[key]
            # Compute the mean over cycles
            if len(cycles) == 0:
                continue
            mean_data, std_data = mean_cycles(cycles, index_to_keep=self.plot_idx, nb_frames_interp=nb_frames_interp)
            all_mean_data[i_condition, :, :] = mean_data
            all_std_data[i_condition, :, :] = std_data
            for i_ax, ax in enumerate(axs):
                ax.fill_between(
                    normalized_time,
                    mean_data[i_ax, :] - std_data[i_ax, :],
                    mean_data[i_ax, :] + std_data[i_ax, :],
                    color=colors[i_condition],
                    alpha=0.3,
                )
                if i_ax == 0:
                    lines_list += ax.plot(normalized_time, mean_data[i_ax, :], label=key, color=colors[i_condition])
                    labels_list += [key]
                else:
                    ax.plot(normalized_time, mean_data[i_ax, :], label=key, color=colors[i_condition])
                this_unit_str = unit_str if isinstance(unit_str, str) else unit_str[i_ax]
                ax.set_ylabel(f"{self.plot_labels[i_ax]} " + this_unit_str)
            axs[-1].set_xlabel("Normalized time [%]")

        axs[0].legend(lines_list, labels_list, bbox_to_anchor=(0.5, 1.6), loc="upper center")
        fig.subplots_adjust(top=0.9)
        fig.tight_layout()
        fig.savefig(f"plot_conditions_{self.plot_type.value}.png")
        self.fig = fig

    def save(self, file_name: str):
        self.fig.savefig(file_name)

    def show(self):
        self.fig.show()
