import os
import pickle
import numpy as np

from gait_analyzer.operator import Operator
from gait_analyzer.plots.plot_utils import split_cycle, split_cycles, mean_cycles
from gait_analyzer.plots.plot_utils import EventIndexType, LegToPlot, PlotType


class OrganizedResult:
    def __init__(
        self,
        result_folder: str,
        conditions_to_compare: list[str],
        plot_type: PlotType,
        groups_to_compare: dict[str, list[str]] | None = None,
        leg_to_plot: LegToPlot = LegToPlot.RIGHT,
        unique_event_to_split: dict = None,
        nb_frames_interp: int = 101
    ):
        # Checks
        if not isinstance(result_folder, str):
            raise ValueError("result_folder must be a string")
        if not os.path.isdir(result_folder):
            raise ValueError(f"The result_folder specified {result_folder} does not exist.")
        if not isinstance(conditions_to_compare, list):
            raise ValueError("conditions_to_compare must be a list")
        if not all(isinstance(cond, str) for cond in conditions_to_compare):
            raise ValueError("conditions_to_compare must be a list of strings")
        if groups_to_compare is not None:
            if not isinstance(groups_to_compare, dict):
                raise ValueError("groups_to_compare must be a list")
            existing_subject_names = []
            for group_name in groups_to_compare:
                if not all(isinstance(subject, str) for subject in groups_to_compare[group_name]):
                    raise ValueError("groups_to_compare must be a dict of lists of strings")
                for subject in groups_to_compare[group_name]:
                    if subject in existing_subject_names:
                        raise ValueError(f"Subject {subject} is already in groups_to_compare. Please check the groups.")
                    else:
                        existing_subject_names.append(subject)
        if not isinstance(plot_type, PlotType):
            raise ValueError("plot_type must be a PlotType type")
        if not isinstance(leg_to_plot, LegToPlot):
            raise ValueError("leg_to_plot must be a LegToPlot type")
        if unique_event_to_split is not None:
            if not isinstance(unique_event_to_split, dict):
                raise ValueError("unique_event_to_split must be a list or None")
            if list(unique_event_to_split.keys()) != ["event_index_type", "start", "stop"]:
                raise ValueError(
                    "unique_event_to_split must be a dict with keys event_index_type (weather to express the index in marker indices of analog indices), start a callable giving the first frame of the cycle, and stop a callable giving the last frame of the cycle."
                )
            if not (callable(unique_event_to_split["start"]) and callable(unique_event_to_split["stop"])):
                raise ValueError("unique_event_to_split must be a dict of callables")
        if not isinstance(nb_frames_interp, int):
            raise ValueError("nb_frames_interp must be an integer")

        # Initial attributes
        self.result_folder = result_folder
        self.conditions_to_compare = conditions_to_compare
        self.groups_to_compare = groups_to_compare
        self.plot_type = plot_type
        self.leg_to_plot = leg_to_plot
        self.unique_event_to_split = unique_event_to_split
        self.nb_frames_interp = nb_frames_interp
        event_index_type = (
            EventIndexType.ANALOGS if self.plot_type in [PlotType.GRF, PlotType.EMG] else EventIndexType.MARKERS
        )
        self.event_index_type = event_index_type

        # Extended attributes
        self.results = ResultObject(groups_to_compare, self.nb_frames_interp)

        # Prepare the plot
        self.prepare_cycles()

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

    def get_event_index(self, event, cycles_to_analyze, analog_time_vector, markers_time_vector):
        if self.event_index_type == EventIndexType.ANALOGS:
            event_index = event
        elif self.event_index_type == EventIndexType.MARKERS:
            event_idx_markers = Operator.from_analog_frame_to_marker_frame(
                analog_time_vector,
                markers_time_vector,
                event,
            )
            start_cycle = 0 if cycles_to_analyze is None else cycles_to_analyze.start
            end_cycle = -1 if cycles_to_analyze is None else cycles_to_analyze.stop
            events_idx_q = np.array(event_idx_markers)[start_cycle:end_cycle]
            events_idx_q -= events_idx_q[0]
            event_index = list(events_idx_q)
        else:
            raise RuntimeError("The event_index_type must be either EventIndexType.ANALOGS or EventIndexType.MARKERS.")
        return event_index

    def get_splitted_cycles(self, current_file: str, partial_output_file_name: str):
        this_cycles_data = None
        condition_name = None
        subject_name = None
        if current_file.endswith("results.pkl"):
            with open(current_file, "rb") as f:
                data = pickle.load(f)
            subject_name = data["subject_name"]
            subject_mass = data["subject_mass"]
            condition_name = partial_output_file_name.replace(subject_name, "").replace("_results.pkl", "")

            if self.groups_to_compare is not None:
                all_subjects = []
                for group in self.groups_to_compare:
                    all_subjects += self.groups_to_compare[group]
                if subject_name not in all_subjects:
                    raise ValueError(
                        f"Subject {subject_name} not found in groups_to_compare. "
                        f"Please check the groups_to_compare dictionary."
                    )
            if condition_name in self.conditions_to_compare:
                if isinstance(data["events"], list):
                    cycle_start = self.unique_event_to_split["start"](data)
                    cycle_end = self.unique_event_to_split["stop"](data)
                    if self.unique_event_to_split["event_index_type"] == EventIndexType.ANALOGS:
                        cycle_start_idx = cycle_start
                        cycle_end_idx = cycle_end
                    elif self.unique_event_to_split["event_index_type"] == EventIndexType.MARKERS:
                        cycle_start_idx = Operator.from_analog_frame_to_marker_frame(
                            data["analogs_time_vector"],
                            data["markers_time_vector"],
                            cycle_start,
                        )
                        cycle_end_idx = Operator.from_analog_frame_to_marker_frame(
                            data["analogs_time_vector"],
                            data["markers_time_vector"],
                            cycle_end,
                        )
                    else:
                        raise ValueError("event_index_type must be a EventIndexType.")

                    data_to_split = self.get_data_to_split(data)
                    this_cycles_data = split_cycle(
                        data_to_split,
                        cycle_start_idx,
                        cycle_end_idx,
                        plot_type=self.plot_type,
                        subject_mass=subject_mass,
                    )
                else:
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
        return this_cycles_data, condition_name, subject_name

    def prepare_cycles(self):
        """
        This function prepares the data to plot.
        """

        # Load the treated data to plot
        for result_file in os.listdir(self.result_folder):
            if os.path.isdir(os.path.join(self.result_folder, result_file)):
                if result_file in ["Geometry", "Geometry_cleaned", "hide_and_seek"]:
                    continue
                for file_in_sub_folder in os.listdir(os.path.join(self.result_folder, result_file)):
                    file_in_sub_folder = os.path.join(self.result_folder, result_file, file_in_sub_folder)
                    partial_output_file_name = file_in_sub_folder.replace(f"{self.result_folder}/{result_file}/", "")
                    if file_in_sub_folder.endswith("results.pkl"):
                        this_cycles_data, condition_name, subject_name = self.get_splitted_cycles(
                            current_file=file_in_sub_folder, partial_output_file_name=partial_output_file_name
                        )
                        if self.groups_to_compare is not None:
                            for group in self.groups_to_compare:
                                if subject_name in self.groups_to_compare[group]:
                                    self.results.add(data=this_cycles_data, subject_name=subject_name, condition_name=condition_name, group_name=group)
                        else:
                            self.results.add(data=this_cycles_data, subject_name=subject_name,
                                             condition_name=condition_name)

            else:
                if result_file.endswith("results.pkl"):
                    this_cycles_data, condition_name, subject_name = self.get_splitted_cycles(
                        current_file=result_file, partial_output_file_name=result_file
                    )
                    self.results.add(data=this_cycles_data, subject_name=subject_name,
                                     condition_name=condition_name)

    def save(self, file_path: str):
        """
        Save the organized result to a file.
        Parameters
        ----------
        file_path: str
            The path to save the organized result.
        """
        if file_path.endswith("_results.pkl"):
            raise RuntimeError("The file_path cannot end with '_results.pkl'. This is reserved for the result from the AnalysisPerformer.")
        self.results.save(file_path=file_path)


class ResultObject:
    def __init__(self, groups_to_compare: dict[str, list[str]], nb_frames_interp):

        # Initial attributes
        self.groups_to_compare = groups_to_compare
        self.nb_frames_interp = nb_frames_interp

        # Extended attributes
        self.data = {}

    def add(self,
            data: list[np.ndarray],
            subject_name: str,
            condition_name: str,
            group_name: str | None = None):
        """
        Add a result to the ResultObject.
        .
        Parameters
        ----------
        data: list[np.ndarray]
            The data to add, typically a list of numpy arrays representing the cycles.
        subject_name: str
            The name of the subject.
        condition_name: str
            The name of the condition.
        group_name: str | None
            The name of the group, if applicable. If None, the data is not grouped.
        """
        if data is None:
            pass
        else:
            # Checks
            if not isinstance(data, list):
                raise ValueError("data must be a list of numpy arrays.")
            if not all(isinstance(d, np.ndarray) for d in data):
                raise ValueError("All elements in data must be numpy arrays.")

            # Add safely
            group_name = group_name if group_name is not None else "all"
            if group_name not in self.data.keys():
                self.data[group_name] = {}
            if condition_name not in self.data[group_name].keys():
                self.data[group_name][condition_name] = {}
            if subject_name not in self.data[group_name][condition_name].keys():
                self.data[group_name][condition_name][subject_name] = []

            self.data[group_name][condition_name][subject_name] += data

    def mean_per_subject(self):
        subject_mean = {}
        subject_std = {}
        for group_name in self.data.keys():
            subject_mean[group_name] = {}
            subject_std[group_name] = {}
            for condition_name in self.data[group_name].keys():
                subject_mean[group_name][condition_name] = {}
                subject_std[group_name][condition_name] = {}
                for subject_name in self.data[group_name][condition_name].keys():
                    if len(self.data[group_name][condition_name][subject_name]) == 0:
                        continue
                    mean_data, std_data = mean_cycles(data=self.data[group_name][condition_name][subject_name], index_to_keep=None, nb_frames_interp=self.nb_frames_interp)
                    subject_mean[group_name][condition_name][subject_name] = mean_data
                    subject_std[group_name][condition_name][subject_name] = std_data
        return subject_mean, subject_std

    def mean_per_group(self, subject_mean: dict[dict[dict[np.ndarray]]]):
        group_mean = {}
        group_std = {}
        for group_name in subject_mean.keys():
            group_mean[group_name] = {}
            group_std[group_name] = {}
            for condition_name in subject_mean[group_name].keys():
                group_data = None
                for subject_name in subject_mean[group_name][condition_name].keys():
                    if group_data is None:
                        group_data = subject_mean[group_name][condition_name][subject_name][:, :, np.newaxis]
                    else:
                        group_data = np.concatenate(
                            (group_data, subject_mean[group_name][condition_name][subject_name][:, :, np.newaxis]), axis=2
                        )
                if group_data is not None:
                    group_mean[group_name][condition_name] = np.nanmean(group_data, axis=2)
                    group_std[group_name][condition_name] = np.nanstd(group_data, axis=2)
        return group_mean, group_std

    def save(self, file_path: str):

        if file_path.endswith("_results.pkl"):
            raise RuntimeError("The file_path cannot end with '_results.pkl'. This is reserved for the result from the AnalysisPerformer.")

        subject_mean, subject_std = self.mean_per_subject()
        group_mean, group_std = self.mean_per_group(subject_mean=subject_mean)
        data = {
            "groups_to_compare": self.groups_to_compare,
            "nb_frames_interp": self.nb_frames_interp,
            "data": self.data,
            "subject_mean": subject_mean,
            "subject_std": subject_std,
            "group_mean": group_mean,
            "group_std": group_std,
        }
        with open(file_path, "wb") as f:
            pickle.dump(data, f)