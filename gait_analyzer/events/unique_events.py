import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

from gait_analyzer.operator import Operator
from gait_analyzer.experimental_data import ExperimentalData


class UniqueEvents:
    """
    This class contains all the events detected from the experimental data.
    """

    def __init__(self, experimental_data: ExperimentalData, skip_if_existing: bool):
        """
        Initialize the UniqueEvents.
        .
        Parameters
        ----------
        experimental_data: ExperimentalData
            The experimental data from the trial
        skip_if_existing: bool
            If True, the events will not be recalculated if they already exist
        plot_phases_flag: bool
            If True, the phases will be plotted
        """
        # Checks
        if not isinstance(experimental_data, ExperimentalData):
            raise ValueError(
                "experimental_data must be an instance of ExperimentalData. You can declare it by running ExperimentalData(file_path)."
            )
        if not isinstance(skip_if_existing, bool):
            raise ValueError("skip_if_existing must be a boolean")

        # Parameters of the detection algorithm
        self.minimal_vertical_force_threshold = 15  # TODO: Charbie -> cite article and make it weight dependent

        # Initial attributes
        self.experimental_data = experimental_data
        self.is_loaded_events = False
        self.type = "unique"

        # Extended attributes
        self.events = [
            {
                "heel_touch": [],  # heel strike
                "toes_off": [],  # beginning of swing
            }
            for _ in range(len(experimental_data.platform_corners))
        ]

        if skip_if_existing and self.check_if_existing():
            self.is_loaded_events = True
        else:
            print("Detecting events...")
            self.find_event_timestamps()
            self.save_events()

    def check_if_existing(self) -> bool:
        """
        Check if the events detection already exists.
        If it exists, load the events.
        .
        Returns
        -------
        bool
            If the events detection already exists
        """
        result_file_full_path = self.get_result_file_full_path()
        if os.path.exists(result_file_full_path):
            with open(result_file_full_path, "rb") as file:
                data = pickle.load(file)
                self.events = data["events"]
            return True
        else:
            return False

    def detect_heel_touch(self):
        """
        Detect the heel touch event when the vertical GRF reaches a certain threshold
        """
        for i_platform in range(len(self.experimental_data.platform_corners)):
            grf_y_filtered = Operator.moving_average(self.experimental_data.f_ext_sorted[i_platform, 8, :], 21)
            index = np.abs(grf_y_filtered) > self.minimal_vertical_force_threshold
            first_indices_of_a_bloc = np.where(index[1:].astype(int) - index[:-1].astype(int) == 1)
            if len(first_indices_of_a_bloc) > 0:
                first_indices_of_a_bloc = first_indices_of_a_bloc[0] + 1
            else:
                first_indices_of_a_bloc = 0

            self.events[i_platform]["heel_touch"] = first_indices_of_a_bloc

    def detect_toes_off(self):
        """
        Detect the toes off event when the vertical GRF is lower than a threshold
        """
        for i_platform in range(len(self.experimental_data.platform_corners)):
            grf_y_filtered = Operator.moving_average(self.experimental_data.f_ext_sorted[i_platform, 8, :], 21)
            index = np.abs(grf_y_filtered) > self.minimal_vertical_force_threshold
            last_indices_of_a_bloc = np.where(index[1:].astype(int) - index[:-1].astype(int) == -1)
            if len(last_indices_of_a_bloc) > 0:
                last_indices_of_a_bloc = last_indices_of_a_bloc[0]
            else:
                last_indices_of_a_bloc = self.experimental_data.f_ext_sorted[i_platform, 8, :].shape[0]
            self.events[i_platform]["toes_off"] = last_indices_of_a_bloc

    def find_event_timestamps(self):
        # Detect events
        self.detect_toes_off()
        self.detect_heel_touch()

    def get_frame_range(self, cycles_to_analyze):
        """
        Get the frame range to analyze.
        """
        if cycles_to_analyze is not None:
            raise NotImplementedError("All frames should be analyzed for now.")
        return np.arange(0, len(self.experimental_data.markers_time_vector))

    def get_result_file_full_path(self, result_folder=None):
        if result_folder is None:
            result_folder = self.experimental_data.result_folder
        trial_name = self.experimental_data.c3d_full_file_path.split("/")[-1][:-4]
        result_file_full_path = f"{result_folder}/events_{trial_name}.pkl"
        return result_file_full_path

    def save_events(self):
        """
        Save the events detected.
        """
        result_file_full_path = self.get_result_file_full_path()
        with open(result_file_full_path, "wb") as file:
            pickle.dump(self.outputs(), file)

    def inputs(self):
        return {
            "experimental_data": self.experimental_data,
        }

    def outputs(self):
        return {
            "events": self.events,
            "is_loaded_events": self.is_loaded_events,
        }
