import ezc3d
import numpy as np

from gait_analyzer.model_creator import ModelCreator
from gait_analyzer.operator import Operator
from gait_analyzer.subject import Subject


class ExperimentalData:
    """
    This class contains all the experimental data from a trial (markers, EMG, force plates data, gait parameters).
    """

    def __init__(
        self,
        c3d_file_name: str,
        result_folder: str,
        model_creator: ModelCreator,
        animate_c3d_flag: bool,
    ):
        """
        Initialize the ExperimentalData.
        .
        Parameters
        ----------
        c3d_file_name: str
            The name of the trial's c3d file.
        subject: Subject
            The subject to analyze.
        result_folder: str
            The folder where the results will be saved. It should look like result_folder/subject_name.
        model_creator: ModelCreator
            The subject's personalized biorbd model.
        animate_c3d: bool
            If True, the c3d file will be animated.
        """
        # Checks
        if not isinstance(c3d_file_name, str):
            raise ValueError("c3d_file_name must be a string")
        if not isinstance(result_folder, str):
            raise ValueError("result_folder must be a string")

        # Threshold for removing force values
        self.force_threshold = 15  # N

        # Initial attributes
        self.c3d_file_name = c3d_file_name
        self.c3d_full_file_path = "../data/" + c3d_file_name
        self.model_creator = model_creator
        self.result_folder = result_folder

        # Extended attributes
        self.c3d = None
        self.model_marker_names = None
        self.marker_sampling_frequency = None
        self.markers_dt = None
        self.marker_units = None
        self.nb_marker_frames = None
        self.markers_sorted = None
        self.markers_sorted_with_virtual = None
        self.analogs_sampling_frequency = None
        self.platform_corners = None
        self.analogs_dt = None
        self.nb_analog_frames = None
        self.f_ext_sorted = None
        self.f_ext_sorted_filtered = None
        self.markers_time_vector = None
        self.analogs_time_vector = None

        # Extract data from the c3d file
        print(f"Reading experimental data from file {self.c3d_file_name} ...")
        self.perform_initial_treatment()
        self.extract_gait_parameters()
        if animate_c3d_flag:
            self.animate_c3d()

    def perform_initial_treatment(self):
        """
        Extract important information and sort the data
        """

        def load_model():
            self.model_marker_names = [m.to_string() for m in self.model_creator.biorbd_model.markerNames()]
            # model_muscle_names = [m.to_string() for m in self.model_creator.biorbd_model.muscleNames()]

        def sort_markers():
            self.c3d = ezc3d.c3d(self.c3d_full_file_path, extract_forceplat_data=True)
            markers = self.c3d["data"]["points"]
            self.marker_sampling_frequency = self.c3d["parameters"]["POINT"]["RATE"]["value"][0]  # Hz
            self.markers_dt = 1 / self.c3d["header"]["points"]["frame_rate"]
            self.nb_marker_frames = markers.shape[2]
            exp_marker_names = self.c3d["parameters"]["POINT"]["LABELS"]["value"]
            self.marker_units = 1
            if self.c3d["parameters"]["POINT"]["UNITS"]["value"][0] == "mm":
                self.marker_units = 0.001
            if len(self.model_marker_names) > len(exp_marker_names):
                supplementary_marker_names = [name for name in self.model_marker_names if name not in exp_marker_names]
                raise ValueError(
                    f"The markers {supplementary_marker_names} are not in the c3d file, but are in the model."
                )
            markers_sorted = np.zeros((3, len(self.model_marker_names), self.nb_marker_frames))
            markers_sorted[:, :, :] = np.nan
            for i_marker, name in enumerate(exp_marker_names):
                marker_idx = self.model_marker_names.index(name)
                markers_sorted[:, marker_idx, :] = markers[:3, i_marker, :] * self.marker_units
            self.markers_sorted = markers_sorted

        def add_virtual_markers():
            """
            This function augments the marker set with virtual markers to improve the extended Kalman filter kinematics reconstruction.
            """
            markers_for_virtual = self.model_creator.markers_for_virtual
            markers_sorted_with_virtual = np.zeros(
                (3, len(self.model_marker_names) + len(markers_for_virtual.keys()), self.nb_marker_frames)
            )
            markers_sorted_with_virtual[:, : len(self.model_marker_names), :] = self.markers_sorted[:, :, :]
            for i_marker, name in enumerate(markers_for_virtual.keys()):
                exp_marker_position = np.zeros((3, len(markers_for_virtual[name]), self.nb_marker_frames))
                for i in range(len(markers_for_virtual[name])):
                    exp_marker_position[:, i, :] = self.markers_sorted[
                        :, self.model_marker_names.index(markers_for_virtual[name][i]), :
                    ]
                markers_sorted_with_virtual[:, len(self.model_marker_names) + i_marker, :] = np.mean(
                    exp_marker_position, axis=1
                )
            self.markers_sorted_with_virtual = markers_sorted_with_virtual

        def sort_analogs():
            """
            TODO: -> treatment of the EMG signal to remove stimulation artifacts here
            """

            # Get an array of the experimental muscle activity
            analogs = self.c3d["data"]["analogs"]
            self.nb_analog_frames = analogs.shape[2]
            self.analogs_sampling_frequency = self.c3d["parameters"]["ANALOG"]["RATE"]["value"][0]  # Hz
            self.analogs_dt = 1 / self.c3d["header"]["analogs"]["frame_rate"]

            # print(analog_names)
            # emg_sorted = np.zeros((len(model_muscle_names), self.nb_analog_frames))
            # for i_muscle, name in enumerate(model_muscle_names):
            #     muscle_idx = analog_names.index(name)
            #     emg_sorted[i_muscle, :] = analogs[muscle_idx, :]
            # self.emg_sorted = emg_sorted
            return

        def extract_force_platform_data():
            """
            Extracts the force platform data from the c3d file and filters it.
            The F_ext output is of the form [cop, moments, forces].
            """

            platforms = self.c3d["data"]["platform"]
            nb_platforms = len(platforms)
            units = self.marker_units  # We assume that the all position units are the same as the markers'
            self.platform_corners = []
            self.platform_corners += [platforms[0]["corners"] * units]
            self.platform_corners += [platforms[1]["corners"] * units]

            # Initialize arrays for storing external forces and moments
            force_filtered = np.zeros((nb_platforms, 3, self.nb_analog_frames))
            moment_filtered = np.zeros((nb_platforms, 3, self.nb_analog_frames))
            tz_filtered = np.zeros((nb_platforms, 3, self.nb_analog_frames))
            cop_filtered = np.zeros((nb_platforms, 3, self.nb_analog_frames))
            f_ext_sorted = np.zeros((2, 9, self.nb_analog_frames))
            f_ext_sorted_filtered = np.zeros((2, 9, self.nb_analog_frames))

            # Process force platform data
            for i_platform in range(nb_platforms):

                # Get the data
                force = platforms[i_platform]["force"]
                moment = platforms[i_platform]["moment"] * units
                tz = platforms[i_platform]["Tz"] * units
                tz[:2, :] = 0  # This is the intended behavior (no moments on X and Y at the CoP)

                # Filter forces and moments
                # TODO: Charbie -> Antoine is supposed to send a ref for this filtering
                force_filtered[i_platform, :, :] = Operator.apply_filtfilt(
                    force, order=2, sampling_rate=self.analogs_sampling_frequency, cutoff_freq=10
                )
                moment_filtered[i_platform, :, :] = Operator.apply_filtfilt(
                    moment, order=2, sampling_rate=self.analogs_sampling_frequency, cutoff_freq=10
                )
                tz_filtered[i_platform, :, :] = Operator.apply_filtfilt(
                    tz, order=2, sampling_rate=self.analogs_sampling_frequency, cutoff_freq=10
                )

                # Remove the values when the force is too small since it is likely only noise
                null_idx = np.where(np.linalg.norm(force_filtered[i_platform, :, :], axis=0) < self.force_threshold)[0]
                moment_filtered[i_platform, :, null_idx] = np.nan
                force_filtered[i_platform, :, null_idx] = np.nan

                # Do not trust the CoP from ezc3d and recompute it after filtering the forces and moments
                cop_ezc3d = platforms[i_platform]["center_of_pressure"] * units

                r_z = 0  # In our case the reference frame of the platform is at its surface, so the height is 0
                cop_filtered[i_platform, 0, :] = - (moment_filtered[i_platform, 1, :] - force_filtered[i_platform, 0, :] * r_z) / force_filtered[i_platform, 2, :]
                cop_filtered[i_platform, 1, :] = (moment_filtered[i_platform, 0, :] + force_filtered[i_platform, 1, :] * r_z) / force_filtered[i_platform, 2, :]
                cop_filtered[i_platform, 2, :] = r_z
                # The CoP must be expressed relatively to the center of the platforms
                cop_filtered[i_platform, :, :] += np.tile(np.mean(self.platform_corners[i_platform], axis=1), (self.nb_analog_frames, 1)).T

                # Store output in a biorbd compatible format
                f_ext_sorted[i_platform, :3, :] = cop_ezc3d[:, :]
                f_ext_sorted_filtered[i_platform, :3, :] = cop_filtered[i_platform, :, :]
                f_ext_sorted[i_platform, 3:6, :] = tz[:, :]
                f_ext_sorted_filtered[i_platform, 3:6, :] = tz_filtered[i_platform, :, :]
                f_ext_sorted[i_platform, 6:9, :] = force[:, :]
                f_ext_sorted_filtered[i_platform, 6:9, :] = force_filtered[i_platform, :, :]

                if np.nanmean(cop_ezc3d[:, :] - cop_filtered[i_platform, :, :]) > 1e-3:
                    import matplotlib.pyplot as plt
                    fig, axs = plt.subplots(3, 1)

                    axs[0].plot(cop_filtered[0, :], '-r', label='CoP recomputed (from filtered F and M)')
                    axs[1].plot(cop_filtered[1, :], '-r')
                    axs[2].plot(cop_filtered[2, :], '-r')

                    axs[0].plot(cop_ezc3d[0, :], '--b', label='CoP ezc3d raw')
                    axs[1].plot(cop_ezc3d[1, :], '--b')
                    axs[2].plot(cop_ezc3d[2, :], '--b')

                    axs[0].set_xlim(0, 25000)
                    axs[1].set_xlim(0, 25000)
                    axs[2].set_xlim(0, 25000)

                    axs[0].set_ylim(-1, 1)
                    axs[1].set_ylim(-1, 1)
                    axs[2].set_ylim(-0.01, 0.01)

                    axs[0].legend()
                    fig.savefig("CoP_filtering_error.png")
                    fig.show()
                    raise NotImplementedError("The force platform data is not computed the same way in ezc3d than in this code, see the CoP graph.")

            self.f_ext_sorted = f_ext_sorted
            self.f_ext_sorted_filtered = f_ext_sorted_filtered

        def compute_time_vectors():
            self.markers_time_vector = np.linspace(0, self.markers_dt * self.nb_marker_frames, self.nb_marker_frames)
            self.analogs_time_vector = np.linspace(0, self.analogs_dt * self.nb_analog_frames, self.nb_analog_frames)

        # Perform the initial treatment
        load_model()
        sort_markers()
        add_virtual_markers()
        sort_analogs()
        extract_force_platform_data()
        compute_time_vectors()

    def animate_c3d(self):
        try:
            from pyorerun import BiorbdModel, PhaseRerun
        except:
            raise RuntimeError("To animate the .c3d, you first need to install Pyorerun.")
        raise NotImplementedError("Aniomation of c3d files is not implemented yet.")
        pass

    def extract_gait_parameters(self):
        """
        TODO: Guys -> please provide code :)
        """
        pass

    def inputs(self):
        return {
            "c3d_full_file_path": self.c3d_full_file_path,
            "model_creator": self.model_creator,
        }

    def outputs(self):
        return {
            "model_marker_names": self.model_marker_names,
            "marker_sampling_frequency": self.marker_sampling_frequency,
            "markers_dt": self.markers_dt,
            "nb_marker_frames": self.nb_marker_frames,
            "markers_sorted": self.markers_sorted,
            "analogs_sampling_frequency": self.analogs_sampling_frequency,
            "analogs_dt": self.analogs_dt,
            "nb_analog_frames": self.nb_analog_frames,
            "f_ext_sorted": self.f_ext_sorted,
            "f_ext_sorted_filtered": self.f_ext_sorted_filtered,
            "markers_time_vector": self.markers_time_vector,
            "analogs_time_vector": self.analogs_time_vector,
        }
