import os
import numpy as np
import pickle
import biorbd

from gait_analyzer.subject import Subject
from gait_analyzer.kinematics_reconstructor import KinematicsReconstructor
from gait_analyzer.experimental_data import ExperimentalData


class AngularMomentumCalculator:
    """
    This class computes the angular momentum of the whole body (total_angular_momentum) and of each segment (segments_angular_momentum).
    The angular momentum is normalised by subject_mass * subject_height * sqrt(gravity * height) to allow comparison across subjects.
    Similarly, the segments' angular momentum is also normalised by segment_mass * segment_length * sqrt(gravity * segment_length).
    """

    def __init__(
        self,
        biorbd_model: biorbd.Model,
        experimental_data: ExperimentalData,
        kinematics_reconstructor: KinematicsReconstructor,
        subject: Subject,
        skip_if_existing: bool,
    ):
        """
        Initialize the AngularMomentumCalculator.
        .
        Parameters
        ----------
        biorbd_model : biorbd.Model
            The biorbd model of the subject.
        experimental_data : ExperimentalData
            The experimental data for this trial.
        kinematics_reconstructor : KinematicsReconstructor
            The kinematics reconstructor object containing the filtered joint angles and velocities.
        subject : Subject
            The subject object containing subject-specific parameters like mass and height.
        # segments_length: dict[str, float]
        #     A dictionary containing the length of each segment in meters.
        skip_if_existing : bool
            If True, skip the angular momentum computations if it already exists.
        """

        # Initial attributes
        self.model = biorbd_model
        self.experimental_data = experimental_data
        self.q = kinematics_reconstructor.q_filtered
        self.qdot = kinematics_reconstructor.qdot
        self.subject_mass = subject.subject_mass
        self.subject_height = subject.subject_height
        self.gravity = biorbd_model.getGravity().to_array()

        # Helper parameters
        self.nb_frames = self.q.shape[1]

        # Extended attributes
        self.total_angular_momentum = None
        self.total_angular_momentum_normalized = None
        self.segments_angular_momentum = None
        # self.segments_angular_momentum_normalized = None
        self.is_loaded_angular_momentum = False

        if skip_if_existing and self.check_if_existing():
            self.is_loaded_angular_momentum = True
        else:
            # Compute the angular momentum values
            self.compute_total_angular_momentum()
            self.normalize_total_angular_momentum()
            self.compute_segments_angular_momentum()

    def compute_total_angular_momentum(self):
        """
        Computes the angular momentum of the whole body around the center of mass on the three axis.
        """
        self.total_angular_momentum = np.zeros((3, self.nb_frames))
        for i_frame in range(self.nb_frames):
            self.total_angular_momentum[:, i_frame] = self.model.angularMomentum(
                self.q[:, i_frame], self.qdot[:, i_frame], True
            ).to_array()

    def normalize_total_angular_momentum(self):
        """
        Normalize the angular momentum with respect to the mass and height of the subject.
        """
        if self.gravity[0] != 0.0 or self.gravity[1] != 0.0 or self.gravity[2] == 0.0:
            raise NotImplementedError(
                f"The gravity of this model is not aligned with the z axis ({self.gravity}), which id not implemented yet."
            )

        gravity_factor = np.array([1.0, 1.0, np.abs(self.gravity[2])])
        normalization_factor = self.subject_mass * self.subject_height * np.sqrt(gravity_factor * self.subject_height)
        self.total_angular_momentum_normalized = self.total_angular_momentum / normalization_factor.reshape(3, 1)

    def extract_last_dof_per_segment(self):
        """
        Extract the last DoF for each segment because biorbd stores the angular momentum of the kinematic chain at
        the index of this DoF (all others are set to zeros).
        TODO: This workaround works for models with "_rotation" and "_translation" in the DoF names, but should be replaced with something else.
        """
        dof_names = [m.to_string() for m in self.model.nameDof()]
        last_dofs = []
        segment_names = []
        last_dof_indices = []
        current_segment = None
        for i, dof_name in enumerate(dof_names):
            if "_translation" in dof_name:
                segment_name = dof_name.split("_translation")[0]
            elif "_rotation" in dof_name:
                segment_name = dof_name.split("_rotation")[0]
            else:
                segment_name = dof_name.split("_")[0]

            if current_segment is not None and segment_name != current_segment:
                last_dofs.append(dof_names[i - 1])
                segment_names.append(segment_name)
                last_dof_indices.append(i - 1)
            current_segment = segment_name

        if dof_names:
            last_dofs.append(dof_names[-1])
            segment_names.append(segment_name)
            last_dof_indices.append(len(dof_names) - 1)

            return last_dofs, segment_names, last_dof_indices

    def compute_segments_angular_momentum(self):
        """
        Computes the angular momentum of each segment around its center of mass on the three axis.
        """
        last_dofs, segment_names, last_dof_indices = self.extract_last_dof_per_segment()

        self.segments_angular_momentum = {segment_name: np.zeros((3, self.nb_frames)) for segment_name in segment_names}

        # TODO: The normalization of the segments' angular momentum is not implemented yet.
        # It would require providing anthropometric measurement from the participant.

        # # Make sure segment_length is of the right type
        # if self.segments_length is None:
        #     self.segments_length = {segment_name: np.nan for segment_name in segment_names}
        # elif not isinstance(self.segments_length, dict):
        #     raise ValueError("segments_length must be a dictionary with segment names as keys and lengths as values.")
        # elif not all(segment_name in self.segments_length for segment_name in segment_names):
        #     raise ValueError("segments_length must contain all segment names from the model.")
        # elif len(self.segments_length) != len(segment_names):
        #     raise ValueError("segments_length must contain the same number of segments as the model.")

        # self.segments_angular_momentum_normalized = {segment_name: np.zeros((3, self.nb_frames)) for segment_name in segment_names}
        # for i_frame in range(self.nb_frames):
        #     segment_angular_momentum = self.model.CalcSegmentsAngularMomentum(self.q[:, i_frame], self.qdot[:, i_frame], True)
        #     for segment_name, index in zip(segment_names, last_dof_indices):
        #         self.segments_angular_momentum[segment_name][: i_frame] = segment_angular_momentum[index].to_array()
        #         self.segments_angular_momentum_normalized[segment_name][:, i_frame] = self.segments_angular_momentum[segment_name][:, i_frame] / (
        #             self.subject_mass * self.segments_length[segment_name] * np.sqrt(self.gravity * self.segments_length[segment_name])
        #         )

        return

    def check_if_existing(self) -> bool:
        """
        Check if the angular momentum value already exists.
        If it exists, load it.
        .
        Returns
        -------
        bool
            If the angular momentum value already exists
        """
        result_file_full_path = self.get_result_file_full_path()
        if os.path.exists(result_file_full_path):
            with open(result_file_full_path, "rb") as file:
                data = pickle.load(file)
                self.total_angular_momentum = data["total_angular_momentum"]
                self.total_angular_momentum_normalized = data["total_angular_momentum_normalized"]
                self.segments_angular_momentum = data["segments_angular_momentum"]
                # self.segments_angular_momentum_normalized = data["segments_angular_momentum_normalized"]
                self.is_loaded_angular_momentum = True
            return True
        else:
            return False

    def get_result_file_full_path(self, result_folder=None):
        if result_folder is None:
            result_folder = self.experimental_data.result_folder
        trial_name = self.experimental_data.c3d_full_file_path.split("/")[-1][:-4]
        result_file_full_path = f"{result_folder}/ang_mom_{trial_name}.pkl"
        return result_file_full_path

    def save_angular_momentum(self):
        """
        Save the angular momentum values.
        """
        result_file_full_path = self.get_result_file_full_path()
        with open(result_file_full_path, "wb") as file:
            pickle.dump(self.outputs(), file)

    def inputs(self):
        return {
            "biorbd_model": self.model.path,
            "q_filtered": self.q,
            "qdot": self.qdot,
            "subject_mass": self.subject_mass,
            "subject_height": self.subject_height,
        }

    def outputs(self):
        return {
            "total_angular_momentum": self.total_angular_momentum,
            "total_angular_momentum_normalized": self.total_angular_momentum_normalized,
            "segments_angular_momentum": self.segments_angular_momentum,
            # "segments_angular_momentum_normalized": self.segments_angular_momentum_normalized,
        }
