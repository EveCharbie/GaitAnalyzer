import os
import pickle
import numpy as np
import casadi as cas
import biorbd
import biobuddy

try:
    import bioptim
except ImportError:
    print("Skipped Bioptim import as it is not installed")

from gait_analyzer.operator import Operator
from gait_analyzer.model_creator import ModelCreator
from gait_analyzer.kinematics_reconstructor import KinematicsReconstructor
from gait_analyzer.inverse_dynamics_performer import InverseDynamicsPerformer
from gait_analyzer.experimental_data import ExperimentalData
from gait_analyzer.events.cyclic_events import CyclicEvents
from gait_analyzer.subject import Subject


class OptimalEstimator:
    """
    This class create an optimal control problem and solve it.
    The goal is to match as closely as possible the experimental data.
    This method allows estimating the muscle forces and the joint torques more accurately.
    However, it is quite long.
    """

    def __init__(
        self,
        cycle_to_analyze: int,
        subject: Subject,
        model_creator: ModelCreator,
        experimental_data: ExperimentalData,
        events: CyclicEvents,
        kinematics_reconstructor: KinematicsReconstructor,
        inverse_dynamic_performer: InverseDynamicsPerformer,
        plot_solution_flag: bool,
        animate_solution_flag: bool,
        implicit_contacts: bool,
        skip_if_existing: bool,
    ):
        """
        Initialize the OptimalEstimator.
        .
        Parameters
        ----------
        cycle_to_analyze: int
            The number of the cycle to analyze.
            TODO: Charbie -> Maybe we should chose the most representative cycle instead of picking one?
        subject: Subject
            The subject to analyze.
        model_creator: ModelCreator
            The model creator for this subject.
        experimental_data: ExperimentalData
            The experimental data to match.
        events: CyclicEvents
            The events of the gait cycle to split the trial into appropriate phases.
        kinematics_reconstructor: KinematicsReconstructor
            The kinematics reconstructor to use.
        inverse_dynamic_performer: InverseDynamicsPerformer
            The inverse dynamics performer to use.
        plot_solution_flag: bool
            If True, the solution will be plotted.
        animate_solution_flag: bool
            If True, the solution will be animated.
        implicit_contacts: bool
            If True, the contacts will be added implicitly to the problem.
        skip_if_existing: bool
            If True, the optimal estimation will be skipped if the results already exist.
            If False, the optimal estimation will be performed even if the results already exist.
        """

        # Checks
        if not isinstance(cycle_to_analyze, int):
            raise ValueError("cycle_to_analyze must be an int")
        if not isinstance(subject, Subject):
            raise ValueError("subject must be a Subject")
        if not isinstance(model_creator, ModelCreator):
            raise ValueError("model_creator must be a ModelCreator")
        if not isinstance(experimental_data, ExperimentalData):
            raise ValueError("experimental_data must be an ExperimentalData")
        if not isinstance(events, CyclicEvents):
            raise ValueError("events must be an CyclicEvents")
        if not isinstance(kinematics_reconstructor, KinematicsReconstructor):
            raise ValueError("kinematics_reconstructor must be a KinematicsReconstructor")
        if not isinstance(inverse_dynamic_performer, InverseDynamicsPerformer):
            raise ValueError("inverse_dynamic_performer must be a InverseDynamicsPerformer")

        # Initial attributes
        self.cycle_to_analyze = cycle_to_analyze
        self.subject = subject
        self.model_creator = model_creator
        self.experimental_data = experimental_data
        self.events = events
        self.kinematics_reconstructor = kinematics_reconstructor
        self.inverse_dynamic_performer = inverse_dynamic_performer

        # Extended attributes
        self.ocp = None
        self.model_ocp = None
        self.q_exp_ocp = None
        self.qdot_exp_ocp = None
        self.tau_exp_ocp = None
        self.emg_normalized_exp_ocp = None
        self.f_ext_exp_ocp = None
        self.markers_exp_ocp = None
        self.emg_exp_ocp = None
        self.n_shooting = None
        self.phase_time = None
        self.solution = None
        self.muscle_forces = None
        self.q_opt = None
        self.qdot_opt = None
        self.tau_opt = None
        self.opt_status = "CVG"
        self.is_loaded_optimal_solution = False

        # Execution
        if skip_if_existing and self.check_if_existing():
            print("Optimal estimation already exists, skipping...")
            self.is_loaded_optimal_solution = True
        else:
            print("Performing optimal estimation...")

            if implicit_contacts:
                self.generate_contact_biomods()
                self.prepare_reduced_experimental_data(plot_exp_data_flag=False, animate_exp_data_flag=True)
                self.prepare_ocp_implicit()
            else:
                self.generate_no_contacts_model()
                self.prepare_reduced_experimental_data(plot_exp_data_flag=False, animate_exp_data_flag=True)
                self.prepare_ocp_fext(with_residual_forces=True)

            self.solve(show_online_optim=True)
            self.save_optimal_reconstruction()

            if plot_solution_flag:
                self.solution.graphs(
                    show_bounds=True,
                    save_name=self.get_result_file_full_path(self.experimental_data.result_folder + "/figures")[:-4],
                )

        if animate_solution_flag:
            self.animate_solution()
        self.extract_muscle_forces()

    def generate_no_contacts_model(self):

        segments_to_remove_dofs_from = [
            "toes_r_rotation_transform",
            "toes_l_rotation_transform",
            "lunate_r_rotation_transform",
            "hand_r_rotation_transform",
            "fingers_r_rotation_transform",
            "lunate_l_rotation_transform",
            "hand_l_rotation_transform",
            "fingers_l_rotation_transform",
        ]

        no_contact_model = biobuddy.BiomechanicalModelReal().from_biomod(self.model_creator.biorbd_model_full_path)
        for segment in no_contact_model.segments:
            if segment.name in segments_to_remove_dofs_from:
                segment.rotations = biobuddy.Rotations.NONE
                segment.translations = biobuddy.Translations.NONE
                segment.dof_names = None
                segment.q_ranges = None
                segment.qdot_ranges = None

        no_contact_model.to_biomod(self.model_creator.biorbd_model_full_path.replace(".bioMod", "_no_contacts.bioMod"))

    # def generate_contact_biomods(self):
    #     """
    #     Create other bioMod files with the addition of the different feet contact conditions.
    #     """
    #
    #     def add_contacts_per_condition(model_to_modify: biobuddy.BiomechanicalModelReal, condition: str) -> str:
    #         # First, remove some degrees of freedom and ranges from the model
    #
    #         # Second, add contacts
    #         if "heelL" in condition and "toesL" in condition:
    #             contact = biobuddy.Contact(name="LCAL",
    #                                        parent_name="calcn_l",
    #                                        function=lambda m, bio: (5*m["LCAL"] + m["LFH1"] + m["LFH5"]) / 8,
    #                                        axis=biobuddy.Translations.XYZ,
    #                                        )
    #             model_to_modify.segments["calcn_l"].add_contact(contact.to_contact(model_to_modify))
    #
    #             def to_contact(self, data: Data) -> ContactReal:
    #
    #             name: str,
    #             parent_name: str,
    #             position: Point = None,
    #             axis: Translations = None,
    #             contact_text += f"contact\tLCAL\n"
    #             contact_text += f"\tparent\tcalcn_l\n"
    #             contact_text += f"\tposition\t-0.018184372684362127\t-0.036183919561541877\t0.010718604411614319\n"
    #             contact_text += f"\taxis\txyz\n"
    #             contact_text += "endcontact\n\n"
    #
    #             contact_text += f"contact\tLMFH1\n"
    #             contact_text += f"\tparent\tcalcn_l\n"
    #             contact_text += f"\tposition\t0.19202791077724868\t-0.013754853217574914\t0.039283237127771042\n"
    #             contact_text += f"\taxis\tz\n"
    #             contact_text += "endcontact\n\n"
    #
    #             contact_text += f"contact\tLMFH5\n"
    #             contact_text += f"\tparent\tcalcn_l\n"
    #             contact_text += f"\tposition\t0.18583815793306013\t-0.0092170000425693677\t-0.072430596752376397\n"
    #             contact_text += f"\taxis\tzy\n"
    #             contact_text += "endcontact\n\n"
    #
    #         elif "heelL" in condition:
    #             contact_text += f"contact\tLCAL\n"
    #             contact_text += f"\tparent\tcalcn_l\n"
    #             contact_text += f"\tposition\t-0.018184372684362127\t-0.036183919561541877\t0.010718604411614319\n"
    #             contact_text += f"\taxis\txyz\n"
    #             contact_text += "endcontact\n\n"
    #
    #         elif "toesL" in condition:
    #             contact_text += f"contact\tLMFH1\n"
    #             contact_text += f"\tparent\tcalcn_l\n"
    #             contact_text += f"\tposition\t0.19202791077724868\t-0.013754853217574914\t0.039283237127771042\n"
    #             contact_text += f"\taxis\txz\n"
    #             contact_text += "endcontact\n\n"
    #
    #             contact_text += f"contact\tLMFH5\n"
    #             contact_text += f"\tparent\tcalcn_l\n"
    #             contact_text += f"\tposition\t0.18583815793306013\t-0.0092170000425693677\t-0.072430596752376397\n"
    #             contact_text += f"\taxis\txyz\n"
    #             contact_text += "endcontact\n\n"
    #
    #         if "heelR" in condition and "toesR" in condition:
    #             contact_text += f"contact\tRCAL\n"
    #             contact_text += f"\tparent\tcalcn_r\n"
    #             contact_text += f"\tposition\t-0.017776522017632024\t-0.030271301561674208\t-0.015068364463032391\n"
    #             contact_text += f"\taxis\txyz\n"
    #             contact_text += "endcontact\n\n"
    #
    #             contact_text += f"contact\tRMFH1\n"
    #             contact_text += f"\tparent\tcalcn_r\n"
    #             contact_text += f"\tposition\t0.20126587479704638\t-0.0099656486276807066\t-0.039248701869426805\n"
    #             contact_text += f"\taxis\tz\n"
    #             contact_text += "endcontact\n\n"
    #
    #             contact_text += f"contact\tRMFH5\n"
    #             contact_text += f"\tparent\tcalcn_r\n"
    #             contact_text += f"\tposition\t0.18449626841163846\t-0.018897872323952902\t0.07033570386440513\n"
    #             contact_text += f"\taxis\tzy\n"
    #             contact_text += "endcontact\n\n"
    #
    #         elif "heelR" in condition:
    #             contact_text += f"contact\tRCAL\n"
    #             contact_text += f"\tparent\tcalcn_r\n"
    #             contact_text += f"\tposition\t-0.017776522017632024\t-0.030271301561674208\t-0.015068364463032391\n"
    #             contact_text += f"\taxis\txyz\n"
    #             contact_text += "endcontact\n\n"
    #
    #         elif "toesR" in condition:
    #             contact_text += f"contact\tRMFH1\n"
    #             contact_text += f"\tparent\tcalcn_r\n"
    #             contact_text += f"\tposition\t0.20126587479704638\t-0.0099656486276807066\t-0.039248701869426805\n"
    #             contact_text += f"\taxis\txz\n"
    #             contact_text += "endcontact\n\n"
    #
    #             contact_text += f"contact\tRMFH5\n"
    #             contact_text += f"\tparent\tcalcn_r\n"
    #             contact_text += f"\tposition\t0.18449626841163846\t-0.018897872323952902\t0.07033570386440513\n"
    #             contact_text += f"\taxis\txyz\n"
    #             contact_text += "endcontact\n\n"
    #
    #         return contact_text
    #
    #     original_model_path = self.biorbd_model_path
    #     conditions = [
    #         "heelR_toesR",
    #         "toesR_heelL",
    #         "toesR",
    #         "toesR_heelL",
    #         "heelL_toesL",
    #         "toesL",
    #         "toesL_heelR",
    #         "toesL_heelR_toesR",
    #         "no_contacts",
    #     ]
    #     for condition in conditions:
    #         new_model_path = original_model_path.replace(".bioMod", f"_{condition}.bioMod")
    #         model_to_modify = biobuddy.BiomechanicalModelReal().from_biomod(original_model_path)
    #             model_to_modify = add_contacts_per_condition(model_to_modify, condition, self.)
    #             model_to_modify.to_biomod(new_model_path)

    def prepare_reduced_experimental_data(self, plot_exp_data_flag: bool = False, animate_exp_data_flag: bool = False):
        """
        To reduce the optimization time, only one cycle is treated at a time
        (and the number of degrees of freedom is reduced?).
        """
        # self.model_ocp = self.biorbd_model_path.replace(".bioMod", "_heelL_toesL.bioMod")
        self.model_ocp = self.model_creator.biorbd_model_full_path.replace(".bioMod", "_no_contacts.bioMod")
        model = biorbd.Model(self.model_ocp)

        # Only one right leg swing (while left leg in flat foot)
        swing_timings = np.where(self.events.phases["heelL_toesL"])[0]
        right_swing_sequence = np.array_split(swing_timings, np.flatnonzero(np.diff(swing_timings) > 1) + 1)
        this_sequence_analogs = right_swing_sequence[self.cycle_to_analyze]
        this_sequence_markers = Operator.from_analog_frame_to_marker_frame(
            analogs_time_vector=self.experimental_data.analogs_time_vector,
            markers_time_vector=self.experimental_data.markers_time_vector,
            analog_idx=this_sequence_analogs,
        )

        # # One full cycle
        # cycle_timings = self.events.events["right_leg_heel_touch"]
        # this_sequence_analogs = list(range(cycle_timings[self.cycle_to_analyze], cycle_timings[self.cycle_to_analyze + 1]))
        # this_sequence_markers = Operator.from_analog_frame_to_marker_frame(
        #     analogs_time_vector=self.experimental_data.analogs_time_vector,
        #     markers_time_vector=self.experimental_data.markers_time_vector,
        #     analog_idx=this_sequence_analogs,
        # )

        # Skipping some frames to lighten the OCP
        marker_start = this_sequence_markers[0]
        marker_end = this_sequence_markers[-1]
        marker_hop = 4
        idx_to_keep = np.arange(marker_start, marker_end, marker_hop)
        index_to_keep_filtered_q = idx_to_keep - self.kinematics_reconstructor.frame_range.start

        # Skipping some DoFs to lighten the OCP
        dof_idx_to_keep = np.array(
            [
                0,  # Pelvis trans X
                1,  # Pelvis trans Y
                2,  # Pelvis trans Z
                3,  # Pelvis rot X
                4,  # Pelvis rot Y
                5,  # Pelvis rot Z
                6,  # femur_r rot X
                7,  # femur_r rot Y
                8,  # femur_r rot Z
                9,  # tibia_r rot X
                10,  # talus_r rot X
                11,  # calc_r rot X (skip toes_r rot X)
                13,  # femur_l rot X
                14,  # femur_l rot Y
                15,  # femur_l rot Z
                16,  # tibia_l rot X
                17,  # talus_l rot X
                18,  # calc_l rot X (skip toes_l rot X)
                20,  # thorax rot X
                21,  # thorax rot Y
                22,  # thorax rot Z
                23,  # head_and_neck rot X
                24,  # head_and_neck rot Y
                25,  # head_and_neck rot Z
                26,  # humerus_r rot X
                27,  # humerus_r rot Y
                28,  # humerus_r rot Z
                29,  # ulna_r rot X
                30,  # radius_r rot X (skip lunate, hand and fingers)
                34,  # humerus_r rot X
                35,  # humerus_r rot Y
                36,  # humerus_r rot Z
                37,  # ulna_l rot X
                38,  # radius_l rot X (skip lunate, hand and fingers)
            ]
        )

        self.n_shooting = idx_to_keep.shape[0] - 1
        self.q_exp_ocp = self.kinematics_reconstructor.q_filtered[np.ix_(dof_idx_to_keep, index_to_keep_filtered_q)]
        self.qdot_exp_ocp = self.kinematics_reconstructor.qdot[np.ix_(dof_idx_to_keep, index_to_keep_filtered_q)]
        self.tau_exp_ocp = self.inverse_dynamic_performer.tau[np.ix_(dof_idx_to_keep, index_to_keep_filtered_q)]
        self.f_ext_exp_ocp = {
            "left_leg": np.zeros((9, self.n_shooting + 1)),
            "right_leg": np.zeros((9, self.n_shooting + 1)),
        }
        muscle_names = [m.to_string() for m in model.muscleNames()]
        nb_muscles = len(muscle_names)
        self.emg_normalized_exp_ocp = np.zeros((nb_muscles, self.n_shooting + 1))
        for i_frame, marker_frame in enumerate(idx_to_keep):
            idx_analogs = Operator.from_marker_frame_to_analog_frame(
                analogs_time_vector=self.experimental_data.analogs_time_vector,
                markers_time_vector=self.experimental_data.markers_time_vector,
                marker_idx=int(marker_frame),
            )
            self.f_ext_exp_ocp["left_leg"][:, i_frame] = np.mean(
                self.experimental_data.f_ext_sorted[0, :, idx_analogs - 5 : idx_analogs + 5], axis=1
            )
            self.f_ext_exp_ocp["right_leg"][:, i_frame] = np.mean(
                self.experimental_data.f_ext_sorted[1, :, idx_analogs - 5 : idx_analogs + 5], axis=1
            )
            for i_muscle, muscle_name in enumerate(muscle_names):
                if muscle_name in self.model_creator.osim_model_type.muscle_name_mapping:
                    muscle_speudo = self.model_creator.osim_model_type.muscle_name_mapping[muscle_name]
                    if muscle_speudo is not None:
                        muscle_index = self.experimental_data.analog_names.index(muscle_speudo)
                        self.emg_normalized_exp_ocp[i_muscle, i_frame] = np.nanmean(
                            self.experimental_data.normalized_emg[muscle_index, idx_analogs - 5 : idx_analogs + 5]
                        )

        self.markers_exp_ocp = self.experimental_data.markers_sorted[:, :, idx_to_keep]
        # Fill NaNs in markers
        for i_marker in range(self.markers_exp_ocp.shape[1]):
            if np.any(np.isnan(self.markers_exp_ocp[:, i_marker, :])):
                marker_mean = np.nanmean(self.markers_exp_ocp[:, i_marker, :], axis=1)
                nan_idx = np.where(np.isnan(self.markers_exp_ocp[0, i_marker, :]))[0]
                for i_nan in nan_idx:
                    if i_nan == 0 or i_nan == self.markers_exp_ocp.shape[2] - 1:
                        raise RuntimeError(
                            "Maybe chose another cycle as there are NaNs at the beginning or the end of this cycle."
                        )
                    if np.isnan(self.markers_exp_ocp[0, i_marker, i_nan - 1]) or np.isnan(
                        self.markers_exp_ocp[0, i_marker, i_nan + 1]
                    ):
                        raise NotImplementedError("TODO: Implement a better NaN filling method.")
                    self.markers_exp_ocp[:, i_marker, i_nan] = (
                        self.markers_exp_ocp[:, i_marker, i_nan - 1] + self.markers_exp_ocp[:, i_marker, i_nan + 1]
                    ) / 2

        self.phase_time = (
            self.experimental_data.markers_time_vector[idx_to_keep[-1]]
            - self.experimental_data.markers_time_vector[idx_to_keep[0]]
        )

        if plot_exp_data_flag:
            import matplotlib.pyplot as plt

            time_plot = np.linspace(0, self.phase_time, self.n_shooting + 1)

            # Plot Q that are susceptible to gimbal lock
            fig, axs = plt.subplots(5, 1)
            for i_dof in range(3, 6):
                axs[0].plot(time_plot, self.q_exp_ocp[i_dof, :], label=f"q_{i_dof}")
            for i_dof in range(10, 13):
                axs[1].plot(time_plot, self.q_exp_ocp[i_dof, :], label=f"q_{i_dof}")
            for i_dof in range(17, 20):
                axs[2].plot(time_plot, self.q_exp_ocp[i_dof, :], label=f"q_{i_dof}")
            for i_dof in range(23, 26):
                axs[3].plot(time_plot, self.q_exp_ocp[i_dof, :], label=f"q_{i_dof}")
            for i_dof in range(29, 32):
                axs[4].plot(time_plot, self.q_exp_ocp[i_dof, :], label=f"q_{i_dof}")
            for i in range(5):
                axs[i].plot(np.array([time_plot[0], time_plot[-1]]), np.array([-np.pi / 2, -np.pi / 2]), "--k")
                axs[i].plot(np.array([time_plot[0], time_plot[-1]]), np.array([np.pi / 2, np.pi / 2]), "--k")
                axs[i].legend()
            plt.savefig("Dofs_exp.png")
            plt.show()
            print(f"There are {np.sum(np.isnan(self.q_exp_ocp))} NaNs in Q.")
            print(f"There are {np.sum(np.isnan(self.qdot_exp_ocp))} NaNs in Qdot.")

            # Plot Tau to see if there are NaNs
            plt.figure()
            for i_dof in range(self.tau_exp_ocp.shape[0]):
                plt.plot(time_plot, self.tau_exp_ocp[i_dof, :], ".", label=f"tau_{i_dof}")
                nan_idx = np.isnan(self.tau_exp_ocp[i_dof, :])
                plt.plot(time_plot[nan_idx], self.tau_exp_ocp[i_dof, nan_idx], "ok")
            plt.legend()
            plt.savefig("Tau_exp.png")
            plt.show()
            print(f"There are {np.sum(np.isnan(self.tau_exp_ocp))} NaNs in Tau.")
            print(f"There are {np.sum(np.isnan(self.emg_normalized_exp_ocp))} NaNs in muscle activation.")

            # Plot Markers to see if some are missing
            plt.figure()
            for i_marker in range(self.markers_exp_ocp.shape[1]):
                for i_coordinate in range(3):
                    plt.plot(
                        time_plot, self.markers_exp_ocp[i_coordinate, i_marker, :], ".", label=f"marker_{i_marker}"
                    )
                    nan_idx = np.isnan(self.markers_exp_ocp[i_coordinate, i_marker, :])
                    plt.plot(time_plot[nan_idx], self.markers_exp_ocp[i_coordinate, i_marker, nan_idx], "ok")
            plt.legend()
            plt.savefig("Markers_exp.png")
            plt.show()
            print(f"There are {np.sum(np.isnan(self.markers_exp_ocp))} markers missing.")

        if animate_exp_data_flag:
            try:
                from pyorerun import BiorbdModel, PhaseRerun, PyoMarkers, PyoMuscles
            except:
                raise RuntimeError("To animate the initial guess, you must install Pyorerun.")

            # Add the model
            model = BiorbdModel(self.model_ocp)
            model.options.transparent_mesh = False
            model.options.show_gravity = True
            model.options.show_marker_labels = False
            model.options.show_muscle_labels = False
            model.options.show_center_of_mass_labels = False

            viz = PhaseRerun(np.linspace(0, self.phase_time, self.n_shooting + 1))

            # Add experimental markers
            markers = PyoMarkers(data=self.markers_exp_ocp, marker_names=list(model.marker_names), show_labels=False)
            emg = PyoMuscles(data=self.emg_normalized_exp_ocp, muscle_names=list(model.muscle_names))

            # Add force plates to the animation
            viz.add_force_plate(num=1, corners=self.experimental_data.platform_corners[0])
            viz.add_force_plate(num=2, corners=self.experimental_data.platform_corners[1])
            viz.add_force_data(
                num=1,
                force_origin=self.f_ext_exp_ocp["left_leg"][:3, :],
                force_vector=self.f_ext_exp_ocp["left_leg"][6:9, :],
            )
            viz.add_force_data(
                num=2,
                force_origin=self.f_ext_exp_ocp["right_leg"][:3, :],
                force_vector=self.f_ext_exp_ocp["right_leg"][6:9, :],
            )

            # Add the kinematics
            viz.add_animated_model(model, self.q_exp_ocp, tracked_markers=markers, muscle_activations_intensity=emg)

            # Play
            viz.rerun_by_frame("OCP initial guess from experimental data")

    def prepare_ocp_fext(self, with_residual_forces: bool = False):
        """
        Let's say swing phase only for now
        """

        try:
            from bioptim import (
                MusclesBiorbdModel,
                ConfigureProblem,
                DynamicsFunctions,
                DynamicsEvaluation,
                InitialGuess,
                InitialGuessList,
                InterpolationType,
                NonLinearProgram,
                ObjectiveFcn,
                ObjectiveList,
                OptimalControlProgram,
                PhaseTransitionFcn,
                PhaseTransitionList,
                PhaseDynamics,
                BoundsList,
                ConstraintFcn,
                ConstraintList,
                Solver,
                OdeSolver,
                ExternalForceSetTimeSeries,
                Node,
                DynamicsOptionsList,
                BiMappingList,
                DefectType,
                PenaltyController,
                ConfigureVariables,
            )

        except:
            raise RuntimeError("To reconstruct optimally, you must install Bioptim")

        class CustomMuscleModelNoContacts(MusclesBiorbdModel):
            def __init__(self, biorbd_model_path, external_force_set=None, with_residual_torque=True):
                """
                Custom Torque model to handle the no contact case.
                """
                super().__init__(
                    biorbd_model_path, external_force_set=external_force_set, with_residual_torque=with_residual_torque
                )
                if with_residual_forces:
                    self.control_configuration += [
                        lambda ocp, nlp, as_states, as_controls, as_algebraic_states: ConfigureVariables.configure_translational_forces(
                            ocp, nlp, as_states=False, as_controls=True, as_algebraic_states=False, n_contacts=2
                        )
                    ]

            def dynamics(
                self,
                time,
                states,
                controls,
                parameters,
                algebraic_states,
                numerical_timeseries,
                nlp,
            ):

                q = DynamicsFunctions.get(nlp.states["q"], states)
                qdot = DynamicsFunctions.get(nlp.states["qdot"], states)

                # Get torques
                tau_residual = DynamicsFunctions.get(nlp.controls["tau"], controls)
                mus_activations = DynamicsFunctions.get(nlp.controls["muscles"], controls)
                tau = tau_residual + DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations, None)

                # Get external forces
                if with_residual_forces:
                    f_ext_residual_value = DynamicsFunctions.get(nlp.controls["contact_forces"], controls)
                    f_ext_residual_position = DynamicsFunctions.get(nlp.controls["contact_positions"], controls)

                external_forces = nlp.get_external_forces(
                    "external_forces", states, controls, algebraic_states, numerical_timeseries
                )
                if with_residual_forces:
                    # Left
                    external_forces[:3] += f_ext_residual_position[:3]
                    external_forces[6:9] += f_ext_residual_value[:3]
                    # Right
                    external_forces[9:12] += f_ext_residual_position[3:6]
                    external_forces[15:18] += f_ext_residual_value[3:6]

                ddq = nlp.model.forward_dynamics()(q, qdot, tau, external_forces, nlp.parameters.cx)

                return DynamicsEvaluation(dxdt=cas.vertcat(qdot, ddq), defects=None)

        print(f"Preparing optimal control problem with platform force applied directly to the CoP...")

        # External force set
        external_force_set = ExternalForceSetTimeSeries(nb_frames=self.n_shooting)
        external_force_set.add(
            force_name="calcn_l",
            segment="calcn_l",
            values=self.f_ext_exp_ocp["left_leg"][3:9, :-1],
            point_of_application=self.f_ext_exp_ocp["left_leg"][:3, :-1],
        )
        external_force_set.add(
            force_name="calcn_r",
            segment="calcn_r",
            values=self.f_ext_exp_ocp["right_leg"][3:9, :-1],
            point_of_application=self.f_ext_exp_ocp["right_leg"][:3, :-1],
        )
        numerical_time_series = {"external_forces": external_force_set.to_numerical_time_series()}
        biorbd_model_path = self.model_creator.biorbd_model_full_path.replace(".bioMod", "_no_contacts.bioMod")
        bio_model = CustomMuscleModelNoContacts(biorbd_model_path, external_force_set=external_force_set)

        nb_q = bio_model.nb_q
        nb_muscles = bio_model.nb_muscles
        r_foot_marker_index = np.array(
            [
                bio_model.marker_index(f"RCAL"),
                bio_model.marker_index(f"RMFH1"),
                bio_model.marker_index(f"RMFH5"),
                bio_model.marker_index(f"R_foot_up"),
            ]
        )
        l_foot_marker_index = np.array(
            [
                bio_model.marker_index(f"LCAL"),
                bio_model.marker_index(f"LMFH1"),
                bio_model.marker_index(f"LMFH5"),
                bio_model.marker_index(f"L_foot_up"),
            ]
        )

        # Declaration of the objectives
        objective_functions = ObjectiveList()
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            key="tau",
            weight=0.1,
        )
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            key="tau",
            weight=1.0,
            index=[0, 1, 2, 3, 4, 5],
        )
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            key="muscles",
            weight=0.01,
            target=self.emg_normalized_exp_ocp[:10, :-1],
            index=[0, 1, 2, 4, 5, 6, 7, 8, 9, 10],  # Only the right leg (except tfl)
        )
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.TRACK_MARKERS, weight=100.0, node=Node.ALL, target=self.markers_exp_ocp
        )
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.TRACK_MARKERS,
            weight=1000.0,
            node=Node.ALL,
            marker_index=["RCAL", "RMFH1", "RMFH5", "R_foot_up", "LCAL", "LMFH1", "LMFH5", "L_foot_up"],
            target=self.markers_exp_ocp[:, np.hstack((r_foot_marker_index, l_foot_marker_index)), :],
        )
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.TRACK_STATE, key="q", weight=1.0, node=Node.ALL, target=self.q_exp_ocp
        )
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.TRACK_STATE,
            key="qdot",
            node=Node.ALL,
            weight=0.01,
            target=self.qdot_exp_ocp,
        )
        if with_residual_forces:
            objective_functions.add(  # Minimize residual contact forces
                objective=ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
                key="contact_forces",
                node=Node.ALL_SHOOTING,
                weight=1000,
            )
            objective_functions.add(  # Track CoP position
                objective=ObjectiveFcn.Lagrange.TRACK_CONTROL,
                key="contact_positions",
                node=Node.ALL_SHOOTING,
                weight=0.01,
                target=np.vstack((self.f_ext_exp_ocp["left_leg"][0:3, :-1], self.f_ext_exp_ocp["right_leg"][0:3, :-1])),
            )

        constraints = ConstraintList()

        dynamics = DynamicsOptionsList()
        dynamics.add(
            numerical_data_timeseries=numerical_time_series,
            phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
            ode_solver=OdeSolver.RK4(),
        )

        x_bounds = BoundsList()
        # Bounds personalized to the subject's current kinematics
        min_q = self.q_exp_ocp[:, :] - 0.3
        min_q[:6, :] = self.q_exp_ocp[:6, :] - 0.05
        max_q = self.q_exp_ocp[:, :] + 0.3
        max_q[:6, :] = self.q_exp_ocp[:6, :] + 0.05
        x_bounds.add("q", min_bound=min_q, max_bound=max_q, interpolation=InterpolationType.EACH_FRAME)
        # Bounds personalized to the subject's current joint velocities (not a real limitation, so it is executed with +-10)
        x_bounds.add(
            "qdot",
            min_bound=self.qdot_exp_ocp - 10,
            max_bound=self.qdot_exp_ocp + 10,
            interpolation=InterpolationType.EACH_FRAME,
        )

        x_init = InitialGuessList()
        x_init.add("q", initial_guess=self.q_exp_ocp, interpolation=InterpolationType.EACH_FRAME)
        x_init.add("qdot", initial_guess=self.qdot_exp_ocp, interpolation=InterpolationType.EACH_FRAME)

        u_bounds = BoundsList()
        # TODO: Charbie -> Change for maximal tau during the trial to simulate limited force
        u_bounds.add("tau", min_bound=[-800] * nb_q, max_bound=[800] * nb_q, interpolation=InterpolationType.CONSTANT)
        u_bounds.add(
            "muscles",
            min_bound=[0.0001] * nb_muscles,
            max_bound=[0.0001] * nb_muscles,
            interpolation=InterpolationType.CONSTANT,
        )
        if with_residual_forces:
            u_bounds.add(
                "contact_forces", min_bound=[-100] * 6, max_bound=[100] * 6, interpolation=InterpolationType.CONSTANT
            )
            u_bounds.add(
                "contact_positions", min_bound=[-2] * 6, max_bound=[2] * 6, interpolation=InterpolationType.CONSTANT
            )

        u_init = InitialGuessList()
        u_init.add("tau", initial_guess=[0] * nb_q, interpolation=InterpolationType.CONSTANT)
        u_init.add(
            "muscles", initial_guess=self.emg_normalized_exp_ocp[:, :-1], interpolation=InterpolationType.EACH_FRAME
        )
        if with_residual_forces:
            u_init.add("contact_forces", initial_guess=[0] * 6, interpolation=InterpolationType.CONSTANT)
            u_init.add(
                "contact_positions",
                initial_guess=np.vstack(
                    (self.f_ext_exp_ocp["left_leg"][0:3, :-1], self.f_ext_exp_ocp["right_leg"][0:3, :-1])
                ),
                interpolation=InterpolationType.EACH_FRAME,
            )

        # TODO: Charbie -> Add phase transition when I have the full cycle
        # phase_transitions = PhaseTransitionList()
        # phase_transitions.add(PhaseTransitionFcn.CYCLIC, phase_pre_idx=0)

        ocp = OptimalControlProgram(
            bio_model=bio_model,
            n_shooting=self.n_shooting,
            phase_time=self.phase_time,
            dynamics=dynamics,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            x_init=x_init,
            u_init=u_init,
            objective_functions=objective_functions,
            constraints=constraints,
            # phase_transitions=phase_transitions,
            use_sx=False,
            n_threads=10,
        )
        ocp.add_plot_penalty()
        ocp.add_plot_ipopt_outputs()
        self.ocp = ocp

    # def prepare_ocp_implicit(self):
    #     """
    #     Let's say swing phase only for now
    #     """
    #     # TODO: Charbie -> extract the common functions for implicit and explicit
    #
    #     def marker_velocity(controller):
    #         marker_velocities = []
    #         for marker_name in ["LCAL", "LMFH1", "LMFH5"]:
    #             marker_index = controller.model.marker_index(marker_name)
    #             qs = cas.horzcat(*([controller.states["q"].cx_start] + controller.states["q"].cx_intermediates))
    #             qdots = cas.horzcat(
    #                 *([controller.states["qdot"].cx_start] + controller.states["qdot"].cx_intermediates)
    #             )
    #             for i_sn in range(len(qs)):
    #                 marker_velocity = controller.model.marker_velocity(marker_index)(qs[i_sn], qdots[i_sn])
    #                 marker_velocities += [marker_velocity]
    #         return cas.vertcat(*marker_velocities)
    #
    #     def get_forces_on_each_point(controller):
    #         contact_forces = controller.algebraic_states["rigid_contact_forces"].cx_start
    #
    #         # Rearrange the forces to get all 3 components for each contact point
    #         forces_on_each_point = None
    #         current_index = 0
    #         for i_contact in range(controller.model.nb_rigid_contacts):
    #             available_axes = np.array(controller.model.rigid_contact_axes_index(i_contact))
    #             contact_force_idx = range(current_index, current_index + available_axes.shape[0])
    #             current_force = cas.MX.zeros(3)
    #             for i, contact_to_add in enumerate(contact_force_idx):
    #                 current_force[available_axes[i]] += contact_forces[contact_to_add]
    #             current_index += available_axes.shape[0]
    #             if forces_on_each_point is not None:
    #                 forces_on_each_point = cas.horzcat(forces_on_each_point, current_force)
    #             else:
    #                 forces_on_each_point = current_force
    #         return forces_on_each_point
    #
    #     def minimize_sum_reaction_forces(
    #         controller,
    #         contact_index: tuple[str, ...] | tuple[int, ...] | list[str | int],
    #     ):
    #
    #         forces_on_each_point = get_forces_on_each_point(controller)
    #
    #         total_force = controller.cx.zeros(3, 1)
    #         for contact in contact_index:
    #             idx = controller.model.contact_index(contact) if isinstance(contact, str) else contact
    #             total_force += forces_on_each_point[:, idx]
    #
    #         return total_force
    #
    #     def minimize_center_of_pressure(
    #         controller,
    #         contact_index: tuple[str, ...] | tuple[int, ...] | list[str | int],
    #     ):
    #
    #         forces_on_each_point = get_forces_on_each_point(controller)
    #
    #         total_force = controller.cx.zeros(3, 1)
    #         position_of_each_point = None
    #         weighted_sum = controller.cx.zeros(3, 1)
    #         for contact in contact_index:
    #             idx = controller.model.contact_index(contact) if isinstance(contact, str) else contact
    #
    #             # Compute the sum of the forces on the points of interest
    #             total_force += forces_on_each_point[:, idx]
    #
    #             # Get the position of all the contact points of interest
    #             this_contact_position = controller.model.rigid_contact_position(idx)(
    #                 controller.q, controller.parameters.cx
    #             )
    #             position_of_each_point = (
    #                 cas.horzcat(position_of_each_point, this_contact_position)
    #                 if position_of_each_point is not None
    #                 else this_contact_position
    #             )
    #
    #             # Weighted sum
    #             weighted_sum += forces_on_each_point[:, idx] * this_contact_position
    #
    #         # Compute the mean position weighted by the force magnitude
    #         center_of_pressure = controller.cx.zeros(3, 1)
    #         for i_component in range(3):
    #             # Avoid division by zero if the force is too small
    #             center_of_pressure[i_component] = cas.if_else(
    #                 total_force[i_component] ** 2 < 1e-8, 0, weighted_sum[i_component] / total_force[i_component]
    #             )
    #
    #         return center_of_pressure
    #
    #     def custom_dynamics_no_contact(
    #         time,
    #         states,
    #         controls,
    #         parameters,
    #         algebraic_states,
    #         numerical_timeseries,
    #         nlp,
    #     ):
    #
    #         q = DynamicsFunctions.get(nlp.states["q"], states)
    #         qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    #         tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
    #         f_ext_residual_value = DynamicsFunctions.get(nlp.controls["contact_forces"], controls)
    #         f_ext_residual_position = DynamicsFunctions.get(nlp.controls["contact_positions"], controls)
    #
    #         external_forces = nlp.get_external_forces(
    #             "external_forces", states, controls, algebraic_states, numerical_timeseries
    #         )
    #         external_forces[:3] += f_ext_residual_position
    #         external_forces[6:9] += f_ext_residual_value
    #
    #         ddq = nlp.model.forward_dynamics()(q, qdot, tau, external_forces, nlp.parameters.cx)
    #
    #         return DynamicsEvaluation(dxdt=cas.vertcat(qdot, ddq), defects=None)
    #
    #     def custom_configure_no_contact(ocp, nlp, numerical_data_timeseries=None):
    #         ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    #         ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
    #         ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
    #         ConfigureProblem.configure_translational_forces(ocp, nlp, as_states=False, as_controls=True, n_contacts=1)
    #         ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamics_no_contact)
    #         return
    #
    #     def custom_dynamics_with_contacts(
    #         time,
    #         states,
    #         controls,
    #         parameters,
    #         algebraic_states,
    #         numerical_timeseries,
    #         nlp,
    #     ):
    #
    #         q = DynamicsFunctions.get(nlp.states["q"], states)
    #         qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    #         tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
    #         # f_ext = DynamicsFunctions.get(nlp.algebraic_states["contact_forces"], algebraic_states)
    #
    #         external_forces = nlp.get_external_forces(
    #             "rigid_contact_forces", states, controls, algebraic_states, numerical_timeseries
    #         )
    #
    #         # q_ddot_computed = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact=False, external_forces=external_forces)
    #         # dxdt = nlp.cx(nlp.states.shape, q_ddot_computed.shape[1])
    #         # dxdt[nlp.states["q"].index, :] = cas.horzcat(*[qdot for _ in range(q_ddot_computed.shape[1])])
    #         # dxdt[nlp.states["qdot"].index, :] = q_ddot_computed
    #
    #         # Defects
    #         slope_q = DynamicsFunctions.get(nlp.states_dot["qdot"], nlp.states_dot.scaled.cx)
    #         slope_qdot = DynamicsFunctions.get(nlp.states_dot["qddot"], nlp.states_dot.scaled.cx)
    #         tau_id = DynamicsFunctions.inverse_dynamics(
    #             nlp, q, slope_q, slope_qdot, with_contact=False, external_forces=external_forces
    #         )
    #         # defects = nlp.cx(slope_q.shape[0] + tau_id.shape[0], tau_id.shape[1])
    #
    #         defects = cas.horzcat(qdot - slope_q, tau - tau_id)
    #
    #         # defects[: dq.shape[0], :] = cas.horzcat(*dq_defects)
    #         # # We modified on purpose the size of the tau to keep the zero in the defects in order to respect the dynamics
    #         # defects[dq.shape[0] :, :] = tau - tau_id
    #
    #         return DynamicsEvaluation(None, defects)
    #
    #     def custom_configure_with_contacts(ocp, nlp, numerical_data_timeseries=None):
    #         ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    #         ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False, as_states_dot=True)
    #         ConfigureProblem.configure_qddot(ocp, nlp, as_states=False, as_controls=False, as_states_dot=True)
    #         ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
    #         ConfigureProblem.configure_rigid_contact_forces(
    #             ocp, nlp, as_states=False, as_algebraic_states=True, as_controls=False
    #         )
    #         ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamics_with_contacts)
    #         return
    #
    #     try:
    #         from bioptim import (
    #             BiorbdModel,
    #             ConfigureProblem,
    #             DynamicsFunctions,
    #             DynamicsEvaluation,
    #             DynamicsFcn,
    #             InitialGuess,
    #             InitialGuessList,
    #             InterpolationType,
    #             NonLinearProgram,
    #             ObjectiveFcn,
    #             ObjectiveList,
    #             OptimalControlProgram,
    #             PhaseTransitionFcn,
    #             PhaseTransitionList,
    #             PhaseDynamics,
    #             BoundsList,
    #             ConstraintFcn,
    #             ConstraintList,
    #             Solver,
    #             OdeSolver,
    #             ExternalForceSetTimeSeries,
    #             Node,
    #             DynamicsList,
    #             BiMappingList,
    #             DefectType,
    #             PenaltyController,
    #         )
    #
    #     except:
    #         raise RuntimeError("To reconstruct optimally, you must install ")
    #
    #     print(f"Preparing optimal control problem...")
    #
    #     polynomial_degree = 3
    #
    #     if with_contact:
    #         biorbd_model_path = self.biorbd_model_path.replace(".bioMod", "_heelL_toesL.bioMod")
    #         bio_model = BiorbdModel(biorbd_model_path)
    #     else:
    #         # External force set
    #         external_force_set = ExternalForceSetTimeSeries(nb_frames=self.n_shooting)
    #         external_force_set.add(
    #             "calcn_l",
    #             self.f_ext_exp_ocp["left_leg"][3:9, :-1],
    #             point_of_application=self.f_ext_exp_ocp["left_leg"][:3, :-1],
    #         )
    #         # external_force_set.add(
    #         #     "calcn_r",
    #         #     self.f_ext_exp_ocp["right_leg"][3:9, :-1],
    #         #     point_of_application=self.f_ext_exp_ocp["right_leg"][:3, :-1],
    #         # )
    #         numerical_time_series = {"external_forces": external_force_set.to_numerical_time_series()}
    #         biorbd_model_path = self.biorbd_model_path.replace(".bioMod", "_no_contacts.bioMod")
    #         bio_model = BiorbdModel(biorbd_model_path, external_force_set=external_force_set)
    #
    #     nb_q = bio_model.nb_q
    #     nb_root = 6  # Necessary because of the ground segment (model.nb_root does not work)
    #     nb_tau = nb_q - nb_root
    #     r_foot_marker_index = np.array(
    #         [bio_model.marker_index(f"RCAL"), bio_model.marker_index(f"RMFH1"), bio_model.marker_index(f"RMFH5")]
    #     )
    #
    #     # Declaration of the objectives
    #     objective_functions = ObjectiveList()
    #     objective_functions.add(
    #         objective=ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
    #         key="tau",
    #         weight=0.001,
    #     )
    #     objective_functions.add(
    #         objective=ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
    #         key="tau",
    #         weight=1.0,
    #         index=[0, 1, 2, 3, 4, 5],
    #     )
    #     objective_functions.add(
    #         objective=ObjectiveFcn.Lagrange.TRACK_MARKERS, weight=100.0, node=Node.ALL, target=self.markers_exp_ocp
    #     )
    #     objective_functions.add(
    #         objective=ObjectiveFcn.Lagrange.TRACK_MARKERS,
    #         weight=1000.0,
    #         node=Node.ALL,
    #         marker_index=["RCAL", "RMFH1", "RMFH5"],
    #         target=self.markers_exp_ocp[:, r_foot_marker_index, :],
    #     )
    #     objective_functions.add(
    #         objective=ObjectiveFcn.Lagrange.TRACK_STATE, key="q", weight=1.0, node=Node.ALL, target=self.q_exp_ocp
    #     )
    #     objective_functions.add(
    #         objective=ObjectiveFcn.Lagrange.TRACK_STATE,
    #         key="qdot",
    #         node=Node.ALL,
    #         weight=0.01,
    #         target=self.qdot_exp_ocp,
    #     )
    #     if with_contact:
    #         # Explicit
    #         # objective_functions.add(
    #         #     objective=ObjectiveFcn.Lagrange.TRACK_SUM_REACTION_FORCES,
    #         #     weight=0.01,
    #         #     target=self.f_ext_exp_ocp["left_leg"][6:9, :-1],
    #         #     contact_index=[0, 1, 2],
    #         # )
    #         # objective_functions.add(
    #         #     objective=ObjectiveFcn.Lagrange.TRACK_CENTER_OF_PRESSURE,
    #         #     weight=0.01,
    #         #     target=self.f_ext_exp_ocp["left_leg"][0:3, :-1],
    #         #     contact_index=[0, 1, 2],
    #         # )
    #         # Implicit
    #         objective_functions.add(
    #             minimize_sum_reaction_forces,
    #             custom_type=ObjectiveFcn.Lagrange,
    #             node=Node.ALL_SHOOTING,
    #             weight=0.01,
    #             target=self.f_ext_exp_ocp["left_leg"][6:9, :-1],
    #             contact_index=[0, 1, 2],
    #         )
    #         objective_functions.add(
    #             minimize_center_of_pressure,
    #             custom_type=ObjectiveFcn.Lagrange,
    #             weight=0.01,
    #             target=self.f_ext_exp_ocp["left_leg"][0:3, :-1],
    #             contact_index=[0, 1, 2],
    #         )
    #     else:
    #         objective_functions.add(  # Minimize residual contact forces
    #             objective=ObjectiveFcn.Lagrange.TRACK_CONTROL,
    #             key="contact_forces",
    #             node=Node.ALL_SHOOTING,
    #             weight=0.01,
    #         )
    #         objective_functions.add(  # Track CoP position
    #             objective=ObjectiveFcn.Lagrange.TRACK_CONTROL,
    #             key="contact_positions",
    #             node=Node.ALL_SHOOTING,
    #             weight=0.01,
    #             target=self.f_ext_exp_ocp["left_leg"][0:3, :-1],
    #         )
    #
    #     constraints = ConstraintList()
    #     if with_contact:
    #         # Explicit
    #         # constraints.add(
    #         #     ConstraintFcn.TRACK_CONTACT_FORCES,  # Only pushing on the floor, no pulling (Z heel)
    #         #     min_bound=0,
    #         #     max_bound=np.inf,
    #         #     node=Node.ALL_SHOOTING,
    #         #     contact_index=2,
    #         # )
    #         # constraints.add(
    #         #     ConstraintFcn.TRACK_CONTACT_FORCES,  # Only pushing on the floor, no pulling (Z LMFH1)
    #         #     min_bound=0,
    #         #     max_bound=np.inf,
    #         #     node=Node.ALL_SHOOTING,
    #         #     contact_index=3,
    #         # )
    #         # constraints.add(
    #         #     ConstraintFcn.TRACK_CONTACT_FORCES,  # Only pushing on the floor, no pulling (Z LMFH5)
    #         #     min_bound=0,
    #         #     max_bound=np.inf,
    #         #     node=Node.ALL_SHOOTING,
    #         #     contact_index=4,
    #         # )
    #         # for marker in ["LCAL", "LMFH1", "LMFH5"]:
    #         #     # Impose treadmill speed
    #         #     constraints.add(
    #         #         ConstraintFcn.TRACK_MARKERS_VELOCITY,
    #         #         min_bound=self.subject.preferential_speed - 0.1,
    #         #         max_bound=self.subject.preferential_speed + 0.1,
    #         #         node=Node.START,  # Actually it's ALL, but the contact dynamics should take care of it (non-acceleration dynamics contraint)
    #         #         marker_index=marker,
    #         #     )
    #
    #         # Implicit
    #         # Impose marker velocity to be the treadmill speed
    #         constraints.add(
    #             marker_velocity,
    #             min_bound=[self.subject.preferential_speed, 0.0, 0.0] * 3 * (polynomial_degree + 1),
    #             max_bound=[self.subject.preferential_speed, 0.0, 0.0] * 3 * (polynomial_degree + 1),
    #             node=Node.ALL,
    #         )
    #
    #     dynamics = DynamicsList()  # TODO: Charbie -> Change for muscles
    #     if with_contact:
    #         dynamics.add(
    #             custom_configure_with_contacts,
    #             dynamic_function=custom_dynamics_with_contacts,
    #             phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
    #         )
    #     else:
    #         dynamics.add(
    #             custom_configure_no_contact,
    #             dynamic_function=custom_dynamics_no_contact,
    #             numerical_data_timeseries=numerical_time_series,
    #             phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
    #         )
    #
    #     x_bounds = BoundsList()
    #     # Bounds from model
    #     # x_bounds["q"] = bio_model.bounds_from_ranges("q")
    #     # x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    #     # Bounds personalized to the subject's current kinematics
    #     min_q = self.q_exp_ocp[:, :] - 0.3
    #     min_q[:6, :] = self.q_exp_ocp[:6, :] - 0.05
    #     max_q = self.q_exp_ocp[:, :] + 0.3
    #     max_q[:6, :] = self.q_exp_ocp[:6, :] + 0.05
    #     x_bounds.add("q", min_bound=min_q, max_bound=max_q, interpolation=InterpolationType.EACH_FRAME)
    #     # Bounds personalized to the subject's current joint velocities (not a real limitation, so it is executed with +-5)
    #     x_bounds.add(
    #         "qdot",
    #         min_bound=self.qdot_exp_ocp - 10,
    #         max_bound=self.qdot_exp_ocp + 10,
    #         interpolation=InterpolationType.EACH_FRAME,
    #     )
    #
    #     x_init = InitialGuessList()
    #     x_init.add("q", initial_guess=self.q_exp_ocp, interpolation=InterpolationType.EACH_FRAME)
    #     x_init.add("qdot", initial_guess=self.qdot_exp_ocp, interpolation=InterpolationType.EACH_FRAME)
    #
    #     u_bounds = BoundsList()
    #     # TODO: Charbie -> Change for maximal tau during the trial to simulate limited force
    #     u_bounds.add("tau", min_bound=[-500] * nb_q, max_bound=[500] * nb_q, interpolation=InterpolationType.CONSTANT)
    #     if not with_contact:
    #         u_bounds.add(
    #             "contact_positions",
    #             min_bound=[0, 0, -0.0001],
    #             max_bound=[1.5, 1.5, 0.0001],
    #             interpolation=InterpolationType.CONSTANT,
    #         )
    #         u_bounds.add(
    #             "contact_forces", min_bound=[-300] * 3, max_bound=[300] * 3, interpolation=InterpolationType.CONSTANT
    #         )
    #
    #     u_init = InitialGuessList()
    #     u_init.add("tau", initial_guess=self.tau_exp_ocp[:, :-1], interpolation=InterpolationType.EACH_FRAME)
    #     if not with_contact:
    #         u_init.add(
    #             "contact_positions",
    #             initial_guess=self.f_ext_exp_ocp["left_leg"][:3, :-1],
    #             interpolation=InterpolationType.EACH_FRAME,
    #         )
    #
    #     # TODO: Charbie -> Add phase transition when I have the full cycle
    #     # phase_transitions = PhaseTransitionList()
    #     # phase_transitions.add(PhaseTransitionFcn.CYCLIC, phase_pre_idx=0)
    #
    #     self.ocp = OptimalControlProgram(
    #         bio_model=bio_model,
    #         n_shooting=self.n_shooting,
    #         phase_time=self.phase_time,
    #         dynamics=dynamics,
    #         x_bounds=x_bounds,
    #         u_bounds=u_bounds,
    #         x_init=x_init,
    #         u_init=u_init,
    #         objective_functions=objective_functions,
    #         constraints=constraints,
    #         # phase_transitions=phase_transitions,
    #         ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3, defects_type=DefectType.IMPLICIT),
    #         # ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3),
    #         use_sx=False,
    #         n_threads=10,
    #     )

    def solve(self, show_online_optim: bool = False):
        from bioptim import SolutionMerge, TimeAlignment, Solver

        solver = Solver.IPOPT(show_online_optim=show_online_optim, show_options=dict(show_bounds=True))
        solver.set_linear_solver("ma57")
        solver.set_maximum_iterations(10_000)  # 10_000
        solver.set_tol(1e-3)  # TODO: Charbie -> Change for a more appropriate value (just to see for now)
        self.solution = self.ocp.solve(solver=solver)
        self.time_opt = self.solution.decision_time(to_merge=SolutionMerge.NODES, time_alignment=TimeAlignment.STATES)
        self.q_opt = self.solution.decision_states(to_merge=SolutionMerge.NODES)["q"]
        self.qdot_opt = self.solution.decision_states(to_merge=SolutionMerge.NODES)["qdot"]
        self.tau_opt = self.solution.decision_controls(to_merge=SolutionMerge.NODES)["tau"]
        self.opt_status = "CVG" if self.solution.status == 0 else "DVG"

    def animate_solution(self):

        try:
            from pyorerun import BiorbdModel, PhaseRerun, PyoMarkers
        except:
            raise RuntimeError("To animate the optimal solution, you must install Pyorerun.")

        # Add the model
        model = BiorbdModel(self.model_ocp)
        model.options.transparent_mesh = False
        model.options.show_gravity = True
        viz = PhaseRerun(np.linspace(0, self.phase_time, self.n_shooting + 1))

        # Add experimental markers
        markers = PyoMarkers(data=self.markers_exp_ocp, channels=list(model.marker_names), show_labels=False)

        # Add force plates to the animation
        viz.add_force_plate(num=1, corners=self.experimental_data.platform_corners[0])
        viz.add_force_plate(num=2, corners=self.experimental_data.platform_corners[1])
        viz.add_force_data(
            num=1,
            force_origin=self.f_ext_exp_ocp["left_leg"][:3, :],
            force_vector=self.f_ext_exp_ocp["left_leg"][6:9, :],
        )
        viz.add_force_data(
            num=2,
            force_origin=self.f_ext_exp_ocp["right_leg"][:3, :],
            force_vector=self.f_ext_exp_ocp["right_leg"][6:9, :],
        )

        # Add the kinematics
        viz.add_animated_model(model, self.q_opt, tracked_markers=markers)

        # Play
        viz.rerun_by_frame("OCP optimal solution")

    def check_if_existing(self):
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
                self.model_ocp = data["model_ocp"]
                self.q_exp_ocp = data["q_exp_ocp"]
                self.qdot_exp_ocp = data["qdot_exp_ocp"]
                self.tau_exp_ocp = data["tau_exp_ocp"]
                self.f_ext_exp_ocp = data["f_ext_exp_ocp"]
                self.markers_exp_ocp = data["markers_exp_ocp"]
                self.emg_exp_ocp = data["emg_exp_ocp"]
                self.n_shooting = data["n_shooting"]
                self.phase_time = data["phase_time"]
                self.q_opt = data["q_opt"]
                self.qdot_opt = data["qdot_opt"]
                self.tau_opt = data["tau_opt"]
            return True
        else:
            return False

    def extract_muscle_forces(self):
        # TODO: Charbie -> Extract muscle forces from the solution
        self.muscle_forces = None

    def get_result_file_full_path(self, result_folder=None):
        if result_folder is None:
            result_folder = self.experimental_data.result_folder
        trial_name = self.experimental_data.c3d_full_file_path.split("/")[-1][:-4]
        result_file_full_path = f"{result_folder}/optim_estim_{trial_name}_{self.opt_status}.pkl"
        return result_file_full_path

    def save_optimal_reconstruction(self):
        """
        Save the optimal estimation reconstruction.
        """
        result_file_full_path = self.get_result_file_full_path()
        with open(result_file_full_path, "wb") as file:
            pickle.dump(self.outputs(), file)

    def inputs(self):
        return {
            "cycle_to_analyze": self.cycle_to_analyze,
            "biorbd_model_path": self.model_creator.biorbd_model_full_path,
            "experimental_data": self.experimental_data,
            "events": self.events,
            "kinematics_reconstructor": self.kinematics_reconstructor,
        }

    def outputs(self):
        return {
            "model_ocp": self.model_ocp,
            "q_exp_ocp": self.q_exp_ocp,
            "qdot_exp_ocp": self.qdot_exp_ocp,
            "tau_exp_ocp": self.tau_exp_ocp,
            "f_ext_exp_ocp": self.f_ext_exp_ocp,
            "markers_exp_ocp": self.markers_exp_ocp,
            "emg_exp_ocp": self.emg_exp_ocp,
            "n_shooting": self.n_shooting,
            "phase_time": self.phase_time,
            "q_opt": self.q_opt,
            "qdot_opt": self.qdot_opt,
            "tau_opt": self.tau_opt,
            "opt_status": self.opt_status,
        }
