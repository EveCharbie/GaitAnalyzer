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
        self.q_opt = None
        self.qdot_opt = None
        self.tau_opt = None
        self.muscles_opt = None
        self.f_ext_value_opt = None
        self.f_ext_position_opt = None
        self.opt_status = "CVG"
        self.muscle_forces = None
        self.is_loaded_optimal_solution = False

        # Execution
        if skip_if_existing and self.check_if_existing():
            print("Optimal estimation already exists, skipping...")
            self.is_loaded_optimal_solution = True
        else:
            print("Performing optimal estimation...")

            self.generate_no_contacts_model()
            self.prepare_reduced_experimental_data(plot_exp_data_flag=False, animate_exp_data_flag=True)
            self.prepare_ocp_fext(with_residual_forces=True)
            self.solve(show_online_optim=True)
            self.extract_muscle_forces()
            self.save_optimal_reconstruction()

            if plot_solution_flag:
                self.solution.graphs(
                    show_bounds=True,
                    save_name=self.get_result_file_full_path(self.experimental_data.result_folder + "/figures")[:-4],
                )

        if animate_solution_flag:
            self.animate_solution()

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

    def prepare_reduced_experimental_data(self, plot_exp_data_flag: bool = False, animate_exp_data_flag: bool = False):
        """
        To reduce the optimization time, only one cycle is treated at a time
        (and the number of degrees of freedom is reduced?).
        """
        # self.model_ocp = self.biorbd_model_path.replace(".bioMod", "_heelL_toesL.bioMod")
        self.model_ocp = self.model_creator.biorbd_model_full_path.replace(".bioMod", "_no_contacts.bioMod")
        model = biorbd.Model(self.model_ocp)

        # One full cycle
        cycle_timings = self.events.events["right_leg_heel_touch"]
        this_sequence_analogs = list(range(cycle_timings[self.cycle_to_analyze], cycle_timings[self.cycle_to_analyze + 1]))
        this_sequence_markers = Operator.from_analog_frame_to_marker_frame(
            analogs_time_vector=self.experimental_data.analogs_time_vector,
            markers_time_vector=self.experimental_data.markers_time_vector,
            analog_idx=this_sequence_analogs,
        )

        # Skipping some frames to lighten the OCP
        marker_start = this_sequence_markers[0]
        marker_end = this_sequence_markers[-1]
        marker_hop = 1
        idx_to_keep = np.arange(marker_start, marker_end, marker_hop)
        print(f"------------------ nb_frames = {len(idx_to_keep)} ------------------")
        index_to_keep_filtered_q = idx_to_keep - self.kinematics_reconstructor.frame_range.start
        nb_frames = len(idx_to_keep)

        frame_index_shifted_half_cycle = list(range(nb_frames))
        frame_index_shifted_half_cycle[0: int(np.floor(nb_frames / 2))] = list(range(int(np.ceil(nb_frames / 2)), nb_frames))
        frame_index_shifted_half_cycle[int(np.floor(nb_frames / 2)):] = list(range(0, int(np.ceil(nb_frames / 2))))

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

        # Copy the right leg activation to the left led with a time delay of 1/2 cycle
        for i_muscle, muscle_name in enumerate(muscle_names):
            if muscle_name in self.model_creator.osim_model_type.muscle_name_mapping:
                muscle_speudo = self.model_creator.osim_model_type.muscle_name_mapping[muscle_name]
                if muscle_speudo is not None:
                    i_muscle_l = muscle_names.index(muscle_name.replace("_r", "_l"))
                    self.emg_normalized_exp_ocp[i_muscle_l, :] = self.emg_normalized_exp_ocp[
                        i_muscle, frame_index_shifted_half_cycle]

        self.markers_exp_ocp = self.experimental_data.markers_sorted[:, :, idx_to_keep]
        # Fill NaNs in markers
        for i_marker in range(self.markers_exp_ocp.shape[1]):
            if np.any(np.isnan(self.markers_exp_ocp[:, i_marker, :])):
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
            emg = PyoMuscles(data=self.emg_normalized_exp_ocp,
                             muscle_names=list(model.muscle_names),
                             mvc=np.ones((len(model.muscle_names), 1)),
                             )

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
            viz.rerun("OCP initial guess from experimental data")

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
            weight=0.001,
        )
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            key="tau",
            weight=0.1,
            index=[0, 1, 2, 3, 4, 5],
        )
        # Note: all muscles have a target except tfl, see if we should hendle it differently
        # index = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            key="muscles",
            weight=1,
            target=self.emg_normalized_exp_ocp[:, :-1],
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
                weight=10,
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
        u_bounds.add("tau", min_bound=[-800] * nb_q, max_bound=[800] * nb_q, interpolation=InterpolationType.CONSTANT)
        u_bounds.add(
            "muscles",
            min_bound=[0.0001] * nb_muscles,
            max_bound=[1.0] * nb_muscles,
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
        u_init.add("tau", initial_guess=self.tau_exp_ocp[:, :-1], interpolation=InterpolationType.EACH_FRAME)
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

        # TODO: Charbie -> Add a cyclic phase transition ?
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

    def solve(self, show_online_optim: bool = False):
        from bioptim import SolutionMerge, TimeAlignment, Solver

        solver = Solver.IPOPT(show_online_optim=show_online_optim, show_options=dict(show_bounds=True))
        solver.set_linear_solver("ma57")
        solver.set_maximum_iterations(1000)  # 10_000
        solver.set_tol(1e-3)  # TODO: Charbie -> Change for a more appropriate value (just to see for now)
        self.solution = self.ocp.solve(solver=solver)
        self.time_opt = self.solution.decision_time(to_merge=SolutionMerge.NODES, time_alignment=TimeAlignment.STATES)
        self.q_opt = self.solution.decision_states(to_merge=SolutionMerge.NODES)["q"]
        self.qdot_opt = self.solution.decision_states(to_merge=SolutionMerge.NODES)["qdot"]
        self.tau_opt = self.solution.decision_controls(to_merge=SolutionMerge.NODES)["tau"]
        self.muscles_opt = self.solution.decision_controls(to_merge=SolutionMerge.NODES)["muscles"]
        self.f_ext_value_opt = self.solution.decision_controls(to_merge=SolutionMerge.NODES)["contact_forces"]
        self.f_ext_position_opt = self.solution.decision_controls(to_merge=SolutionMerge.NODES)["contact_positions"]
        self.opt_status = "CVG" if self.solution.status == 0 else "DVG"

    def animate_solution(self):

        try:
            from pyorerun import BiorbdModel, PhaseRerun, PyoMarkers, PyoMuscles
        except:
            raise RuntimeError("To animate the optimal solution, you must install Pyorerun.")

        # Add the model
        model = BiorbdModel(self.model_ocp)
        model.options.transparent_mesh = False
        model.options.show_gravity = True
        viz = PhaseRerun(np.linspace(0, self.phase_time, self.n_shooting + 1))

        # Add experimental markers
        markers = PyoMarkers(data=self.markers_exp_ocp, marker_names=list(model.marker_names), show_labels=False)
        nb_muscles = len(model.muscle_names)
        emgs = PyoMuscles(data=np.hstack((self.muscles_opt, np.zeros((nb_muscles, 1)))),
                          muscle_names=list(model.muscle_names),
                          mvc=np.ones((nb_muscles, 1)),
                          )

        # Add force plates to the animation
        viz.add_force_plate(num=1, corners=self.experimental_data.platform_corners[0])
        viz.add_force_plate(num=2, corners=self.experimental_data.platform_corners[1])
        viz.add_force_plate(num=3, corners=self.experimental_data.platform_corners[0])
        viz.add_force_plate(num=4, corners=self.experimental_data.platform_corners[1])
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
        viz.add_force_data(
            num=3,
            force_origin=np.hstack((self.f_ext_position_opt[:3, :], np.zeros((3, 1)))),
            force_vector=np.hstack((self.f_ext_value_opt[:3, :], np.zeros((3, 1)))),
        )
        viz.add_force_data(
            num=3,
            force_origin=np.hstack((self.f_ext_position_opt[3:6, :], np.zeros((3, 1)))),
            force_vector=np.hstack((self.f_ext_value_opt[3:6, :], np.zeros((3, 1)))),
        )

        # Add the kinematics
        viz.add_animated_model(model, self.q_opt, tracked_markers=markers, muscle_activations_intensity=emgs)

        # Play
        viz.rerun("OCP optimal solution")

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
                self.muscles_opt = data["muscles_opt"]
                self.f_ext_value_opt = data["f_ext_value_opt"]
                self.f_ext_position_opt = data["f_ext_position_opt"]
                self.opt_status = data["opt_status"]
                self.muscle_forces = data["muscle_forces"]
            return True
        else:
            return False

    def extract_muscle_forces(self):
        model = biorbd.Model(self.model_ocp)
        self.muscle_forces = np.zeros((model.nbMuscles(), self.n_shooting))
        for i_frame in range(self.n_shooting):
            muscles = model.stateSet()
            for i_muscle, muscle in enumerate(muscles):
                muscle.setActivation(self.muscles_opt[i_muscle, i_frame])
            self.muscle_forces[:, i_frame] = model.muscleForces(muscles, self.q_opt[:, i_frame], self.qdot_opt[:, i_frame]).to_array()

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
            "muscles_opt": self.muscles_opt,
            "f_ext_value_opt": self.f_ext_value_opt,
            "f_ext_position_opt": self.f_ext_position_opt,
            "opt_status": self.opt_status,
            "muscle_forces": self.muscle_forces,
        }
