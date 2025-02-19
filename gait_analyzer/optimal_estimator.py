import pickle
import numpy as np
import casadi as cas

from pyomeca import Markers
try:
    import bioptim
except ImportError:
    print("Skipped Bioptim import as it is not installed")

from gait_analyzer.operator import Operator
from gait_analyzer.kinematics_reconstructor import KinematicsReconstructor
from gait_analyzer.inverse_dynamics_performer import InverseDynamicsPerformer
from gait_analyzer.experimental_data import ExperimentalData
from gait_analyzer.events import Events
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
        biorbd_model_path: str,
        experimental_data: ExperimentalData,
        events: Events,
        kinematics_reconstructor: KinematicsReconstructor,
        inverse_dynamic_performer: InverseDynamicsPerformer,
        plot_solution_flag: bool,
        animate_solution_flag: bool,
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
        biorbd_model_path: str
            The full path to the biorbd model.
        experimental_data: ExperimentalData
            The experimental data to match.
        events: Events
            The events of the gait cycle to split the trial into appropriate phases.
        kinematics_reconstructor: KinematicsReconstructor
            The kinematics reconstructor to use.
        inverse_dynamic_performer: InverseDynamicsPerformer
            The inverse dynamics performer to use.
        plot_solution_flag: bool
            If True, the solution will be plotted.
        animate_solution_flag: bool
            If True, the solution will be animated.
        """

        # Checks
        if not isinstance(cycle_to_analyze, int):
            raise ValueError("cycle_to_analyze must be an int")
        if not isinstance(subject, Subject):
            raise ValueError("subject must be a Subject")
        if not isinstance(biorbd_model_path, str):
            raise ValueError("biorbd_model_path must be a string")
        if not isinstance(experimental_data, ExperimentalData):
            raise ValueError("experimental_data must be an ExperimentalData")
        if not isinstance(events, Events):
            raise ValueError("events must be an Events")
        if not isinstance(kinematics_reconstructor, KinematicsReconstructor):
            raise ValueError("kinematics_reconstructor must be a KinematicsReconstructor")
        if not isinstance(inverse_dynamic_performer, InverseDynamicsPerformer):
            raise ValueError("inverse_dynamic_performer must be a InverseDynamicsPerformer")

        # Initial attributes
        self.cycle_to_analyze = cycle_to_analyze
        self.subject = subject
        self.biorbd_model_path = biorbd_model_path
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
        self.generate_contact_biomods()
        self.prepare_reduced_experimental_data(plot_exp_data_flag=False, animate_exp_data_flag=False)
        self.prepare_ocp(with_contact=False)
        self.solve(show_online_optim=False)
        self.save_optimal_reconstruction()
        if plot_solution_flag:
            self.solution.graphs(show_bounds=True, save_name=self.get_result_file_full_path(self.experimental_data.result_folder + "/figures"))
        if animate_solution_flag:
            self.animate_solution()
        self.extract_muscle_forces()

    def generate_contact_biomods(self):
        """
        Create other bioMod files with the addition of the different feet contact conditions.
        """

        def add_txt_per_condition(condition: str) -> str:
            # TODO: Charbie -> Until biorbd is fixed to read biomods, I will hard code the position of the contacts
            contact_text = "\n/*-------------- CONTACTS---------------\n*/\n"
            if "heelL" in condition and "toesL" in condition:
                contact_text += f"contact\tLCAL\n"
                contact_text += f"\tparent\tcalcn_l\n"
                contact_text += f"\tposition\t-0.018184372684362127\t-0.036183919561541877\t0.010718604411614319\n"
                contact_text += f"\taxis\txyz\n"
                contact_text += "endcontact\n\n"

                contact_text += f"contact\tLMFH1\n"
                contact_text += f"\tparent\tcalcn_l\n"
                contact_text += f"\tposition\t0.19202791077724868\t-0.013754853217574914\t0.039283237127771042\n"
                contact_text += f"\taxis\tz\n"
                contact_text += "endcontact\n\n"

                contact_text += f"contact\tLMFH5\n"
                contact_text += f"\tparent\tcalcn_l\n"
                contact_text += f"\tposition\t0.18583815793306013\t-0.0092170000425693677\t-0.072430596752376397\n"
                contact_text += f"\taxis\tzy\n"
                contact_text += "endcontact\n\n"

            elif "heelL" in condition:
                contact_text += f"contact\tLCAL\n"
                contact_text += f"\tparent\tcalcn_l\n"
                contact_text += f"\tposition\t-0.018184372684362127\t-0.036183919561541877\t0.010718604411614319\n"
                contact_text += f"\taxis\txyz\n"
                contact_text += "endcontact\n\n"

            elif "toesL" in condition:
                contact_text += f"contact\tLMFH1\n"
                contact_text += f"\tparent\tcalcn_l\n"
                contact_text += f"\tposition\t0.19202791077724868\t-0.013754853217574914\t0.039283237127771042\n"
                contact_text += f"\taxis\txz\n"
                contact_text += "endcontact\n\n"

                contact_text += f"contact\tLMFH5\n"
                contact_text += f"\tparent\tcalcn_l\n"
                contact_text += f"\tposition\t0.18583815793306013\t-0.0092170000425693677\t-0.072430596752376397\n"
                contact_text += f"\taxis\txyz\n"
                contact_text += "endcontact\n\n"

            if "heelR" in condition and "toesR" in condition:
                contact_text += f"contact\tRCAL\n"
                contact_text += f"\tparent\tcalcn_r\n"
                contact_text += f"\tposition\t-0.017776522017632024\t-0.030271301561674208\t-0.015068364463032391\n"
                contact_text += f"\taxis\txyz\n"
                contact_text += "endcontact\n\n"

                contact_text += f"contact\tRMFH1\n"
                contact_text += f"\tparent\tcalcn_r\n"
                contact_text += f"\tposition\t0.20126587479704638\t-0.0099656486276807066\t-0.039248701869426805\n"
                contact_text += f"\taxis\tz\n"
                contact_text += "endcontact\n\n"

                contact_text += f"contact\tRMFH5\n"
                contact_text += f"\tparent\tcalcn_r\n"
                contact_text += f"\tposition\t0.18449626841163846\t-0.018897872323952902\t0.07033570386440513\n"
                contact_text += f"\taxis\tzy\n"
                contact_text += "endcontact\n\n"

            elif "heelR" in condition:
                contact_text += f"contact\tRCAL\n"
                contact_text += f"\tparent\tcalcn_r\n"
                contact_text += f"\tposition\t-0.017776522017632024\t-0.030271301561674208\t-0.015068364463032391\n"
                contact_text += f"\taxis\txyz\n"
                contact_text += "endcontact\n\n"

            elif "toesR" in condition:
                contact_text += f"contact\tRMFH1\n"
                contact_text += f"\tparent\tcalcn_r\n"
                contact_text += f"\tposition\t0.20126587479704638\t-0.0099656486276807066\t-0.039248701869426805\n"
                contact_text += f"\taxis\txz\n"
                contact_text += "endcontact\n\n"

                contact_text += f"contact\tRMFH5\n"
                contact_text += f"\tparent\tcalcn_r\n"
                contact_text += f"\tposition\t0.18449626841163846\t-0.018897872323952902\t0.07033570386440513\n"
                contact_text += f"\taxis\txyz\n"
                contact_text += "endcontact\n\n"

            return contact_text

        original_model_path = self.biorbd_model_path
        conditions = [
            "heelR_toesR",
            "toesR_heelL",
            "toesR",
            "toesR_heelL",
            "heelL_toesL",
            "toesL",
            "toesL_heelR",
            "toesL_heelR_toesR",
            "no_contacts",
        ]
        for condition in conditions:
            new_model_path = original_model_path.replace(".bioMod", f"_{condition}.bioMod")
            with open(original_model_path, "r+", encoding="utf-8") as file:
                lines = file.readlines()
            with open(new_model_path, "w+", encoding="utf-8") as file:
                for i_line, line in enumerate(lines):
                    if i_line + 1 in [558, 559, 560, 1014, 1015, 1016]:
                        pass  # Remove the toes rotations
                    elif i_line + 1 in [1586, 1587, 1588, 1654, 1655, 1656, 1914, 1915, 1916, 2417, 2418, 2419, 2485, 2486, 2487, 2745, 2746, 2747]:
                        pass  # Remove the hands rotations
                    else:
                        file.write(line)
                file.write(add_txt_per_condition(condition))

    def prepare_reduced_experimental_data(self, plot_exp_data_flag: bool = False, animate_exp_data_flag: bool = False):
        """
        To reduce the optimization time, only one cycle is treated at a time
        (and the number of degrees of freedom is reduced?).
        """
        # self.model_ocp = self.biorbd_model_path.replace(".bioMod", "_heelL_toesL.bioMod")
        self.model_ocp = self.biorbd_model_path.replace(".bioMod", "_no_contacts.bioMod")

        # Only one right leg swing (while left leg in flat foot)
        swing_timings = np.where(self.events.phases["heelL_toesL"])[0]
        right_swing_sequence = np.array_split(swing_timings, np.flatnonzero(np.diff(swing_timings) > 1) + 1)
        this_sequence_analogs = right_swing_sequence[self.cycle_to_analyze]
        this_sequence_markers = Operator.from_analog_frame_to_marker_frame(analogs_time_vector=self.experimental_data.analogs_time_vector,
                                                                            markers_time_vector=self.experimental_data.markers_time_vector,
                                                                            analog_idx=this_sequence_analogs)

        # Skipping some frames to lighten the OCP
        marker_start = this_sequence_markers[0]
        marker_end = this_sequence_markers[-1]
        marker_hop = 4
        idx_to_keep = np.arange(marker_start, marker_end, marker_hop)
        index_to_keep_filtered_q = idx_to_keep - self.kinematics_reconstructor.frame_range.start

        # Skipping some DoFs to lighten the OCP
        dof_idx_to_keep = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 34, 35, 36, 37, 38])

        self.n_shooting = idx_to_keep.shape[0] - 1
        self.q_exp_ocp = self.kinematics_reconstructor.q_filtered[np.ix_(dof_idx_to_keep, index_to_keep_filtered_q)]
        self.qdot_exp_ocp = self.kinematics_reconstructor.qdot[np.ix_(dof_idx_to_keep, index_to_keep_filtered_q)]
        self.tau_exp_ocp = self.inverse_dynamic_performer.tau[np.ix_(dof_idx_to_keep, index_to_keep_filtered_q)]
        self.f_ext_exp_ocp = {
            "left_leg": np.zeros((9, self.n_shooting+1)),
            "right_leg": np.zeros((9, self.n_shooting+1)),
        }
        for i_frame, marker_frame in enumerate(idx_to_keep):
            idx_analogs = Operator.from_marker_frame_to_analog_frame(analogs_time_vector=self.experimental_data.analogs_time_vector,
                                                                    markers_time_vector=self.experimental_data.markers_time_vector,
                                                                    marker_idx=int(marker_frame))
            self.f_ext_exp_ocp["left_leg"][:, i_frame] = np.mean(
                self.experimental_data.f_ext_sorted[0, :, idx_analogs-5:idx_analogs+5], axis=1
            )
            self.f_ext_exp_ocp["right_leg"][:, i_frame] = np.mean(
                self.experimental_data.f_ext_sorted[1, :, idx_analogs-5:idx_analogs+5], axis=1
            )
        self.markers_exp_ocp = self.experimental_data.markers_sorted_with_virtual[:, :, idx_to_keep]
        self.phase_time = self.experimental_data.markers_time_vector[idx_to_keep[-1]] - self.experimental_data.markers_time_vector[idx_to_keep[0]]

        if plot_exp_data_flag:
            import matplotlib.pyplot as plt
            time_plot = np.linspace(0, self.phase_time, self.n_shooting+1)

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
                axs[i].plot(np.array([time_plot[0], time_plot[-1]]), np.array([-np.pi/2, -np.pi/2]), '--k')
                axs[i].plot(np.array([time_plot[0], time_plot[-1]]), np.array([np.pi/2, np.pi/2]), '--k')
                axs[i].legend()
            plt.savefig('Dofs_exp.png')
            plt.show()
            print(f"There are {np.sum(np.isnan(self.q_exp_ocp))} NaNs in Q.")
            print(f"There are {np.sum(np.isnan(self.qdot_exp_ocp))} NaNs in Qdot.")

            # Plot Tau to see if there are NaNs
            plt.figure()
            for i_dof in range(self.tau_exp_ocp.shape[0]):
                plt.plot(time_plot, self.tau_exp_ocp[i_dof, :], '.', label=f"tau_{i_dof}")
                nan_idx = np.isnan(self.tau_exp_ocp[i_dof, :])
                plt.plot(time_plot[nan_idx], self.tau_exp_ocp[i_dof, nan_idx], 'ok')
            plt.legend()
            plt.savefig('Tau_exp.png')
            plt.show()
            print(f"There are {np.sum(np.isnan(self.tau_exp_ocp))} NaNs in Tau.")

            # Plot Markers to see if some are missing
            plt.figure()
            for i_marker in range(self.markers_exp_ocp.shape[1]):
                for i_coordinate in range(3):
                    plt.plot(time_plot, self.markers_exp_ocp[i_coordinate, i_marker, :], '.', label=f"marker_{i_marker}")
                    nan_idx = np.isnan(self.markers_exp_ocp[i_coordinate, i_marker, :])
                    plt.plot(time_plot[nan_idx], self.markers_exp_ocp[i_coordinate, i_marker, nan_idx], 'ok')
            plt.legend()
            plt.savefig('Markers_exp.png')
            plt.show()
            print(f"There are {np.sum(np.isnan(self.markers_exp_ocp))} markers missing.")

        if animate_exp_data_flag:
            try:
                from pyorerun import BiorbdModel, PhaseRerun
            except:
                raise RuntimeError("To animate the initial guess, you must install Pyorerun.")

            # Add the model
            model = BiorbdModel(self.model_ocp)
            model.options.transparent_mesh = False
            viz = PhaseRerun(np.linspace(0, self.phase_time, self.n_shooting+1))

            # Add experimental markers
            markers = Markers(data=self.markers_exp_ocp, channels=list(model.marker_names))

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
            viz.add_animated_model(model, self.q_exp_ocp, tracked_markers=markers)

            # Play
            viz.rerun("OCP initial guess from experimental data")

    def prepare_ocp(self, with_contact: bool = False):
        """
        Let's say swing phase only for now
        """

        def custom_dynamics(
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
            tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
            f_ext_residual_value = DynamicsFunctions.get(nlp.controls["contact_forces"], controls)
            f_ext_residual_position = DynamicsFunctions.get(nlp.controls["contact_positions"], controls)

            external_forces = nlp.get_external_forces(states, controls, algebraic_states, numerical_timeseries)
            external_forces[:3] += f_ext_residual_position
            external_forces[6:9] += f_ext_residual_value

            ddq = nlp.model.forward_dynamics()(q, qdot, tau, external_forces, nlp.parameters.cx)

            return DynamicsEvaluation(dxdt=cas.vertcat(qdot, ddq), defects=None)

        def custom_configure(
                ocp, nlp, numerical_data_timeseries=None
        ):
            ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
            ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
            ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
            ConfigureProblem.configure_contact_forces(ocp, nlp, as_states=False, as_controls=True, n_contacts=1)
            ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamics)
            return


        try:
            from bioptim import (BiorbdModel,
                                    ConfigureProblem,
                                    DynamicsFunctions,
                                    DynamicsEvaluation,
                                    DynamicsFcn,
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
                                    DynamicsList,
                                    BiMappingList,
                                 DefectType,
                                     )
            
        except:
            raise RuntimeError("To reconstruct optimally, you must install ")

        print(f"Preparing optimal control problem...")

        if with_contact:
            biorbd_model_path = self.biorbd_model_path.replace(".bioMod", "_heelL_toesL.bioMod")
            bio_model = BiorbdModel(biorbd_model_path)
        else:
            # External force set
            external_force_set = ExternalForceSetTimeSeries(nb_frames=self.n_shooting)
            external_force_set.add(
                "calcn_l",
                self.f_ext_exp_ocp["left_leg"][3:9, :-1],
                point_of_application=self.f_ext_exp_ocp["left_leg"][:3, :-1],
            )
            # external_force_set.add(
            #     "calcn_r",
            #     self.f_ext_exp_ocp["right_leg"][3:9, :-1],
            #     point_of_application=self.f_ext_exp_ocp["right_leg"][:3, :-1],
            # )
            numerical_time_series = {"external_forces": external_force_set.to_numerical_time_series()}
            biorbd_model_path = self.biorbd_model_path.replace(".bioMod", "_no_contacts.bioMod")
            bio_model = BiorbdModel(biorbd_model_path, external_force_set=external_force_set)

        nb_q = bio_model.nb_q
        nb_root = 6  # Necessary because of the ground segment (model.nb_root does not work)
        nb_tau = nb_q - nb_root
        r_foot_marker_index = np.array([bio_model.marker_index(f"RCAL"), bio_model.marker_index(f"RMFH1"), bio_model.marker_index(f"RMFH5")])

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
            weight=1.0,
            index=[0, 1, 2, 3, 4, 5],
        )
        objective_functions.add(objective=ObjectiveFcn.Lagrange.TRACK_MARKERS, weight=100.0, node=Node.ALL, target=self.markers_exp_ocp)
        objective_functions.add(objective=ObjectiveFcn.Lagrange.TRACK_MARKERS, weight=1000.0, node=Node.ALL, marker_index=["RCAL", "RMFH1", "RMFH5"], target=self.markers_exp_ocp[:, r_foot_marker_index, :])
        objective_functions.add(objective=ObjectiveFcn.Lagrange.TRACK_STATE, key="q", weight=1.0, node=Node.ALL, target=self.q_exp_ocp)
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.TRACK_STATE, key="qdot", node=Node.ALL, weight=0.01, target=self.qdot_exp_ocp
        )
        if not with_contact:
            objective_functions.add(  # Minimize residual contact forces
                objective=ObjectiveFcn.Lagrange.TRACK_CONTROL, key="contact_forces", node=Node.ALL_SHOOTING, weight=0.01,
            )
            objective_functions.add(  # Track CoP position
                objective=ObjectiveFcn.Lagrange.TRACK_CONTROL, key="contact_positions", node=Node.ALL_SHOOTING, weight=0.01, target=self.f_ext_exp_ocp["left_leg"][0:3, :-1]
            )
        else:
            objective_functions.add(
                objective=ObjectiveFcn.Lagrange.TRACK_GROUND_REACTION_FORCES,
                weight=0.01,
                target=self.f_ext_exp_ocp["left_leg"][6:9, :-1],
                contact_index=[0, 1, 2],
            )
            objective_functions.add(
                objective=ObjectiveFcn.Lagrange.TRACK_CENTER_OF_PRESSURE,
                weight=0.01,
                target=self.f_ext_exp_ocp["left_leg"][0:3, :-1],
                contact_index=[0, 1, 2],
                associated_marker_index=["LCAL", "LMFH1", "LMFH5"],
            )

        constraints = ConstraintList()
        if with_contact:
            constraints.add(
                ConstraintFcn.TRACK_CONTACT_FORCES,  # Only pushing on the floor, no pulling (Z heel)
                min_bound=0,
                max_bound=np.inf,
                node=Node.ALL_SHOOTING,
                contact_index=2,
            )
            constraints.add(
                ConstraintFcn.TRACK_CONTACT_FORCES,  # Only pushing on the floor, no pulling (Z LMFH1)
                min_bound=0,
                max_bound=np.inf,
                node=Node.ALL_SHOOTING,
                contact_index=3,
            )
            constraints.add(
                ConstraintFcn.TRACK_CONTACT_FORCES,  # Only pushing on the floor, no pulling (Z LMFH5)
                min_bound=0,
                max_bound=np.inf,
                node=Node.ALL_SHOOTING,
                contact_index=4,
            )
            for marker in ["LCAL", "LMFH1", "LMFH5"]:
                # Impose treadmill speed
                constraints.add(
                    ConstraintFcn.TRACK_MARKERS_VELOCITY,
                    min_bound=self.subject.preferential_speed-0.1,
                    max_bound=self.subject.preferential_speed+0.1,
                    node=Node.START,  # Actually it's ALL, but the contact dynamics should take care of it (non-acceleration dynamics contraint)
                    marker_index=marker,
                )

        dynamics = DynamicsList()  # TODO: Charbie -> Change for muscles
        if with_contact:
            dynamics.add(DynamicsFcn.TORQUE_DRIVEN,
                         with_contact=True,
                         phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE)
        else:
            dynamics.add(custom_configure,
                         dynamic_function=custom_dynamics,
                         numerical_data_timeseries=numerical_time_series,
                         phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
                         )

        # TODO: Charbie
        x_bounds = BoundsList()
        # Bounds from model
        # x_bounds["q"] = bio_model.bounds_from_ranges("q")
        # x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
        # Bounds personalized to the subject's current kinematics
        min_q = self.q_exp_ocp[:, :] - 0.3
        min_q[:6, :] = self.q_exp_ocp[:6, :] - 0.05
        max_q = self.q_exp_ocp[:, :] + 0.3
        max_q[:6, :] = self.q_exp_ocp[:6, :] + 0.05
        x_bounds.add("q", min_bound=min_q, max_bound=max_q, interpolation=InterpolationType.EACH_FRAME)
        # Bounds personalized to the subject's current joint velocities (not a real limitation, so it is executed with +-5)
        x_bounds.add("qdot", min_bound=self.qdot_exp_ocp-10, max_bound=self.qdot_exp_ocp+10, interpolation=InterpolationType.EACH_FRAME)

        x_init = InitialGuessList()
        x_init.add("q", initial_guess=self.q_exp_ocp, interpolation=InterpolationType.EACH_FRAME)
        x_init.add("qdot", initial_guess=self.qdot_exp_ocp, interpolation=InterpolationType.EACH_FRAME)

        u_bounds = BoundsList()
        # TODO: Charbie -> Change for maximal tau during the trial to simulate limited force
        u_bounds.add(
            "tau", min_bound=[-500] * nb_q, max_bound=[500] * nb_q, interpolation=InterpolationType.CONSTANT
        )
        if not with_contact:
            u_bounds.add(
                "contact_positions", min_bound=[0, 0, -0.0001], max_bound=[1.5, 1.5, 0.0001], interpolation=InterpolationType.CONSTANT
            )
            u_bounds.add(
                "contact_forces", min_bound=[-300] * 3, max_bound=[300] * 3, interpolation=InterpolationType.CONSTANT
            )

        u_init = InitialGuessList()
        u_init.add("tau", initial_guess=self.tau_exp_ocp[:, :-1], interpolation=InterpolationType.EACH_FRAME)
        if not with_contact:
            u_init.add("contact_positions", initial_guess=self.f_ext_exp_ocp["left_leg"][:3, :-1], interpolation=InterpolationType.EACH_FRAME)

        # TODO: Charbie -> Add phase transition when I have the full cycle
        # phase_transitions = PhaseTransitionList()
        # phase_transitions.add(PhaseTransitionFcn.CYCLIC, phase_pre_idx=0)

        self.ocp = OptimalControlProgram(
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
            # ode_solver=OdeSolver.COLLOCATION(polynomial_degree=3),
            use_sx=False,
            n_threads=10,
        )

    def solve(self, show_online_optim: bool = False):
        from bioptim import SolutionMerge, TimeAlignment, Solver
        solver = Solver.IPOPT(show_online_optim=show_online_optim)
        solver.set_tol(1e-3)  # TODO: Charbie -> Change for a more appropriate value (just to see for now)
        self.solution = self.ocp.solve(solver=solver)
        self.time_opt = self.solution.decision_time(to_merge=SolutionMerge.NODES, time_alignment=TimeAlignment.STATES)
        self.q_opt = self.solution.decision_states(to_merge=SolutionMerge.NODES)["q"]
        self.qdot_opt = self.solution.decision_states(to_merge=SolutionMerge.NODES)["qdot"]
        self.tau_opt = self.solution.decision_controls(to_merge=SolutionMerge.NODES)["tau"]
        self.opt_status = "CVG" if self.solution.status == 0 else "DVG"


    def animate_solution(self):

        try:
            from pyorerun import BiorbdModel, PhaseRerun
        except:
            raise RuntimeError("To animate the optimal solution, you must install Pyorerun.")

        # Add the model
        model = BiorbdModel(self.model_ocp)
        model.options.transparent_mesh = False
        viz = PhaseRerun(np.linspace(0, self.phase_time, self.n_shooting+1))

        # Add experimental markers
        markers = Markers(data=self.markers_exp_ocp, channels=list(model.marker_names))

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
        viz.rerun("OCP optimal solution")


    def extract_muscle_forces(self):
        # TODO: Charbie -> Extract muscle forces from the solution
        self.muscle_forces = None

    def get_result_file_full_path(self, result_folder=None):
        if result_folder is None:
            result_folder = self.experimental_data.result_folder
        trial_name = self.experimental_data.c3d_file_name.split("/")[-1][:-4]
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
            "biorbd_model_path": self.biorbd_model_path,
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
        }