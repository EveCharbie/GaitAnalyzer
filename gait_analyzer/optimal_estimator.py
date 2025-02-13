import pickle
import numpy as np

try:
    import bioptim
except ImportError:
    print("Skipped Bioptim import as it is not installed")

from gait_analyzer import Operator
from gait_analyzer.experimental_data import ExperimentalData
from gait_analyzer.events import Events


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
        biorbd_model_path: str,
        experimental_data: ExperimentalData,
        events: Events,
        q_filtered: np.ndarray,
        qdot: np.ndarray,
        tau: np.ndarray,
        plot_solution_flag: bool = False,
        animate_solution_flag: bool = False,
    ):
        """
        Initialize the OptimalEstimator.
        .
        Parameters
        ----------
        cycle_to_analyze: int
            The number of the cycle to analyze.
            TODO: Charbie -> Maybe we should chose the most representative cycle instead of picking one?
        biorbd_model_path: str
            The full path to the biorbd model.
        experimental_data: ExperimentalData
            The experimental data to match.
        events: Events
            The events of the gait cycle to split the trial into appropriate phases.
        q_filtered: np.ndarray
            The filtered joint angles.
        qdot: np.ndarray
            The joint velocities (finite difference).
        tau: np.ndarray
            The joint torques (inverse dynamics).
        plot_solution_flag: bool
            If True, the solution will be plotted.
        animate_solution_flag: bool
            If True, the solution will be animated.
        """

        # Checks
        if not isinstance(cycle_to_analyze, int):
            raise ValueError("cycle_to_analyze must be an int")
        if not isinstance(biorbd_model_path, str):
            raise ValueError("biorbd_model_path must be a string")
        if not isinstance(experimental_data, ExperimentalData):
            raise ValueError("experimental_data must be an ExperimentalData")
        if not isinstance(events, Events):
            raise ValueError("events must be an Events")
        if not isinstance(q_filtered, np.ndarray):
            raise ValueError("q_filtered must be a np.ndarray")
        if not isinstance(qdot, np.ndarray):
            raise ValueError("qdot must be a np.ndarray")
        if not isinstance(tau, np.ndarray):
            raise ValueError("tau must be a np.ndarray")

        # Initial attributes
        self.cycle_to_analyze = cycle_to_analyze
        self.biorbd_model_path = biorbd_model_path
        self.experimental_data = experimental_data
        self.events = events
        self.q_filtered = q_filtered
        self.qdot = qdot
        self.tau = tau

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
        self.prepare_reduced_experimental_data()
        self.prepare_ocp()
        self.solve()
        if plot_solution_flag:
            self.solution.graphs(show_bounds=True, save_name=self.get_result_file_full_path(self.experimental_data.result_folder + "/figures"))
        if animate_solution_flag:
            self.solution.animate(n_frames=0, viewer="pyorerun", show_now=True)
        self.extract_muscle_forces()

    def generate_contact_biomods(self):
        """
        Create other bioMod files with the addition of the different feet contact conditions.
        """

        def add_txt_per_condition(condition: str) -> str:
            # TODO: Charbie -> Until biorbd is fixed to read biomods, I will hard code the position of the contacts
            contact_text = "\n/*-------------- CONTACTS---------------\n*/\n"
            if "heelL" in condition:
                contact_text += f"contact\tLCAL\n"
                contact_text += f"\tparent\tcalcn_l\n"
                contact_text += f"\tposition\t-0.018184372684362127\t-0.036183919561541877\t0.010718604411614319\n"
                contact_text += f"\taxis\txyz\n"
                contact_text += "endcontact\n\n"
            if "toesL" in condition:
                contact_text += f"contact\tLMFH1\n"
                contact_text += f"\tparent\tcalcn_l\n"
                contact_text += f"\tposition\t0.19202791077724868\t-0.013754853217574914\t0.039283237127771042\n"
                contact_text += f"\taxis\tz\n"
                contact_text += "endcontact\n\n"

                contact_text += f"contact\tLMFH5\n"
                contact_text += f"\tparent\tcalcn_l\n"
                contact_text += f"\tposition\t0.18583815793306013\t-0.0092170000425693677\t-0.072430596752376397\n"
                contact_text += f"\taxis\tz\n"
                contact_text += "endcontact\n\n"
            if "heelR" in condition:
                contact_text += f"contact\tRCAL\n"
                contact_text += f"\tparent\tcalcn_r\n"
                contact_text += f"\tposition\t-0.017776522017632024\t-0.030271301561674208\t-0.015068364463032391\n"
                contact_text += f"\taxis\txyz\n"
                contact_text += "endcontact\n\n"
            if "toesR" in condition:
                contact_text += f"contact\tRMFH1\n"
                contact_text += f"\tparent\tcalcn_r\n"
                contact_text += f"\tposition\t0.20126587479704638\t-0.0099656486276807066\t-0.039248701869426805\n"
                contact_text += f"\taxis\tz\n"
                contact_text += "endcontact\n\n"

                contact_text += f"contact\tRMFH5\n"
                contact_text += f"\tparent\tcalcn_r\n"
                contact_text += f"\tposition\t0.18449626841163846\t-0.018897872323952902\t0.07033570386440513\n"
                contact_text += f"\taxis\tz\n"
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
        ]
        for condition in conditions:
            new_model_path = original_model_path.replace(".bioMod", f"_{condition}.bioMod")
            with open(original_model_path, "r+", encoding="utf-8") as file:
                lines = file.readlines()
            with open(new_model_path, "w+", encoding="utf-8") as file:
                for line in lines:
                    file.write(line)
                file.write(add_txt_per_condition(condition))

    def prepare_reduced_experimental_data(self):
        """
        To reduce the optimization time, only one cycle is treated at a time
        (and the number of degrees of freedom is reduced?).
        """
        self.model_ocp = self.biorbd_model_path.replace(".bioMod", "_heelL_toesL.bioMod")

        # Only the 10th right leg swing (while left leg in flat foot)
        swing_timings = np.where(self.events.phases["heelL_toesL"])[0]
        right_swing_sequence = np.array_split(swing_timings, np.flatnonzero(np.diff(swing_timings) > 1) + 1)
        this_sequence_analogs = right_swing_sequence[self.cycle_to_analyze]
        this_sequence_markers = Operator.from_analog_frame_to_marker_frame(analogs_time_vector=self.experimental_data.analogs_time_vector,
                                                                            markers_time_vector=self.experimental_data.markers_time_vector,
                                                                            analog_idx=this_sequence_analogs)
        marker_idx_range = range(this_sequence_markers[0], this_sequence_markers[-1])
        self.n_shooting = len(list(marker_idx_range)) - 1
        self.q_exp_ocp = self.q_filtered[:, list(marker_idx_range)]
        self.qdot_exp_ocp = self.qdot[:, marker_idx_range]
        self.tau_exp_ocp = self.tau[:, marker_idx_range]
        self.f_ext_exp_ocp = {
            "left_leg": np.zeros((9, self.n_shooting+1)),
            "right_leg": np.zeros((9, self.n_shooting+1)),
        }
        for i_frame, marker_frame in enumerate(marker_idx_range):
            idx_analogs = Operator.from_marker_frame_to_analog_frame(analogs_time_vector=self.experimental_data.analogs_time_vector,
                                                                    markers_time_vector=self.experimental_data.markers_time_vector,
                                                                    marker_idx=marker_frame)
            self.f_ext_exp_ocp["left_leg"][:, i_frame] = np.mean(
                self.experimental_data.f_ext_sorted[0, :, idx_analogs-5:idx_analogs+5], axis=1
            )
            self.f_ext_exp_ocp["right_leg"][:, i_frame] = np.mean(
                self.experimental_data.f_ext_sorted[1, :, idx_analogs-5:idx_analogs+5], axis=1
            )
        self.markers_exp_ocp = self.experimental_data.markers_sorted_with_virtual[:, :, list(marker_idx_range)]
        self.phase_time = self.n_shooting * self.experimental_data.markers_dt


    def prepare_ocp(self):
        """
        Let's say swing phase only for now
        """
        try:
            bio_model = bioptim.BiorbdModel(self.model_ocp)
        except:
            raise RuntimeError("To reconstruct optimally, you must install Bioptim.")

        print(f"Preparing optimal control problem...")

        nb_q = bio_model.nb_q
        nb_root = 6
        nb_tau = nb_q - nb_root

        # Declaration of the objectives
        objective_functions = bioptim.ObjectiveList()
        objective_functions.add(
            objective=bioptim.ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            key="tau",
            weight=1.0,
        )
        objective_functions.add(objective=bioptim.ObjectiveFcn.Lagrange.TRACK_MARKERS, weight=10.0, node=bioptim.Node.ALL, target=self.markers_exp_ocp)
        objective_functions.add(objective=bioptim.ObjectiveFcn.Lagrange.TRACK_STATE, key="q", weight=0.1, node=bioptim.Node.ALL, target=self.q_exp_ocp)
        objective_functions.add(
            objective=bioptim.ObjectiveFcn.Lagrange.TRACK_STATE, key="qdot", node=bioptim.Node.ALL, weight=0.01, target=self.qdot_exp_ocp
        )
        objective_functions.add(
            objective=bioptim.ObjectiveFcn.Lagrange.TRACK_GROUND_REACTION_FORCES,
            weight=0.01,
            target=self.f_ext_exp_ocp["left_leg"][6:9, :-1],
            contact_index=[0, 1, 2, 3, 4],
        )
        # TODO: Charbie -> track CoP ?
        # TODO: Charbie -> constrain the initial ground contact velocity to be the preferential treadmill speed ?

        constraints = bioptim.ConstraintList()
        constraints.add(
            bioptim.ConstraintFcn.TRACK_CONTACT_FORCES,  # Only pushing on the floor, no pulling (Z heel)
            min_bound=0,
            max_bound=np.inf,
            node=bioptim.Node.ALL_SHOOTING,
            contact_index=2,
        )
        constraints.add(
            bioptim.ConstraintFcn.TRACK_CONTACT_FORCES,  # Only pushing on the floor, no pulling (Z LMFH1)
            min_bound=0,
            max_bound=np.inf,
            node=bioptim.Node.ALL_SHOOTING,
            contact_index=3,
        )
        constraints.add(
            bioptim.ConstraintFcn.TRACK_CONTACT_FORCES,  # Only pushing on the floor, no pulling (Z LMFH5)
            min_bound=0,
            max_bound=np.inf,
            node=bioptim.Node.ALL_SHOOTING,
            contact_index=4,
        )

        dynamics = bioptim.DynamicsList()  # TODO: Charbie -> Change for muscles
        dynamics.add(bioptim.DynamicsFcn.TORQUE_DRIVEN, with_contact=True)

        dof_mappings = bioptim.BiMappingList()
        dof_mappings.add(
            "tau", to_second=[None] * nb_root + list(range(nb_tau)), to_first=list(range(nb_root, nb_tau + nb_root))
        )

        # TODO: Charbie
        x_bounds = bioptim.BoundsList()
        # TODO: Charbie -> Change for maximal range of motion during the trial to simulate limited RoM
        # TODO: Charbie -> Change for maximal velocity during the trial to simulate limited Power
        x_bounds["q"] = bio_model.bounds_from_ranges("q")
        x_bounds["qdot"] = bio_model.bounds_from_ranges("q")

        x_init = bioptim.InitialGuessList()
        x_init.add("q", initial_guess=self.q_exp_ocp, interpolation=bioptim.InterpolationType.EACH_FRAME)
        x_init.add("qdot", initial_guess=self.qdot_exp_ocp, interpolation=bioptim.InterpolationType.EACH_FRAME)

        u_bounds = bioptim.BoundsList()
        # TODO: Charbie -> Change for maximal tau during the trial to simulate limited force
        u_bounds.add(
            "tau", min_bound=[-1000] * nb_tau, max_bound=[1000] * nb_tau, interpolation=bioptim.InterpolationType.CONSTANT
        )

        u_init = bioptim.InitialGuessList()
        u_init.add("tau", initial_guess=self.tau_exp_ocp[6:, :-1], interpolation=bioptim.InterpolationType.EACH_FRAME)

        # TODO: Charbie -> Add phase transition when I have the full cycle
        # phase_transitions = PhaseTransitionList()
        # phase_transitions.add(PhaseTransitionFcn.CYCLIC, phase_pre_idx=0)

        self.ocp = bioptim.OptimalControlProgram(
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
            variable_mappings=dof_mappings,
            use_sx=False,
        )

    def solve(self, show_online_optim: bool = False):
        solver = bioptim.Solver.IPOPT(show_online_optim=show_online_optim)
        self.solution = self.ocp.solve(solver=solver)

    def extract_muscle_forces(self):
        # TODO: Charbie -> Extract muscle forces from the solution
        self.muscle_forces = None

    def get_result_file_full_path(self, result_folder=None):
        if result_folder is None:
            result_folder = self.experimental_data.result_folder
        trial_name = self.experimental_data.c3d_file_name.split("/")[-1][:-4]
        result_file_full_path = f"{result_folder}/optim_estim_{trial_name}.pkl"
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
            "q_filtered": self.q_filtered,
            "qdot": self.qdot,
            "tau": self.tau,
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