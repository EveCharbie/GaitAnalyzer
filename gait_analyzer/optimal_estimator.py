import numpy as np

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
        biorbd_model_path: str,
        experimental_data: ExperimentalData,
        events: Events,
        q_filtered: np.ndarray,
        qdot: np.ndarray,
        tau: np.ndarray,
    ):
        """
        Initialize the OptimalEstimator.
        .
        Parameters
        ----------
        biorbd_model_path: str
            The full path to the biorbd model.
        """
        try:
            from bioptim import (
                BiorbdModel,
                OptimalControlProgram,
                DynamicsList,
                DynamicsFcn,
                BoundsList,
                InitialGuessList,
                ObjectiveList,
                ObjectiveFcn,
                InterpolationType,
                Node,
                Solver,
                ConstraintList,
                ConstraintFcn,
                PhaseTransitionList,
                PhaseTransitionFcn,
                BiMappingList,
            )
        except:
            raise RuntimeError("To animate the kinematics, you must install Bioptim.")

        # Checks
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
        self.biorbd_model_path = biorbd_model_path
        self.experimental_data = experimental_data
        self.events = events
        self.q_filtered = q_filtered
        self.qdot = qdot
        self.tau = tau

        # Extended attributes
        self.ocp = None
        self.solution = None
        self.muscle_forces = None

        # Perform the optimal estimation
        self.model_ocp = None
        self.q_exp_ocp = None
        self.qdot_exp_ocp = None
        self.f_ext_exp_ocp = None
        self.markers_exp_ocp = None
        self.emg_exp_ocp = None
        self.n_shooting = None
        self.phase_time = None
        self.generate_contact_biomods()
        self.prepare_reduced_experimental_data()
        self.prepare_ocp()
        self.solve()
        self.extract_muscle_forces()

    def generate_contact_biomods(self):
        """
        Create other bioMod files with the addition of the different feet contact conditions.
        """

        def add_txt_per_condition(condition: str) -> str:
            # TODO: Charbie -> Until biorbd is fixed to read biomods, I will hard code the position of the contacts
            contact_text = "\n/*-------------- CONTACTS---------------\n*/\n"
            if "heelL" in condition:
                contact_text = f"contact\tLCAL\n"
                contact_text += f"\tparent\tcalcn_l\n"
                contact_text += f"\tposition\t-0.018184372684362127\t-0.036183919561541877\t0.010718604411614319\n"
                contact_text += f"\taxis\txyz\n"
                contact_text += "endcontact\n\n"
            if "toesL" in condition:
                contact_text = f"contact\tLMFH1\n"
                contact_text += f"\tparent\tcalcn_l\n"
                contact_text += f"\tposition\t0.19202791077724868\t-0.013754853217574914\t0.039283237127771042\n"
                contact_text += f"\taxis\tz\n"
                contact_text += "endcontact\n\n"

                contact_text = f"contact\tLMFH5\n"
                contact_text += f"\tparent\tcalcn_l\n"
                contact_text += f"\tposition\t0.18583815793306013\t-0.0092170000425693677\t-0.072430596752376397\n"
                contact_text += f"\taxis\tz\n"
                contact_text += "endcontact\n\n"
            if "heelR" in condition:
                contact_text = f"contact\tRCAL\n"
                contact_text += f"\tparent\tcalcn_r\n"
                contact_text += f"\tposition\t-0.017776522017632024\t-0.030271301561674208\t-0.015068364463032391\n"
                contact_text += f"\taxis\txyz\n"
                contact_text += "endcontact\n\n"
            if "toesR" in condition:
                contact_text = f"contact\tRMFH1\n"
                contact_text += f"\tparent\tcalcn_r\n"
                contact_text += f"\tposition\t0.20126587479704638\t-0.0099656486276807066\t-0.039248701869426805\n"
                contact_text += f"\taxis\tz\n"
                contact_text += "endcontact\n\n"

                contact_text = f"contact\tRMFH5\n"
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
        # Temporarily I will try with everything!
        self.model_ocp = self.biorbd_model_path.replace(".bioMod", "_heelL_toesL.bioMod")

        # Only the 10th right leg swing (while left leg in flat foot)
        swing_timings = np.where(self.phases["heelL_toesL"])[0]
        right_swing_sequence = np.array_split(swing_timings, np.flatnonzero(np.diff(swing_timings) > 1) + 1)
        this_sequence_analogs = right_swing_sequence[9]
        this_sequence_markers = np.arange(
            int(this_sequence_analogs[0] * self.experimental_data.analogs_dt / self.experimental_data.markers_dt),
            int(this_sequence_analogs[-1] * self.experimental_data.analogs_dt / self.experimental_data.markers_dt),
        )
        self.q_exp_ocp = self.q[:, this_sequence_markers]
        self.qdot_exp_ocp = self.qdot[:, this_sequence_markers]
        self.f_ext_exp_ocp = {
            "left_leg": np.zeros((6, self.q_exp_ocp.shape[1])),
            "right_leg": np.zeros((6, self.q_exp_ocp.shape[1])),
        }
        for i_node in range(self.q_exp_ocp.shape[1]):
            # TODO: Guys/Charbie -> This method allows for rounding errors, there should be a mapping of frames instead
            idx_beginning = (
                int(
                    this_sequence_markers[i_node]
                    * self.experimental_data.markers_dt
                    / self.experimental_data.analogs_dt
                )
                - 5
            )
            idx_end = (
                int(
                    this_sequence_markers[i_node]
                    * self.experimental_data.markers_dt
                    / self.experimental_data.analogs_dt
                )
                + 5
            )
            self.f_ext_exp_ocp["left_leg"][:, i_node] = np.mean(
                self.experimental_data.f_ext_sorted[0, 3:, idx_beginning:idx_end], axis=1
            )
            self.f_ext_exp_ocp["right_leg"][:, i_node] = np.mean(
                self.experimental_data.f_ext_sorted[1, 3:, idx_beginning:idx_end], axis=1
            )
        self.markers_exp_ocp = self.experimental_data.markers_sorted[:, :, this_sequence_markers]

        self.n_shooting = self.q_exp_ocp.shape[1]
        self.phase_time = self.n_shooting * self.experimental_data.markers_dt

    def prepare_ocp(self):
        """
        Let's say swing phase only for now
        """

        bio_model = BiorbdModel(self.model_ocp)
        nb_q = bio_model.nb_q
        nb_root = 6
        nb_tau = nb_q - nb_root

        # Declaration of the objectives
        objective_functions = ObjectiveList()
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            key="tau_joints",
            weight=1.0,
        )
        objective_functions.add(objective=ObjectiveFcn.Lagrange.TRACK_MARKERS, weight=10.0, target=self.markers_exp_ocp)
        objective_functions.add(objective=ObjectiveFcn.Lagrange.TRACK_STATE, key="q", weight=0.1, target=self.q_exp_ocp)
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.TRACK_STATE, key="qdot", weight=0.01, target=self.qdot_exp_ocp
        )
        objective_functions.add(
            objective=ObjectiveFcn.Lagrange.TRACK_CONTACT_FORCES,
            weight=0.01,
            target=self.f_ext_exp_ocp,
        )

        constraints = ConstraintList()
        constraints.add(
            ConstraintFcn.TRACK_CONTACT_FORCES,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL_SHOOTING,
            contact_index=2,
        )
        constraints.add(
            ConstraintFcn.TRACK_CONTACT_FORCES,
            min_bound=0,
            max_bound=np.inf,
            node=Node.ALL_SHOOTING,
            contact_index=3,
        )

        dynamics = DynamicsList()  # Change for muscles
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)

        dof_mappings = BiMappingList()
        dof_mappings.add(
            "tau", to_second=[None] * nb_root + list(range(nb_tau)), to_first=list(range(nb_root, nb_tau + nb_root))
        )

        # TODO: Charbie
        x_bounds = BoundsList()
        x_initial_guesses = InitialGuessList()

        u_bounds = BoundsList()
        u_initial_guesses = InitialGuessList()
        u_bounds.add(
            "tau", min_bound=[-1000] * nb_tau, max_bound=[1000] * nb_tau, interpolation=InterpolationType.CONSTANT
        )

        phase_transitions = PhaseTransitionList()
        phase_transitions.add(PhaseTransitionFcn.CYCLIC, phase_pre_idx=0)

        self.ocp = OptimalControlProgram(
            bio_model=bio_model,
            n_shooting=self.n_shooting,
            phase_time=self.phase_time,
            dynamics=dynamics,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            x_init=x_initial_guesses,
            u_init=u_initial_guesses,
            objective_functions=objective_functions,
            constraints=constraints,
            phase_transitions=phase_transitions,
            variable_mappings=dof_mappings,
            use_sx=False,
        )

    def solve(self, show_online_optim: bool = False):
        solver = Solver.IPOPT(show_online_optim=show_online_optim)
        self.solution = self.ocp.solve(solver=solver)

    def extract_muscle_forces(self):
        # TODO: Charbie -> Extract muscle forces from the solution
        self.muscle_forces = None
