import os
from copy import deepcopy
import numpy as np
import biorbd

from biobuddy import (
    BiomechanicalModelReal,
    MuscleType,
    MuscleStateType,
    ScaleTool,
    RangeOfMotion,
    Ranges,
    C3dData,
    RotoTransMatrix,
    MarkerReal,
    JointCenterTool,
    Score,
    Sara,
)
from gait_analyzer.subject import Subject


class OsimModels:

    @property
    def osim_model_name(self):
        raise RuntimeError(
            "This method is implemented in the child class. You should call OsimModels.[mode type name].osim_model_name."
        )

    @property
    def original_osim_model_full_path(self):
        raise RuntimeError(
            "This method is implemented in the child class. You should call OsimModels.[mode type name].original_osim_model_full_path."
        )

    @property
    def xml_setup_file(self):
        raise RuntimeError(
            "This method is implemented in the child class. You should call OsimModels.[mode type name].xml_setup_file."
        )

    @property
    def muscles_to_ignore(self):
        raise RuntimeError(
            "This method is implemented in the child class. You should call OsimModels.[mode type name].muscles_to_ignore."
        )

    @property
    def markers_to_ignore(self):
        raise RuntimeError(
            "This method is implemented in the child class. You should call OsimModels.[mode type name].markers_to_ignore."
        )

    # Child classes acting as an enum
    class WholeBody:
        """This is a hole body model that consists of 23 bodies, 42 degrees of freedom and 30 muscles.
        The whole-body geometric model and the lower limbs, pelvis and upper limbs anthropometry are based on the running model of Hammer et al. 2010 which consists of 12 segments and 29 degrees of freedom.
        Extra segments and degrees of freedom were later added based on Dumas et al. 2007.
        Each lower extremity had seven degrees-of-freedom; the hip was modeled as a ball-and-socket joint (3 DoFs), the knee was modeled as a revolute joint with 1 dof, the ankle was modeled as 2 revolute joints and feet toes with one revolute joint.
        The pelvis joint was model as a free flyer joint (6 DoFs) to allow the model to translate and rotate in the 3D space, the lumbar motion was modeled as a ball-and-socket joint (3 DoFs) (Anderson and Pandy, 1999) and the neck joint was modeled as a ball-and-socket joint (3 DoFs).
        Mass properties of the torso and head (including the neck) segments were estimated from Dumas et al., 2007. Each arm consisted of 8 degrees-of-freedom; the shoulder was modeled as a ball-and-socket joint (3 DoFs), the elbow and forearm rotation were each modeled with revolute joints (1 dof) (Holzbaur et al., 2005), the wrist flexion and deviation were each modeled with revolute joints and the hand fingers were modeled with 1 revolute joint.
        Mass properties for the arms were estimated from 1 and de Leva, 1996. The model also include 30 superficial muscles of the whole body.
        [Charbie -> Link ?]
        """

        @property
        def osim_model_name(self):
            return "wholebody"

        @property
        def original_osim_model_full_path(self):
            return "../models/OpenSim_models/wholebody.osim"

        @property
        def xml_setup_file(self):
            return "../models/OpenSim_models/wholebody.xml"

        @property
        def muscles_to_ignore(self):
            return [
                "ant_delt_r",
                "ant_delt_l",
                "medial_delt_l",
                "post_delt_r",
                "post_delt_l",
                "medial_delt_r",
                "ercspn_r",
                "ercspn_l",
                "rect_abd_r",
                "rect_abd_l",
                "r_stern_mast",
                "l_stern_mast",
                "r_trap_acr",
                "l_trap_acr",
                "TRIlong",
                "TRIlong_l",
                "TRIlat",
                "TRIlat_l",
                "BIClong",
                "BIClong_l",
                "BRD",
                "BRD_l",
                "FCR",
                "FCR_l",
                "ECRL",
                "ECRL_l",
                "PT",
                "PT_l",
                "LAT2",
                "LAT2_l",
                "PECM2",
                "PECM2_l",
            ] + [
                "glut_med1_r",
                "semiten_r",
                "bifemlh_r",
                "sar_r",
                "tfl_r",
                "vas_med_r",
                "vas_lat_r",
                "glut_med1_l",
                "semiten_l",
                "bifemlh_l",
                "sar_l",
                "tfl_l",
                "vas_med_l",
                "vas_lat_l",
            ]

        @property
        def markers_to_ignore(self):
            return []

        @property
        def ranges_to_adjust(self):
            return {
                "pelvis_translation": [
                    [-3, 3],
                    [-3, 3],
                    [-3, 3],
                ],
                "pelvis_rotation_transform": [
                    [-np.pi / 4, np.pi / 4],
                    [-np.pi / 4, np.pi / 4],
                    [-np.pi, np.pi],
                ],
                "femur_r_rotation_transform": [
                    [-40 * np.pi / 180, 120 * np.pi / 180],
                    [-60 * np.pi / 180, 30 * np.pi / 180],
                    [-30 * np.pi / 180, 30 * np.pi / 180],
                ],
                "tibia_r_rotation_transform": [
                    [-150 * np.pi / 180, 0.0],
                ],
                "talus_r_ankle_angle_r": [
                    [-50 * np.pi / 180, 30 * np.pi / 180],  # Ankle Flexion
                ],
                "calcn_r_subtalar_angle_r": [
                    [-15 * np.pi / 180, 15 * np.pi / 180],  # Ankle Inversion
                ],
                "toes_r_rotation_transform": [
                    [-50 * np.pi / 180, 60 * np.pi / 180],  # Toes Flexion
                ],
                "femur_l_rotation_transform": [
                    [-40 * np.pi / 180, 120 * np.pi / 180],
                    [-60 * np.pi / 180, 30 * np.pi / 180],
                    [-30 * np.pi / 180, 30 * np.pi / 180],
                ],
                "tibia_l_rotation_transform": [
                    [0.0, 150 * np.pi / 180],
                ],
                "talus_l_ankle_angle_l": [
                    [-50 * np.pi / 180, 30 * np.pi / 180],  # Ankle Flexion
                ],
                "calcn_l_subtalar_angle_l": [
                    [-15 * np.pi / 180, 15 * np.pi / 180],  # Ankle Inversion
                ],
                "toes_l_rotation_transform": [
                    [-50 * np.pi / 180, 60 * np.pi / 180],  # Toes Flexion
                ],
                "torso_rotation_transform": [
                    [-90 * np.pi / 180, 45 * np.pi / 180],
                    [-35 * np.pi / 180, 35 * np.pi / 180],
                    [-45 * np.pi / 180, 45 * np.pi / 180],
                ],
                "head_neck_rotation_transform": [[-50 * np.pi / 180, 45 * np.pi / 180], [-0.6, 0.6], [-1.2217, 1.2217]],
                "humerus_r_rotation_transform": [
                    [-np.pi / 2, np.pi],
                    [-3.8397, np.pi / 2],
                    [-np.pi / 2, np.pi / 2],
                ],
                "ulna_r_elbow_flex_r": [
                    [0.0, np.pi],
                ],
                "radius_r_pro_sup_r": [
                    [-np.pi, np.pi],
                ],
                "lunate_r_rotation_transform": [
                    [-np.pi / 2, np.pi / 2],
                ],
                "hand_r_rotation_transform": [
                    [-0.43633231, 0.61086524],
                ],
                "fingers_r_rotation_transform": [
                    [-np.pi / 2, np.pi / 2],
                ],
                "humerus_l_rotation_transform": [
                    [-np.pi / 2, np.pi],
                    [-3.8397, np.pi / 2],
                    [-np.pi / 2, np.pi / 2],
                ],
                "ulna_l_elbow_flex_l": [
                    [0.0, np.pi],
                ],
                "radius_l_pro_sup_l": [
                    [-np.pi, np.pi],
                ],
                "lunate_l_rotation_transform": [
                    [-np.pi / 2, np.pi / 2],
                ],
                "hand_l_rotation_transform": [
                    [-0.43633231, 0.61086524],
                ],
                "fingers_l_rotation_transform": [
                    [-np.pi / 2, np.pi / 2],
                ],
            }

        @property
        def markers_to_add(self):
            return {
                "femur_r": ["RTHI1", "RTHI2", "RTHI3"],
                "femur_l": ["LTHI1", "LTHI2", "LTHI3"],
                "tibia_r": ["RLEG1", "RLEG2", "RLEG3"],
                "tibia_l": ["LLEG1", "LLEG2", "LLEG3"],
                "humerus_r": ["RAMR1", "RARM2", "RARM3"],
                "radius_r": ["RFARM1", "RFARM2", "RFARM3"],
                "humerus_l": ["LARM1", "LARM2", "LARM3"],
                "radius_l": ["LFARM1", "LFARM2", "LFARM3"],
            }

        def perform_modifications(self, model, static_trial):
            """
            1. Remove the markers that are not needed (markers_to_ignore)
            2. Change the ranges of motion for the segments (ranges_to_adjust)
            3. Remove the muscles/via_points/muscle_groups that are not needed (muscles_to_ignore)
            4. Add the marker clusters (markers_to_add)
            """

            # Modify segments
            for segment in model.segments:
                markers = deepcopy(segment.markers)
                for marker in markers:
                    if marker in self.markers_to_ignore:
                        segment.remove_marker(marker)
                if segment.name in self.ranges_to_adjust.keys():
                    min_bounds = [r[0] for r in self.ranges_to_adjust[segment.name]]
                    max_bounds = [r[1] for r in self.ranges_to_adjust[segment.name]]
                    segment.q_ranges = RangeOfMotion(Ranges.Q, min_bounds, max_bounds)

            # Modify muscles
            via_points = deepcopy(model.via_points)
            for via_point in via_points:
                if via_point.muscle_name in self.muscles_to_ignore:
                    model.remove_via_point(via_point.name)

            muscle_groups = deepcopy(model.muscle_groups)
            for muscle_group in muscle_groups:
                if muscle_group.origin_parent_name in self.muscles_to_ignore:
                    model.remove_muscle_group(muscle_group.name)
                elif muscle_group.insertion_parent_name in self.muscles_to_ignore:
                    model.remove_muscle_group(muscle_group.name)

            for muscle in self.muscles_to_ignore:
                model.remove_muscle(muscle)

            # Add the marker clusters
            jcs_in_global = model.forward_kinematics()
            c3d_data = C3dData(static_trial, first_frame=100, last_frame=200)
            for segment_name in self.markers_to_add.keys():
                for marker in self.markers_to_add[segment_name]:
                    position_in_global = c3d_data.mean_marker_position(marker)
                    rt = RotoTransMatrix()
                    rt.from_rt_matrix(jcs_in_global[segment_name])
                    position_in_local = rt.inverse @ position_in_global
                    model.segments[segment_name].add_marker(
                        MarkerReal(
                            name=marker,
                            parent_name=segment_name,
                            position=position_in_local,
                            is_anatomical=False,
                            is_technical=True,
                        )
                    )

            return model


class ModelCreator:
    def __init__(
        self,
        subject: Subject,
        static_trial: str,
        functional_trials_path: str,
        models_result_folder: str,
        osim_model_type,
        skip_if_existing: bool,
        animate_model_flag: bool,
    ):

        # Checks
        if not isinstance(subject, Subject):
            raise ValueError("subject must be a Subject.")
        if not isinstance(static_trial, str):
            raise ValueError("static_trial must be a string.")
        # if not isinstance(functional_trials_path, str):
        #     raise ValueError("functional_trials_path must be a string.")
        # if not os.path.exists(functional_trials_path):
        #     raise RuntimeError(f"Functional trials path {functional_trials_path} does not exist.")
        if not isinstance(models_result_folder, str):
            raise ValueError("models_result_folder must be a string.")
        if not isinstance(skip_if_existing, bool):
            raise ValueError("skip_if_existing must be a boolean.")
        if not isinstance(animate_model_flag, bool):
            raise ValueError("animate_model_flag must be a boolean.")

        # Initial attributes
        self.subject = subject
        self.osim_model_type = osim_model_type
        self.static_trial = static_trial
        self.functional_trials_path = functional_trials_path
        self.models_result_folder = models_result_folder

        # Extended attributes
        self.trc_file_path = None
        self.vtp_geometry_path = "../../Geometry_cleaned"
        self.biorbd_model_full_path = (
            self.models_result_folder
            + "/"
            + osim_model_type.osim_model_name
            + "_"
            + self.subject.subject_name
            + ".bioMod"
        )
        self.model = None  # This is the object that will be modified to be personalized to the subject
        self.new_model_created = False

        # Create the models
        if not (skip_if_existing and os.path.isfile(self.biorbd_model_full_path)):
            print(f"The model {self.biorbd_model_full_path} is being created...")
            self.read_osim_model()
            self.scale_model()
            self.osim_model_type.perform_modifications(self.model, self.static_trial)
            self.relocate_joint_centers_functionally()
            self.create_biorbd_model()
        else:
            print(f"The model {self.biorbd_model_full_path} already exists, so it is being used.")
        self.biorbd_model = biorbd.Model(self.biorbd_model_full_path)

        if animate_model_flag:
            self.animate_model()

    def read_osim_model(self):
        self.model = BiomechanicalModelReal.from_osim(
            filepath=self.osim_model_type.original_osim_model_full_path,
            muscle_type=MuscleType.HILL_DE_GROOTE,
            muscle_state_type=MuscleStateType.DEGROOTE,
            mesh_dir=self.vtp_geometry_path,
        )

    def scale_model(self):
        scale_tool = ScaleTool(original_model=self.model).from_xml(filepath=self.osim_model_type.xml_setup_file)
        self.model = scale_tool.scale(
            filepath=self.static_trial,
            first_frame=100,
            last_frame=200,
            mass=self.subject.subject_mass,
            q_regularization_weight=0.01,
            make_static_pose_the_models_zero=True,
            visualize_optimal_static_pose=False,
            method="lm",
        )

    def relocate_joint_centers_functionally(self):

        # Move the model's joint centers
        joint_center_tool = JointCenterTool(self.model, animate_reconstruction=True)

        # Hip Right
        joint_center_tool.add(
            Score(
                filepath=self.functional_trials_path + "right_hip.c3d",
                parent_name="pelvis",
                child_name="femur_r",
                parent_marker_names=["RASIS", "LASIS", "LPSIS", "RPSIS"],
                child_marker_names=["RLFE", "RMFE"] + self.osim_model_type.markers_to_add["femur_r"],
                initialize_whole_trial_reconstruction=False,
                animate_rt=False,
            )
        )
        joint_center_tool.add(
            Sara(
                filepath=self.functional_trials_path + "right_knee.c3d",
                parent_name="femur_r",
                child_name="tibia_r",
                parent_marker_names=["RGT"] + self.osim_model_type.markers_to_add["femur_r"],
                child_marker_names=["RATT", "RLM", "RSPH"] + self.osim_model_type.markers_to_add["tibia_r"],
                joint_center_markers=["RLFE", "RMFE"],
                distal_markers=["RLM", "RSPH"],
                is_longitudinal_axis_from_jcs_to_distal_markers=False,
                initialize_whole_trial_reconstruction=False,
                animate_rt=False,
            )
        )
        # Ankle right
        joint_center_tool.add(
            Score(
                filepath=self.functional_trials_path + "right_ankle.c3d",
                parent_name="tibia_r",
                child_name="calcn_r",
                parent_marker_names=["RATT", "RSPH", "RLM"] + self.osim_model_type.markers_to_add["tibia_r"],
                child_marker_names=["RCAL", "RMFH1", "RMFH5"],  # toes_r: "RTT2"
                initialize_whole_trial_reconstruction=False,
                animate_rt=False,
            )
        )
        # Hip Left
        joint_center_tool.add(
            Score(
                filepath=self.functional_trials_path + "left_hip.c3d",
                parent_name="pelvis",
                child_name="femur_l",
                parent_marker_names=["RASIS", "LASIS", "LPSIS", "RPSIS"],
                child_marker_names=["LGT", "LMFE", "LLFE"] + self.osim_model_type.markers_to_add["femur_l"],
                initialize_whole_trial_reconstruction=False,
                animate_rt=False,
            )
        )
        # Knee Left
        joint_center_tool.add(
            Score(
                filepath=self.functional_trials_path + "left_knee.c3d",
                parent_name="femur_l",
                child_name="tibia_l",
                parent_marker_names=["LGT"] + self.osim_model_type.markers_to_add["femur_l"],
                child_marker_names=["LATT", "LSPH", "LLM"] + self.osim_model_type.markers_to_add["tibia_l"],
                initialize_whole_trial_reconstruction=False,
                animate_rt=False,
            )
        )
        # Ankle Left
        joint_center_tool.add(
            Score(
                filepath=self.functional_trials_path + "left_ankle.c3d",
                parent_name="tibia_l",
                child_name="calcn_l",
                parent_marker_names=["LATT", "LSPH", "LLM"] + self.osim_model_type.markers_to_add["tibia_l"],
                child_marker_names=["LCAL", "LMFH1", "LMFH5"],  # toes_r: "LTT2"
                initialize_whole_trial_reconstruction=False,
                animate_rt=False,
            )
        )
        # Shoulder Right
        joint_center_tool.add(
            Score(
                filepath=self.functional_trials_path + "shoulders.c3d",
                parent_name="torso",
                child_name="humerus_r",
                parent_marker_names=["STR", "C7", "T10", "SUP"],
                child_marker_names=["RLHE", "RMHE"] + self.osim_model_type.markers_to_add["humerus_r"],
                initialize_whole_trial_reconstruction=False,
                animate_rt=False,
            )
        )
        # Elbow Right
        joint_center_tool.add(
            Score(
                filepath=self.functional_trials_path + "elbows.c3d",
                parent_name="humerus_r",
                child_name="radius_r",
                parent_marker_names=self.osim_model_type.markers_to_add["humerus_r"],
                child_marker_names=["RUS", "RRS"] + self.osim_model_type.markers_to_add["radius_r"],
                initialize_whole_trial_reconstruction=False,
                animate_rt=False,
            )
        )
        # Shoulder Left
        joint_center_tool.add(
            Score(
                filepath=self.functional_trials_path + "shoulders.c3d",
                parent_name="torso",
                child_name="humerus_l",
                parent_marker_names=["STR", "C7", "T10", "SUP"],
                child_marker_names=["LLHE", "LMHE"] + self.osim_model_type.markers_to_add["humerus_l"],
                initialize_whole_trial_reconstruction=False,
                animate_rt=False,
            )
        )
        # Elbow Left
        joint_center_tool.add(
            Score(
                filepath=self.functional_trials_path + "elbows.c3d",
                parent_name="humerus_l",
                child_name="radius_l",
                parent_marker_names=self.osim_model_type.markers_to_add["humerus_l"],
                child_marker_names=["LUS", "LRS"] + self.osim_model_type.markers_to_add["radius_l"],
                initialize_whole_trial_reconstruction=False,
                animate_rt=False,
            )
        )
        # Neck
        joint_center_tool.add(
            Score(
                filepath=self.functional_trials_path + "neck.c3d",
                parent_name="torso",
                child_name="head_and_neck",
                parent_marker_names=["STR", "RA", "LA", "C7", "T10", "SUP"],
                child_marker_names=["SEL", "OCC", "RTEMP", "LTEMP", "HV"],
                initialize_whole_trial_reconstruction=False,
                animate_rt=False,
            )
        )

        self.model = joint_center_tool.replace_joint_centers(self.osim_model_type.marker_weights)

    def create_biorbd_model(self):
        self.model.to_biomod(self.biorbd_model_full_path, with_mesh=True)
        self.new_model_created = True

    def animate_model(self):
        """
        Animate the model
        """
        try:
            from pyorerun import BiorbdModel, PhaseRerun
        except:
            raise RuntimeError("To animate the model, you must install Pyorerun.")

        # Model
        model = BiorbdModel(self.biorbd_model_full_path)
        model.options.transparent_mesh = False
        model.options.show_gravity = True

        # Visualization
        viz = PhaseRerun(np.linspace(0, 1, 10))
        viz.add_animated_model(model, np.zeros((model.nb_q, 10)))
        viz.rerun_by_frame("Kinematics reconstruction")

    def inputs(self):
        return {
            "subject_name": self.subject.subject_name,
            "subject_mass": self.subject.subject_mass,
            "osim_model_type": self.osim_model_type,
            "static_trial": self.static_trial,
        }

    def outputs(self):
        return {
            "biorbd_model_full_path": self.biorbd_model_full_path,
            "biorbd_model": self.biorbd_model,
            "new_model_created": self.new_model_created,
        }
