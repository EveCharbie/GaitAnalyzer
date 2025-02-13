from .analysis_performer import AnalysisPerformer
from .model_creator import ModelCreator, OsimModels
from .experimental_data import ExperimentalData
from .helper import helper
from .kinematics_reconstructor import KinematicsReconstructor
from .operator import Operator
from .optimal_estimator import OptimalEstimator
from .plots.plot_leg_joint_angles import PlotLegData, LegToPlot, PlotType
from .result_manager import ResultManager
from .subject import Subject, Side

# Check if there are models and data where they should be
import os

if not os.path.exists("../data"):
    os.makedirs("../data")
    full_path = os.path.abspath("../data")
    raise FileNotFoundError(
        f"I have created the data folder for you here: {full_path}. " f"Please put your c3d files to analyze in there."
    )
if not os.path.exists("../models"):
    os.makedirs("../models")
    os.makedirs("../models/biorbd_models")
    os.makedirs("../models/biorbd_models/Geometry")
    os.makedirs("../models/OpenSim_models")
    full_path = os.path.abspath("../models")
    osim_full_path = os.path.abspath("../models/OpenSim_models")
    geometry_full_path = os.path.abspath("../models/biorbd_models/Geometry")
    raise FileNotFoundError(
        f"I have created the model folders for you here: {full_path}. "
        f"Please put your OpenSim model scaled to the subjects' anthropometry in {osim_full_path} and"
        f"the vtp files from OpenSim in here {geometry_full_path}."
    )
