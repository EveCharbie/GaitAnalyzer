from .analysis_performer import AnalysisPerformer
from .biomechanics_quantities.angular_momentum_calculator import AngularMomentumCalculator
from .model_creator import ModelCreator, OsimModels
from .experimental_data import ExperimentalData
from .events.cyclic_events import CyclicEvents
from .events.unique_events import UniqueEvents
from .helper import helper
from .kinematics_reconstructor import KinematicsReconstructor, ReconstructionType
from .operator import Operator
from .optimal_estimator import OptimalEstimator
from .plots.plot_leg_joint_angles import PlotLegData, LegToPlot, PlotType, EventIndexType
from .utils.marker_labeling_handler import MarkerLabelingHandler
from .result_manager import ResultManager
from .subject import Subject, Side

# Check if there are models and data where they should be
import os

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(parent_path + "/data"):
    os.makedirs(parent_path + "/data")
    full_path = os.path.abspath(parent_path + "/data")
    raise FileNotFoundError(
        f"I have created the data folder for you here: {full_path}. " f"Please put your c3d files to analyze in there."
    )
if not os.path.exists(parent_path + "/models"):
    os.makedirs(parent_path + "/models")
    os.makedirs(parent_path + "/models/biorbd_models")
    os.makedirs(parent_path + "/models/biorbd_models/Geometry")
    os.makedirs(parent_path + "/models/OpenSim_models")
    full_path = os.path.abspath(parent_path + "/models")
    osim_full_path = os.path.abspath(parent_path + "/models/OpenSim_models")
    geometry_full_path = os.path.abspath(parent_path + "/models/biorbd_models/Geometry")
    raise FileNotFoundError(
        f"I have created the model folders for you here: {full_path}. "
        f"Please put your OpenSim model scaled to the subjects' anthropometry in {osim_full_path} and"
        f"the vtp files from OpenSim in here {geometry_full_path}."
    )
