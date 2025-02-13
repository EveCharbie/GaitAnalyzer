import os
import pickle
from scipy.io import savemat
import git
from datetime import date
import subprocess
import json

from gait_analyzer.subject import Subject


class AnalysisPerformer:
    def __init__(
        self,
        analysis_to_perform: callable,
        subjects_to_analyze: list[Subject],
        cycles_to_analyze: range = range(5, -5),
        result_folder: str = "../results/",
        trails_to_analyze: list[str] = None,
        skip_if_existing: bool = False,
    ):
        """
        Initialize the AnalysisPerformer.
        .
        Parameters
        ----------
        analysis_to_perform: callable(subject_name: str, subject_mass: float, c3d_file_name: str)
            The analysis to perform
        subjects_to_analyze: list[Subject]
            The list of subjects to analyze
        cycles_to_analyze: range
            The range of cycles to analyze
        result_folder: str
            The folder where the results will be saved. It will look like result_folder/subject_name.
        trails_to_analyze: list[str]
            The list of trails to analyze. If None, all the trails will be analyzed.
        skip_if_existing: bool
            If True, the analysis will not be performed if the results already exist.
        """

        # Checks:
        if not callable(analysis_to_perform):
            raise ValueError("analysis_to_perform must be a callable")
        if not isinstance(subjects_to_analyze, list):
            raise ValueError("subjects_to_analyze must be a list of Subject")
        for subject in subjects_to_analyze:
            if not isinstance(subject, Subject):
                raise ValueError("All elements of subjects_to_analyze must be Subject")
        if not isinstance(cycles_to_analyze, range):
            raise ValueError("cycles_to_analyze must be a range of cycles to analyze")
        if not isinstance(result_folder, str):
            raise ValueError("result_folder must be a string")
        if not isinstance(trails_to_analyze, list) and trails_to_analyze is not None:
            raise ValueError("trails_to_analyze must be a list of strings")
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
            print(f"Result folder did not exist, I have created it here {os.path.abspath(result_folder)}")

        # Initial attributes
        self.analysis_to_perform = analysis_to_perform
        self.subjects_to_analyze = subjects_to_analyze
        self.cycles_to_analyze = cycles_to_analyze
        self.result_folder = result_folder
        self.trails_to_analyze = trails_to_analyze
        self.skip_if_existing = skip_if_existing

        # Extended attributes
        self.figures_result_folder = None
        self.models_result_folder = None

        # Run the analysis
        self.run_analysis()

    @staticmethod
    def get_version():
        """
        Save the version of the code and the date of the analysis for future reference
        """

        # Packages installed in the env
        # Running 'conda list' command and parse it as JSON
        result = subprocess.run(["conda", "list", "--json"], capture_output=True, text=True)
        packages = json.loads(result.stdout)
        packages_versions = {elt["name"]: elt["version"] for elt in packages}

        # Get the version of the current package
        repo = git.Repo(search_parent_directories=True)
        commit_id = str(repo.commit())
        branch = str(repo.active_branch)
        try:
            tag = repo.git.describe("--tags")
        except git.exc.GitCommandError:
            tag = "No tag"
        gait_analyzer_version = repo.git.version_info
        git_date = repo.git.log("-1", "--format=%cd")
        version_dic = {
            "commit_id": commit_id,
            "git_date": git_date,
            "branch": branch,
            "tag": tag,
            "gait_analyzer_version": gait_analyzer_version,
            "date_of_the_analysis": date.today().strftime("%b-%d-%Y-%H-%M-%S"),
            "biorbd_version": packages_versions["biorbd"],
            "pyomeca_version": packages_versions["pyomeca"] if "pyomeca" in packages_versions else "Not installed",
            "ezc3d_version": packages_versions["ezc3d"],
            "bioptim_version": (
                packages_versions["bioptim"] if "bioptim" in packages_versions else "Not installed through conda-forge"
            ),
        }
        return version_dic

    def save_subject_results(self, results, result_file_name: str):
        """
        Save the results of the analysis in a pickle file and a matlab file.
        .
        Parameters
        ----------
        results: ResultManager
            The ResultManager containing the results of the analysis performed by analysis_to_perform
        result_file_name: str
            The name of the file where the results will be saved. The file will be saved as result_file_name.pkl and result_file_name.mat
        """

        result_dict = self.get_version()
        for attr_name in dir(results):
            attr = getattr(results, attr_name)
            if not callable(attr) and not attr_name.startswith("__"):
                if hasattr(attr, "outputs") and callable(getattr(attr, "outputs")):
                    this_output_dict = attr.outputs()
                    for key, value in this_output_dict.items():
                        if key in result_dict:
                            raise ValueError(
                                f"Key {key} from class {attr_name} already exists in the result dictionary, please change the key to differentiate them."
                            )
                        elif key == "biorbd_model":
                            pass  # biorbd models are not picklable
                        elif value is None:
                            pass  # Nones are not picklable
                        else:
                            result_dict[key] = value

        # Save the results
        # For python analysis
        with open(result_file_name + ".pkl", "wb") as f:
            pickle.dump(result_dict, f)
        # For matlab analysis
        savemat(result_file_name + ".mat", result_dict)

    def run_analysis(self):
        """
        Loops over the data files and perform the analysis specified by the user (on the subjects specified by the user).
        """
        # Loop over all subjects
        for subject in self.subjects_to_analyze:

            subject_name = subject.subject_name
            subject_data_folder = f"../data/{subject_name}"

            # Checks
            if not os.path.exists(subject_data_folder):
                os.makedirs(subject_data_folder)
                tempo_subject_path = os.path.abspath(subject_data_folder)
                raise RuntimeError(
                    f"Data folder for subject {subject_name} does not exist. I have created it here {tempo_subject_path}, please put the data files in here."
                )

            # Loop over files to find the static trial
            static_trial_full_file_path = None
            for data_file in os.listdir(subject_data_folder):
                if data_file.endswith("static.c3d"):
                    static_trial_full_file_path = f"../data/{subject_name}/{data_file}"
                    break
            if not static_trial_full_file_path:
                raise FileNotFoundError(
                    f"Please put the static trial file here {os.path.abspath(subject_data_folder)} and name it [...]_static.c3d"
                )

            # Define subject specific paths
            result_folder = f"{self.result_folder}/{subject_name}"
            self.figures_result_folder = f"{result_folder}/figures"
            self.models_result_folder = f"{result_folder}/models"
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
                os.makedirs(self.figures_result_folder)
                os.makedirs(self.models_result_folder)
                print("The results folder was created here: ", os.path.abspath(result_folder))
            if not os.path.exists(self.figures_result_folder):
                os.makedirs(self.figures_result_folder)
            if not os.path.exists(self.models_result_folder):
                os.makedirs(self.models_result_folder)

            # Loop over all data files
            for data_file in os.listdir(subject_data_folder):
                # Files that we should not analyze
                if data_file.endswith("Statique.c3d") or not data_file.endswith(".c3d"):
                    continue
                if self.trails_to_analyze is not None and not any(
                    trail in data_file for trail in self.trails_to_analyze
                ):
                    continue

                c3d_file_name = f"../data/{subject_name}/{data_file}"
                result_file_name = f"{result_folder}/{data_file.replace('.c3d', '_results')}"

                # Skip if already exists
                if self.skip_if_existing and os.path.exists(result_file_name + ".pkl"):
                    print(f"Skipping {subject_name} - {data_file} because it already exists.")
                    continue

                # Actually perform the analysis
                print("Analyzing ", subject_name, " : ", data_file)
                results = self.analysis_to_perform(
                    subject,
                    self.cycles_to_analyze,
                    static_trial_full_file_path,
                    c3d_file_name,
                    result_folder,
                )
                self.save_subject_results(results, result_file_name)
