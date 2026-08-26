from gait_analyzer import (
    ResultManager,
    OsimModels,
    AnalysisPerformer,
    Subject,
    Side,
    ReconstructionType,
)


def analysis_to_perform(
    subject: Subject,
    cycles_to_analyze: range | None,
    static_trial: str,
    c3d_file_name: str,
    result_folder: str,
):

    # --- Defining full paths for C3D files ---
    base_data_path = f"/Users/floethv/Desktop/Doctorat/Fork/GaitAnalyzer/data/{subject.subject_name}"
    c3d_dynamic_path = f"{base_data_path}/{c3d_file_name}"
    c3d_static_path = f"{static_trial}"

    results = ResultManager(
        subject=subject,
        cycles_to_analyze=cycles_to_analyze,
        static_trial=c3d_static_path,
        result_folder=result_folder,
        trial_full_file_path=c3d_dynamic_path,
        static_trial_full_file_path=c3d_static_path,
    )

    # Creation of model
    results.create_model(
        osim_model_type=OsimModels.WholeBody(),
        mvc_trials_path=f"{base_data_path}/mvc_trials/",
        functional_trials_path=None,
        q_regularization_weight=1,
        skip_if_existing=True,
        animate_model_flag=False,
    )

    # --- Ignore certain markers and channels ---
    markers_to_ignore = ["U1", "U2", "U3", "U4", "U5"]
    analogs_to_ignore = [
        "Channel_01",
        "Channel_02",
        "Channel_03",
        "Channel_04",
        "Channel_05",
        "Channel_06",
        "Channel_07",
        "Channel_08",
        "Channel_09",
        "Channel_10",
        "Channel_11",
        "Channel_12",
        "Bertec_treadmill_speed",
        "BICEPS_FEM_L",
        "RECTUS_FEM_L",
        "SEMITENDINOUS_L",
        "VASTM_L",
        "GM_L",
        "SOL_L",
        "TIB_L",
        "TFL",
        "TFL_L",
    ]
    results.add_experimental_data(
        c3d_file_name=c3d_file_name, markers_to_ignore=markers_to_ignore, analogs_to_ignore=analogs_to_ignore
    )

    # --- Detection of cyclic events ---
    results.add_cyclic_events(force_plate_sides=[Side.LEFT, Side.RIGHT], skip_if_existing=True, plot_phases_flag=False)

    # --- Reconstruction of the kinematics ---
    results.reconstruct_kinematics(
        reconstruction_type=[
            ReconstructionType.LSQ,
            ReconstructionType.ONLY_LM,
            ReconstructionType.LM,
            ReconstructionType.TRF,
        ],
        animate_kinematics_flag=False,
        plot_kinematics_flag=False,
        skip_if_existing=True,
    )

    # --- Biomechanical/Stability calculations ---
    results.compute_angular_momentum()
    results.compute_mechanical_energy(skip_if_existing=True)
    results.compute_marginofstability(skip_if_existing=True)
    results.compute_com_mma_distance()
    results.compute_probalityofstability(skip_if_existing=True)

    # --- Inverse dynamics ---
    results.perform_inverse_dynamics(
        skip_if_existing=True,
        reintegrate_flag=False,
        animate_dynamics_flag=False,
    )

    results.estimate_optimally(
        cycles_to_analyze=[6, 7, 8, 9, 10],
        plot_solution_flag=False,
        animate_solution_flag=False,
        skip_if_existing=True,
    )

    return results


def main():
    subjects_to_analyze = [
        Subject(subject_name="AOT43", subject_height=1.85, subject_mass=74.8),
        Subject(subject_name="BEC20", subject_height=1.68, subject_mass=57.7),
        Subject(subject_name="BEL44", subject_height=1.78, subject_mass=66.6),
        Subject(subject_name="BRP24", subject_height=1.74, subject_mass=77.0),
        Subject(subject_name="CAA21", subject_height=1.73, subject_mass=61.5),
        Subject(subject_name="CAM51", subject_height=1.68, subject_mass=56.5),
        Subject(subject_name="DEE47", subject_height=1.71, subject_mass=74.6),
        Subject(subject_name="DEG25", subject_height=1.82, subject_mass=87.7),
        Subject(subject_name="DOC08", subject_height=1.65, subject_mass=62.9),
        Subject(subject_name="EMC23", subject_height=1.61, subject_mass=59.1),
        Subject(subject_name="GRF37", subject_height=1.81, subject_mass=71.2),
        Subject(subject_name="GRW38", subject_height=1.83, subject_mass=70.9),
        Subject(subject_name="HIB10", subject_height=1.66, subject_mass=59.0),
        Subject(subject_name="HOL48", subject_height=1.66, subject_mass=72.7),
        Subject(subject_name="HOS31", subject_height=1.83, subject_mass=85.0),
        Subject(subject_name="KEI45", subject_height=1.76, subject_mass=75.7),
        Subject(subject_name="LAO01", subject_height=1.65, subject_mass=58.6),
        Subject(subject_name="LAT17", subject_height=1.61, subject_mass=48.4),
        Subject(subject_name="LAV11", subject_height=1.61, subject_mass=57.0),
        Subject(subject_name="LEA50", subject_height=1.71, subject_mass=69.8),
        Subject(subject_name="LED09", subject_height=1.71, subject_mass=69.5),
        Subject(subject_name="LEJ33", subject_height=1.82, subject_mass=79.4),
        Subject(subject_name="LEM19", subject_height=1.63, subject_mass=52.7),
        Subject(subject_name="MAF28", subject_height=1.61, subject_mass=53.3),
        Subject(subject_name="MAK49", subject_height=1.71, subject_mass=64.8),
        Subject(subject_name="MAR40", subject_height=1.84, subject_mass=83.7),
        Subject(subject_name="MAV14", subject_height=1.62, subject_mass=59.6),
        Subject(subject_name="PAM12", subject_height=1.62, subject_mass=65.8),
        Subject(subject_name="PAY18", subject_height=1.68, subject_mass=82.3),
        Subject(subject_name="PEE42", subject_height=1.68, subject_mass=59.2),
        Subject(subject_name="PII30", subject_height=1.55, subject_mass=52.8),
        Subject(subject_name="PLA36", subject_height=1.79, subject_mass=70.2),
        Subject(subject_name="ROA34", subject_height=1.86, subject_mass=82.3),
        Subject(subject_name="SCJ35", subject_height=1.74, subject_mass=73.0),
        Subject(subject_name="SYA41", subject_height=1.71, subject_mass=57.5),
        Subject(subject_name="TEJ26", subject_height=1.83, subject_mass=78.1),
        Subject(subject_name="VAS13", subject_height=1.59, subject_mass=67.3),
        Subject(subject_name="VIM46", subject_height=1.75, subject_mass=49.0),
        Subject(subject_name="VIO15", subject_height=1.70, subject_mass=74.1),
        Subject(subject_name="ZAZ39", subject_height=1.81, subject_mass=90.1),
    ]

    AnalysisPerformer(
        analysis_to_perform,
        subjects_to_analyze=subjects_to_analyze,
        cycles_to_analyze=None,
        result_folder="results",
        trails_to_analyze=["Cond0001", "Cond0002", "Cond0003", "Cond0004"],
        skip_if_existing=False,
    )


if __name__ == "__main__":
    main()
