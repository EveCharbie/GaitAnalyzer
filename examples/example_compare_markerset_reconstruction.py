import os

from biobuddy import C3dData
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
    markers_to_ignore: list[str] | None = None,
):

    # --- Example of analysis that must be performed in order --- #
    results = ResultManager(
        subject=subject,
        cycles_to_analyze=cycles_to_analyze,
        static_trial=static_trial,  # TODO: Charbie -> C3dData
        result_folder=result_folder,
    )

    results.create_model(osim_model_type=OsimModels.WholeBody(), skip_if_existing=False, animate_model_flag=False)

    results.add_experimental_data(c3d_file_name=c3d_file_name, markers_to_ignore=markers_to_ignore)

    results.add_cyclic_events(force_plate_sides=[Side.RIGHT, Side.LEFT], skip_if_existing=False, plot_phases_flag=False)

    results.reconstruct_kinematics(
        reconstruction_type=[ReconstructionType.TRF],
        animate_kinematics_flag=True,
        plot_kinematics_flag=False,
        skip_if_existing=False,
    )

    return results


if __name__ == "__main__":

    # --- Create the list of participants --- #
    subjects_to_analyze = []
    subject_name = "ECH"
    subjects_to_analyze.append(
        Subject(subject_name=subject_name, subject_mass=64.59, dominant_leg=Side.RIGHT, preferential_speed=1.06)
    )

    # --- Example of how to run the analysis --- #
    result_folder = "results"
    AnalysisPerformer(
        analysis_to_perform,
        subjects_to_analyze=subjects_to_analyze,
        cycles_to_analyze=range(5, -5),
        result_folder=result_folder,
        skip_if_existing=False,
        markers_to_ignore=[],
    )
    os.rename(
        f"{result_folder}/{subject_name}/inv_kin_gait_self_speed.pkl",
        f"{result_folder}/{subject_name}/inv_kin_gait_self_speed_with_clusters.pkl",
    )

    # # --- Example of how to run the analysis --- #
    # AnalysisPerformer(
    #     analysis_to_perform,
    #     subjects_to_analyze=subjects_to_analyze,
    #     cycles_to_analyze=range(5, -5),
    #     result_folder="results",
    #     skip_if_existing=False,
    #     markers_to_ignore=[
    #         "RTHI1",
    #         "RTHI2",
    #         "RTHI3",
    #         "LTHI1",
    #         "LTHI2",
    #         "LTHI3",
    #         "RLEG1",
    #         "RLEG2",
    #         "RLEG3",
    #         "LLEG1",
    #         "LLEG2",
    #         "LLEG3",
    #         "RAMR1",
    #         "RARM2",
    #         "RARM3",
    #         "RFARM1",
    #         "RFARM2",
    #         "RFARM3",
    #         "LARM1",
    #         "LARM2",
    #         "LARM3",
    #         "LFARM1",
    #         "LFARM2",
    #         "LFARM3",
    #     ],
    # )
    # os.rename("results/inv_kin_gait_self_speed.pkl", "results/inv_kin_gait_self_speed_no_clusters.pkl")
