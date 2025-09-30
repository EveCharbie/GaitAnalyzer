import logging

from gait_analyzer import (
    ResultManager,
    OsimModels,
    AnalysisPerformer,
    PlotBiomechanicsQuantity,
    PlotType,
    Subject,
    Side,
    ReconstructionType,
    OrganizedResult,
    StatsPerformer,
    QuantityToExtractType,
    StatsType,
    MarkerLabelingHandler,
)


def analysis_to_perform(
    subject: Subject,
    cycles_to_analyze: range | None,
    static_trial: str,
    c3d_file_name: str,
    result_folder: str,
):

    # --- Example of analysis that must be performed in order --- #
    results = ResultManager(
        subject=subject,
        cycles_to_analyze=cycles_to_analyze,
        static_trial=static_trial,
        result_folder=result_folder,
    )

    # The hip SCoREs did not seem to help improve the models, so they were not use din this example.
    results.create_model(
        osim_model_type=OsimModels.WholeBody(),
        mvc_trials_path=f"..\\data\\{subject.subject_name}\\mvc_trials\\",
        # functional_trials_path=None,  # If you want to skip the functional trials for this example
        functional_trials_path=f"..\\data\\{subject.subject_name}\\functional_trials\\",
        skip_if_existing=True,
        animate_model_flag=False,
    )

    markers_to_ignore = []
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
    ]
    results.add_experimental_data(
        c3d_file_name=c3d_file_name, markers_to_ignore=markers_to_ignore, analogs_to_ignore=analogs_to_ignore
    )

    results.add_cyclic_events(force_plate_sides=[Side.LEFT, Side.RIGHT], skip_if_existing=True, plot_phases_flag=False)

    results.reconstruct_kinematics(
        reconstruction_type=[
            ReconstructionType.LSQ,
            ReconstructionType.ONLY_LM,
            ReconstructionType.LM,
            ReconstructionType.TRF,
        ],
        animate_kinematics_flag=True,
        plot_kinematics_flag=False,
        skip_if_existing=True,
    )

    results.compute_angular_momentum()

    return results



def main():
    # Récupérer infos sujet
    subjects_to_analyze = []
    subjects_to_analyze.append(
        Subject(
            subject_name="CAC_01",
            subject_height=1.66,
        )
    )

    # --- Example of how to run the analysis --- #
    AnalysisPerformer(
        analysis_to_perform,
        subjects_to_analyze=subjects_to_analyze,
        cycles_to_analyze=None,
        result_folder="results",
        trails_to_analyze=["_Marche050", "_Marche075", "_Marche100"],
        skip_if_existing=True,
    )

    # --- Example of how to create a OrganizedResult object --- #
    organized_result = OrganizedResult(
        result_folder="results",
        conditions_to_compare=["_Marche050", "_Marche075", "_Marche100"],
        plot_type=PlotType.ANGULAR_MOMENTUM,
        nb_frames_interp=101,
    )
    organized_result.save("results\\AngMom_organized.pkl")

    # --- Example of how to plot the angular momentum --- #
    plot = PlotBiomechanicsQuantity(
        organized_result=organized_result,
    )
    plot.draw_plot()
    plot.save("results\\AngMom_temporary.svg")
    plot.show()

    # --- Example of how compare peak-to-peak angular momentum with a paired t-test --- #
    stats_results = StatsPerformer(
        organized_result=organized_result,
        stats_type=StatsType.PAIRED_T_TEST(QuantityToExtractType.PEAK_TO_PEAK),
    )
    stats_results.perform_stats()
    stats_results.plot_stats(
        save_plot_name="results\\AngMom_paired_t_test.svg", order=["_Marche050", "_Marche075", "_Marche100"]
    )


if __name__ == "__main__":
    main()