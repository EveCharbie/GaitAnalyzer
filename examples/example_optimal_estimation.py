from gait_analyzer import (
    helper,
    ResultManager,
    OsimModels,
    Operator,
    AnalysisPerformer,
    PlotLegData,
    LegToPlot,
    PlotType,
    Subject,
    Side,
    ReconstructionType,
    EventIndexType,
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

    results.create_model(
        osim_model_type=OsimModels.WholeBody(),
        functional_trials_path=f"../data/{subject.subject_name}/functional_trials/",
        mvc_trials_path=f"../data/{subject.subject_name}/maximal_voluntary_contractions/",
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

    results.add_cyclic_events(force_plate_sides=[Side.RIGHT, Side.LEFT], skip_if_existing=True, plot_phases_flag=False)
    # results.add_unique_events(skip_if_existing=True, plot_phases_flag=False)

    results.reconstruct_kinematics(
        reconstruction_type=[ReconstructionType.ONLY_LM],  # , ReconstructionType.LM, ReconstructionType.TRF],
        animate_kinematics_flag=False,
        plot_kinematics_flag=False,
        skip_if_existing=True,
    )

    results.perform_inverse_dynamics(skip_if_existing=True, reintegrate_flag=False, animate_dynamics_flag=False)

    # --- Example of analysis that can be performed in any order --- #
    results.estimate_optimally(
        cycle_to_analyze=5,
        plot_solution_flag=True,
        animate_solution_flag=True,
        implicit_contacts=False,
        skip_if_existing=False,
    )

    return results


def parameters_to_extract_for_statistical_analysis():
    # TODO: Add the parameters you want to extract for statistical analysis
    pass


if __name__ == "__main__":

    # --- Example of how to get help on a GaitAnalyzer class --- #
    # helper(Operator)

    # --- Create the list of participants --- #
    subjects_to_analyze = []
    # subjects_to_analyze.append(
    #     Subject(subject_name="AOT_01", subject_mass=69.2, dominant_leg=Side.RIGHT, preferential_speed=1.06)
    # )
    # subjects_to_analyze.append(
    #     Subject(subject_name="CAR_17", subject_mass=69.5, dominant_leg=Side.RIGHT, preferential_speed=1.06)
    # )
    subjects_to_analyze.append(Subject(subject_name="LEM_PRE_chev", dominant_leg=Side.RIGHT, preferential_speed=1.25))
    # ... add other participants here

    # --- Example of how to run the analysis --- #
    AnalysisPerformer(
        analysis_to_perform,
        subjects_to_analyze=subjects_to_analyze,
        cycles_to_analyze=range(5, -5),
        # cycles_to_analyze=None,
        result_folder="results",
        # trails_to_analyze=["_ManipStim_L400_F50_I60"],  # If not specified, all trials will be analyzed
        skip_if_existing=False,
    )
    #
    # # --- Example of how to plot the joint angles --- #
    # # plot = PlotLegData(
    # #     result_folder="results",
    # #     leg_to_plot=LegToPlot.RIGHT,
    # #     plot_type=PlotType.Q,
    # #     unique_event_to_split={
    # #         "event_index_type": EventIndexType.MARKERS,
    # #         "start": lambda data: int(data["events"][0]["heel_touch"][0]),
    # #         "stop": lambda data: int(data["events"][2]["heel_touch"][0]),
    # #     },
    # #     conditions_to_compare=["_Cond0006"],
    # # )
    # # plot.draw_plot()
    # # plot.save("results/AOT_01_Q_plot_temporary.png")
    # # plot.show()
    #
    # # --- Example of how to plot the joint angular velocities--- #
    # plot = PlotLegData(
    #     result_folder="results",
    #     leg_to_plot=LegToPlot.RIGHT,
    #     plot_type=PlotType.QDOT,
    #     conditions_to_compare=["_ManipStim_L400_F50_I20"],
    # )
    # plot.draw_plot()
    # plot.save("results/AOT_01_QDOT_plot_temporary.png")
    # plot.show()
    #
    # # --- Example of how to plot the joint torques --- #
    # plot = PlotLegData(
    #     result_folder="results",
    #     leg_to_plot=LegToPlot.RIGHT,
    #     plot_type=PlotType.TAU,
    #     conditions_to_compare=["_ManipStim_L200_F30_I20"],
    # )
    # plot.draw_plot()
    # plot.save("results/AOT_01_Tau_plot_temporary.png")
    # plot.show()
    #
    # # --- Example of how to plot the ground reaction forces --- #
    # plot = PlotLegData(
    #     result_folder="results",
    #     leg_to_plot=LegToPlot.RIGHT,
    #     plot_type=PlotType.GRF,
    #     conditions_to_compare=["_ManipStim_L200_F30_I20"],
    # )
    # plot.draw_plot()
    # plot.save("results/AOT_01_GRF_plot_temporary.png")
    # plot.show()
