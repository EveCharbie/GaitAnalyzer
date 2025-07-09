from gait_analyzer import (
    ResultManager,
    OsimModels,
    AnalysisPerformer,
    PlotLegData,
    LegToPlot,
    PlotType,
    Subject,
    Side,
    ReconstructionType,
    MarkerLabelingHandler,
)


def analysis_to_perform(
    subject: Subject,
    cycles_to_analyze: range | None,
    static_trial: str,
    c3d_file_name: str,
    result_folder: str,
):

    # # This step is to show the markers and eventually change their labeling manually
    # marker_handler = MarkerLabelingHandler("path_to_the_c3d_you_want_to_check.c3d")
    # marker_handler.show_marker_labeling_plot()
    # marker_handler.invert_marker_labeling([name_of_the_marker, name_of_another_marker], frame_start=0, frame_end=100)
    # marker_handler.save_c3d(output_c3d_path)

    results = ResultManager(
        subject=subject,
        cycles_to_analyze=cycles_to_analyze,
        static_trial=static_trial,
        result_folder=result_folder,
    )

    results.create_model(
        osim_model_type=OsimModels.WholeBody(),
        functional_trials_path=f"../../data/{subject.subject_name}/functional_trials/",
        mvc_trials_path=f"../../data/{subject.subject_name}/maximal_voluntary_contractions/",
        skip_if_existing=True,
        animate_model_flag=False,
        vtp_geometry_path="../../Geometry_cleaned",
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
        c3d_file_name=c3d_file_name,
        markers_to_ignore=markers_to_ignore,
        analogs_to_ignore=analogs_to_ignore,
    )

    results.add_cyclic_events(force_plate_sides=[Side.RIGHT, Side.LEFT], skip_if_existing=True, plot_phases_flag=False)

    results.reconstruct_kinematics(
        reconstruction_type=[ReconstructionType.ONLY_LM],
        animate_kinematics_flag=False,
        plot_kinematics_flag=False,
        skip_if_existing=True,
    )

    results.perform_inverse_dynamics(skip_if_existing=True, reintegrate_flag=False, animate_dynamics_flag=False)

    return results


if __name__ == "__main__":

    # --- Create the list of participants --- #
    subjects_to_analyze = []
    subjects_to_analyze.append(
        Subject(subject_name="PRE", subject_mass=72.0, dominant_leg=Side.RIGHT, preferential_speed=1.5)
    )
    subjects_to_analyze.append(
        Subject(subject_name="POST", subject_mass=72.0, dominant_leg=Side.RIGHT, preferential_speed=1.5)
    )

    # --- Run the analysis --- #
    AnalysisPerformer(
        analysis_to_perform,
        subjects_to_analyze=subjects_to_analyze,
        cycles_to_analyze=range(5, -5),
        result_folder="../results",
        trails_to_analyze=["_condition1"],
        skip_if_existing=True,
    )

    # --- Plot --- #
    conditions_to_compare = ["_condition1"]
    groups_to_compare = {
        "Control": ["PRE"],
        "Chev": ["POST"],
    }

    # Joint angles
    plot = PlotLegData(
        result_folder="../results",
        leg_to_plot=LegToPlot.RIGHT,
        plot_type=PlotType.Q,
        conditions_to_compare=conditions_to_compare,
        groups_to_compare=groups_to_compare,
    )
    plot.draw_plot()
    plot.save("../results/Q_plot.png")
    plot.show()

    # GRF
    plot = PlotLegData(
        result_folder="../results",
        leg_to_plot=LegToPlot.RIGHT,
        plot_type=PlotType.GRF,
        conditions_to_compare=conditions_to_compare,
        groups_to_compare=groups_to_compare,
    )
    plot.draw_plot()
    plot.save("../results/GRF_plot.png")
    plot.show()

    # EMG
    plot = PlotLegData(
        result_folder="../results",
        leg_to_plot=LegToPlot.RIGHT,
        plot_type=PlotType.EMG,
        conditions_to_compare=conditions_to_compare,
        groups_to_compare=groups_to_compare,
    )
    plot.draw_plot()
    plot.save("../results/EMG_plot.png")
    plot.show()

    # Ankle torque
    plot = PlotLegData(
        result_folder="../results",
        leg_to_plot=LegToPlot.RIGHT,
        plot_type=PlotType.TAU,
        conditions_to_compare=conditions_to_compare,
        groups_to_compare=groups_to_compare,
    )
    plot.draw_plot()
    plot.save("../results/TAU_plot.png")
    plot.show()
