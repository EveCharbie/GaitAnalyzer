from gait_analyzer import (
    helper,
    ResultManager,
    OsimModels,
    Operator,
    AnalysisPerformer,
    PlotLegData,
    LegToPlot,
    PlotType,
)
from gait_analyzer.kinematics_reconstructor import ReconstructionType


def analysis_to_perform(
    subject_name: str,
    subject_mass: float,
    cycles_to_analyze: range,
    static_trial: str,
    c3d_file_name: str,
    result_folder: str,
):

    # --- Example of analysis that must be performed in order --- #
    results = ResultManager(
        subject_name=subject_name,
        subject_mass=subject_mass,
        cycles_to_analyze=cycles_to_analyze,
        static_trial=static_trial,
        result_folder=result_folder,
    )
    results.create_model(osim_model_type=OsimModels.WholeBody(), skip_if_existing=False, animate_model_flag=False)
    results.add_experimental_data(c3d_file_name=c3d_file_name, animate_c3d_flag=False)
    results.add_events(plot_phases_flag=False)
    results.reconstruct_kinematics(
        reconstruction_type=ReconstructionType.ONLY_LM,  # [ReconstructionType.ONLY_LM, ReconstructionType.LM, ReconstructionType.TRF],
        animate_kinematics_flag=True,
        plot_kinematics_flag=True,
        skip_if_existing=False,
    )
    results.perform_inverse_dynamics(reintegrate_flag=False, animate_dynamics_flag=False)

    # --- Example of analysis that can be performed in any order --- #
    # results.estimate_optimally()

    return results


def parameters_to_extract_for_statistical_analysis():
    # TODO: Add the parameters you want to extract for statistical analysis
    pass


if __name__ == "__main__":

    # --- Example of how to get help on a GaitAnalyzer class --- #
    # helper(Operator)

    # --- Example of how to run the analysis --- #
    AnalysisPerformer(
        analysis_to_perform,
        subjects_to_analyze={"AOT_01": 69.2},  # add participants
        cycles_to_analyze=range(5, -5),
        result_folder="results",
        trails_to_analyze=["_ManipStim_L200_F30_I20"],
        skip_if_existing=False,
    )

    # --- Example of how to plot the joint angles --- #
    plot = PlotLegData(
        result_folder="results",
        leg_to_plot=LegToPlot.RIGHT,
        plot_type=PlotType.Q,
        conditions_to_compare=["_ManipStim_L200_F30_I20"],
    )
    plot.draw_plot()
    plot.save("results/AOT_01_Q_plot_temporary.png")
    plot.show()

    # --- Example of how to plot the joint torques --- #
    plot = PlotLegData(
        result_folder="results",
        leg_to_plot=LegToPlot.RIGHT,
        plot_type=PlotType.TAU,
        conditions_to_compare=["_ManipStim_L200_F30_I20"],
    )
    plot.draw_plot()
    plot.save("results/AOT_01_Tau_plot_temporary.png")
    plot.show()

    # --- Example of how to plot the ground reaction forces --- #
    plot = PlotLegData(
        result_folder="results",
        leg_to_plot=LegToPlot.RIGHT,
        plot_type=PlotType.GRF,
        conditions_to_compare=["_ManipStim_L200_F30_I20"],
    )
    plot.draw_plot()
    plot.save("results/AOT_01_GRF_plot_temporary.png")
    plot.show()
