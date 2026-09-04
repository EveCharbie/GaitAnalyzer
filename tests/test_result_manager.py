import pytest

from gait_analyzer.result_manager import ResultManager
from gait_analyzer.subject import Subject, Side


def _make_result_manager(tmp_path):
    return ResultManager(
        subject=Subject(subject_name="P01", subject_mass=70.0),
        cycles_to_analyze=None,
        static_trial=str(tmp_path / "static.c3d"),
        result_folder=str(tmp_path),
        trial_full_file_path=str(tmp_path / "trial.c3d"),
        static_trial_full_file_path=str(tmp_path / "static.c3d"),
    )


def test_constructor_requires_subject_instance(tmp_path):
    with pytest.raises(ValueError, match="subject must be a Subject"):
        ResultManager(
            subject="not_a_subject",
            cycles_to_analyze=None,
            static_trial=str(tmp_path / "static.c3d"),
            result_folder=str(tmp_path),
            trial_full_file_path=str(tmp_path / "trial.c3d"),
            static_trial_full_file_path=str(tmp_path / "static.c3d"),
        )


def test_constructor_requires_cycles_to_analyze_range_or_none(tmp_path):
    with pytest.raises(ValueError, match="cycles_to_analyze must be a range"):
        ResultManager(
            subject=Subject(subject_name="P01", subject_mass=70.0),
            cycles_to_analyze="not_a_range",
            static_trial=str(tmp_path / "static.c3d"),
            result_folder=str(tmp_path),
            trial_full_file_path=str(tmp_path / "trial.c3d"),
            static_trial_full_file_path=str(tmp_path / "static.c3d"),
        )


def test_constructor_accepts_none_or_range_cycles_to_analyze(tmp_path):
    rm_none = _make_result_manager(tmp_path)
    assert rm_none.cycles_to_analyze is None

    rm_range = ResultManager(
        subject=Subject(subject_name="P01", subject_mass=70.0),
        cycles_to_analyze=range(0, 5),
        static_trial=str(tmp_path / "static.c3d"),
        result_folder=str(tmp_path),
        trial_full_file_path=str(tmp_path / "trial.c3d"),
        static_trial_full_file_path=str(tmp_path / "static.c3d"),
    )
    assert rm_range.cycles_to_analyze == range(0, 5)


def test_add_experimental_data_requires_model_first(tmp_path):
    result_manager = _make_result_manager(tmp_path)
    with pytest.raises(Exception, match="Please add the biorbd model first"):
        result_manager.add_experimental_data(c3d_file_name="trial.c3d")


def test_add_cyclic_events_requires_model_first(tmp_path):
    result_manager = _make_result_manager(tmp_path)
    with pytest.raises(Exception, match="Please add the biorbd model first"):
        result_manager.add_cyclic_events(force_plate_sides=[Side.LEFT, Side.RIGHT], skip_if_existing=False)


def test_add_cyclic_events_requires_experimental_data_after_model(tmp_path):
    result_manager = _make_result_manager(tmp_path)
    result_manager.model_creator = object()  # sentinel: "a model has been added"
    with pytest.raises(Exception, match="Please add the experimental data first"):
        result_manager.add_cyclic_events(force_plate_sides=[Side.LEFT, Side.RIGHT], skip_if_existing=False)


def test_add_cyclic_events_forbids_adding_events_twice(tmp_path):
    result_manager = _make_result_manager(tmp_path)
    result_manager.model_creator = object()
    result_manager.experimental_data = object()
    result_manager.events = object()  # sentinel: "events have already been added"
    with pytest.raises(Exception, match="already added"):
        result_manager.add_cyclic_events(force_plate_sides=[Side.LEFT, Side.RIGHT], skip_if_existing=False)


def test_reconstruct_kinematics_requires_events_first(tmp_path):
    result_manager = _make_result_manager(tmp_path)
    result_manager.model_creator = object()
    result_manager.experimental_data = object()
    with pytest.raises(Exception, match="Please run the events detection first"):
        result_manager.reconstruct_kinematics()


def test_perform_inverse_dynamics_requires_kinematics_first(tmp_path):
    result_manager = _make_result_manager(tmp_path)
    result_manager.model_creator = object()
    result_manager.experimental_data = object()
    with pytest.raises(Exception, match="Please add the kinematics reconstructor first"):
        result_manager.perform_inverse_dynamics(skip_if_existing=False)


def test_compute_angular_momentum_requires_kinematics_first(tmp_path):
    result_manager = _make_result_manager(tmp_path)
    result_manager.model_creator = object()
    result_manager.experimental_data = object()
    with pytest.raises(Exception, match="Please add the kinematics reconstructor first"):
        result_manager.compute_angular_momentum()


def test_compute_angular_momentum_forbids_computing_twice(tmp_path):
    result_manager = _make_result_manager(tmp_path)
    result_manager.model_creator = object()
    result_manager.experimental_data = object()
    result_manager.kinematics_reconstructor = object()
    result_manager.angular_momentum_calculator = object()
    with pytest.raises(Exception, match="already been calculated"):
        result_manager.compute_angular_momentum()


def test_compute_com_mma_distance_requires_angular_momentum_first(tmp_path):
    result_manager = _make_result_manager(tmp_path)
    with pytest.raises(Exception, match="Compute angular momentum first"):
        result_manager.compute_com_mma_distance()


def test_compute_com_mma_distance_requires_kinematics_after_angular_momentum(tmp_path):
    result_manager = _make_result_manager(tmp_path)
    result_manager.angular_momentum_calculator = object()
    with pytest.raises(Exception, match="Compute kinematics first"):
        result_manager.compute_com_mma_distance()


def test_compute_mechanical_energy_requires_angular_momentum_first(tmp_path):
    result_manager = _make_result_manager(tmp_path)
    result_manager.model_creator = object()
    result_manager.experimental_data = object()
    result_manager.kinematics_reconstructor = object()
    with pytest.raises(Exception, match="Please compute angular momentum first"):
        result_manager.compute_mechanical_energy()


def test_estimate_optimally_requires_full_pipeline(tmp_path):
    result_manager = _make_result_manager(tmp_path)
    with pytest.raises(Exception, match="Please add the biorbd model first"):
        result_manager.estimate_optimally(cycles_to_analyze=0)

    result_manager.model_creator = object()
    with pytest.raises(Exception, match="Please add the experimental data first"):
        result_manager.estimate_optimally(cycles_to_analyze=0)

    result_manager.experimental_data = object()
    with pytest.raises(Exception, match="Please run the events detection first"):
        result_manager.estimate_optimally(cycles_to_analyze=0)

    result_manager.events = object()
    with pytest.raises(Exception, match="Please run the kinematics reconstruction first"):
        result_manager.estimate_optimally(cycles_to_analyze=0)
