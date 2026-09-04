import pytest

from gait_analyzer.analysis_performer import AnalysisPerformer
from gait_analyzer.subject import Subject


def _do_nothing(*args, **kwargs):
    return None


def test_constructor_requires_callable_analysis_to_perform(tmp_path):
    with pytest.raises(ValueError, match="analysis_to_perform must be a callable"):
        AnalysisPerformer(
            analysis_to_perform="not_callable",
            subjects_to_analyze=[],
            result_folder=str(tmp_path),
        )


def test_constructor_requires_list_of_subjects(tmp_path):
    with pytest.raises(ValueError, match="subjects_to_analyze must be a list of Subject"):
        AnalysisPerformer(
            analysis_to_perform=_do_nothing,
            subjects_to_analyze="not_a_list",
            result_folder=str(tmp_path),
        )


def test_constructor_requires_subject_elements(tmp_path):
    with pytest.raises(ValueError, match="All elements of subjects_to_analyze must be Subject"):
        AnalysisPerformer(
            analysis_to_perform=_do_nothing,
            subjects_to_analyze=["not_a_subject"],
            result_folder=str(tmp_path),
        )


def test_constructor_requires_valid_cycles_to_analyze_type(tmp_path):
    with pytest.raises(ValueError, match="cycles_to_analyze must be a range"):
        AnalysisPerformer(
            analysis_to_perform=_do_nothing,
            subjects_to_analyze=[],
            cycles_to_analyze="not_valid",
            result_folder=str(tmp_path),
        )


def test_constructor_cycles_to_analyze_dict_requires_string_keys(tmp_path):
    with pytest.raises(ValueError, match="Keys of cycles_to_analyze must be strings"):
        AnalysisPerformer(
            analysis_to_perform=_do_nothing,
            subjects_to_analyze=[],
            cycles_to_analyze={123: range(0, 3)},
            result_folder=str(tmp_path),
        )


def test_constructor_requires_string_result_folder():
    with pytest.raises(ValueError, match="result_folder must be a string"):
        AnalysisPerformer(
            analysis_to_perform=_do_nothing,
            subjects_to_analyze=[],
            result_folder=123,
        )


def test_constructor_requires_list_trails_to_analyze(tmp_path):
    with pytest.raises(ValueError, match="trails_to_analyze must be a list of strings"):
        AnalysisPerformer(
            analysis_to_perform=_do_nothing,
            subjects_to_analyze=[],
            result_folder=str(tmp_path),
            trails_to_analyze="not_a_list",
        )


def test_constructor_creates_result_folder_if_missing(tmp_path):
    result_folder = tmp_path / "brand_new_result_folder"
    assert not result_folder.exists()

    # subjects_to_analyze=[] means run_analysis() has nothing to loop over, so this cannot
    # touch the real repository's data/ folder (see AnalysisPerformer.run_analysis, which
    # only reads from data/<subject_name> once a subject is present in the list).
    AnalysisPerformer(
        analysis_to_perform=_do_nothing,
        subjects_to_analyze=[],
        result_folder=str(result_folder),
    )
    assert result_folder.exists()


def test_cycles_to_analyze_none_is_expanded_per_subject(tmp_path, monkeypatch):
    # NOTE: AnalysisPerformer.run_analysis() and check_for_geometry_files() resolve paths
    # relative to the installed gait_analyzer package itself (not result_folder), so any
    # subject actually present in subjects_to_analyze would make the real constructor touch
    # the real repository's data/ folder. Both are monkeypatched to no-ops here so this test
    # exercises the real __init__ validation/normalization logic in isolation.
    monkeypatch.setattr(AnalysisPerformer, "check_for_geometry_files", lambda self: None)
    monkeypatch.setattr(AnalysisPerformer, "run_analysis", lambda self: None)

    subject = Subject(subject_name="P01", subject_mass=70.0)
    performer = AnalysisPerformer(
        analysis_to_perform=_do_nothing,
        subjects_to_analyze=[subject],
        cycles_to_analyze=None,
        result_folder=str(tmp_path),
    )
    assert performer.cycles_to_analyze == {"P01": None}


def test_cycles_to_analyze_range_is_expanded_per_subject(tmp_path, monkeypatch):
    monkeypatch.setattr(AnalysisPerformer, "check_for_geometry_files", lambda self: None)
    monkeypatch.setattr(AnalysisPerformer, "run_analysis", lambda self: None)

    subject_1 = Subject(subject_name="P01", subject_mass=70.0)
    subject_2 = Subject(subject_name="P02", subject_mass=60.0)
    performer = AnalysisPerformer(
        analysis_to_perform=_do_nothing,
        subjects_to_analyze=[subject_1, subject_2],
        cycles_to_analyze=range(2, 5),
        result_folder=str(tmp_path),
    )
    assert performer.cycles_to_analyze == {"P01": range(2, 5), "P02": range(2, 5)}


def test_get_cycles_to_analyze_for_this_trial_with_none():
    performer = AnalysisPerformer.__new__(AnalysisPerformer)
    performer.cycles_to_analyze = None
    assert performer.get_cycles_to_analyze_for_this_trial("P01", "P01_trial.c3d") is None


def test_get_cycles_to_analyze_for_this_trial_with_range_per_subject():
    performer = AnalysisPerformer.__new__(AnalysisPerformer)
    performer.cycles_to_analyze = {"P01": range(3, 7)}
    assert performer.get_cycles_to_analyze_for_this_trial("P01", "P01_trial.c3d") == range(3, 7)


def test_get_cycles_to_analyze_for_this_trial_missing_subject_raises():
    performer = AnalysisPerformer.__new__(AnalysisPerformer)
    performer.cycles_to_analyze = {"P01": None}
    with pytest.raises(ValueError, match="Please provide a cycles_to_analyze for each subject"):
        performer.get_cycles_to_analyze_for_this_trial("P02", "P02_trial.c3d")


def test_get_cycles_to_analyze_for_this_trial_with_dict_matches_by_suffix():
    performer = AnalysisPerformer.__new__(AnalysisPerformer)
    performer.cycles_to_analyze = {"P01": {"walk": range(0, 10)}}
    assert performer.get_cycles_to_analyze_for_this_trial("P01", "P01_walk.c3d") == range(0, 10)
