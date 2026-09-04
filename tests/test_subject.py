import pytest
import numpy.testing as npt

from gait_analyzer.subject import Subject, Side


def test_side_enum_values():
    assert Side.LEFT.value == "left"
    assert Side.RIGHT.value == "right"


def test_subject_minimal_construction():
    subject = Subject(subject_name="P01")
    assert subject.subject_name == "P01"
    assert subject.subject_mass is None
    assert subject.subject_height is None
    assert subject.dominant_leg is None
    assert subject.preferential_speed is None


def test_subject_full_construction():
    subject = Subject(
        subject_name="P02",
        subject_mass=65.5,
        subject_height=1.68,
        dominant_leg=Side.RIGHT,
        preferential_speed=1.2,
    )
    npt.assert_allclose(subject.subject_mass, 65.5)
    npt.assert_allclose(subject.subject_height, 1.68)
    assert subject.dominant_leg == Side.RIGHT
    npt.assert_allclose(subject.preferential_speed, 1.2)


def test_subject_outputs_dict_matches_attributes():
    subject = Subject(subject_name="P03", subject_mass=80.0, dominant_leg=Side.LEFT, preferential_speed=0.9)
    outputs = subject.outputs()
    assert outputs["subject_name"] == "P03"
    npt.assert_allclose(outputs["subject_mass"], 80.0)
    assert outputs["dominant_leg"] == Side.LEFT
    npt.assert_allclose(outputs["preferential_speed"], 0.9)
    # subject_height is intentionally not part of outputs()
    assert "subject_height" not in outputs


def test_subject_name_must_be_string():
    with pytest.raises(ValueError, match="subject_name must be a string"):
        Subject(subject_name=123)


def test_subject_mass_must_be_float():
    with pytest.raises(ValueError, match="subject_mass must be an float"):
        Subject(subject_name="P04", subject_mass=70)  # int, not float


@pytest.mark.parametrize("bad_mass", [10.0, 29.9, 100.1, 500.0])
def test_subject_mass_out_of_range_raises(bad_mass):
    with pytest.raises(ValueError, match="must be a expressed in \\[30, 100\\] kg"):
        Subject(subject_name="P05", subject_mass=bad_mass)


@pytest.mark.parametrize("good_mass", [30.0, 65.0, 100.0])
def test_subject_mass_boundary_values_accepted(good_mass):
    subject = Subject(subject_name="P06", subject_mass=good_mass)
    npt.assert_allclose(subject.subject_mass, good_mass)


def test_subject_dominant_leg_must_be_side():
    with pytest.raises(ValueError, match="dominant_leg must be a Side"):
        Subject(subject_name="P07", dominant_leg="right")


def test_subject_preferential_speed_must_be_float():
    with pytest.raises(ValueError, match="preferential_speed must be a float"):
        Subject(subject_name="P08", preferential_speed=1)  # int, not float
