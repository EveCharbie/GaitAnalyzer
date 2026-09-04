import pytest

from gait_analyzer.helper import helper
from gait_analyzer.subject import Subject
from gait_analyzer.operator import Operator


def test_helper_raises_on_non_callable_input():
    with pytest.raises(ValueError, match="The input must be a class instance."):
        helper(42)


def test_helper_prints_signature_and_methods(capsys):
    # helper() requires its input to be callable itself, so a class (not an instance of a
    # non-callable class like Subject) is passed here, as done for the classes exposed by
    # gait_analyzer/__init__.py.
    helper(Subject)
    captured = capsys.readouterr()
    assert "Subject" in captured.out
    assert "outputs" in captured.out
    assert "The following functions are available" in captured.out


def test_helper_documents_docstring_content(capsys):
    helper(Operator)
    captured = capsys.readouterr()
    # Operator.moving_average has a real docstring that should be surfaced
    assert "moving_average" in captured.out
    assert "Compute the moving average of a signal" in captured.out


def test_helper_reports_no_documentation_when_docstring_missing(capsys):
    # Subject.outputs has no docstring in the source, unlike Subject.__init__
    helper(Subject)
    captured = capsys.readouterr()
    assert "No documentation available." in captured.out
