import pytest

from .. import MaxEvalError


def test_max_eval_error():
    with pytest.raises(MaxEvalError):
        raise MaxEvalError
