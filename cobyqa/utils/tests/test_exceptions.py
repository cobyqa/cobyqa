import pytest

from cobyqa.utils import MaxEvalError


def test_max_eval_error():
    with pytest.raises(MaxEvalError):
        raise MaxEvalError
