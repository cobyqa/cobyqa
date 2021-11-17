import numpy as np
import pytest
from numpy.testing import assert_, assert_equal, assert_raises

from cobyqa import OptimizeResult


class TestOptimizeResult:

    @pytest.fixture
    def kwargs(self):
        return dict(
            x=np.zeros(3),
            success=True,
            status=0,
            message='Optimization terminated successfully.',
            fun=1e-4,
            jac=np.ones(3),
            nfev=123,
            nit=132,
        )

    def test_simple(self, kwargs):
        res = OptimizeResult(**kwargs)
        for key, value in kwargs.items():
            assert_equal(getattr(res, key), kwargs.get(key))
        assert_equal(dir(res), sorted(kwargs.keys()))
        res.nfev += 1
        assert_equal(res.nfev, kwargs.get('nfev') + 1)
        del res.nfev
        assert_('nfev' not in dir(res))
        assert_(isinstance(res.__repr__(), str))
        assert_(isinstance(res.__str__(), str))

    def test_exceptions(self):
        res = OptimizeResult()
        with assert_raises(AttributeError):
            _ = res.nfev
        with assert_raises(KeyError):
            del res.nfev
