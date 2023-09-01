import re
from abc import ABC

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cobyqa import minimize


class TestMinimizeBase(ABC):

    @staticmethod
    def arwhead(x):
        return np.sum((x[:-1] ** 2.0 + x[-1] ** 2.0) ** 2.0 - 4.0 * x[:-1] + 3.0)

    @staticmethod
    def rosen(x):
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1.0 - x[:-1]) ** 2.0)

    @staticmethod
    def rothyp(x):
        return np.sum(np.cumsum(x ** 2.0))

    @staticmethod
    def sphere(x):
        return np.sum(x ** 2.0)

    @staticmethod
    def sumsqu(x):
        return np.sum(np.arange(1, x.size + 1) * x ** 2.0)

    @staticmethod
    def trid(x):
        return np.sum((x - 1.0) ** 2.0) - np.sum(x[1:] * x[:-1])

    @staticmethod
    def _tough(fun, x):
        test = np.cos(1e12 * np.sum(x))
        if test >= 0.95:
            return np.nan
        elif test >= 0.9:
            return np.inf
        else:
            return fun(x)

    def __getattr__(self, item):
        tough = re.compile('(?P<fun>[a-z]+)_tough')
        tough_match = tough.match(item)
        if tough_match:
            try:
                fun = getattr(self, tough_match.group('fun'))
                return lambda x: self._tough(fun, x)
            except AttributeError as exc:
                raise AttributeError(item) from exc
        else:
            raise AttributeError(item)

    @staticmethod
    def assert_result(res, n, x_best, fun_best, status, maxcv):
        assert res.nfev <= 500 * n
        assert res.status == status
        assert res.success, res.message
        if status == 0:
            assert_allclose(res.x, x_best, atol=1e-3)
            assert_allclose(res.fun, fun_best, atol=1e-3)
            assert_allclose(res.maxcv, maxcv, atol=1e-3)

    @pytest.fixture
    def x0(self, fun, n):
        return {
            'arwhead': np.zeros(n),
            'rosen': np.zeros(n),
            'rothyp': np.ones(n),
            'sphere': np.ones(n),
            'sumsqu': np.ones(n),
            'trid': np.zeros(n),
        }.get(fun)

    @pytest.fixture
    def xl(self, fun, n):
        return {
            'arwhead': -5.12 * np.ones(n),
            'rosen': -2.048 * np.ones(n),
            'rothyp': -65.536 * np.ones(n),
            'sphere': -5.12 * np.ones(n),
            'sumsqu': -5.12 * np.ones(n),
            'trid': -n ** 2 * np.ones(n),
        }.get(fun)

    @pytest.fixture
    def xu(self, fun, n):
        return {
            'arwhead': 5.12 * np.ones(n),
            'rosen': 2.048 * np.ones(n),
            'rothyp': 65.536 * np.ones(n),
            'sphere': 5.12 * np.ones(n),
            'sumsqu': 5.12 * np.ones(n),
            'trid': n ** 2 * np.ones(n),
        }.get(fun)

    @pytest.fixture
    def x_best(self, fun, n):
        return {
            'arwhead': np.r_[np.ones(n - 1), 0.0],
            'rosen': np.ones(n),
            'rothyp': np.zeros(n),
            'sphere': np.zeros(n),
            'sumsqu': np.zeros(n),
            'trid': np.arange(1, n + 1) * np.arange(n, 0, -1),
        }.get(fun)

    @pytest.fixture
    def fun_best(self, fun, n):
        return {
            'arwhead': 0.0,
            'rosen': 0.0,
            'rothyp': 0.0,
            'sphere': 0.0,
            'sumsqu': 0.0,
            'trid': -n * (n + 4) * (n - 1) / 6,
        }.get(fun)


class TestMinimizeUnconstrained(TestMinimizeBase):

    @pytest.mark.parametrize('fun', ['arwhead', 'rosen', 'rothyp', 'sphere', 'sumsqu', 'trid'])
    @pytest.mark.parametrize('n', [2, 5, 10])
    def test_simple(self, fun, n, x0, x_best, fun_best):
        res = minimize(fun=getattr(self, fun), x0=x0, options={'debug': True})
        self.assert_result(res, n, x_best, fun_best, 0, 0.0)

    @pytest.mark.parametrize('fun', ['arwhead', 'rosen', 'rothyp', 'sphere', 'sumsqu', 'trid'])
    @pytest.mark.parametrize('n', [2, 5, 10])
    def test_tough(self, fun, n, x0, x_best, fun_best):
        minimize(fun=getattr(self, fun + '_tough'), x0=x0)

    @pytest.mark.parametrize('fun', ['arwhead', 'rosen', 'rothyp', 'sphere', 'sumsqu', 'trid'])
    @pytest.mark.parametrize('n', [2, 5, 10])
    def test_target(self, fun, n, x0, x_best, fun_best):
        res = minimize(fun=getattr(self, fun), x0=x0, options={'target': fun_best + 1.0, 'debug': True})
        self.assert_result(res, n, x_best, fun_best, 1, 0.0)


class TestMinimizeBoundConstrained(TestMinimizeBase):

    @pytest.fixture
    def xl_restricted(self, fun, n):
        return {
            'arwhead': -0.5 * np.ones(n),
            'power': 0.5 * np.ones(n),
            'sphere': np.arange(n),
        }.get(fun)

    @pytest.fixture
    def xu_restricted(self, fun, n):
        return {
            'arwhead': 0.5 * np.ones(n),
            'sphere': 2.0 * n * np.ones(n),
            'sumsqu': np.ones(n),
        }.get(fun)

    @pytest.fixture
    def x_best_restricted(self, fun, n):
        return {
            'arwhead': np.r_[0.5 * np.ones(n - 1), 0.0],
            'sphere': np.arange(n),
            'sumsqu': 0.5 * np.ones(n),
        }.get(fun)

    @pytest.fixture
    def fun_best_restricted(self, fun, n):
        return {
            'arwhead': 1.0625 * (n - 1),
            'sphere': n * (n - 1) * (2 * n - 1) // 6,
            'sumsqu': 0.25 * (n * (n + 1) // 2),
        }.get(fun)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['arwhead', 'rosen', 'rothyp', 'sphere', 'sumsqu', 'trid'])
    def test_simple(self, fun, n, x0, xl, xu, x_best, fun_best):
        res = minimize(fun=getattr(self, fun), x0=x0, xl=xl, xu=xu, options={'debug': True})
        self.assert_result(res, n, x_best, fun_best, 0, 0.0)

    @pytest.mark.parametrize('fun', ['arwhead', 'rosen', 'rothyp', 'sphere', 'sumsqu', 'trid'])
    @pytest.mark.parametrize('n', [2, 5, 10])
    def test_tough(self, fun, n, x0, xl, xu, x_best, fun_best):
        minimize(fun=getattr(self, fun + '_tough'), x0=x0, xl=xl, xu=xu)

    @pytest.mark.parametrize('fun', ['arwhead', 'rosen', 'rothyp', 'sphere', 'sumsqu', 'trid'])
    @pytest.mark.parametrize('n', [2, 5, 10])
    def test_target(self, fun, n, x0, xl, xu, x_best, fun_best):
        res = minimize(fun=getattr(self, fun), x0=x0, xl=xl, xu=xu, options={'target': fun_best + 1.0, 'debug': True})
        self.assert_result(res, n, x_best, fun_best, 1, 0.0)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['arwhead', 'sphere', 'sumsqu'])
    def test_restricted(self, fun, n, x0, xl_restricted, xu_restricted, x_best_restricted, fun_best_restricted):
        res = minimize(fun=getattr(self, fun), x0=x0, xl=xl_restricted, xu=xu_restricted, options={'debug': True})
        self.assert_result(res, n, x_best_restricted, fun_best_restricted, 0, 0.0)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['arwhead', 'rosen', 'rothyp', 'sphere', 'sumsqu', 'trid'])
    def test_fixed(self, fun, n, x0, xl, x_best, fun_best):
        res = minimize(fun=getattr(self, fun), x0=x0, xl=xl, xu=xl, options={'debug': True})
        self.assert_result(res, n, x0, getattr(self, fun)(x0), 2, 0.0)
