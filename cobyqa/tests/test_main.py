from abc import ABC

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cobyqa import minimize


class TestMinimizeBase(ABC):

    @staticmethod
    def rosen(x):
        return np.sum(100.0 * np.square(x[1:] - np.square(x[:-1])) + np.square(x[:-1] - 1.0))

    @staticmethod
    def rothyp(x):
        return np.sum(np.cumsum(np.square(x)))

    @staticmethod
    def sphere(x):
        return np.sum(np.square(x))

    @staticmethod
    def sumpow(x):
        return np.sum(np.power(np.abs(x), np.arange(2, x.size + 2)))

    @staticmethod
    def sumsqu(x):
        return np.sum(np.arange(1, x.size + 1) * np.square(x))

    @staticmethod
    def trid(x):
        return np.sum(np.square(x - 1.0)) - np.sum(x[1:] * x[:-1])

    @staticmethod
    def assert_result(res, n, x_best, fun_best, status, maxcv):
        assert res.nfev <= 500 * n
        assert res.status == status
        assert res.success, res.message
        if status == 0:
            assert_allclose(res.x, x_best, atol=1e-3)
            assert_allclose(res.fun, fun_best, atol=1e-3)
            assert_allclose(res.maxcv, 0.0, atol=1e-3)

    @pytest.fixture
    def x0(self, fun, n):
        return {
            'rosen': np.zeros(n),
            'rothyp': np.ones(n),
            'sphere': np.ones(n),
            'sumpow': np.ones(n),
            'sumsqu': np.ones(n),
            'trid': np.zeros(n),
        }.get(fun)

    @pytest.fixture
    def xl(self, fun, n):
        return {
            'rosen': -2.048 * np.ones(n),
            'rothyp': -65.536 * np.ones(n),
            'sphere': -5.12 * np.ones(n),
            'sumpow': -np.ones(n),
            'sumsqu': -5.12 * np.ones(n),
            'trid': -n ** 2 * np.ones(n),
        }.get(fun)

    @pytest.fixture
    def xu(self, fun, n):
        return {
            'rosen': 2.048 * np.ones(n),
            'rothyp': 65.536 * np.ones(n),
            'sphere': 5.12 * np.ones(n),
            'sumpow': np.ones(n),
            'sumsqu': 5.12 * np.ones(n),
            'trid': n ** 2 * np.ones(n),
        }.get(fun)

    @pytest.fixture
    def x_best(self, fun, n):
        return {
            'rosen': np.ones(n),
            'rothyp': np.zeros(n),
            'sphere': np.zeros(n),
            'sumpow': np.zeros(n),
            'sumsqu': np.zeros(n),
            'trid': np.arange(1, n + 1) * np.arange(n, 0, -1),
        }.get(fun)

    @pytest.fixture
    def fun_best(self, fun, n):
        return {
            'rosen': 0.0,
            'rothyp': 0.0,
            'sphere': 0.0,
            'sumpow': 0.0,
            'sumsqu': 0.0,
            'trid': -n * (n + 4) * (n - 1) / 6,
        }.get(fun)


class TestMinimizeUnconstrained(TestMinimizeBase):

    @pytest.mark.parametrize('fun', ['rosen', 'rothyp', 'sphere', 'sumpow', 'sumsqu', 'trid'])
    @pytest.mark.parametrize('n', [2, 5, 10])
    def test_simple(self, fun, n, x0, x_best, fun_best):
        res = minimize(fun=getattr(self, fun), x0=x0, options={'debug': True})
        self.assert_result(res, n, x_best, fun_best, 0, 0.0)
