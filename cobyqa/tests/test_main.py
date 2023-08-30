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
    def perm0d(x):
        n_range = np.arange(1, x.size + 1)
        fx = 0.0
        for i in range(x.size):
            fx += np.sum((n_range + 10.0) * (x ** i - 1.0 / n_range ** i)) ** 2.0
        return fx

    @staticmethod
    def permd(x):
        n_range = np.arange(1, x.size + 1)
        fx = 0.0
        for i in range(x.size):
            fx += np.sum((n_range ** i + 0.5) * ((x / n_range) ** i - 1.0)) ** 2.0
        return fx

    @staticmethod
    def powell(x):
        fx = 10.0 * (x[-4] - x[-1]) ** 4.0 if x.size % 4 == 0 else 0.0
        for i in range(x.size - 1):
            if i % 4 == 0:
                fx += (x[i] + 10.0 * x[i + 1]) ** 2.0
            elif i % 4 == 1:
                fx += (x[i] - 2.0 * x[i + 1]) ** 4.0
            elif i % 4 == 2:
                fx += 5.0 * (x[i] - x[i + 1]) ** 2.0
            else:
                fx += 10.0 * (x[i - 3] - x[i]) ** 4.0
        return fx

    @staticmethod
    def power(x):
        return np.sum(np.arange(1, x.size + 1) * x ** 2.0)

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
    def sumpow(x):
        return np.sum(np.abs(x) ** np.arange(2, x.size + 2))

    @staticmethod
    def sumsqu(x):
        return np.sum(np.arange(1, x.size + 1) * x ** 2.0)

    @staticmethod
    def trid(x):
        return np.sum((x - 1.0) ** 2.0) - np.sum(x[1:] * x[:-1])

    @staticmethod
    def zakharov(x):
        weighted_sum = 0.5 * np.sum(np.arange(1, x.size + 1) * x)
        return np.inner(x, x) + weighted_sum ** 2.0 + weighted_sum ** 4.0

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
            'perm0d': np.ones(n),
            'permd': np.ones(n),
            'powell': np.ones(n),
            'power': np.ones(n),
            'rosen': np.zeros(n),
            'rothyp': np.ones(n),
            'sphere': np.ones(n),
            'sumpow': np.ones(n),
            'sumsqu': np.ones(n),
            'trid': np.zeros(n),
            'zakharov': np.ones(n),
        }.get(fun)

    @pytest.fixture
    def xl(self, fun, n):
        return {
            'arwhead': -5.12 * np.ones(n),
            'perm0d': -n * np.ones(n),
            'permd': -n * np.ones(n),
            'powell': -4.0 * np.ones(n),
            'power': -5.12 * np.ones(n),
            'rosen': -2.048 * np.ones(n),
            'rothyp': -65.536 * np.ones(n),
            'sphere': -5.12 * np.ones(n),
            'sumpow': -np.ones(n),
            'sumsqu': -5.12 * np.ones(n),
            'trid': -n ** 2 * np.ones(n),
            'zakharov': -5.0 * np.ones(n),
        }.get(fun)

    @pytest.fixture
    def xu(self, fun, n):
        return {
            'arwhead': 5.12 * np.ones(n),
            'perm0d': n * np.ones(n),
            'permd': n * np.ones(n),
            'powell': 3.0 * np.ones(n),
            'power': 5.12 * np.ones(n),
            'rosen': 2.048 * np.ones(n),
            'rothyp': 65.536 * np.ones(n),
            'sphere': 5.12 * np.ones(n),
            'sumpow': np.ones(n),
            'sumsqu': 5.12 * np.ones(n),
            'trid': n ** 2 * np.ones(n),
            'zakharov': 10.0 * np.ones(n),
        }.get(fun)

    @pytest.fixture
    def x_best(self, fun, n):
        return {
            'arwhead': np.r_[np.ones(n - 1), 0.0],
            'perm0d': 1.0 / np.arange(1, n + 1),
            'permd': np.arange(1, n + 1),
            'powell': np.zeros(n),
            'power': np.zeros(n),
            'rosen': np.ones(n),
            'rothyp': np.zeros(n),
            'sphere': np.zeros(n),
            'sumpow': np.zeros(n),
            'sumsqu': np.zeros(n),
            'trid': np.arange(1, n + 1) * np.arange(n, 0, -1),
            'zakharov': np.zeros(n),
        }.get(fun)

    @pytest.fixture
    def fun_best(self, fun, n):
        return {
            'arwhead': 0.0,
            'perm0d': 0.0,
            'permd': 0.0,
            'powell': 0.0,
            'power': 0.0,
            'rosen': 0.0,
            'rothyp': 0.0,
            'sphere': 0.0,
            'sumpow': 0.0,
            'sumsqu': 0.0,
            'trid': -n * (n + 4) * (n - 1) / 6,
            'zakharov': 0.0,
        }.get(fun)


class TestMinimizeUnconstrained(TestMinimizeBase):

    @pytest.mark.parametrize('fun', ['rothyp', 'sphere', 'sumsqu', 'trid'])
    @pytest.mark.parametrize('n', [2, 5, 10])
    def test_simple(self, fun, n, x0, x_best, fun_best):
        res = minimize(fun=getattr(self, fun), x0=x0, options={'debug': True})
        self.assert_result(res, n, x_best, fun_best, 0, 0.0)

    @pytest.mark.parametrize('fun', ['rothyp', 'sphere', 'sumsqu', 'trid'])
    @pytest.mark.parametrize('n', [2, 5, 10])
    def test_target(self, fun, n, x0, x_best, fun_best):
        res = minimize(fun=getattr(self, fun), x0=x0, options={'target': fun_best + 1.0, 'debug': True})
        self.assert_result(res, n, x_best, fun_best, 1, 0.0)
