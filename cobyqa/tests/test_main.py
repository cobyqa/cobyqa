import re
from abc import ABC

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint

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
            assert_allclose(res.x, x_best, atol=1e-4 * n)
            assert_allclose(res.fun, fun_best, atol=1e-6)
        if status == 1:
            assert res.fun <= fun_best + res.maxcv
        assert res.maxcv <= maxcv

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
    def test_tough(self, fun, n, x0):
        minimize(fun=getattr(self, fun + '_tough'), x0=x0)

    @pytest.mark.parametrize('fun', ['arwhead', 'rosen', 'rothyp', 'sphere', 'sumsqu', 'trid'])
    @pytest.mark.parametrize('n', [2, 5, 10])
    def test_target(self, fun, n, x0, x_best, fun_best):
        res = minimize(fun=getattr(self, fun), x0=x0, options={'target': fun_best + 1.0, 'debug': True})
        self.assert_result(res, n, x_best, fun_best + 1.0, 1, 0.0)


class TestMinimizeBoundConstrained(TestMinimizeBase):

    @pytest.fixture
    def xl_restricted(self, fun, n):
        return {
            'arwhead': -0.5 * np.ones(n),
            'sphere': np.arange(n),
            'sumsqu': 0.5 * np.ones(n),
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
        res = minimize(fun=getattr(self, fun), x0=x0, bounds=Bounds(xl, xu), options={'debug': True})
        self.assert_result(res, n, x_best, fun_best, 0, 0.0)

    @pytest.mark.parametrize('fun', ['arwhead', 'rosen', 'rothyp', 'sphere', 'sumsqu', 'trid'])
    @pytest.mark.parametrize('n', [2, 5, 10])
    def test_tough(self, fun, n, x0, xl, xu):
        minimize(fun=getattr(self, fun + '_tough'), x0=x0, bounds=Bounds(xl, xu))

    @pytest.mark.parametrize('fun', ['arwhead', 'rosen', 'rothyp', 'sphere', 'sumsqu', 'trid'])
    @pytest.mark.parametrize('n', [2, 5, 10])
    def test_target(self, fun, n, x0, xl, xu, x_best, fun_best):
        res = minimize(fun=getattr(self, fun), x0=x0, bounds=Bounds(xl, xu), options={'target': fun_best + 1.0, 'debug': True})
        self.assert_result(res, n, x_best, fun_best + 1.0, 1, 0.0)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['arwhead', 'sphere', 'sumsqu'])
    def test_restricted(self, fun, n, x0, xl_restricted, xu_restricted, x_best_restricted, fun_best_restricted):
        res = minimize(fun=getattr(self, fun), x0=x0, bounds=Bounds(xl_restricted, xu_restricted), options={'debug': True})
        self.assert_result(res, n, x_best_restricted, fun_best_restricted, 0, 0.0)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['arwhead', 'rosen', 'rothyp', 'sphere', 'sumsqu', 'trid'])
    def test_fixed(self, fun, n, x0, xl, x_best, fun_best):
        res = minimize(fun=getattr(self, fun), x0=x0, bounds=Bounds(xl, xl), options={'debug': True})
        self.assert_result(res, n, x0, getattr(self, fun)(x0), 2, 0.0)


class TestMinimizeLinearInequalityConstrained(TestMinimizeBase):

    @pytest.fixture
    def aub(self, fun, n):
        return {
            'sphere': -np.ones((1, n)),
            'sumsqu': -np.ones((1, n)),
        }.get(fun)

    @pytest.fixture
    def bub(self, fun):
        return {
            'sphere': -np.ones(1),
            'sumsqu': -np.ones(1),
        }.get(fun)

    @pytest.fixture
    def x_best(self, fun, n):
        n_range = np.arange(1, n + 1)
        return {
            'sphere': (1.0 / n) * np.ones(n),
            'sumsqu': (1.0 / np.sum(1.0 / n_range)) / n_range,
        }.get(fun)

    @pytest.fixture
    def fun_best(self, fun, n):
        return {
            'sphere': 1.0 / n,
            'sumsqu': 1.0 / np.sum(1.0 / np.arange(1, n + 1)),
        }.get(fun)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['sphere', 'sumsqu'])
    def test_simple(self, fun, n, x0, xl, xu, aub, bub, x_best, fun_best):
        res = minimize(fun=getattr(self, fun), x0=x0, constraints=LinearConstraint(aub, -np.inf, bub), options={'debug': True})
        self.assert_result(res, n, x_best, fun_best, 0, np.sqrt(np.finfo(float).eps))

        res = minimize(fun=getattr(self, fun), x0=x0, bounds=Bounds(xl, xu), constraints=LinearConstraint(aub, -np.inf, bub), options={'debug': True})
        self.assert_result(res, n, x_best, fun_best, 0, np.sqrt(np.finfo(float).eps))

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['sphere', 'sumsqu'])
    def test_tough(self, fun, n, x0, xl, xu, aub, bub):
        minimize(fun=getattr(self, fun + '_tough'), x0=x0, constraints=LinearConstraint(aub, -np.inf, bub))

        minimize(fun=getattr(self, fun + '_tough'), x0=x0, bounds=Bounds(xl, xu), constraints=LinearConstraint(aub, -np.inf, bub))

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['sphere', 'sumsqu'])
    def test_target(self, fun, n, x0, xl, xu, aub, bub, x_best, fun_best):
        res = minimize(fun=getattr(self, fun), x0=x0, constraints=LinearConstraint(aub, -np.inf, bub), options={'debug': True, 'target': fun_best + 1.0})
        self.assert_result(res, n, x_best, fun_best + 1.0, 1, np.sqrt(np.finfo(float).eps))

        res = minimize(fun=getattr(self, fun), x0=x0, bounds=Bounds(xl, xu), constraints=LinearConstraint(aub, -np.inf, bub), options={'debug': True, 'target': fun_best + 1.0})
        self.assert_result(res, n, x_best, fun_best + 1.0, 1, np.sqrt(np.finfo(float).eps))


class TestMinimizeLinearEqualityConstrained(TestMinimizeBase):

    @pytest.fixture
    def aeq(self, fun, n):
        return {
            'arwhead': np.c_[np.ones((1, n - 1)), 0.0],
            'sphere': np.ones((1, n)),
            'sumsqu': np.ones((1, n)),
        }.get(fun)

    @pytest.fixture
    def beq(self, fun):
        return {
            'arwhead': np.ones(1),
            'sphere': np.ones(1),
            'sumsqu': np.ones(1),
        }.get(fun)

    @pytest.fixture
    def x_best(self, fun, n):
        n_range = np.arange(1, n + 1)
        return {
            'arwhead': np.r_[(1.0 / (n - 1.0)) * np.ones(n - 1), 0.0],
            'sphere': (1.0 / n) * np.ones(n),
            'sumsqu': (1.0 / np.sum(1.0 / n_range)) / n_range,
        }.get(fun)

    @pytest.fixture
    def fun_best(self, fun, n):
        return {
            'arwhead': 1.0 / (n - 1.0) ** 3.0 + 3.0 * (n - 1.0) - 4.0,
            'sphere': 1.0 / n,
            'sumsqu': 1.0 / np.sum(1.0 / np.arange(1, n + 1)),
        }.get(fun)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['arwhead', 'sphere', 'sumsqu'])
    def test_simple(self, fun, n, x0, xl, xu, aeq, beq, x_best, fun_best):
        res = minimize(fun=getattr(self, fun), x0=x0, constraints=LinearConstraint(aeq, beq, beq), options={'debug': True})
        self.assert_result(res, n, x_best, fun_best, 0, np.sqrt(np.finfo(float).eps))

        res = minimize(fun=getattr(self, fun), x0=x0, bounds=Bounds(xl, xu), constraints=LinearConstraint(aeq, beq, beq), options={'debug': True})
        self.assert_result(res, n, x_best, fun_best, 0, np.sqrt(np.finfo(float).eps))

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['arwhead', 'sphere', 'sumsqu'])
    def test_tough(self, fun, n, x0, xl, xu, aeq, beq):
        minimize(fun=getattr(self, fun + '_tough'), x0=x0, constraints=LinearConstraint(aeq, beq, beq))

        minimize(fun=getattr(self, fun + '_tough'), x0=x0, bounds=Bounds(xl, xu), constraints=LinearConstraint(aeq, beq, beq))

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['arwhead', 'sphere', 'sumsqu'])
    def test_target(self, fun, n, x0, xl, xu, aeq, beq, x_best, fun_best):
        res = minimize(fun=getattr(self, fun), x0=x0, constraints=LinearConstraint(aeq, beq, beq), options={'debug': True, 'target': fun_best + 1.0})
        self.assert_result(res, n, x_best, fun_best + 1.0, 1, np.sqrt(np.finfo(float).eps))

        res = minimize(fun=getattr(self, fun), x0=x0, bounds=Bounds(xl, xu), constraints=LinearConstraint(aeq, beq, beq), options={'debug': True, 'target': fun_best + 1.0})
        self.assert_result(res, n, x_best, fun_best + 1.0, 1, np.sqrt(np.finfo(float).eps))


class TestMinimizeNonlinearInequalityConstrained(TestMinimizeBase):

    @staticmethod
    def cub_base(fun):
        return lambda x: {
            'sphere': 1.0 - np.sum(x),
            'sumsqu': 1.0 - np.sum(x),
        }.get(fun)

    @pytest.fixture
    def cub(self, fun):
        return self.cub_base(fun)

    @pytest.fixture
    def cub_tough(self, fun):
        return lambda x: self._tough(self.cub_base(fun), x)

    @pytest.fixture
    def x_best(self, fun, n):
        n_range = np.arange(1, n + 1)
        return {
            'sphere': (1.0 / n) * np.ones(n),
            'sumsqu': (1.0 / np.sum(1.0 / n_range)) / n_range,
        }.get(fun)

    @pytest.fixture
    def fun_best(self, fun, n):
        return {
            'sphere': 1.0 / n,
            'sumsqu': 1.0 / np.sum(1.0 / np.arange(1, n + 1)),
        }.get(fun)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['sphere', 'sumsqu'])
    def test_simple(self, fun, n, x0, xl, xu, cub, x_best, fun_best):
        res = minimize(fun=getattr(self, fun), x0=x0, constraints=NonlinearConstraint(cub, -np.inf, 0.0), options={'debug': True})
        self.assert_result(res, n, x_best, fun_best, 0, np.sqrt(np.finfo(float).eps))

        res = minimize(fun=getattr(self, fun), x0=x0, bounds=Bounds(xl, xu), constraints=NonlinearConstraint(cub, -np.inf, 0.0), options={'debug': True})
        self.assert_result(res, n, x_best, fun_best, 0, np.sqrt(np.finfo(float).eps))

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['sphere', 'sumsqu'])
    def test_tough(self, fun, n, x0, xl, xu, cub_tough):
        minimize(fun=getattr(self, fun + '_tough'), x0=x0, constraints=NonlinearConstraint(cub_tough, -np.inf, 0.0))

        minimize(fun=getattr(self, fun + '_tough'), x0=x0, bounds=Bounds(xl, xu), constraints=NonlinearConstraint(cub_tough, -np.inf, 0.0))

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['sphere', 'sumsqu'])
    def test_target(self, fun, n, x0, xl, xu, cub, x_best, fun_best):
        res = minimize(fun=getattr(self, fun), x0=x0, constraints=NonlinearConstraint(cub, -np.inf, 0.0), options={'debug': True, 'target': fun_best + 1.0})
        self.assert_result(res, n, x_best, fun_best + 1.0, 1, np.sqrt(np.finfo(float).eps))

        res = minimize(fun=getattr(self, fun), x0=x0, bounds=Bounds(xl, xu), constraints=NonlinearConstraint(cub, -np.inf, 0.0), options={'debug': True, 'target': fun_best + 1.0})
        self.assert_result(res, n, x_best, fun_best + 1.0, 1, np.sqrt(np.finfo(float).eps))


class TestMinimizeNonlinearEqualityConstrained(TestMinimizeBase):

    @staticmethod
    def ceq_base(fun):
        return lambda x: {
            'sphere': np.sum(x) - 1.0,
            'sumsqu': np.sum(x) - 1.0,
        }.get(fun)

    @pytest.fixture
    def ceq(self, fun):
        return self.ceq_base(fun)

    @pytest.fixture
    def ceq_tough(self, fun):
        return lambda x: self._tough(self.ceq_base(fun), x)

    @pytest.fixture
    def x_best(self, fun, n):
        n_range = np.arange(1, n + 1)
        return {
            'sphere': (1.0 / n) * np.ones(n),
            'sumsqu': (1.0 / np.sum(1.0 / n_range)) / n_range,
        }.get(fun)

    @pytest.fixture
    def fun_best(self, fun, n):
        return {
            'sphere': 1.0 / n,
            'sumsqu': 1.0 / np.sum(1.0 / np.arange(1, n + 1)),
        }.get(fun)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['sphere', 'sumsqu'])
    def test_simple(self, fun, n, x0, xl, xu, ceq, x_best, fun_best):
        res = minimize(fun=getattr(self, fun), x0=x0, constraints=NonlinearConstraint(ceq, 0.0, 0.0), options={'debug': True})
        self.assert_result(res, n, x_best, fun_best, 0, np.sqrt(np.finfo(float).eps))

        res = minimize(fun=getattr(self, fun), x0=x0, bounds=Bounds(xl, xu), constraints=NonlinearConstraint(ceq, 0.0, 0.0), options={'debug': True})
        self.assert_result(res, n, x_best, fun_best, 0, np.sqrt(np.finfo(float).eps))

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['sphere', 'sumsqu'])
    def test_tough(self, fun, n, x0, xl, xu, ceq_tough):
        minimize(fun=getattr(self, fun + '_tough'), x0=x0, constraints=NonlinearConstraint(ceq_tough, 0.0, 0.0))

        minimize(fun=getattr(self, fun + '_tough'), x0=x0, bounds=Bounds(xl, xu), constraints=NonlinearConstraint(ceq_tough, 0.0, 0.0))

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['sphere', 'sumsqu'])
    def test_target(self, fun, n, x0, xl, xu, ceq, x_best, fun_best):
        res = minimize(fun=getattr(self, fun), x0=x0, constraints=NonlinearConstraint(ceq, 0.0, 0.0), options={'debug': True, 'target': fun_best + 1.0})
        self.assert_result(res, n, x_best, fun_best + 1.0, 1, np.sqrt(np.finfo(float).eps))

        res = minimize(fun=getattr(self, fun), x0=x0, bounds=Bounds(xl, xu), constraints=NonlinearConstraint(ceq, 0.0, 0.0), options={'debug': True, 'target': fun_best + 1.0})
        self.assert_result(res, n, x_best, fun_best + 1.0, 1, np.sqrt(np.finfo(float).eps))
