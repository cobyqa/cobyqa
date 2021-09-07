from abc import ABC

import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose

from cobyqa import minimize


class TestBase(ABC):

    @staticmethod
    def arwhead(x):
        """
        The ARWHEAD function.
        """
        x = np.asarray(x)
        return np.sum((x[:-1] ** 2. + x[-1] ** 2.) ** 2. - 4. * x[:-1] + 3.)

    @staticmethod
    def dixonpr(x):
        """
        The Dixon-Price function.
        """
        x = np.asarray(x)
        n = x.size
        ssq = np.sum(np.arange(2, n + 1) * (2. * x[1:] ** 2. - x[:-1]) ** 2.)
        return (x[0] - 1.) ** 2. + ssq

    @staticmethod
    def perm0d(x):
        """
        The Perm 0, d function.
        """
        x = np.asarray(x)
        n = x.size
        nrg = np.arange(1, n + 1)
        fx = 0.
        for i in range(n):
            fx += np.sum((nrg + 10.) * (x ** i - 1. / nrg ** i)) ** 2.
        return fx

    @staticmethod
    def permd(x):
        """
        The Perm d function.
        """
        x = np.asarray(x)
        n = x.size
        nrg = np.arange(1, n + 1)
        fx = 0.
        for i in range(n):
            fx += np.sum((nrg ** i + .5) * ((x / nrg) ** i - 1.)) ** 2.
        return fx

    @staticmethod
    def powell(x):
        """
        The Powell function.
        """
        x = np.asarray(x)
        n = x.size
        fx = 1e1 * (x[-4] - x[-1]) ** 4. if n % 4 == 0 else 0.
        for i in range(n - 1):
            if i % 4 == 0:
                fx += (x[i] + 10. * x[i + 1]) ** 2.
            elif i % 4 == 1:
                fx += (x[i] - 2. * x[i + 1]) ** 4.
            elif i % 4 == 2:
                fx += 5. * (x[i] - x[i + 1]) ** 2.
            else:
                fx += 10. * (x[i - 3] - x[i]) ** 4.
        return fx

    @staticmethod
    def power(x):
        """
        The sum of squares function.
        """
        x = np.asarray(x)
        n = x.size
        return np.sum(np.arange(1, n + 1) * x ** 2.)

    @staticmethod
    def rosen(x):
        """
        The Rosenbrock function.
        """
        x = np.asarray(x)
        return np.sum(100. * (x[1:] - x[:-1] ** 2.) ** 2. + (1. - x[:-1]) ** 2.)

    @staticmethod
    def rothyp(x):
        """
        The rotated hyper-ellipsoid function.
        """
        x = np.asarray(x)
        n = x.size
        return np.sum(np.arange(n, 0, -1) * x ** 2.)

    @staticmethod
    def sphere(x):
        """
        The sphere function.
        """
        x = np.asarray(x)
        return np.inner(x, x)

    @staticmethod
    def stybtang(x):
        """
        The Styblinski-Tang function.
        """
        x = np.asarray(x)
        return .5 * np.sum(x ** 4. - 16. * x ** 2. + 5. * x)

    @staticmethod
    def sumpow(x):
        """
        The sum of different powers function.
        """
        x = np.asarray(x)
        n = x.size
        return np.sum(np.abs(x) ** np.arange(2, n + 2))

    @staticmethod
    def trid(x):
        """
        The Trid function.
        """
        x = np.asarray(x)
        return np.sum((x - 1.) ** 2.) - np.sum(x[1:] * x[:-1])

    @staticmethod
    def zakharov(x):
        """
        The Zakharov function.
        """
        x = np.asarray(x)
        n = x.size
        swi = .5 * np.sum(np.arange(1, n + 1) * x)
        return np.inner(x, x) + swi ** 2. + swi ** 4.

    @staticmethod
    def assert_(res, n, x_sol, f_sol, maxcv=False):
        assert_allclose(res.x, x_sol, atol=1e-4)
        assert_allclose(res.fun, f_sol, atol=1e-4)
        assert_(res.nfev <= 500 * n)
        assert_(res.status in [0, 1])
        assert_(res.success, res.message)
        if maxcv:
            assert_allclose(res.maxcv, 0., atol=1e-4)

    @pytest.fixture
    def x0(self, fun, n):
        return {
            'arwhead': np.zeros(n),
            'dixonpr': np.ones(n),
            'perm0d': np.ones(n),
            'permd': np.ones(n),
            'powell': np.ones(n),
            'power': np.ones(n),
            'rosen': np.zeros(n),
            'rothyp': np.ones(n),
            'sphere': np.ones(n),
            'stybtang': np.zeros(n),
            'sumpow': np.ones(n),
            'trid': np.zeros(n),
            'zakharov': np.ones(n),
        }.get(fun)

    @pytest.fixture
    def xl(self, fun, n):
        return {
            'dixonpr': -10. * np.ones(n),
            'perm0d': -n * np.ones(n),
            'permd': -n * np.ones(n),
            'powell': -4. * np.ones(n),
            'power': -5.12 * np.ones(n),
            'rosen': -2.048 * np.ones(n),
            'rothyp': -65.536 * np.ones(n),
            'sphere': -5.12 * np.ones(n),
            'stybtang': -5. * np.ones(n),
            'sumpow': -np.ones(n),
            'trid': -n ** 2. * np.ones(n),
            'zakharov': -5. * np.ones(n),
        }.get(fun)

    @pytest.fixture
    def xu(self, fun, n):
        return {
            'dixonpr': 10. * np.ones(n),
            'perm0d': n * np.ones(n),
            'permd': n * np.ones(n),
            'powell': 3. * np.ones(n),
            'power': 5.12 * np.ones(n),
            'rosen': 2.048 * np.ones(n),
            'rothyp': 65.536 * np.ones(n),
            'sphere': 5.12 * np.ones(n),
            'stybtang': 5. * np.ones(n),
            'sumpow': np.ones(n),
            'trid': n ** 2. * np.ones(n),
            'zakharov': 10. * np.ones(n),
        }.get(fun)

    @pytest.fixture
    def x_sol(self, fun, n):
        return {
            'arwhead': np.r_[np.ones(n - 1), 0.],
            'dixonpr': 2. ** (2. ** (1. - np.arange(1, n + 1)) - 1.),
            'perm0d': 1. / np.arange(1, n + 1),
            'permd': np.arange(1, n + 1),
            'powell': np.zeros(n),
            'power': np.zeros(n),
            'rosen': np.ones(n),
            'rothyp': np.zeros(n),
            'sphere': np.zeros(n),
            'stybtang': -2.90353402777118 * np.ones(n),
            'sumpow': np.zeros(n),
            'trid': np.arange(1, n + 1) * np.arange(n, 0, -1),
            'zakharov': np.zeros(n),
        }.get(fun)

    @pytest.fixture
    def f_sol(self, fun, n):
        return {
            'arwhead': 0.,
            'dixonpr': 0.,
            'perm0d': 0.,
            'permd': 0.,
            'powell': 0.,
            'power': 0.,
            'rosen': 0.,
            'rothyp': 0.,
            'sphere': 0.,
            'sumpow': 0.,
            'stybtang': -39.1661657037714 * n,
            'trid': -n * (n + 4.) * (n - 1.) / 6.,
            'zakharov': 0.,
        }.get(fun)


class TestUnconstrained(TestBase):

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['arwhead', 'dixonpr', 'power', 'rosen',
                                     'rothyp', 'sphere', 'stybtang', 'trid'])
    def test_simple(self, fun, n, x0, x_sol, f_sol):
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            options={'debug': True},
        )
        self.assert_(res, n, x_sol, f_sol)


class TestBoundConstrained(TestBase):

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['dixonpr', 'power', 'rosen', 'rothyp',
                                     'sphere', 'stybtang', 'trid'])
    def test_simple(self, fun, n, x0, xl, xu, x_sol, f_sol):
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            xl=xl,
            xu=xu,
            options={'debug': True},
        )
        self.assert_(res, n, x_sol, f_sol, maxcv=True)


class TestLinearEqualityConstrained(TestBase):

    @pytest.fixture
    def aeq(self, fun, n):
        return {
            'arwhead': np.c_[np.ones((1, n - 1)), 0.],
            'power': np.ones((1, n)),
            'sphere': np.ones((1, n)),
        }.get(fun)

    @pytest.fixture
    def beq(self, fun):
        return {
            'arwhead': np.ones(1),
            'power': np.ones(1),
            'sphere': np.ones(1),
        }.get(fun)

    @pytest.fixture
    def x_sol(self, fun, n):
        nrg = np.arange(1, n + 1)
        return {
            'arwhead': np.r_[(1. / (n - 1.)) * np.ones(n - 1), 0.],
            'power': (1. / np.sum(1. / nrg)) / nrg,
            'sphere': (1. / n) * np.ones(n),
        }.get(fun)

    @pytest.fixture
    def f_sol(self, fun, n):
        return {
            'arwhead': 1. / (n - 1.) ** 3. + 3. * (n - 1.) - 4.,
            'power': 1. / np.sum(1. / np.arange(1, n + 1)),
            'sphere': 1. / n
        }.get(fun)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['arwhead', 'power', 'sphere'])
    def test_simple(self, fun, n, x0, aeq, beq, x_sol, f_sol):
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            Aeq=aeq,
            beq=beq,
            options={'debug': True},
        )
        self.assert_(res, n, x_sol, f_sol, maxcv=True)

