import re
from abc import ABC

import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_warns

from cobyqa import minimize


class TestBase(ABC):

    @staticmethod
    def arwhead(x):
        x = np.asarray(x)
        fvx = (x[:-1] ** 2.0 + x[-1] ** 2.0) ** 2.0 - 4.0 * x[:-1] + 3.0
        return np.sum(fvx)

    @staticmethod
    def perm0d(x):
        x = np.asarray(x)
        n = x.size
        nrg = np.arange(1, n + 1)
        fx = 0.0
        for i in range(n):
            fx += np.sum((nrg + 10.0) * (x ** i - 1.0 / nrg ** i)) ** 2.0
        return fx

    @staticmethod
    def permd(x):
        x = np.asarray(x)
        n = x.size
        nrg = np.arange(1, n + 1)
        fx = 0.0
        for i in range(n):
            fx += np.sum((nrg ** i + 0.5) * ((x / nrg) ** i - 1.0)) ** 2.0
        return fx

    @staticmethod
    def powell(x):
        x = np.asarray(x)
        n = x.size
        fx = 10.0 * (x[-4] - x[-1]) ** 4.0 if n % 4 == 0 else 0.0
        for i in range(n - 1):
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
        x = np.asarray(x)
        n = x.size
        return np.sum(np.arange(1, n + 1) * x ** 2.0)

    @staticmethod
    def rosen(x):
        x = np.asarray(x)
        fvx = 100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1.0 - x[:-1]) ** 2.0
        return np.sum(fvx)

    @staticmethod
    def rothyp(x):
        x = np.asarray(x)
        n = x.size
        return np.sum(np.arange(n, 0, -1) * x ** 2.0)

    @staticmethod
    def sphere(x):
        x = np.asarray(x)
        return np.inner(x, x)

    @staticmethod
    def stybtang(x):
        x = np.asarray(x)
        return 0.5 * np.sum(x ** 4.0 - 16.0 * x ** 2.0 + 5.0 * x)

    @staticmethod
    def sumpow(x):
        x = np.asarray(x)
        n = x.size
        return np.sum(np.abs(x) ** np.arange(2, n + 2))

    @staticmethod
    def trid(x):
        x = np.asarray(x)
        return np.sum((x - 1.0) ** 2.0) - np.sum(x[1:] * x[:-1])

    @staticmethod
    def zakharov(x):
        x = np.asarray(x)
        n = x.size
        swi = 0.5 * np.sum(np.arange(1, n + 1) * x)
        return np.inner(x, x) + swi ** 2.0 + swi ** 4.0

    @staticmethod
    def _unstable(fun, x):
        test = np.cos(1e12 * np.sum(x))
        if test >= 0.95:
            return np.nan
        else:
            return fun(x)

    def __getattr__(self, item):
        unstable = re.compile('(?P<fun>[a-z]+)_unstable')
        stable = unstable.match(item)
        if stable:
            try:
                fun = getattr(self, stable.group('fun'))
                return lambda x: self._unstable(fun, x)
            except AttributeError as exc:
                raise AttributeError(item) from exc
        else:
            raise AttributeError(item)

    @staticmethod
    def assert_optimize(res, n, x_sol, f_sol, status=0, maxcv=False):
        assert_(res.nfev <= 500 * n)
        assert_(res.status == status)
        assert_(res.success, res.message)
        if status == 0:
            assert_allclose(res.x, x_sol, atol=1e-3)
            assert_allclose(res.fun, f_sol, atol=1e-3)
            if maxcv:
                assert_allclose(res.maxcv, 0.0, atol=1e-3)

    @pytest.fixture
    def x0(self, fun, n):
        return {
            'arwhead': np.zeros(n),
            'perm0d': np.ones(n),
            'permd': np.ones(n),
            'powell': np.ones(n),
            'power': np.ones(n),
            'rosen': 0.5 * np.ones(n),
            'rothyp': np.ones(n),
            'sphere': np.ones(n),
            'stybtang': -np.ones(n),
            'sumpow': np.ones(n),
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
            'stybtang': -5.0 * np.ones(n),
            'sumpow': -np.ones(n),
            'trid': -n ** 2.0 * np.ones(n),
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
            'stybtang': 5.0 * np.ones(n),
            'sumpow': np.ones(n),
            'trid': n ** 2.0 * np.ones(n),
            'zakharov': 10.0 * np.ones(n),
        }.get(fun)

    @pytest.fixture
    def x_sol(self, fun, n):
        return {
            'arwhead': np.r_[np.ones(n - 1), 0.0],
            'perm0d': 1.0 / np.arange(1, n + 1),
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
            'arwhead': 0.0,
            'perm0d': 0.0,
            'permd': 0.0,
            'powell': 0.0,
            'power': 0.0,
            'rosen': 0.0,
            'rothyp': 0.0,
            'sphere': 0.0,
            'sumpow': 0.0,
            'stybtang': -39.1661657037714 * n,
            'trid': -n * (n + 4.0) * (n - 1.0) / 6.0,
            'zakharov': 0.0,
        }.get(fun)


class TestUnconstrained(TestBase):

    def get_npt(self, n, npt):
        npt_max = (n + 1) * (n + 2) // 2
        npt_min = min(n + 2, npt_max)
        return max(npt_min, min(npt_max, npt))

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['arwhead', 'power', 'rosen', 'rothyp',
                                     'sphere', 'stybtang', 'trid'])
    def test_simple(self, fun, n, x0, x_sol, f_sol):
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            options={'debug': True},
        )
        self.assert_optimize(res, n, x_sol, f_sol)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['arwhead', 'power', 'rosen', 'rothyp',
                                     'sphere', 'stybtang', 'trid'])
    def test_unstable(self, fun, n, x0, x_sol, f_sol):
        minimize(
            fun=getattr(self, fun + '_unstable'),
            x0=x0,
        )

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['arwhead', 'power', 'rosen', 'rothyp',
                                     'sphere', 'stybtang', 'trid'])
    def test_target(self, fun, n, x0, x_sol, f_sol):
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            options={'debug': True, 'target': f_sol + 1.0},
        )
        self.assert_optimize(res, n, x_sol, f_sol, 1)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['arwhead', 'power', 'rosen', 'rothyp',
                                     'sphere', 'stybtang', 'trid'])
    def test_options(self, fun, n, x0, x_sol, f_sol):
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
        )
        self.assert_optimize(res, n, x_sol, f_sol)
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            options={'rhobeg': 1.5, 'debug': True},
        )
        self.assert_optimize(res, n, x_sol, f_sol)
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            options={'rhoend': 1e-7, 'debug': True},
        )
        self.assert_optimize(res, n, x_sol, f_sol)
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            options={'npt': self.get_npt(n, n + 3), 'debug': True},
        )
        self.assert_optimize(res, n, x_sol, f_sol)
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            options={'npt': self.get_npt(n, n + 3), 'debug': True},
        )
        self.assert_optimize(res, n, x_sol, f_sol)
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            options={'maxfev': 300 * n, 'debug': True},
        )
        self.assert_optimize(res, n, x_sol, f_sol)
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            options={'maxiter': 500 * n, 'debug': True},
        )
        self.assert_optimize(res, n, x_sol, f_sol)
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            options={'disp': True, 'debug': True},
        )
        self.assert_optimize(res, n, x_sol, f_sol)
        with assert_warns(RuntimeWarning):
            minimize(
                fun=getattr(self, fun),
                x0=x0,
                options={'npt': n},
            )
        with assert_warns(RuntimeWarning):
            minimize(
                fun=getattr(self, fun),
                x0=x0,
                options={'npt': (n + 2) ** 2.0 // 2},
            )
        with assert_warns(RuntimeWarning):
            minimize(
                fun=getattr(self, fun),
                x0=x0,
                options={'maxfev': n},
            )
        with assert_warns(RuntimeWarning):
            minimize(
                fun=getattr(self, fun),
                x0=x0,
                options={'rhobeg': 1e-3, 'rhoend': 1e-2},
            )


class TestBoundConstrained(TestBase):

    @pytest.fixture
    def xl2(self, fun, n):
        return {
            'arwhead': -0.5 * np.ones(n),
            'power': 0.5 * np.ones(n),
            'sphere': np.arange(n),
        }.get(fun)

    @pytest.fixture
    def xu2(self, fun, n):
        return {
            'arwhead': 0.5 * np.ones(n),
            'power': np.ones(n),
            'sphere': 2.0 * n * np.ones(n),
        }.get(fun)

    @pytest.fixture
    def x_sol2(self, fun, n):
        return {
            'arwhead': np.r_[0.5 * np.ones(n - 1), 0.0],
            'power': 0.5 * np.ones(n),
            'sphere': np.arange(n),
        }.get(fun)

    @pytest.fixture
    def f_sol2(self, fun, n):
        return {
            'arwhead': 1.0625 * (n - 1),
            'power': 0.25 * (n * (n + 1) // 2),
            'sphere': n * (n - 1) * (2 * n - 1) // 6,
        }.get(fun)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['arwhead', 'power', 'rosen', 'rothyp',
                                     'sphere', 'stybtang', 'trid'])
    def test_simple(self, fun, n, x0, xl, xu, x_sol, f_sol):
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            xl=xl,
            xu=xu,
            options={'debug': True},
        )
        self.assert_optimize(res, n, x_sol, f_sol, maxcv=True)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['arwhead', 'power', 'rosen', 'rothyp',
                                     'sphere', 'stybtang', 'trid'])
    def test_relax(self, fun, n, x0, xl, xu, x_sol, f_sol):
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            xl=xl,
            xu=xu,
            options={'debug': True, 'respect_bounds': False},
        )
        self.assert_optimize(res, n, x_sol, f_sol, maxcv=True)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['arwhead', 'power', 'rosen', 'rothyp',
                                     'sphere', 'stybtang', 'trid'])
    def test_unstable(self, fun, n, x0, xl, xu, x_sol, f_sol):
        minimize(
            fun=getattr(self, fun + '_unstable'),
            x0=x0,
            xl=xl,
            xu=xu,
        )

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['arwhead', 'power', 'rosen', 'rothyp',
                                     'sphere', 'stybtang', 'trid'])
    def test_target(self, fun, n, x0, xl, xu, x_sol, f_sol):
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            xl=xl,
            xu=xu,
            options={'debug': True, 'target': f_sol + 1.0},
        )
        self.assert_optimize(res, n, x_sol, f_sol, 1, True)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['arwhead', 'power', 'sphere'])
    def test_restricted(self, fun, n, x0, xl2, xu2, x_sol2, f_sol2):
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            xl=xl2,
            xu=xu2,
            options={'debug': True},
        )
        self.assert_optimize(res, n, x_sol2, f_sol2, maxcv=True)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['arwhead', 'power', 'sphere'])
    def test_restricted_relax(self, fun, n, x0, xl2, xu2, x_sol2, f_sol2):
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            xl=xl2,
            xu=xu2,
            options={'debug': True, 'respect_bounds': False},
        )
        self.assert_optimize(res, n, x_sol2, f_sol2, maxcv=True)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['arwhead', 'power', 'rosen', 'rothyp',
                                     'sphere', 'stybtang', 'trid'])
    def test_fixed(self, fun, n, x0, xl):
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            xl=xl,
            xu=xl,
            options={'debug': True},
        )
        assert_(res.status == 9)
        assert_(res.success, res.message)


class TestLinearEqualityConstrained(TestBase):

    @pytest.fixture
    def aeq(self, fun, n):
        return {
            'arwhead': np.c_[np.ones((1, n - 1)), 0.0],
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
            'arwhead': np.r_[(1.0 / (n - 1.0)) * np.ones(n - 1), 0.0],
            'power': (1.0 / np.sum(1.0 / nrg)) / nrg,
            'sphere': (1.0 / n) * np.ones(n),
        }.get(fun)

    @pytest.fixture
    def f_sol(self, fun, n):
        return {
            'arwhead': 1.0 / (n - 1.0) ** 3.0 + 3.0 * (n - 1.0) - 4.0,
            'power': 1.0 / np.sum(1.0 / np.arange(1, n + 1)),
            'sphere': 1.0 / n
        }.get(fun)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['arwhead', 'power', 'sphere'])
    def test_simple(self, fun, n, x0, xl, xu, aeq, beq, x_sol, f_sol):
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            Aeq=aeq,
            beq=beq,
            options={'debug': True},
        )
        self.assert_optimize(res, n, x_sol, f_sol, maxcv=True)

        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            Aeq=aeq,
            beq=beq,
            options={'debug': True},
            exact_normal_step=True,
        )
        self.assert_optimize(res, n, x_sol, f_sol, maxcv=True)

        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            xl=xl,
            xu=xu,
            Aeq=aeq,
            beq=beq,
            options={'debug': True},
        )
        self.assert_optimize(res, n, x_sol, f_sol, maxcv=True)

        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            xl=xl,
            xu=xu,
            Aeq=aeq,
            beq=beq,
            options={'debug': True},
            exact_normal_step=True,
        )
        self.assert_optimize(res, n, x_sol, f_sol, maxcv=True)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['arwhead', 'power', 'sphere'])
    def test_unstable(self, fun, n, x0, xl, xu, aeq, beq, x_sol, f_sol):
        minimize(
            fun=getattr(self, fun + '_unstable'),
            x0=x0,
            Aeq=aeq,
            beq=beq,
        )

        minimize(
            fun=getattr(self, fun + '_unstable'),
            x0=x0,
            xl=xl,
            xu=xu,
            Aeq=aeq,
            beq=beq,
        )

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['arwhead', 'power', 'sphere'])
    def test_target(self, fun, n, x0, xl, xu, aeq, beq, x_sol, f_sol):
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            Aeq=aeq,
            beq=beq,
            options={'debug': True, 'target': f_sol + 1.0},
        )
        self.assert_optimize(res, n, x_sol, f_sol, 1, True)

        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            xl=xl,
            xu=xu,
            Aeq=aeq,
            beq=beq,
            options={'debug': True, 'target': f_sol + 1.0},
        )
        self.assert_optimize(res, n, x_sol, f_sol, 1, True)


class TestLinearInequalityConstrained(TestBase):

    @pytest.fixture
    def aub(self, fun, n):
        return {
            'power': -np.ones((1, n)),
            'sphere': -np.ones((1, n)),
        }.get(fun)

    @pytest.fixture
    def bub(self, fun):
        return {
            'power': -np.ones(1),
            'sphere': -np.ones(1),
        }.get(fun)

    @pytest.fixture
    def x_sol(self, fun, n):
        nrg = np.arange(1, n + 1)
        return {
            'power': (1.0 / np.sum(1.0 / nrg)) / nrg,
            'sphere': (1.0 / n) * np.ones(n),
        }.get(fun)

    @pytest.fixture
    def f_sol(self, fun, n):
        return {
            'power': 1.0 / np.sum(1.0 / np.arange(1, n + 1)),
            'sphere': 1.0 / n
        }.get(fun)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['power', 'sphere'])
    def test_simple(self, fun, n, x0, xl, xu, aub, bub, x_sol, f_sol):
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            Aub=aub,
            bub=bub,
            options={'debug': True},
        )
        self.assert_optimize(res, n, x_sol, f_sol, maxcv=True)

        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            Aub=aub,
            bub=bub,
            options={'debug': True},
            exact_normal_step=True,
        )
        self.assert_optimize(res, n, x_sol, f_sol, maxcv=True)

        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            xl=xl,
            xu=xu,
            Aub=aub,
            bub=bub,
            options={'debug': True},
        )
        self.assert_optimize(res, n, x_sol, f_sol, maxcv=True)

        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            xl=xl,
            xu=xu,
            Aub=aub,
            bub=bub,
            options={'debug': True},
            exact_normal_step=True,
        )
        self.assert_optimize(res, n, x_sol, f_sol, maxcv=True)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['power', 'sphere'])
    def test_unstable(self, fun, n, x0, xl, xu, aub, bub, x_sol, f_sol):
        minimize(
            fun=getattr(self, fun + '_unstable'),
            x0=x0,
            Aub=aub,
            bub=bub,
        )

        minimize(
            fun=getattr(self, fun + '_unstable'),
            x0=x0,
            xl=xl,
            xu=xu,
            Aub=aub,
            bub=bub,
        )

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['power', 'sphere'])
    def test_target(self, fun, n, x0, xl, xu, aub, bub, x_sol, f_sol):
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            Aub=aub,
            bub=bub,
            options={'debug': True, 'target': f_sol + 1.0},
        )
        self.assert_optimize(res, n, x_sol, f_sol, 1, True)

        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            xl=xl,
            xu=xu,
            Aub=aub,
            bub=bub,
            options={'debug': True, 'target': f_sol + 1.0},
        )
        self.assert_optimize(res, n, x_sol, f_sol, 1, True)


class TestNonlinearEqualityConstrained(TestBase):

    @staticmethod
    def ceq_base(fun):
        return lambda x: {
            'power': np.sum(x) - 1.0,
            'sphere': np.sum(x) - 1.0,
        }.get(fun)

    @pytest.fixture
    def ceq(self, fun):
        return self.ceq_base(fun)

    @pytest.fixture
    def ceq_unstable(self, fun):
        return lambda x: self._unstable(self.ceq_base(fun), x)

    @pytest.fixture
    def x_sol(self, fun, n):
        nrg = np.arange(1, n + 1)
        return {
            'power': (1.0 / np.sum(1.0 / nrg)) / nrg,
            'sphere': (1.0 / n) * np.ones(n),
        }.get(fun)

    @pytest.fixture
    def f_sol(self, fun, n):
        return {
            'power': 1.0 / np.sum(1.0 / np.arange(1, n + 1)),
            'sphere': 1.0 / n
        }.get(fun)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['power', 'sphere'])
    def test_simple(self, fun, n, x0, xl, xu, ceq, x_sol, f_sol):
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            ceq=ceq,
            options={'debug': True},
        )
        self.assert_optimize(res, n, x_sol, f_sol, maxcv=True)

        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            ceq=ceq,
            options={'debug': True},
            exact_normal_step=True,
        )
        self.assert_optimize(res, n, x_sol, f_sol, maxcv=True)

        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            xl=xl,
            xu=xu,
            ceq=ceq,
            options={'debug': True},
        )
        self.assert_optimize(res, n, x_sol, f_sol, maxcv=True)

        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            xl=xl,
            xu=xu,
            ceq=ceq,
            options={'debug': True},
            exact_normal_step=True,
        )
        self.assert_optimize(res, n, x_sol, f_sol, maxcv=True)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['power', 'sphere'])
    def test_unstable(self, fun, n, x0, xl, xu, ceq_unstable, x_sol, f_sol):
        minimize(
            fun=getattr(self, fun + '_unstable'),
            x0=x0,
            ceq=ceq_unstable,
        )

        minimize(
            fun=getattr(self, fun + '_unstable'),
            x0=x0,
            xl=xl,
            xu=xu,
            ceq=ceq_unstable,
        )

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['power', 'sphere'])
    def test_target(self, fun, n, x0, xl, xu, ceq, x_sol, f_sol):
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            ceq=ceq,
            options={'debug': True, 'target': f_sol + 1.0},
        )
        self.assert_optimize(res, n, x_sol, f_sol, 1, True)

        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            xl=xl,
            xu=xu,
            ceq=ceq,
            options={'debug': True, 'target': f_sol + 1.0},
        )
        self.assert_optimize(res, n, x_sol, f_sol, 1, True)


class TestNonlinearInequalityConstrained(TestBase):

    @staticmethod
    def cub_base(fun):
        return lambda x: {
            'power': 1.0 - np.sum(x),
            'sphere': 1.0 - np.sum(x),
        }.get(fun)

    @pytest.fixture
    def cub(self, fun):
        return self.cub_base(fun)

    @pytest.fixture
    def cub_unstable(self, fun):
        return lambda x: self._unstable(self.cub_base(fun), x)

    @pytest.fixture
    def x_sol(self, fun, n):
        nrg = np.arange(1, n + 1)
        return {
            'power': (1.0 / np.sum(1.0 / nrg)) / nrg,
            'sphere': (1.0 / n) * np.ones(n),
        }.get(fun)

    @pytest.fixture
    def f_sol(self, fun, n):
        return {
            'power': 1.0 / np.sum(1.0 / np.arange(1, n + 1)),
            'sphere': 1.0 / n
        }.get(fun)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['power', 'sphere'])
    def test_simple(self, fun, n, x0, xl, xu, cub, x_sol, f_sol):
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            cub=cub,
            options={'debug': True},
        )
        self.assert_optimize(res, n, x_sol, f_sol, maxcv=True)

        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            cub=cub,
            options={'debug': True},
            exact_normal_step=True,
        )
        self.assert_optimize(res, n, x_sol, f_sol, maxcv=True)

        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            xl=xl,
            xu=xu,
            cub=cub,
            options={'debug': True},
        )
        self.assert_optimize(res, n, x_sol, f_sol, maxcv=True)

        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            xl=xl,
            xu=xu,
            cub=cub,
            options={'debug': True},
            exact_normal_step=True,
        )
        self.assert_optimize(res, n, x_sol, f_sol, maxcv=True)

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['power', 'sphere'])
    def test_unstable(self, fun, n, x0, xl, xu, cub_unstable, x_sol, f_sol):
        minimize(
            fun=getattr(self, fun + '_unstable'),
            x0=x0,
            cub=cub_unstable,
        )

        minimize(
            fun=getattr(self, fun + '_unstable'),
            x0=x0,
            xl=xl,
            xu=xu,
            cub=cub_unstable,
        )

    @pytest.mark.parametrize('n', [2, 5, 10])
    @pytest.mark.parametrize('fun', ['power', 'sphere'])
    def test_target(self, fun, n, x0, xl, xu, cub, x_sol, f_sol):
        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            cub=cub,
            options={'debug': True, 'target': f_sol + 1.0},
        )
        self.assert_optimize(res, n, x_sol, f_sol, 1, True)

        res = minimize(
            fun=getattr(self, fun),
            x0=x0,
            xl=xl,
            xu=xu,
            cub=cub,
            options={'debug': True, 'target': f_sol + 1.0},
        )
        self.assert_optimize(res, n, x_sol, f_sol, 1, True)
