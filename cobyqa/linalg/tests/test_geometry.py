import numpy as np
import pytest
from numpy.testing import assert_, assert_raises

from cobyqa.linalg import bvcs, bvlag
from cobyqa.tests import assert_array_less_equal, assert_dtype_equal


class TestBVCS:

    @staticmethod
    def curv(x, Hq):
        return np.inner(x, np.dot(Hq, x))

    @pytest.mark.parametrize('n', [1, 5, 10, 100])
    def test_simple(self, n):
        rng = np.random.default_rng(n)
        kopt = rng.integers(2 * n + 1)
        gq = rng.standard_normal(n)
        Hq = rng.standard_normal((n, n))
        Hq = .5 * (Hq + Hq.T)
        xl = rng.standard_normal(n)
        xu = rng.standard_normal(n)
        xl, xu = np.minimum(xl, xu), np.maximum(xl, xu)
        xpt = rng.uniform(xl, xu, (2 * n + 1, n))
        delta = rng.uniform(1e-1, 1.)
        step, cauchy = bvcs(xpt, kopt, gq, self.curv, (Hq,), xl, xu, delta)
        assert_dtype_equal(step, float)
        assert_(step.ndim == 1)
        assert_(step.size == n)
        assert_(isinstance(cauchy, float))

        # Ensure the feasibility of the output.
        eps = np.finfo(float).eps
        tol = 10. * eps * n
        bdtol = tol * np.max(np.abs(np.r_[xl, xu]), initial=1.)
        assert_array_less_equal(xl - xpt[kopt, :] - step, bdtol)
        assert_array_less_equal(xpt[kopt, :] + step - xu, bdtol)
        assert_(cauchy >= 0.)

    def test_exceptions(self):
        xpt = np.ones((11, 5), dtype=float)
        kopt = 1
        gq = np.ones(5, dtype=float)
        Hq = np.ones((5, 5), dtype=float)
        xl = np.zeros(5, dtype=float)
        xu = 2. * np.ones(5, dtype=float)
        delta = 1.
        with assert_raises(AssertionError):
            bvcs(xpt, kopt, gq, self.curv, (Hq,), xl, xu, -1.)
        xpt[kopt, 2] = 2.1
        with assert_raises(AssertionError):
            bvcs(xpt, kopt, gq, self.curv, (Hq,), xl, xu, delta)
        xpt[kopt, 2], xl[2], xu[2] = 1., 1.1, 0.9
        with assert_raises(AssertionError):
            bvcs(xpt, kopt, gq, self.curv, (Hq,), xl, xu, delta)


class TestBVLAG:

    @pytest.mark.parametrize('n', [1, 5, 10, 100])
    def test_simple(self, n):
        rng = np.random.default_rng(n)
        kopt, klag = rng.choice(2 * n + 1, 2, replace=False)
        gq = rng.standard_normal(n)
        xl = rng.standard_normal(n)
        xu = rng.standard_normal(n)
        xl, xu = np.minimum(xl, xu), np.maximum(xl, xu)
        xpt = rng.uniform(xl, xu, (2 * n + 1, n))
        delta = rng.uniform(1e-1, 1.)
        alpha = rng.uniform(1e-1, 1.)
        step = bvlag(xpt, kopt, klag, gq, xl, xu, delta, alpha)
        assert_dtype_equal(step, float)
        assert_(step.ndim == 1)
        assert_(step.size == n)

        # Ensure the feasibility of the output.
        eps = np.finfo(float).eps
        tol = 10. * eps * n
        bdtol = tol * np.max(np.abs(np.r_[xl, xu]), initial=1.)
        assert_array_less_equal(xl - xpt[kopt, :] - step, bdtol)
        assert_array_less_equal(xpt[kopt, :] + step - xu, bdtol)

    def test_exceptions(self):
        xpt = np.ones((11, 5), dtype=float)
        kopt, klag = 1, 2
        gq = np.ones(5, dtype=float)
        xl = np.zeros(5, dtype=float)
        xu = 2. * np.ones(5, dtype=float)
        delta = 1.
        alpha = 1.
        with assert_raises(AssertionError):
            bvlag(xpt, kopt, klag, gq, xl, xu, -1., alpha)
        xpt[kopt, 2] = 2.1
        with assert_raises(AssertionError):
            bvlag(xpt, kopt, klag, gq, xl, xu, delta, alpha)
        xpt[kopt, 2], xl[2], xu[2] = 1., 1.1, 0.9
        with assert_raises(AssertionError):
            bvlag(xpt, kopt, klag, gq, xl, xu, -1., alpha)
