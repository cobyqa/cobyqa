import numpy as np
import pytest
from numpy.testing import assert_, assert_raises

from cobyqa.linalg import bvtcg, cpqp, lctcg, nnls
from cobyqa.tests import assert_array_less_equal, assert_dtype_equal


class TestBVTCG:

    @pytest.mark.parametrize('n', [1, 5, 10, 100])
    def test_simple(self, n):
        rng = np.random.default_rng(n)
        gq = rng.standard_normal(n)
        Hq = rng.standard_normal((n, n))
        Hq = .5 * (Hq + Hq.T)
        xl = rng.standard_normal(n)
        xu = rng.standard_normal(n)
        xl, xu = np.minimum(xl, xu), np.maximum(xl, xu)
        x0 = rng.uniform(xl, xu)
        delta = rng.uniform(0.1, 1.0)
        step = bvtcg(x0, gq, np.dot, xl, xu, delta, Hq)
        assert_dtype_equal(step, float)
        assert_(step.ndim == 1)
        assert_(step.size == n)

        # Ensure the feasibility of the output.
        tol = 10.0 * np.finfo(float).eps * n
        assert_array_less_equal(xl - x0 - step, tol)
        assert_array_less_equal(x0 + step - xu, tol)
        assert_(np.linalg.norm(step) - delta <= tol)

        # Ensure that no increase occurred in the objective function.
        reduct = -np.inner(gq, step) - 0.5 * np.inner(step, np.dot(Hq, step))
        assert_(reduct >= -tol)

    def test_exceptions(self):
        x0 = np.ones(5, dtype=float)
        gq = np.ones(5, dtype=float)
        Hq = np.ones((5, 5), dtype=float)
        xl = np.zeros(5, dtype=float)
        xu = 2.0 * np.ones(5, dtype=float)
        delta = 1.0
        with assert_raises(ValueError):
            bvtcg(x0, gq, np.dot, xl, xu, -1.0, Hq, debug=True)
        x0[2] = 2.1
        with assert_raises(ValueError):
            bvtcg(x0, gq, np.dot, xl, xu, delta, Hq, debug=True)
        x0[2], xl[2], xu[2] = 1.0, 1.1, 0.9
        with assert_raises(ValueError):
            bvtcg(x0, gq, np.dot, xl, xu, delta, Hq, debug=True)


class TestCPQP:

    @pytest.mark.parametrize('mleq', [1, 5, 10, 100])
    @pytest.mark.parametrize('mlub', [1, 5, 10, 100])
    @pytest.mark.parametrize('n', [1, 5, 10, 100])
    def test_simple(self, n, mleq, mlub):
        rng = np.random.default_rng(n + mlub + mleq)
        Aub = rng.standard_normal((mlub, n))
        bub = rng.standard_normal(mlub)
        Aeq = rng.standard_normal((mleq, n))
        beq = rng.standard_normal(mleq)
        xl = rng.standard_normal(n)
        xu = rng.standard_normal(n)
        xl, xu = np.minimum(xl, xu), np.maximum(xl, xu)
        x0 = rng.uniform(xl, xu)
        delta = rng.uniform(0.1, 1.0)
        step = cpqp(x0, Aub, bub, Aeq, beq, xl, xu, delta)
        assert_dtype_equal(step, float)
        assert_(step.ndim == 1)
        assert_(step.size == n)

        # Ensure the feasibility of the output.
        tol = 10.0 * np.finfo(float).eps * max((n, mlub, mleq))
        assert_array_less_equal(xl - x0 - step, tol)
        assert_array_less_equal(x0 + step - xu, tol)
        assert_(np.linalg.norm(step) - delta <= tol)

        # Assert that no increase occurs in the objective function.
        rub = np.maximum(0.0, np.dot(Aub, x0) - bub)
        reduct = np.inner(rub, rub)
        leq = np.dot(Aeq, x0) - beq
        reduct += np.inner(leq, leq)
        rub = np.maximum(0.0, np.dot(Aub, x0 + step) - bub)
        reduct -= np.inner(rub, rub)
        leq = np.dot(Aeq, x0 + step) - beq
        reduct -= np.inner(leq, leq)
        assert_(reduct >= -tol)

    def test_exceptions(self):
        x0 = np.ones(5, dtype=float)
        Aub = np.ones((3, 5), dtype=float)
        bub = np.ones(3, dtype=float)
        Aeq = np.ones((2, 5), dtype=float)
        beq = np.zeros(2, dtype=float)
        xl = np.zeros(5, dtype=float)
        xu = 2.0 * np.ones(5, dtype=float)
        delta = 1.0
        with assert_raises(ValueError):
            cpqp(x0, Aub, bub, Aeq, beq, xl, xu, -1.0, debug=True)
        x0[2] = 2.1
        with assert_raises(ValueError):
            cpqp(x0, Aub, bub, Aeq, beq, xl, xu, delta, debug=True)
        x0[2], xl[2], xu[2] = 1.0, 1.1, 0.9
        with assert_raises(ValueError):
            cpqp(x0, Aub, bub, Aeq, beq, xl, xu, delta, debug=True)


class TestLCTCG:

    @pytest.mark.parametrize('mleq', [1, 5, 10, 100])
    @pytest.mark.parametrize('mlub', [1, 5, 10, 100])
    @pytest.mark.parametrize('n', [1, 5, 10, 100])
    def test_simple(self, n, mlub, mleq):
        rng = np.random.default_rng(n + mlub + mleq)
        gq = rng.standard_normal(n)
        Hq = rng.standard_normal((n, n))
        Hq = .5 * (Hq + Hq.T)
        Aub = rng.standard_normal((mlub, n))
        Aeq = rng.standard_normal((mleq, n))
        xl = rng.standard_normal(n)
        xu = rng.standard_normal(n)
        xl, xu = np.minimum(xl, xu), np.maximum(xl, xu)
        x0 = rng.uniform(xl, xu)
        bub = np.dot(Aub, x0) + rng.uniform(0.0, 1.0, mlub)
        beq = np.dot(Aeq, x0)
        delta = rng.uniform(0.1, 1.0)
        step = lctcg(x0, gq, np.dot, Aub, bub, Aeq, beq, xl, xu, delta, Hq)
        assert_dtype_equal(step, float)
        assert_(step.ndim == 1)
        assert_(step.size == n)

        # Ensure the feasibility of the output.
        tol = 10.0 * np.finfo(float).eps * max((n, mlub, mleq))
        assert_array_less_equal(xl - x0 - step, tol)
        assert_array_less_equal(x0 + step - xu, tol)
        if mlub > 0:
            assert_array_less_equal(np.dot(Aub, x0 + step) - bub, tol)
        if mleq > 0:
            assert_array_less_equal(np.abs(np.dot(Aeq, step)), tol)
        assert_(np.linalg.norm(step) - delta <= tol)

        # Ensure that no increase occurred in the objective function.
        reduct = -np.inner(gq, step) - 0.5 * np.inner(step, np.dot(Hq, step))
        assert_(reduct >= -tol)

    def test_exceptions(self):
        x0 = np.zeros(5, dtype=float)
        gq = np.ones(5, dtype=float)
        Hq = np.ones((5, 5), dtype=float)
        Aub = np.ones((3, 5), dtype=float)
        bub = np.ones(3, dtype=float)
        Aeq = np.ones((2, 5), dtype=float)
        beq = np.zeros(2, dtype=float)
        xl = -np.ones(5, dtype=float)
        xu = np.ones(5, dtype=float)
        delta = 1.
        with assert_raises(ValueError):
            lctcg(x0, gq, np.dot, Aub, bub, Aeq, beq, xl, xu, -1., Hq,
                  debug=True)
        x0[2] = 1.1
        with assert_raises(ValueError):
            lctcg(x0, gq, np.dot, Aub, bub, Aeq, beq, xl, xu, delta, Hq,
                  debug=True)
        x0[2] = -1.1
        with assert_raises(ValueError):
            lctcg(x0, gq, np.dot, Aub, bub, Aeq, beq, xl, xu, delta, Hq,
                  debug=True)
        x0[2], xl[2], xu[2] = 0.0, 1.1, 0.9
        with assert_raises(ValueError):
            lctcg(x0, gq, np.dot, Aub, bub, Aeq, beq, xl, xu, delta, Hq,
                  debug=True)
        xl[2], xu[2] = -1.0, 1.0
        bub = -np.ones(3)
        with assert_raises(ValueError):
            lctcg(x0, gq, np.dot, Aub, bub, Aeq, beq, xl, xu, delta, Hq,
                  debug=True)
        bub = np.ones(3)
        beq = np.ones(2)
        with assert_raises(ValueError):
            lctcg(x0, gq, np.dot, Aub, bub, Aeq, beq, xl, xu, delta, Hq,
                  debug=True)


class TestNNLS:

    @staticmethod
    def nnls_test(n, m):
        rng = np.random.default_rng(n + m)
        A = rng.standard_normal((m, n))
        b = rng.standard_normal(m)
        k = rng.integers(n + 1)
        x = nnls(A, b, k)
        assert_dtype_equal(x, float)
        assert_(x.ndim == 1)
        assert_(x.size == n)

        # Ensure that the KKT conditions approximately hold at the solution.
        tol = 1e5 * np.finfo(float).eps * max(m, n)
        grad = np.dot(A.T, np.dot(A, x) - b)
        assert_array_less_equal(np.abs(grad[k:]), tol)
        assert_array_less_equal(np.abs(grad[:k] * x[:k]), tol)
        assert_array_less_equal(-tol, x[:k])
        assert_array_less_equal(-tol, grad[:k])

    @pytest.mark.parametrize('n', [1, 5, 10, 100])
    def test_simple(self, n):
        self.nnls_test(n, n)

    @pytest.mark.parametrize('n,m', [(1, 4), (5, 12), (10, 23), (100, 250)])
    def test_simple_trap(self, n, m):
        self.nnls_test(n, m)

    @pytest.mark.parametrize('n,m', [(4, 1), (12, 5), (23, 10), (250, 100)])
    def test_simple_tall(self, n, m):
        self.nnls_test(n, m)

    def test_exceptions(self):
        A = np.ones((4, 3), dtype=float)
        b = np.ones(4, dtype=float)
        assert_raises(ValueError, nnls, A, b, -1)
        assert_raises(ValueError, nnls, A, b, 4)
