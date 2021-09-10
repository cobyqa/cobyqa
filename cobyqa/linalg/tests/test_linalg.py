import numpy as np
import pytest
from numpy.testing import assert_, assert_almost_equal, assert_raises

from cobyqa.linalg import bvtcg, cpqp, lctcg, nnls

EPS = np.finfo(float).eps


class TestNNLS:

    @pytest.mark.parametrize('n_max, m_max, rep', [
        (100, 400, 10),
        (400, 100, 10),
    ])
    def test_standard(self, n_max, m_max, rep):
        rng = np.random.default_rng(0)
        for n in rng.integers(1, n_max, rep):
            for m in rng.integers(1, m_max, rep):
                # Solve a randomly generated problem.
                A = rng.standard_normal((m, n))
                b = rng.standard_normal(m)
                k = rng.integers(n + 1)
                x, rnorm = nnls(A, b, k)

                # Assert the basic properties of the output.
                assert_(x.dtype == np.float64)
                assert_(x.ndim == 1)
                assert_(x.size == n)
                assert_almost_equal(rnorm, np.linalg.norm(np.dot(A, x) - b))

                # Ensure that the KKT conditions approximately hold. The dual
                # solution, in W[:K], is computed by supposing that the
                # stationary conditions hold for the first K components of the
                # gradient of the Lagrangian of the problem. We check
                # 1. the remaining stationary conditions;
                # 2. the complementary slackness conditions;
                # 3. the primal feasibility conditions; and
                # 4. the dual feasibility conditions.
                tol = 1e1 * EPS * max(n, m) * np.max(np.abs(b), initial=1.)
                w = np.dot(A.T, np.dot(A, x) - b)
                assert_(np.all(np.abs(w[k:]) <= tol))
                assert_(np.all(np.abs(w[:k] * x[:k]) <= tol))
                assert_(np.all(x[:k] >= 0.))
                assert_(np.all(w[:k] >= -tol))

    @pytest.mark.parametrize('n, m', [
        (100, 400),
        (400, 100),
    ])
    def test_raises(self, n, m):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((m, n))
        b = rng.standard_normal(m)
        assert_raises(ValueError, nnls, A, b, -1)
        assert_raises(ValueError, nnls, A, b, n + 1)


class TestBVTCG:

    @pytest.mark.parametrize('n_max, rep', [
        (10, 100),
        (100, 100),
        (400, 100),
    ])
    def test_standard(self, n_max, rep):
        rng = np.random.default_rng(0)
        for n in rng.integers(1, n_max, rep):
            # Solve a randomly generated problem.
            gq = rng.standard_normal(n)
            Hq = rng.standard_normal((n, n))
            Hq = .5 * (Hq + Hq.T)
            xl = rng.standard_normal(n)
            xu = rng.standard_normal(n)
            xl, xu = np.minimum(xl, xu), np.maximum(xl, xu)
            xopt = rng.uniform(xl, xu)
            delta = rng.uniform(1e-1, 1.)
            step = bvtcg(xopt, gq, lambda x: np.dot(Hq, x), (), xl, xu, delta)

            # Assert the basic properties of the output.
            assert_(step.dtype == np.float64)
            assert_(step.ndim == 1)
            assert_(step.size == n)

            # Assert the feasibility of the output.
            tol = 1e1 * EPS * n * np.max(np.abs(np.r_[xl, xu]), initial=1.)
            assert_(np.max(xl - xopt - step) < tol)
            assert_(np.max(xopt + step - xu) < tol)
            assert_(np.linalg.norm(step) - delta < tol)

            # Assert that no increase occurs in the objective function.
            desc = np.inner(gq, step) + .5 * np.inner(step, np.dot(Hq, step))
            assert_(desc <= 0.)

    @pytest.mark.parametrize('n', [
        (100,),
        (400,),
    ])
    def test_raises(self, n):
        # Generate an infeasible problem due to the trust region.
        rng = np.random.default_rng(0)
        gq = rng.standard_normal(n)
        xl = rng.standard_normal(n)
        xu = rng.standard_normal(n)
        xl, xu = np.minimum(xl, xu), np.maximum(xl, xu)
        xopt = rng.uniform(xl, xu)
        assert_raises(AssertionError, bvtcg, xopt, gq, None, (), xl, xu, -1.)

        # Generate a feasible problem with an initial guess having a component
        # exceeding the upper bound.
        delta = rng.uniform(1e-1, 1.)
        xopt[0] = xu[0] + 1e-1
        assert_raises(AssertionError, bvtcg, xopt, gq, None, (), xl, xu, delta)

        # Generate a feasible problem with an initial guess having a component
        # below the lower bound.
        xopt[0] = xl[0] - 1e-1
        assert_raises(AssertionError, bvtcg, xopt, gq, None, (), xl, xu, delta)

        # Generate an infeasible problem due to the bounds.
        xl[0], xu[0] = xu[0] + 1e-1, xl[0] - 1e-1
        assert_raises(AssertionError, bvtcg, xopt, gq, None, (), xl, xu, delta)


class TestLCTCG:

    @pytest.mark.slow
    @pytest.mark.parametrize('n_max, mub_max, meq_max, rep', [
        (10, 10, 10, 10),
        (400, 200, 100, 2),
    ])
    def test_standard(self, n_max, mub_max, meq_max, rep):
        rng = np.random.default_rng(0)
        for n in rng.integers(1, n_max, rep):
            for mub in rng.integers(0, mub_max, rep):
                for meq in rng.integers(0, meq_max, rep):
                    # Solve a randomly generated problem.
                    gq = rng.standard_normal(n)
                    Hq = rng.standard_normal((n, n))
                    Hq = .5 * (Hq + Hq.T)
                    Aub = rng.standard_normal((mub, n))
                    Aeq = rng.standard_normal((meq, n))
                    U, S, Vh = np.linalg.svd(Aeq)
                    rk = sum(S > EPS * max(meq, n) * np.max(S, initial=1.))
                    Aeq = np.dot(U[:rk, :rk] * S[:rk], Vh[:rk, :])
                    xl = rng.standard_normal(n)
                    xu = rng.standard_normal(n)
                    xl, xu = np.minimum(xl, xu), np.maximum(xl, xu)
                    xopt = rng.uniform(xl, xu)
                    bub = np.dot(Aub, xopt) + rng.uniform(0., 1., mub)
                    beq = np.dot(Aeq, xopt)
                    delta = rng.uniform(1e-1, 1.)
                    step = lctcg(xopt, gq, lambda x: np.dot(Hq, x), (), Aub,
                                 bub, Aeq, beq, xl, xu, delta)

                    # Assert the basic properties of the output.
                    assert_(step.dtype == np.float64)
                    assert_(step.ndim == 1)
                    assert_(step.size == n)

                    # Assert the feasibility of the output.
                    tol = 1e1 * EPS * n
                    tolbd = tol * np.max(np.abs(np.r_[xl, xu]), initial=1.)
                    tollc = tol * np.max(np.abs(bub), initial=1.)
                    assert_(np.max(xl - xopt - step) < tolbd)
                    assert_(np.max(xopt + step - xu) < tolbd)
                    if mub > 0:
                        reslub = bub - np.dot(Aub, xopt + step)
                        assert_(np.min(reslub) > -tollc)
                    if meq > 0:
                        assert_(np.max(np.abs(np.dot(Aeq, step))) < tollc)
                    assert_(np.linalg.norm(step) - delta < max(tolbd, tollc))

                    # Assert that no increase occurs in the objective function.
                    desc = np.inner(gq, step)
                    desc += .5 * np.inner(step, np.dot(Hq, step))
                    assert_(desc < max(tolbd, tollc))

    @pytest.mark.parametrize('n, mub, meq', [
        (100, 50, 30),
        (400, 450, 10),
    ])
    def test_raises(self, n, mub, meq):
        # Generate an infeasible problem due to the trust region.
        rng = np.random.default_rng(0)
        gq = rng.standard_normal(n)
        Aub = rng.standard_normal((mub, n))
        Aeq = rng.standard_normal((meq, n))
        U, S, Vh = np.linalg.svd(Aeq)
        rk = sum(S > EPS * max(meq, n) * np.max(S, initial=1.))
        Aeq = np.dot(U[:rk, :rk] * S[:rk], Vh[:rk, :])
        xl = rng.standard_normal(n)
        xu = rng.standard_normal(n)
        xl, xu = np.minimum(xl, xu), np.maximum(xl, xu)
        xopt = rng.uniform(xl, xu)
        bub = np.dot(Aub, xopt) + rng.uniform(0., 1., mub)
        beq = np.dot(Aeq, xopt)
        assert_raises(AssertionError, lctcg, xopt, gq, None, (), Aub, bub, Aeq,
                      beq, xl, xu, -1.)

        # Generate a feasible problem with an initial guess having a component
        # exceeding the upper bound.
        delta = rng.uniform(1e-1, 1.)
        xopt[0] = xu[0] + 1e-1
        assert_raises(AssertionError, lctcg, xopt, gq, None, (), Aub, bub, Aeq,
                      beq, xl, xu, delta)

        # Generate a feasible problem with an initial guess having a component
        # below the lower bound.
        xopt[0] = xl[0] - 1e-1
        assert_raises(AssertionError, lctcg, xopt, gq, None, (), Aub, bub, Aeq,
                      beq, xl, xu, delta)

        # Generate an infeasible problem due to the bounds.
        xl[0], xu[0] = xu[0] + 1e-1, xl[0] - 1e-1
        assert_raises(AssertionError, lctcg, xopt, gq, None, (), Aub, bub, Aeq,
                      beq, xl, xu, delta)

        # Generate an infeasible problem due to the linear equalities.
        xl = rng.standard_normal(n)
        xu = rng.standard_normal(n)
        xl, xu = np.minimum(xl, xu), np.maximum(xl, xu)
        xopt = rng.uniform(xl, xu)
        bub = np.dot(Aub, xopt) + rng.uniform(0., 1., mub)
        beq = np.dot(Aeq, xopt) + 1e-5
        assert_raises(AssertionError, lctcg, xopt, gq, None, (), Aub, bub, Aeq,
                      beq, xl, xu, delta)

        # Generate an infeasible problem due to the linear inequalities.
        bub = np.dot(Aub, xopt) + rng.uniform(0., 1., mub)
        bub[0] -= 1.1
        beq = np.dot(Aeq, xopt)
        assert_raises(AssertionError, lctcg, xopt, gq, None, (), Aub, bub, Aeq,
                      beq, xl, xu, delta)

        # Generate a feasible problem with a rank deficient matrix AEQ.
        while np.linalg.matrix_rank(Aeq) < 2:
            Aeq = rng.standard_normal((meq, n))
            U, S, Vh = np.linalg.svd(Aeq)
            rk = sum(S > EPS * max(meq, n) * np.max(S, initial=1.))
            Aeq = np.dot(U[:rk, :rk] * S[:rk], Vh[:rk, :])
        bub = np.dot(Aub, xopt) + rng.uniform(0., 1., mub)
        beq = np.dot(Aeq, xopt)
        lc = rng.uniform(1e-3, .1, meq - 1)
        Aeq[0, :] = np.dot(Aeq[1:, :].T, lc)
        beq[0] = np.inner(beq[1:], lc)
        assert_raises(AssertionError, lctcg, xopt, gq, None, (), Aub, bub, Aeq,
                      beq, xl, xu, delta)


class TestCPQP:

    @pytest.mark.slow
    @pytest.mark.parametrize('n_max, mub_max, meq_max, rep', [
        (10, 10, 10, 10),
        (400, 200, 100, 2),
    ])
    def test_standard(self, n_max, mub_max, meq_max, rep):
        rng = np.random.default_rng(0)
        for n in rng.integers(1, n_max, rep):
            for mub in rng.integers(0, mub_max, rep):
                for meq in rng.integers(0, meq_max, rep):
                    Aub = rng.standard_normal((mub, n))
                    bub = rng.standard_normal(mub)
                    Aeq = rng.standard_normal((meq, n))
                    beq = rng.standard_normal(meq)
                    xl = rng.standard_normal(n)
                    xu = rng.standard_normal(n)
                    xl, xu = np.minimum(xl, xu), np.maximum(xl, xu)
                    xopt = rng.uniform(xl, xu)
                    delta = rng.uniform(1e-1, 1.)
                    step = cpqp(xopt, Aub, bub, Aeq, beq, xl, xu, delta)

                    # Assert the basic properties of the output.
                    assert_(step.dtype == np.float64)
                    assert_(step.ndim == 1)
                    assert_(step.size == n)

                    # Assert the feasibility of the output.
                    tol = 1e1 * EPS * n
                    tol *= np.max(np.abs(np.r_[xl, xu]), initial=1.)
                    snorm = np.linalg.norm(step)
                    assert_(np.max(xl - xopt - step) < tol)
                    assert_(np.max(xopt + step - xu) < tol)
                    assert_(snorm - delta < tol)

                    # Assert that no increase occurs in the objective function.
                    if snorm > tol:
                        lub = np.maximum(0., np.dot(Aub, xopt) - bub)
                        desc = np.inner(lub, lub)
                        leq = np.dot(Aeq, xopt) - beq
                        desc += np.inner(leq, leq)
                        lub = np.maximum(0., np.dot(Aub, xopt + step) - bub)
                        desc -= np.inner(lub, lub)
                        leq = np.dot(Aeq, xopt + step) - beq
                        desc -= np.inner(leq, leq)
                        assert_(desc > -tol)
