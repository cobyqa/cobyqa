import numpy as np
import pytest

from cobyqa.subsolvers import tangential_byrd_omojokun, constrained_tangential_byrd_omojokun, normal_byrd_omojokun


class TestTangentialByrdOmojokun:

    @pytest.mark.parametrize('n', [1, 2, 10, 50])
    def test_simple(self, n):
        tol = 10.0 * np.finfo(float).eps * n
        for seed in range(100):
            # Construct and solve a random subproblem.
            rng = np.random.default_rng(seed)
            grad, hess, xl, xu, delta = _tangential_subproblem(rng, n)
            step = tangential_byrd_omojokun(grad, lambda s: hess @ s, xl, xu, delta, True)

            # Check whether the solution is valid and feasible.
            assert step.shape == (n,)
            assert np.isfinite(step).all()
            assert np.all(xl <= step)
            assert np.all(step <= xu)
            assert np.linalg.norm(step) < delta + tol

            # Check whether the solution decreases the objective function value.
            assert grad @ step + 0.5 * step @ hess @ step <= 0.0

    @pytest.mark.parametrize('n', [1, 2, 10, 50])
    def test_improve(self, n):
        tol = 10.0 * np.finfo(float).eps * n
        for seed in range(100):
            # Construct and solve a random subproblem.
            rng = np.random.default_rng(seed)
            grad, hess, xl, xu, delta = _tangential_subproblem(rng, n)
            step = tangential_byrd_omojokun(grad, lambda s: hess @ s, xl, xu, delta, True)

            # Solve the same subproblem without the improving strategy.
            step_base = tangential_byrd_omojokun(grad, lambda s: hess @ s, xl, xu, delta, True, improve=False)

            # Check whether the solution is valid and feasible.
            assert step_base.shape == (n,)
            assert np.isfinite(step_base).all()
            assert np.all(xl <= step_base)
            assert np.all(step_base <= xu)
            assert np.linalg.norm(step_base) < delta + tol

            # Check whether the improving strategy is effective.
            assert grad @ step + 0.5 * step @ hess @ step <= grad @ step_base + 0.5 * step_base @ hess @ step_base

    def test_exception(self):
        # Construct a random subproblem.
        rng = np.random.default_rng(0)
        grad, hess, xl, xu, delta = _tangential_subproblem(rng, 5)

        # We must have xl <= 0.
        with pytest.raises(AssertionError):
            xl_wrong = np.copy(xl)
            xl_wrong[0] = 0.1
            tangential_byrd_omojokun(grad, lambda s: hess @ s, xl_wrong, xu, delta, True)

        # We must have 0 <= xu.
        with pytest.raises(AssertionError):
            xu_wrong = np.copy(xu)
            xu_wrong[0] = -0.1
            tangential_byrd_omojokun(grad, lambda s: hess @ s, xl, xu_wrong, delta, True)

        # We must have delta < inf.
        with pytest.raises(AssertionError):
            tangential_byrd_omojokun(grad, lambda s: hess @ s, xl, xu, np.inf, True)

        # We must have delta > 0.
        with pytest.raises(AssertionError):
            tangential_byrd_omojokun(grad, lambda s: hess @ s, xl, xu, -1.0, True)


class TestConstrainedTangentialByrdOmojokun:

    @pytest.mark.parametrize('n', [1, 2, 10, 50])
    @pytest.mark.parametrize('m_ub', [1, 2, 10, 50])
    @pytest.mark.parametrize('m_eq', [1, 2, 10, 50])
    def test_simple(self, n, m_ub, m_eq):
        tol = 10.0 * np.finfo(float).eps * max(n, m_ub, m_eq)
        for seed in range(100):
            # Construct and solve a random subproblem.
            rng = np.random.default_rng(seed)
            grad, hess, xl, xu, delta = _tangential_subproblem(rng, n)
            aub = rng.standard_normal((m_ub, n))
            aeq = rng.standard_normal((m_eq, n))
            bub = rng.random(m_ub)
            step = constrained_tangential_byrd_omojokun(grad, lambda s: hess @ s, xl, xu, aub, bub, aeq, delta, True)

            # Check whether the solution is valid and feasible.
            assert step.shape == (n,)
            assert np.isfinite(step).all()
            assert np.all(xl <= step)
            assert np.all(step <= xu)
            assert np.all(aub @ step < bub + tol)
            assert np.all(np.abs(aeq @ step) < tol)
            assert np.linalg.norm(step) < delta + tol

            # Check whether the solution decreases the objective function value.
            assert grad @ step + 0.5 * step @ hess @ step <= 0.0

    @pytest.mark.parametrize('n', [1, 2, 10, 50])
    @pytest.mark.parametrize('m_ub', [1, 2, 10, 50])
    @pytest.mark.parametrize('m_eq', [1, 2, 10, 50])
    def test_improve(self, n, m_ub, m_eq):
        tol = 10.0 * np.finfo(float).eps * max(n, m_ub, m_eq)
        for seed in range(100):
            # Construct and solve a random subproblem.
            rng = np.random.default_rng(seed)
            grad, hess, xl, xu, delta = _tangential_subproblem(rng, n)
            aub = rng.standard_normal((m_ub, n))
            aeq = rng.standard_normal((m_eq, n))
            bub = rng.random(m_ub)
            step = constrained_tangential_byrd_omojokun(grad, lambda s: hess @ s, xl, xu, aub, bub, aeq, delta, True)

            # Solve the same subproblem without the improving strategy.
            step_base = constrained_tangential_byrd_omojokun(grad, lambda s: hess @ s, xl, xu, aub, bub, aeq, delta, True, improve=False)

            # Check whether the solution is valid and feasible.
            assert step_base.shape == (n,)
            assert np.isfinite(step_base).all()
            assert np.all(xl <= step_base)
            assert np.all(step_base <= xu)
            assert np.all(aub @ step_base < bub + tol)
            assert np.all(np.abs(aeq @ step_base) < tol)
            assert np.linalg.norm(step_base) < delta + tol

            # Check whether the improving strategy is effective.
            assert grad @ step + 0.5 * step @ hess @ step < grad @ step_base + 0.5 * step_base @ hess @ step_base + tol

    def test_exception(self):
        # Construct a random subproblem.
        rng = np.random.default_rng(0)
        grad, hess, xl, xu, delta = _tangential_subproblem(rng, 5)
        aub = rng.standard_normal((1, 5))
        aeq = rng.standard_normal((1, 5))
        bub = rng.random(1)

        # We must have xl <= 0.
        with pytest.raises(AssertionError):
            xl_wrong = np.copy(xl)
            xl_wrong[0] = 0.1
            constrained_tangential_byrd_omojokun(grad, lambda s: hess @ s, xl_wrong, xu, aub, bub, aeq, delta, True)

        # We must have 0 <= xu.
        with pytest.raises(AssertionError):
            xu_wrong = np.copy(xu)
            xu_wrong[0] = -0.1
            constrained_tangential_byrd_omojokun(grad, lambda s: hess @ s, xl, xu_wrong, aub, bub, aeq, delta, True)

        # We must have 0 <= bub.
        with pytest.raises(AssertionError):
            bub_wrong = np.copy(bub)
            bub_wrong[0] = -0.1
            constrained_tangential_byrd_omojokun(grad, lambda s: hess @ s, xl, xu, aub, bub_wrong, aeq, delta, True)

        # We must have delta < inf.
        with pytest.raises(AssertionError):
            constrained_tangential_byrd_omojokun(grad, lambda s: hess @ s, xl, xu, aub, bub, aeq, np.inf, True)

        # We must have delta > 0.
        with pytest.raises(AssertionError):
            constrained_tangential_byrd_omojokun(grad, lambda s: hess @ s, xl, xu, aub, bub, aeq, -1.0, True)


class TestNormalByrdOmojokun:

    @pytest.mark.parametrize('n', [1, 2, 10, 50])
    @pytest.mark.parametrize('m_ub', [1, 2, 10, 50])
    @pytest.mark.parametrize('m_eq', [1, 2, 10, 50])
    def test_simple(self, n, m_ub, m_eq):
        tol = 10.0 * np.finfo(float).eps * max(n, m_ub, m_eq)
        for seed in range(100):
            # Construct and solve a random subproblem.
            rng = np.random.default_rng(seed)
            aub, bub, aeq, beq, xl, xu, delta = _normal_subproblem(rng, n, m_ub, m_eq)
            step = normal_byrd_omojokun(aub, bub, aeq, beq, xl, xu, delta, True)

            # Check whether the solution is valid and feasible.
            assert step.shape == (n,)
            assert np.isfinite(step).all()
            assert np.all(xl <= step)
            assert np.all(step <= xu)
            assert np.linalg.norm(step) < delta + tol

            # Check whether the solution decreases the objective function value.
            resid_ub = np.maximum(aub @ step - bub, 0.0)
            resid_ub_zero = np.maximum(-bub, 0.0)
            resid_eq = aeq @ step - beq
            assert resid_ub @ resid_ub + resid_eq @ resid_eq <= resid_ub_zero @ resid_ub_zero + beq @ beq

    @pytest.mark.parametrize('n', [1, 2, 10, 50])
    @pytest.mark.parametrize('m_ub', [1, 2, 10, 50])
    @pytest.mark.parametrize('m_eq', [1, 2, 10, 50])
    def test_improve(self, n, m_ub, m_eq):
        tol = 10.0 * np.finfo(float).eps * max(n, m_ub, m_eq)
        for seed in range(100):
            # Construct and solve a random subproblem.
            rng = np.random.default_rng(seed)
            aub, bub, aeq, beq, xl, xu, delta = _normal_subproblem(rng, n, m_ub, m_eq)
            step = normal_byrd_omojokun(aub, bub, aeq, beq, xl, xu, delta, True)

            # Solve the same subproblem without the improving strategy.
            step_base = normal_byrd_omojokun(aub, bub, aeq, beq, xl, xu, delta, True, improve=False)

            # Check whether the solution is valid and feasible.
            assert step_base.shape == (n,)
            assert np.isfinite(step_base).all()
            assert np.all(xl <= step_base)
            assert np.all(step_base <= xu)
            assert np.linalg.norm(step_base) < delta + tol

            # Check whether the improving strategy is effective.
            resid_ub = np.maximum(aub @ step - bub, 0.0)
            resid_ub_base = np.maximum(aub @ step_base - bub, 0.0)
            resid_eq = aeq @ step - beq
            resid_eq_base = aeq @ step_base - beq
            assert resid_ub @ resid_ub + resid_eq @ resid_eq <= resid_ub_base @ resid_ub_base + resid_eq_base @ resid_eq_base

    def test_exception(self):
        # Construct a random subproblem.
        rng = np.random.default_rng(0)
        aub, bub, aeq, beq, xl, xu, delta = _normal_subproblem(rng, 5, 1, 1)

        # We must have xl <= 0.
        with pytest.raises(AssertionError):
            xl_wrong = np.copy(xl)
            xl_wrong[0] = 0.1
            normal_byrd_omojokun(aub, bub, aeq, beq, xl_wrong, xu, delta, True)

        # We must have 0 <= xu.
        with pytest.raises(AssertionError):
            xu_wrong = np.copy(xu)
            xu_wrong[0] = -0.1
            normal_byrd_omojokun(aub, bub, aeq, beq, xl, xu_wrong, delta, True)

        # We must have delta < inf.
        with pytest.raises(AssertionError):
            normal_byrd_omojokun(aub, bub, aeq, beq, xl, xu, np.inf, True)

        # We must have delta > 0.
        with pytest.raises(AssertionError):
            normal_byrd_omojokun(aub, bub, aeq, beq, xl, xu, -1.0, True)


def _tangential_subproblem(rng, n):
    grad = rng.standard_normal(n)
    hess = rng.standard_normal((n, n))
    hess = 0.5 * (hess + hess.T)
    xl = -rng.random(n)
    xu = rng.random(n)
    delta = rng.random()
    return grad, hess, xl, xu, delta


def _normal_subproblem(rng, n, m_ub, m_eq):
    aub = rng.standard_normal((m_ub, n))
    aeq = rng.standard_normal((m_eq, n))
    bub = rng.standard_normal(m_ub)
    beq = rng.standard_normal(m_eq)
    xl = -rng.random(n)
    xu = rng.random(n)
    delta = rng.random()
    return aub, bub, aeq, beq, xl, xu, delta
