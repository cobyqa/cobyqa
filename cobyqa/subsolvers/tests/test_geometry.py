import numpy as np
import pytest

from cobyqa.subsolvers import cauchy_geometry, spider_geometry


class TestCauchyGeometry:

    @pytest.mark.parametrize('n', [1, 2, 10, 50])
    def test_simple(self, n):
        tol = 10.0 * np.finfo(float).eps * n
        for seed in range(100):
            # Construct and solve a random subproblem.
            rng = np.random.default_rng(seed)
            const, grad, hess, xl, xu, delta = _subproblem(rng, n)
            step = cauchy_geometry(const, grad, lambda s: s @ hess @ s, xl, xu, delta, True)

            # Check whether the solution is valid and feasible.
            assert step.shape == (n,)
            assert np.isfinite(step).all()
            assert np.all(xl <= step)
            assert np.all(step <= xu)
            assert np.linalg.norm(step) < delta + tol

            # Check whether the solution increases the objective function value.
            assert abs(const + step @ grad + 0.5 * step @ hess @ step) >= abs(const)

    def test_exception(self):
        # Construct a random subproblem.
        rng = np.random.default_rng(0)
        const, grad, hess, xl, xu, delta = _subproblem(rng, 5)

        # We must have xl <= 0.
        with pytest.raises(AssertionError):
            xl_wrong = np.copy(xl)
            xl_wrong[0] = 0.1
            cauchy_geometry(const, grad, lambda s: s @ hess @ s, xl_wrong, xu, delta, True)

        # We must have 0 <= xu.
        with pytest.raises(AssertionError):
            xu_wrong = np.copy(xu)
            xu_wrong[0] = -0.1
            cauchy_geometry(const, grad, lambda s: s @ hess @ s, xl, xu_wrong, delta, True)

        # We must have delta < inf.
        with pytest.raises(AssertionError):
            cauchy_geometry(const, grad, lambda s: s @ hess @ s, xl, xu, np.inf, True)

        # We must have delta > 0.
        with pytest.raises(AssertionError):
            cauchy_geometry(const, grad, lambda s: s @ hess @ s, xl, xu, -1.0, True)


class TestSpiderGeometry:

    @pytest.mark.parametrize('n', [1, 2, 10, 50])
    @pytest.mark.parametrize('npt_f', [
        lambda n: n + 1,
        lambda n: n + 2,
        lambda n: 2 * n + 1,
        lambda n: (n + 1) * (n + 2) // 2,
    ])
    def test_simple(self, n, npt_f):
        npt = npt_f(n)
        tol = 10.0 * np.finfo(float).eps * n
        for seed in range(100):
            # Construct and solve a random subproblem.
            rng = np.random.default_rng(seed)
            const, grad, hess, xl, xu, delta = _subproblem(rng, n)
            xpt = rng.standard_normal((n, npt))
            step = spider_geometry(const, grad, lambda s: s @ hess @ s, xpt, xl, xu, delta, True)

            # Check whether the solution is valid and feasible.
            assert step.shape == (n,)
            assert np.isfinite(step).all()
            assert np.all(xl <= step)
            assert np.all(step <= xu)
            assert np.linalg.norm(step) < delta + tol

            # Check whether the solution increases the objective function value
            # compared to the origin and the feasible interpolation points.
            q_val = const + step @ grad + 0.5 * step @ hess @ step
            assert abs(q_val) >= abs(const)
            for k in range(npt):
                if np.linalg.norm(xpt[:, k]) < delta + tol and np.all(xl <= xpt[:, k]) and np.all(xpt[:, k] <= xu):
                    assert abs(q_val) >= abs(const + xpt[:, k] @ grad + 0.5 * xpt[:, k] @ hess @ xpt[:, k])

    def test_exception(self):
        # Construct a random subproblem.
        rng = np.random.default_rng(0)
        const, grad, hess, xl, xu, delta = _subproblem(rng, 5)
        xpt = rng.standard_normal((5, 11))

        # We must have xl <= 0.
        with pytest.raises(AssertionError):
            xl_wrong = np.copy(xl)
            xl_wrong[0] = 0.1
            spider_geometry(const, grad, lambda s: s @ hess @ s, xpt, xl_wrong, xu, delta, True)

        # We must have 0 <= xu.
        with pytest.raises(AssertionError):
            xu_wrong = np.copy(xu)
            xu_wrong[0] = -0.1
            spider_geometry(const, grad, lambda s: s @ hess @ s, xpt, xl, xu_wrong, delta, True)

        # We must have delta < inf.
        with pytest.raises(AssertionError):
            spider_geometry(const, grad, lambda s: s @ hess @ s, xpt, xl, xu, np.inf, True)

        # We must have delta > 0.
        with pytest.raises(AssertionError):
            spider_geometry(const, grad, lambda s: s @ hess @ s, xpt, xl, xu, -1.0, True)


def _subproblem(rng, n):
    const = rng.standard_normal()
    grad = rng.standard_normal(n)
    hess = rng.standard_normal((n, n))
    hess = 0.5 * (hess + hess.T)
    xl = -rng.random(n)
    xu = rng.random(n)
    delta = rng.random()
    return const, grad, hess, xl, xu, delta
