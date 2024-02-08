import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint

from ..main import minimize


class TestMinimize:

    def setup_method(self):
        self.x0 = [4.0, 1.0]
        self.options = {'debug': True}

    @staticmethod
    def fun(x, c=1.0):
        return x[0]**2 + c * abs(x[1])**3

    @staticmethod
    def con(x):
        return x[0]**2 + x[1]**2 - 25.0

    def test_simple(self):
        constraints = NonlinearConstraint(self.con, 0.0, 0.0)
        res = minimize(
            self.fun,
            self.x0,
            constraints=constraints,
            options=self.options,
        )
        solution = [np.sqrt(25.0 - 4.0 / 9.0), 2.0 / 3.0]
        np.testing.assert_allclose(res.x, solution, atol=1e-4)
        assert res.success, res.message
        assert res.maxcv < 1e-7, res
        assert res.nfev <= 100, res
        assert res.fun < self.fun(solution) + 1e-3, res

    def test_bounds(self):
        # Case where the bounds are not active at the solution.
        bounds = Bounds([4.5, 0.6], [5.0, 0.7])
        constraints = NonlinearConstraint(self.con, 0.0, 0.0)
        res = minimize(
            self.fun,
            self.x0,
            bounds=bounds,
            constraints=constraints,
            options=self.options,
        )
        solution = [np.sqrt(25.0 - 4.0 / 9.0), 2.0 / 3.0]
        np.testing.assert_allclose(res.x, solution, atol=1e-4)
        assert res.success, res.message
        assert res.maxcv < 1e-8, res
        assert res.nfev <= 100, res
        assert res.fun < self.fun(solution) + 1e-3, res

        # Case where the bounds are active at the solution.
        bounds = Bounds([5.0, 0.6], [5.5, 0.65])
        res = minimize(
            self.fun,
            self.x0,
            bounds=bounds,
            constraints=constraints,
            options=self.options,
        )
        assert not res.success, res.message
        assert res.maxcv > 0.35, res
        assert res.nfev <= 100, res

    def test_minimize_linear_constraints(self):
        constraints = LinearConstraint([1.0, 1.0], 1.0, 1.0)
        res = minimize(
            self.fun,
            self.x0,
            constraints=constraints,
            options=self.options,
        )
        solution = [(4 - np.sqrt(7)) / 3, (np.sqrt(7) - 1) / 3]
        np.testing.assert_allclose(res.x, solution, atol=1e-4)
        assert res.success, res.message
        assert res.maxcv < 1e-8, res
        assert res.nfev <= 100, res
        assert res.fun < self.fun(solution) + 1e-3, res

    def test_minimize_args(self):
        constraints = NonlinearConstraint(self.con, 0.0, 0.0)
        res = minimize(
            self.fun,
            self.x0,
            (2.0,),
            constraints=constraints,
            options=self.options,
        )
        solution = [np.sqrt(25.0 - 4.0 / 36.0), 2.0 / 6.0]
        np.testing.assert_allclose(res.x, solution, atol=1e-4)
        assert res.success, res.message
        assert res.maxcv < 1e-8, res
        assert res.nfev <= 100, res
        assert res.fun < self.fun(solution, 2.0) + 1e-3, res
