import numpy as np
from scipy.optimize import NonlinearConstraint

from ..main import minimize


class TestMinimize:

    @staticmethod
    def fun(x, c=1.0):
        return x[0]**2 + c * abs(x[1])**3

    @staticmethod
    def con(x):
        return x[0]**2 + x[1]**2 - 25.0

    def test_simple(self):
        constraints = NonlinearConstraint(self.con, 0.0, 0.0)
        res = minimize(self.fun, [4.0, 1.0], constraints=constraints)
        solution = [np.sqrt(25.0 - 4.0 / 9.0), 2.0 / 3.0]
        np.testing.assert_allclose(res.x, solution, atol=1e-4)
        assert res.success
        assert res.maxcv < 1e-7
        assert res.nfev <= 100
        assert res.fun < self.fun(solution) + 1e-3
