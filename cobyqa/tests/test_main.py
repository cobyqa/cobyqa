import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint
from scipy.optimize._minimize import standardize_constraints

from ..main import minimize


class TestMinimize:

    def setup_method(self):
        self.x0 = [4.0, 1.0]
        self.options = {"debug": True}

    @staticmethod
    def fun(x, c=1.0):
        return x[0] ** 2 + c * abs(x[1]) ** 3

    @staticmethod
    def con(x):
        return x[0] ** 2 + x[1] ** 2 - 25.0

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
        assert res.status == 0, res
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
        assert res.status == 0, res
        assert res.maxcv > 0.35, res
        assert res.nfev <= 100, res
        bounds_alt = [[5.0, 5.5], [0.6, 0.65]]
        res_alt = minimize(
            self.fun,
            self.x0,
            bounds=bounds_alt,
            constraints=constraints,
            options=self.options,
        )
        np.testing.assert_array_equal(res.x, res_alt.x)
        assert res.status == res_alt.status, res
        assert res.nfev == res_alt.nfev, res
        assert res.nit == res_alt.nit, res

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
        assert res.status == 0, res
        assert res.maxcv < 1e-8, res
        assert res.nfev <= 100, res
        assert res.fun < self.fun(solution) + 1e-3, res

    def test_nonlinear_constraints(self):
        constraints = NonlinearConstraint(self.con, 0.0, 0.0)
        res = minimize(
            self.fun,
            self.x0,
            constraints=constraints,
            options=self.options,
        )
        constraints_alt = {"fun": self.con, "type": "eq"}
        constraints_alt = standardize_constraints(
            (constraints_alt,), self.x0, "new"
        )
        res_alt = minimize(
            self.fun,
            self.x0,
            constraints=constraints_alt,
            options=self.options,
        )
        np.testing.assert_array_equal(res.x, res_alt.x)
        assert res.status == res_alt.status, res
        assert res.nfev == res_alt.nfev, res
        assert res.nit == res_alt.nit, res

    def test_minimize_args(self):
        constraints = NonlinearConstraint(self.con, 0.0, 0.0)
        res = minimize(
            self.fun,
            self.x0,
            2.0,
            constraints=constraints,
            options=self.options,
        )
        solution = [np.sqrt(25.0 - 4.0 / 36.0), 2.0 / 6.0]
        np.testing.assert_allclose(res.x, solution, atol=1e-4)
        assert res.success, res.message
        assert res.status == 0, res
        assert res.maxcv < 1e-8, res
        assert res.nfev <= 100, res
        assert res.fun < self.fun(solution, 2.0) + 1e-3, res

    def test_callback(self):
        class Callback:
            def __init__(self):
                self.n_eval = 0

            def __call__(self, x):
                self.n_eval += 1
                solution = np.array([np.sqrt(25.0 - 4.0 / 9.0), 2.0 / 3.0])
                if np.allclose(x, solution, atol=1e-4):
                    raise StopIteration

        callback = Callback()
        constraints = NonlinearConstraint(self.con, 0.0, 0.0)
        res = minimize(
            self.fun,
            self.x0,
            constraints=constraints,
            callback=callback,
            options=self.options,
        )
        assert callback.n_eval == res.nfev, res
        assert res.status > 0, res

    def test_infeasible_bounds(self):
        bounds = Bounds([4.5, 0.6], [4.0, 0.7])
        res = minimize(
            self.fun,
            self.x0,
            bounds=bounds,
            options=self.options,
        )
        assert not res.success, res.message
        assert res.status < 0, res
        assert res.maxcv >= 0.5, res

    def test_fixed(self):
        bounds = Bounds([4.5, 0.6], [4.5, 0.6])
        res = minimize(
            self.fun,
            self.x0,
            bounds=bounds,
            options=self.options,
        )
        np.testing.assert_allclose(res.x, [4.5, 0.6], atol=1e-4)
        assert res.success, res.message
        assert res.status > 0, res
        assert res.maxcv < 1e-8, res

    def test_target(self):
        options = dict(self.options)
        options["target"] = 40.0
        res = minimize(
            self.fun,
            self.x0,
            (2.0,),
            options=options,
        )
        assert res.success, res.message

    def test_max_eval(self):
        constraints = NonlinearConstraint(self.con, 0.0, 0.0)
        options = dict(self.options)
        options["maxfev"] = 10
        res = minimize(
            self.fun,
            self.x0,
            constraints=constraints,
            options=options,
        )
        assert not res.success, res.message

    def test_max_iter(self):
        constraints = NonlinearConstraint(self.con, 0.0, 0.0)
        options = dict(self.options)
        options["maxiter"] = 5
        res = minimize(
            self.fun,
            self.x0,
            constraints=constraints,
            options=options,
        )
        assert not res.success, res.message

    def test_feasibility_problem(self):
        bounds = Bounds([4.5, 0.6], [5.0, 0.7])
        res = minimize(
            None,
            self.x0,
            bounds=bounds,
            options=self.options,
        )
        assert res.success, res.message
        assert res.status > 0, res

    def test_exceptions(self):
        constraints = NonlinearConstraint(self.con, 0.0, 0.0)
        with pytest.raises(ValueError):
            minimize(
                self.fun,
                self.x0,
                (2.0,),
                constraints=constraints,
                options={"history_size": 0},
            )
        with pytest.raises(ValueError):
            minimize(
                self.fun,
                self.x0,
                (2.0,),
                constraints=constraints,
                options={"filter_size": 0},
            )
        with pytest.raises(ValueError):
            minimize(
                self.fun,
                self.x0,
                (2.0,),
                constraints=constraints,
                options={"radius_init": 0},
            )
        with pytest.raises(ValueError):
            minimize(
                self.fun,
                self.x0,
                (2.0,),
                constraints=constraints,
                options={"radius_final": -1},
            )
        with pytest.raises(ValueError):
            minimize(
                self.fun,
                self.x0,
                (2.0,),
                constraints=constraints,
                options={"radius_init": 1, "radius_final": 2},
            )
        with pytest.raises(ValueError):
            minimize(
                self.fun,
                self.x0,
                (2.0,),
                constraints=constraints,
                options={"nb_points": 0},
            )
        with pytest.raises(ValueError):
            minimize(
                self.fun,
                self.x0,
                (2.0,),
                constraints=constraints,
                options={"nb_points": 7},
            )
        with pytest.raises(ValueError):
            minimize(
                self.fun,
                self.x0,
                (2.0,),
                constraints=constraints,
                options={"maxfev": 0},
            )
        with pytest.raises(ValueError):
            minimize(
                self.fun,
                self.x0,
                (2.0,),
                constraints=constraints,
                options={"maxiter": 0},
            )

    def test_warning(self):
        with pytest.warns(RuntimeWarning):
            minimize(
                self.fun,
                self.x0,
                options={"unknown": 0},
            )
