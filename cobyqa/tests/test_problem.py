import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint

from ..problem import (
    ObjectiveFunction,
    BoundConstraints,
    LinearConstraints,
    # NonlinearConstraints,
    # Problem,
)
from ..settings import PRINT_OPTIONS


class TestObjectiveFunction:

    @staticmethod
    def rosen(x, c=100.0):
        x = np.asarray(x)
        return np.sum(c * (x[1:] - x[:-1]**2.0)**2.0 + (1.0 - x[:-1])**2.0)

    class Rosen:

        def __call__(self, x):
            return TestObjectiveFunction.rosen(x)

    def test_simple(self, capsys):
        obj = ObjectiveFunction(self.rosen, False, True)
        x = [1.5, 1.5]
        assert obj.n_eval == 0
        assert obj(x) == self.rosen(x)
        assert obj.n_eval == 1
        assert obj.name == "rosen"
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_args(self):
        obj = ObjectiveFunction(self.rosen, False, True, 2.0)
        x = [1.5, 1.5]
        assert obj(x) == self.rosen(x, 2.0)

    def test_none(self):
        obj = ObjectiveFunction(None, False, True)
        assert obj([1.5, 1.5]) == 0.0
        assert obj.n_eval == 0
        assert obj.name == ""

    def test_wrapper(self):
        obj = ObjectiveFunction(self.Rosen(), False, True)
        assert obj.name == "fun"

    def test_verbose(self, capsys):
        obj = ObjectiveFunction(self.rosen, True, True)
        x = np.array([1.5, 1.5])
        obj(x)
        captured = capsys.readouterr()
        with np.printoptions(**PRINT_OPTIONS):
            assert captured.out == f"rosen({x}) = {self.rosen(x)}\n"


class TestBoundConstraints:

    def test_simple(self):
        bounds = Bounds([0.0, 0.0], [1.0, 1.0])
        constraints = BoundConstraints(bounds)
        np.testing.assert_array_equal(constraints.xl, bounds.lb)
        np.testing.assert_array_equal(constraints.xu, bounds.ub)
        assert constraints.m == 4
        assert constraints.is_feasible
        assert constraints.maxcv([0.5, 0.5]) == 0.0
        assert constraints.maxcv(constraints.xl) == 0.0
        assert constraints.maxcv(constraints.xu) == 0.0
        x = [2.0, 2.0]
        assert np.all(constraints.project(x) >= constraints.xl)
        assert np.all(constraints.project(x) <= constraints.xu)

    def test_nan(self):
        bounds = Bounds([np.nan, 0.0], [1.0, 1.0])
        constraints = BoundConstraints(bounds)
        np.testing.assert_array_equal(constraints.xl, [-np.inf, 0.0])
        assert constraints.m == 3
        bounds = Bounds([0.0, 0.0], [1.0, np.nan])
        constraints = BoundConstraints(bounds)
        np.testing.assert_array_equal(constraints.xu, [1.0, np.inf])
        assert constraints.m == 3

    def test_infeasible(self):
        bounds = Bounds([2.0, 0.0], [1.0, 1.0])
        constraints = BoundConstraints(bounds)
        assert not constraints.is_feasible
        x = [2.0, 2.0]
        np.testing.assert_array_equal(constraints.project(x), x)
        np.testing.assert_allclose(constraints.maxcv(x), 1.0)


class TestLinearConstraints:

    def test_simple(self):
        linear_constraints = [
            LinearConstraint([[1.0, 1.0]], [0.0], [1.0]),
            LinearConstraint([[2.0, 1.0]], [1.0], [1.0]),
        ]
        constraints = LinearConstraints(linear_constraints, 2, True)
        np.testing.assert_array_equal(
            constraints.a_ub,
            [[1.0, 1.0], [-1.0, -1.0]],
        )
        np.testing.assert_array_equal(constraints.b_ub, [1.0, 0.0])
        np.testing.assert_array_equal(constraints.a_eq, [[2.0, 1.0]])
        np.testing.assert_array_equal(constraints.b_eq, [1.0])
        assert constraints.m_ub == 2
        assert constraints.m_eq == 1
        np.testing.assert_allclose(constraints.maxcv([0.5, 0.0]), 0.0)

    def test_nan(self):
        linear_constraints = [LinearConstraint([[1.0, np.nan]], [0.0], [1.0])]
        constraints = LinearConstraints(linear_constraints, 2, True)
        np.testing.assert_array_equal(
            constraints.a_ub,
            [[1.0, 0.0], [-1.0, 0.0]],
        )
        np.testing.assert_array_equal(constraints.b_ub, [1.0, 0.0])
        assert constraints.m_ub == 2
        assert constraints.m_eq == 0
        linear_constraints = [LinearConstraint([[1.0, 1.0]], [np.nan], [1.0])]
        constraints = LinearConstraints(linear_constraints, 2, True)
        np.testing.assert_array_equal(constraints.a_ub, [[1.0, 1.0]])
        np.testing.assert_array_equal(constraints.b_ub, [1.0])
        assert constraints.m_ub == 1
        assert constraints.m_eq == 0
        linear_constraints = [LinearConstraint([[1.0, 1.0]], [0.0], [np.nan])]
        constraints = LinearConstraints(linear_constraints, 2, True)
        np.testing.assert_array_equal(constraints.a_ub, [[-1.0, -1.0]])
        np.testing.assert_array_equal(constraints.b_ub, [0.0])
        assert constraints.m_ub == 1
        assert constraints.m_eq == 0

    def test_inf(self):
        linear_constraints = [
            LinearConstraint([[1.0, 1.0]], [0.0], [np.inf]),
        ]
        constraints = LinearConstraints(linear_constraints, 2, True)
        np.testing.assert_array_equal(constraints.a_ub, [[-1.0, -1.0]])
        np.testing.assert_array_equal(constraints.b_ub, [0.0])
        assert constraints.m_ub == 1

    def test_exceptions(self):
        linear_constraints = [
            LinearConstraint([[1.0, 1.0]], [0.0], [1.0]),
            LinearConstraint([[3.0, 2.0, 1.0]], [1.0], [1.0]),
        ]
        with pytest.raises(ValueError):
            LinearConstraints(linear_constraints, 2, True)
        with pytest.raises(ValueError):
            LinearConstraints(linear_constraints, 3, True)
