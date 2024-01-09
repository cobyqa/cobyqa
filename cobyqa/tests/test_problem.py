import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, rosen

from cobyqa.problem import ObjectiveFunction, BoundConstraints, LinearConstraints, NonlinearConstraints, Problem
from cobyqa.settings import PRINT_OPTIONS


class TestObjectiveFunction:

    def test_simple(self, capsys):
        # Check an objective function with verbose=False.
        obj = ObjectiveFunction(rosen, False, True)
        x = np.zeros(5)
        fun = obj(x)
        captured = capsys.readouterr()
        assert obj.name == 'rosen'
        assert fun == rosen(x)
        assert obj.n_eval == 1
        assert captured.out == ''

        # Check an objective function with verbose=True.
        obj = ObjectiveFunction(rosen, True, True)
        fun = obj(x)
        captured = capsys.readouterr()
        assert fun == rosen(x)
        assert obj.n_eval == 1
        with np.printoptions(**PRINT_OPTIONS):
            assert captured.out == f'rosen({x}) = {rosen(x)}\n'

    def test_none(self, capsys):
        # Check an objective function with no function and verbose=False.
        obj = ObjectiveFunction(None, False, True)
        x = np.zeros(5)
        fun = obj(x)
        captured = capsys.readouterr()
        assert obj.name == ''
        assert fun == 0.0
        assert obj.n_eval == 0
        assert captured.out == ''

        # Check an objective function with no function and verbose=True.
        obj = ObjectiveFunction(None, True, True)
        fun = obj(x)
        captured = capsys.readouterr()
        assert fun == 0.0
        assert obj.n_eval == 0
        assert captured.out == ''


class TestBoundConstraints:

    def test_simple(self):
        bounds = BoundConstraints(Bounds(np.zeros(5), np.ones(5)))
        assert bounds.is_feasible
        assert bounds.m == 10
        assert np.all(bounds.xl == 0.0)
        assert np.all(bounds.xu == 1.0)
        assert bounds.maxcv(0.5 * np.ones(5)) == 0.0

    def test_exceptions(self):
        with pytest.raises(ValueError):
            BoundConstraints(Bounds(np.zeros(5), np.zeros(4)))


class TestLinearConstraints:

    def test_simple(self):
        linear = LinearConstraints([LinearConstraint(np.ones((2, 5)), -np.inf, np.ones(2))], 5, True)
        assert linear.m_ub == 2
        assert linear.m_eq == 0
        assert np.all(linear.a_ub == 1.0)
        assert np.all(linear.b_ub == 1.0)
        assert linear.maxcv(np.zeros(5)) == 0.0

    def test_exceptions(self):
        with pytest.raises(ValueError):
            LinearConstraints([LinearConstraint(np.ones((2, 5)), -np.inf, np.ones(3))], 5, True)


class TestNonlinearConstraints:

    def test_simple(self, capsys):
        # Check a constraint function with verbose=False.
        nonlinear = NonlinearConstraints([NonlinearConstraint(np.sin, -np.inf, np.zeros(5))], False, True)
        x = np.zeros(5)
        cub, ceq = nonlinear(x)
        captured = capsys.readouterr()
        assert nonlinear.m_ub == 5
        assert nonlinear.m_eq == 0
        assert np.all(cub == np.sin(x))
        assert nonlinear.n_eval == 1
        assert captured.out == ''
        assert nonlinear.maxcv(x) < 50.0 * np.finfo(float).eps
        assert nonlinear.maxcv(x, np.zeros(5)) == 0.0

        # Check a constraint function with verbose=True.
        nonlinear = NonlinearConstraints([NonlinearConstraint(np.sin, np.zeros(5), np.zeros(5))], True, True)
        cub, ceq = nonlinear(x)
        captured = capsys.readouterr()
        assert np.all(ceq == np.sin(x))
        assert nonlinear.n_eval == 1
        with np.printoptions(**PRINT_OPTIONS):
            assert captured.out == f'sin({x}) = {np.sin(x)}\n'


class TestProblem:

    def test_simple(self):
        obj = ObjectiveFunction(rosen, False, True)
        x0 = np.zeros(5)
        bounds = BoundConstraints(Bounds(np.zeros(5), np.ones(5)))
        linear = LinearConstraints([LinearConstraint(np.ones((2, 5)), -np.inf, np.ones(2)), LinearConstraint(np.ones((2, 5)), np.ones(2), np.ones(2))], 5, True)
        nonlinear = NonlinearConstraints([NonlinearConstraint(np.sin, -np.inf, 0), NonlinearConstraint(np.sin, 0, 0)], False, True)
        pb = Problem(obj, x0, bounds, linear, nonlinear, None, 1e-8, False, False, 1, 1, True)
        fun_x0, cub_x0, ceq_x0 = pb(pb.x0)
        assert pb.n == 5
        assert pb.n_orig == 5
        assert pb.n_eval == 1
        assert pb.m_bounds == 10
        assert pb.m_linear_ub == 2
        assert pb.m_linear_eq == 2
        assert pb.m_nonlinear_ub == 5
        assert pb.m_nonlinear_eq == 5
        assert pb.type == 'nonlinearly constrained'
        assert pb.fun_name == 'rosen'
        assert not pb.is_feasibility
        assert pb.maxcv(pb.x0, cub_x0, ceq_x0) < 1.0 + 50.0 * np.finfo(float).eps
        assert fun_x0 == rosen(x0)
        assert np.all(cub_x0 == np.sin(x0))
        assert np.all(ceq_x0 == np.sin(x0))

    def test_fixed(self):
        obj = ObjectiveFunction(rosen, False, True)
        x0 = np.zeros(5)
        bounds = BoundConstraints(Bounds(np.block([np.zeros(4), 1.0]), np.ones(5)))
        linear = LinearConstraints([LinearConstraint(np.ones((2, 5)), -np.inf, np.ones(2)), LinearConstraint(np.ones((2, 5)), np.ones(2), np.ones(2))], 5, True)
        nonlinear = NonlinearConstraints([], False, True)
        pb = Problem(obj, x0, bounds, linear, nonlinear, None, 1e-8, False, False, 1, 1, True)
        pb(pb.x0)
        assert pb.n == 4
        assert pb.n_orig == 5
        assert pb.n_eval == 1
        assert pb.m_bounds == 8
        assert pb.m_linear_ub == 2
        assert pb.m_linear_eq == 2
        assert pb.m_nonlinear_ub == 0
        assert pb.m_nonlinear_eq == 0
        assert pb.type == 'linearly constrained'
