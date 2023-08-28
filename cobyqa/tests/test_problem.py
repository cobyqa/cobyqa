import numpy as np
import pytest
from scipy.optimize import rosen

from cobyqa.problem import ObjectiveFunction, BoundConstraints, LinearConstraints, NonlinearConstraints, Problem


class TestObjectiveFunction:

    def test_simple(self, capsys):
        # Check an objective function with verbose=False.
        obj = ObjectiveFunction(rosen, False, False, 0)
        x = np.zeros(5)
        fun = obj(x)
        captured = capsys.readouterr()
        assert obj.name == 'rosen'
        assert fun == rosen(x)
        assert obj.n_eval == 1
        assert captured.out == ''

        # Check an objective function with verbose=True.
        obj = ObjectiveFunction(rosen, True, False, 0)
        fun = obj(x)
        captured = capsys.readouterr()
        assert fun == rosen(x)
        assert obj.n_eval == 1
        assert captured.out == f'rosen({x}) = {rosen(x)}\n'

        # Check that no storage is performed.
        assert obj.f_hist.size == 0
        assert obj.x_hist.size == 0

    def test_none(self, capsys):
        # Check an objective function with no function and verbose=False.
        obj = ObjectiveFunction(None, False, False, 0)
        x = np.zeros(5)
        fun = obj(x)
        captured = capsys.readouterr()
        assert obj.name == ''
        assert fun == 0.0
        assert obj.n_eval == 0
        assert captured.out == ''

        # Check an objective function with no function and verbose=True.
        obj = ObjectiveFunction(None, True, False, 0)
        fun = obj(x)
        captured = capsys.readouterr()
        assert fun == 0.0
        assert obj.n_eval == 0
        assert captured.out == ''

    def test_barrier(self):
        # Check an objective function with an infinite value.
        obj = ObjectiveFunction(lambda x: np.inf, False, False, 0)
        x = np.zeros(5)
        fun = obj(x)
        assert obj.name == '<lambda>'
        assert np.isfinite(fun)
        assert obj.n_eval == 1

        # Check an objective function with a NaN value.
        obj = ObjectiveFunction(lambda x: np.nan, False, False, 0)
        x = np.zeros(5)
        fun = obj(x)
        assert obj.name == '<lambda>'
        assert np.isfinite(fun)
        assert obj.n_eval == 1

    def test_store(self):
        obj = ObjectiveFunction(rosen, False, True, 1)
        x = np.zeros(5)
        obj(x)
        assert obj.f_hist.size == 1
        assert obj.x_hist.shape[0] == 1
        assert obj.f_hist[0] == rosen(x)
        assert np.all(obj.x_hist[0, :] == x)


class TestBoundConstraints:

    def test_simple(self):
        bounds = BoundConstraints(np.zeros(5), np.ones(5))
        assert bounds.is_feasible
        assert bounds.m == 10
        assert np.all(bounds.xl == 0.0)
        assert np.all(bounds.xu == 1.0)
        assert bounds.resid(0.5 * np.ones(5)) == 0.0

    def test_exceptions(self):
        with pytest.raises(ValueError):
            BoundConstraints(np.zeros(5), np.zeros(4))


class TestLinearConstraints:

    def test_simple(self):
        linear = LinearConstraints(np.ones((2, 5)), np.ones(2), False)
        assert linear.m == 2
        assert not linear.is_equality
        assert np.all(linear.a == 1.0)
        assert np.all(linear.b == 1.0)
        assert linear.resid(np.zeros(5)) == 0.0

    def test_exceptions(self):
        with pytest.raises(ValueError):
            LinearConstraints(np.ones((2, 5)), np.ones(3), False)


class TestNonlinearConstraints:

    def test_simple(self, capsys):
        # Check a constraint function with verbose=False.
        nonlinear = NonlinearConstraints(np.sin, False, False, False, 0)
        x = np.zeros(5)
        fun = nonlinear(x)
        captured = capsys.readouterr()
        assert nonlinear.m == 5
        assert not nonlinear.is_equality
        assert nonlinear.name == 'sin'
        assert np.all(fun == np.sin(x))
        assert nonlinear.n_eval == 1
        assert captured.out == ''
        assert nonlinear.resid(x) < 50.0 * np.finfo(float).eps
        assert nonlinear.resid(x, np.zeros(5)) == 0.0

        # Check a constraint function with verbose=True.
        nonlinear = NonlinearConstraints(np.sin, False, True, False, 0)
        fun = nonlinear(x)
        captured = capsys.readouterr()
        assert np.all(fun == np.sin(x))
        assert nonlinear.n_eval == 1
        assert captured.out == f'sin({x}) = {np.sin(x)}\n'

        # Check that no storage is performed.
        assert nonlinear.f_hist.size == 0
        assert nonlinear.x_hist.size == 0

    def test_none(self, capsys):
        # Check a constraint function with no function and verbose=False.
        nonlinear = NonlinearConstraints(None, False, False, False, 0)
        x = np.zeros(5)
        fun = nonlinear(x)
        captured = capsys.readouterr()
        assert nonlinear.name == ''
        assert fun.size == 0
        assert nonlinear.n_eval == 0
        assert captured.out == ''

        # Check an objective function with no function and verbose=True.
        nonlinear = NonlinearConstraints(None, False, True, False, 0)
        fun = nonlinear(x)
        captured = capsys.readouterr()
        assert fun.size == 0
        assert nonlinear.n_eval == 0
        assert captured.out == ''

    def test_barrier(self):
        # Check an objective function with an infinite value.
        nonlinear = NonlinearConstraints(lambda x: [0, np.inf], False, False, False, 0)
        x = np.zeros(5)
        fun = nonlinear(x)
        assert nonlinear.name == '<lambda>'
        assert np.all(np.isfinite(fun))
        assert nonlinear.n_eval == 1

        # Check an objective function with a NaN value.
        nonlinear = NonlinearConstraints(lambda x: [0, np.nan], False, False, False, 0)
        x = np.zeros(5)
        fun = nonlinear(x)
        assert nonlinear.name == '<lambda>'
        assert np.all(np.isfinite(fun))
        assert nonlinear.n_eval == 1

    def test_store(self):
        nonlinear = NonlinearConstraints(np.sin, False, False, True, 1)
        x = np.zeros(5)
        nonlinear(x)
        assert nonlinear.f_hist.shape[0] == 1
        assert nonlinear.x_hist.size == 0
        assert np.all(nonlinear.f_hist[0, :] == np.sin(x))


class TestProblem:

    def test_simple(self):
        obj = ObjectiveFunction(rosen, False, False, 0)
        x0 = np.zeros(5)
        bounds = BoundConstraints(np.zeros(5), np.ones(5))
        linear_ub = LinearConstraints(np.ones((2, 5)), np.ones(2), False)
        linear_eq = LinearConstraints(np.ones((2, 5)), np.ones(2), True)
        nonlinear_ub = NonlinearConstraints(np.sin, False, False, False, 0)
        nonlinear_eq = NonlinearConstraints(np.sin, True, False, False, 0)
        pb = Problem(obj, x0, bounds, linear_ub, linear_eq, nonlinear_ub, nonlinear_eq)
        fun_x0 = pb.fun(pb.x0)
        cub_x0 = pb.cub(pb.x0)
        ceq_x0 = pb.ceq(pb.x0)
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
        assert pb.resid(pb.x0, cub_x0, ceq_x0) < 1.0 + 50.0 * np.finfo(float).eps
        assert fun_x0 == rosen(x0)
        assert np.all(cub_x0 == np.sin(x0))
        assert np.all(ceq_x0 == np.sin(x0))

    def test_fixed(self):
        obj = ObjectiveFunction(rosen, False, False, 0)
        x0 = np.zeros(5)
        bounds = BoundConstraints(np.block([np.zeros(4), 1.0]), np.ones(5))
        linear_ub = LinearConstraints(np.ones((2, 5)), np.ones(2), False)
        linear_eq = LinearConstraints(np.ones((2, 5)), np.ones(2), True)
        nonlinear_ub = NonlinearConstraints(None, False, False, False, 0)
        nonlinear_eq = NonlinearConstraints(None, True, False, False, 0)
        pb = Problem(obj, x0, bounds, linear_ub, linear_eq, nonlinear_ub, nonlinear_eq)
        pb.cub(pb.x0)
        pb.ceq(pb.x0)
        assert pb.n == 4
        assert pb.n_orig == 5
        assert pb.n_eval == 0
        assert pb.m_bounds == 8
        assert pb.m_linear_ub == 2
        assert pb.m_linear_eq == 2
        assert pb.m_nonlinear_ub == 0
        assert pb.m_nonlinear_eq == 0
        assert pb.type == 'linearly constrained'