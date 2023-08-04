import numpy as np
import pytest
from scipy.optimize import rosen

from cobyqa.problem import ObjectiveFunction, BoundConstraints, LinearConstraints, NonlinearConstraints, Problem


class TestObjectiveFunction:

    def test_simple(self, capsys):
        # Check an objective function with verbose=False.
        obj = ObjectiveFunction(rosen, False, False)
        x = np.zeros(5)
        fun = obj(x)
        captured = capsys.readouterr()
        assert obj.name == 'rosen'
        assert fun == rosen(x)
        assert obj.n_eval == 1
        assert captured.out == ''

        # Check an objective function with verbose=True.
        obj = ObjectiveFunction(rosen, True, False)
        fun = obj(x)
        captured = capsys.readouterr()
        assert fun == rosen(x)
        assert obj.n_eval == 1
        assert captured.out == f'rosen({x}) = {rosen(x)}\n'

        # Check that no storage is performed.
        assert obj.f_hist == []
        assert obj.x_hist == []

    def test_none(self, capsys):
        # Check an objective function with no function and verbose=False.
        obj = ObjectiveFunction(None, False, False)
        x = np.zeros(5)
        fun = obj(x)
        captured = capsys.readouterr()
        assert obj.name == ''
        assert fun == 0.0
        assert obj.n_eval == 0
        assert captured.out == ''

        # Check an objective function with no function and verbose=True.
        obj = ObjectiveFunction(None, True, False)
        fun = obj(x)
        captured = capsys.readouterr()
        assert fun == 0.0
        assert obj.n_eval == 0
        assert captured.out == ''

    def test_barrier(self):
        # Check an objective function with an infinite value.
        obj = ObjectiveFunction(lambda x: np.inf, False, False)
        x = np.zeros(5)
        fun = obj(x)
        assert obj.name == '<lambda>'
        assert np.isfinite(fun)
        assert obj.n_eval == 1

        # Check an objective function with a NaN value.
        obj = ObjectiveFunction(lambda x: np.nan, False, False)
        x = np.zeros(5)
        fun = obj(x)
        assert obj.name == '<lambda>'
        assert np.isfinite(fun)
        assert obj.n_eval == 1


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
    pass


class TestNonlinearConstraints:
    pass


class TestProblem:
    pass
