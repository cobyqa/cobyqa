import numpy as np
import pytest

from ..problem import (
    ObjectiveFunction,
    BoundConstraints,
    LinearConstraints,
    NonlinearConstraints,
    Problem,
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

    pass
