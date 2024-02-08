import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, rosen

from ..models import Interpolation, Quadratic
from ..problem import (
    ObjectiveFunction,
    BoundConstraints,
    LinearConstraints,
    NonlinearConstraints,
    Problem,
)
from ..settings import Options


class TestInterpolation:

    def test_simple(self):
        problem = get_problem([0.5, 0.5])
        options = {
            Options.RHOBEG.value: 0.5,
            Options.RHOEND.value: 1e-6,
            Options.NPT: ((problem.n + 1) * (problem.n + 2)) // 2,
            Options.DEBUG.value: True,
        }
        interpolation = Interpolation(problem, options)
        assert interpolation.n == problem.n
        assert interpolation.npt == options[Options.NPT]
        np.testing.assert_allclose(
            interpolation.x_base,
            problem.x0,
            atol=1e-15,
        )
        for k in range(interpolation.npt):
            point = interpolation.point(k)
            np.testing.assert_allclose(
                np.maximum(point, problem.bounds.xl),
                point,
                atol=1e-15,
            )
            np.testing.assert_allclose(
                np.minimum(point, problem.bounds.xu),
                point,
                atol=1e-15,
            )

    def test_close(self):
        problem = get_problem([0.0, 0.5])
        options = {
            Options.RHOBEG.value: 0.5,
            Options.RHOEND.value: 1e-6,
            Options.NPT: ((problem.n + 1) * (problem.n + 2)) // 2,
            Options.DEBUG.value: True,
        }
        interpolation = Interpolation(problem, options)
        np.testing.assert_allclose(
            interpolation.x_base,
            problem.x0,
            atol=1e-15,
        )
        problem = get_problem([0.1, 0.5])
        interpolation = Interpolation(problem, options)
        np.testing.assert_allclose(
            interpolation.x_base,
            [0.0, 0.5],
            atol=1e-15,
        )
        problem = get_problem([0.3, 0.5])
        interpolation = Interpolation(problem, options)
        np.testing.assert_allclose(
            interpolation.x_base,
            [0.5, 0.5],
            atol=1e-15,
        )
        problem = get_problem([0.9, 0.5])
        interpolation = Interpolation(problem, options)
        np.testing.assert_allclose(
            interpolation.x_base,
            [1.0, 0.5],
            atol=1e-15,
        )
        problem = get_problem([0.7, 0.5])
        interpolation = Interpolation(problem, options)
        np.testing.assert_allclose(
            interpolation.x_base,
            [0.5, 0.5],
            atol=1e-15,
        )


class TestQuadratic:

    def test_simple(self):
        problem = get_problem([0.5, 0.5])
        options = {
            Options.RHOBEG.value: 0.5,
            Options.RHOEND.value: 1e-6,
            Options.NPT: ((problem.n + 1) * (problem.n + 2)) // 2,
            Options.DEBUG.value: True,
        }
        interpolation = Interpolation(problem, options)
        values = np.arange(interpolation.npt)
        model = Quadratic(interpolation, values, True)
        assert model.n == problem.n
        assert model.npt == interpolation.npt
        for k in range(interpolation.npt):
            np.testing.assert_allclose(
                model(interpolation.point(k), interpolation),
                values[k],
                atol=1e-15,
            )
        hess = model.hess(interpolation)
        for i in range(model.n):
            np.testing.assert_allclose(
                hess[:, i],
                model.hess_prod(
                    np.squeeze(np.eye(1, model.n, i)),
                    interpolation,
                ),
                atol=1e-15,
            )
            np.testing.assert_allclose(
                hess[i, i],
                model.curv(
                    np.squeeze(np.eye(1, model.n, i)),
                    interpolation,
                ),
                atol=1e-15,
            )

    def test_exceptions(self):
        problem = get_problem([0.5, 0.5])
        options = {
            Options.RHOBEG.value: 0.5,
            Options.RHOEND.value: 1e-6,
            Options.NPT: problem.n,
            Options.DEBUG.value: True,
        }
        interpolation = Interpolation(problem, options)
        values = np.zeros(interpolation.npt)
        with pytest.raises(ValueError):
            model = Quadratic(interpolation, values, True)


def get_problem(x0):
    obj = ObjectiveFunction(rosen, False, True)
    bounds = BoundConstraints(Bounds([0.0, 0.0], [1.0, 1.0]))
    linear_constraints = LinearConstraints(
        [
            LinearConstraint([[1.0, 1.0]], [0.0], [1.0]),
            LinearConstraint([[2.0, 1.0]], [1.0], [1.0]),
        ],
        2,
        True,
    )
    nonlinear_constraints = NonlinearConstraints(
        [
            NonlinearConstraint(np.cos, [-0.5, -0.5], [0.0, 0.0]),
            NonlinearConstraint(np.sin, [0.0, 0.0], [0.0, 0.0]),
        ],
        False,
        True,
    )
    return Problem(
        obj,
        x0,
        bounds,
        linear_constraints,
        nonlinear_constraints,
        None,
        0.0,
        False,
        False,
        0,
        1,
        True,
    )
