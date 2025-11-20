import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, rosen
from concurrent.futures import ThreadPoolExecutor

from ..models import Interpolation, Quadratic, Models, build_system
from ..problem import (
    ObjectiveFunction,
    BoundConstraints,
    LinearConstraints,
    NonlinearConstraints,
    Problem,
)
from ..settings import Options
from ..utils import MaxEvalError, FeasibleSuccess, TargetSuccess


class TestInterpolation:

    def test_simple(self):
        problem = get_problem([0.5, 0.5])
        options = {
            Options.RHOBEG.value: 0.5,
            Options.RHOEND.value: 1e-6,
            Options.NPT.value: ((problem.n + 1) * (problem.n + 2)) // 2,
            Options.DEBUG.value: True,
        }
        interpolation = Interpolation(problem, options)
        assert interpolation.n == problem.n
        assert interpolation.npt == options[Options.NPT]
        np.testing.assert_allclose(
            interpolation.x_base,
            problem.x0,
            atol=1e-13,
        )
        for k in range(interpolation.npt):
            point = interpolation.point(k)
            np.testing.assert_allclose(
                np.maximum(point, problem.bounds.xl),
                point,
                atol=1e-13,
            )
            np.testing.assert_allclose(
                np.minimum(point, problem.bounds.xu),
                point,
                atol=1e-13,
            )

    def test_close(self):
        problem = get_problem([0.0, 0.5])
        options = {
            Options.RHOBEG.value: 0.5,
            Options.RHOEND.value: 1e-6,
            Options.NPT.value: ((problem.n + 1) * (problem.n + 2)) // 2,
            Options.DEBUG.value: True,
        }
        interpolation = Interpolation(problem, options)
        np.testing.assert_allclose(
            interpolation.x_base,
            problem.x0,
            atol=1e-13,
        )
        problem = get_problem([0.1, 0.5])
        interpolation = Interpolation(problem, options)
        np.testing.assert_allclose(
            interpolation.x_base,
            [0.0, 0.5],
            atol=1e-13,
        )
        problem = get_problem([0.3, 0.5])
        interpolation = Interpolation(problem, options)
        np.testing.assert_allclose(
            interpolation.x_base,
            [0.5, 0.5],
            atol=1e-13,
        )
        problem = get_problem([0.9, 0.5])
        interpolation = Interpolation(problem, options)
        np.testing.assert_allclose(
            interpolation.x_base,
            [1.0, 0.5],
            atol=1e-13,
        )
        problem = get_problem([0.7, 0.5])
        interpolation = Interpolation(problem, options)
        np.testing.assert_allclose(
            interpolation.x_base,
            [0.5, 0.5],
            atol=1e-13,
        )

    def test_multithread_build_system(self):
        problem_list = [
            get_problem([0.0, 0.5]),
            get_problem([0.5, 0.5]),
            get_problem([0.5, 0.5]),
        ]
        n = problem_list[0].n
        options = {
            Options.RHOBEG.value: 0.5,
            Options.RHOEND.value: 1e-6,
            Options.NPT.value: ((n + 1) * (n + 2)) // 2,
            Options.DEBUG.value: True,
        }
        interpolation_list = [
            Interpolation(problem, options)
            for problem in problem_list
        ]
        expected_x_values = [
            [0.0, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ]
        for interpolation, expected_x in \
                zip(interpolation_list, expected_x_values):
            np.testing.assert_allclose(
                interpolation.x_base,
                expected_x,
                atol=1e-13,
            )
        # Check that multithreaded build is equivalent to single threaded build
        systems = [
            build_system(interpolation)
            for interpolation in interpolation_list
        ]
        inner_iterations = 10
        outer_iterations = 10
        systems = systems * inner_iterations
        interpolation_list = interpolation_list * inner_iterations
        with ThreadPoolExecutor(max_workers=4) as executor:
            for j in range(outer_iterations):
                results = executor.map(build_system, interpolation_list)
                for i, actual in enumerate(results):
                    self.check_system_equal(actual, systems[i])

    def check_system_equal(self, actual, expected):
        expected_a, expected_right_scaling, \
            (expected_eig_values, expected_eig_vectors) = expected
        actual_a, actual_right_scaling, \
            (actual_eig_values, actual_eig_vectors) = actual
        np.testing.assert_allclose(
            actual_a,
            expected_a,
            rtol=0
        )
        np.testing.assert_allclose(
            actual_right_scaling,
            expected_right_scaling,
            rtol=0,
        )
        np.testing.assert_allclose(
            actual_eig_values,
            expected_eig_values,
            rtol=0,
        )
        np.testing.assert_allclose(
            actual_eig_vectors,
            expected_eig_vectors,
            rtol=0,
        )


class TestQuadratic:

    def test_simple(self):
        problem = get_problem([0.5, 0.5])
        options = {
            Options.RHOBEG.value: 0.5,
            Options.RHOEND.value: 1e-6,
            Options.NPT.value: ((problem.n + 1) * (problem.n + 2)) // 2,
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
                atol=1e-13,
            )
        hess = model.hess(interpolation)
        for i in range(model.n):
            np.testing.assert_allclose(
                hess[:, i],
                model.hess_prod(
                    np.squeeze(np.eye(1, model.n, i)),
                    interpolation,
                ),
                atol=1e-13,
            )
            np.testing.assert_allclose(
                hess[i, i],
                model.curv(
                    np.squeeze(np.eye(1, model.n, i)),
                    interpolation,
                ),
                atol=1e-13,
            )

    def test_exceptions(self):
        problem = get_problem([0.5, 0.5])
        options = {
            Options.RHOBEG.value: 0.5,
            Options.RHOEND.value: 1e-6,
            Options.NPT.value: problem.n,
            Options.DEBUG.value: True,
        }
        interpolation = Interpolation(problem, options)
        values = np.zeros(interpolation.npt)
        with pytest.raises(ValueError):
            Quadratic(interpolation, values, True)


class TestModels:

    def test_simple(self):
        problem = get_problem([0.5, 0.5])
        options = {
            Options.RHOBEG.value: 0.5,
            Options.RHOEND.value: 1e-6,
            Options.NPT.value: 2 * problem.n + 1,
            Options.MAX_EVAL.value: 1000,
            Options.FEASIBILITY_TOL.value: 1e-8,
            Options.TARGET.value: 0.0,
            Options.DEBUG.value: True,
        }
        models = Models(problem, options, 0.0)
        assert models.n == problem.n
        assert models.npt == options[Options.NPT]
        assert models.m_nonlinear_ub == problem.m_nonlinear_ub
        assert models.m_nonlinear_eq == problem.m_nonlinear_eq

    def test_max_eval(self):
        problem = get_problem([0.5, 0.5])
        options = {
            Options.RHOBEG.value: 0.5,
            Options.RHOEND.value: 1e-6,
            Options.NPT.value: 2 * problem.n + 1,
            Options.MAX_EVAL.value: problem.n,
            Options.FEASIBILITY_TOL.value: 1e-8,
            Options.TARGET.value: 0.0,
            Options.DEBUG.value: True,
        }
        with pytest.raises(MaxEvalError):
            Models(problem, options, 0.0)

    def test_feasibility_problem(self):
        obj = ObjectiveFunction(None, False, True)
        bounds = BoundConstraints(Bounds(2 * [-np.inf], 2 * [np.inf]))
        linear_constraints = LinearConstraints([], 2, True)
        nonlinear_constraints = NonlinearConstraints([], False, True)
        problem = Problem(
            obj,
            [0.0, 0.0],
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
        options = {
            Options.RHOBEG.value: 0.5,
            Options.RHOEND.value: 1e-6,
            Options.NPT.value: 2 * problem.n + 1,
            Options.MAX_EVAL.value: 1000,
            Options.FEASIBILITY_TOL.value: 1e-8,
            Options.TARGET.value: 0.0,
            Options.DEBUG.value: True,
        }
        with pytest.raises(FeasibleSuccess):
            Models(problem, options, 0.0)

    def test_target(self):
        obj = ObjectiveFunction(rosen, False, True)
        bounds = BoundConstraints(Bounds(2 * [-np.inf], 2 * [np.inf]))
        linear_constraints = LinearConstraints([], 2, True)
        nonlinear_constraints = NonlinearConstraints([], False, True)
        problem = Problem(
            obj,
            [0.0, 0.0],
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
        options = {
            Options.RHOBEG.value: 0.5,
            Options.RHOEND.value: 1e-6,
            Options.NPT.value: 2 * problem.n + 1,
            Options.MAX_EVAL.value: 1000,
            Options.FEASIBILITY_TOL.value: 1e-8,
            Options.TARGET.value: 1.0,
            Options.DEBUG.value: True,
        }
        with pytest.raises(TargetSuccess):
            Models(problem, options, 0.0)


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
