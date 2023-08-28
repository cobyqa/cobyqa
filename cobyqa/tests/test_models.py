import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import rosen

from cobyqa.models import Interpolation, Quadratic, Models
from cobyqa.problem import ObjectiveFunction, BoundConstraints, LinearConstraints, NonlinearConstraints, Problem


class TestInterpolation:

    @pytest.mark.parametrize('n', [1, 2, 10, 50])
    def test_simple(self, n):
        x0 = 0.5 * np.ones(n)
        pb = _problem(x0)
        options = {'radius_init': 0.5, 'radius_final': 1e-6, 'npt': (n + 1) * (n + 2) // 2}
        interpolation = Interpolation(pb, options)
        assert_allclose(interpolation.x_base, 0.5)
        for k in range(options['npt']):
            assert options['radius_init'] == 0.5
            assert np.all(pb.bounds.xl <= interpolation.point(k))
            assert np.all(interpolation.point(k) <= pb.bounds.xu)

    @pytest.mark.parametrize('n', [1, 2, 10, 50])
    def test_close(self, n):
        x0 = 0.5 * np.ones(n)
        x0[0] = 0.4
        if n > 1:
            x0[1] = 0.6
        pb = _problem(x0)
        options = {'radius_init': 0.5, 'radius_final': 1e-6, 'npt': (n + 1) * (n + 2) // 2}
        interpolation = Interpolation(pb, options)
        assert_allclose(interpolation.x_base, 0.5)
        for k in range(options['npt']):
            assert options['radius_init'] == 0.5
            assert np.all(pb.bounds.xl <= interpolation.point(k))
            assert np.all(interpolation.point(k) <= pb.bounds.xu)

    @pytest.mark.parametrize('n', [1, 2, 10, 50])
    def test_very_close(self, n):
        x0 = 0.5 * np.ones(n)
        x0[0] = 0.1
        if n > 1:
            x0[1] = 0.9
        pb = _problem(x0)
        options = {'radius_init': 0.5, 'radius_final': 1e-6, 'npt': (n + 1) * (n + 2) // 2}
        interpolation = Interpolation(pb, options)
        assert_allclose(interpolation.x_base[0], 0.0)
        if n > 1:
            assert_allclose(interpolation.x_base[1], 1.0)
            assert_allclose(interpolation.x_base[2:], 0.5)
        for k in range(options['npt']):
            assert options['radius_init'] == 0.5
            assert np.all(pb.bounds.xl <= interpolation.point(k))
            assert np.all(interpolation.point(k) <= pb.bounds.xu)

    @pytest.mark.parametrize('n', [1, 2, 10, 50])
    def test_reduce_radius(self, n):
        x0 = 0.5 * np.ones(n)
        pb = _problem(x0)
        options = {'radius_init': 1.0, 'radius_final': 1e-6, 'npt': (n + 1) * (n + 2) // 2}
        interpolation = Interpolation(pb, options)
        assert_allclose(interpolation.x_base, 0.5)
        for k in range(options['npt']):
            assert options['radius_init'] == 0.5
            assert np.all(pb.bounds.xl <= interpolation.point(k))
            assert np.all(interpolation.point(k) <= pb.bounds.xu)


class TestQuadratic:

    @pytest.mark.parametrize('n', [1, 2, 10, 50])
    @pytest.mark.parametrize('npt_f', [
        lambda n: n + 1,
        lambda n: n + 2,
        lambda n: 2 * n + 1,
        lambda n: (n + 1) * (n + 2) // 2,
    ])
    def test_simple(self, n, npt_f):
        x0 = 0.5 * np.ones(n)
        pb = _problem(x0)
        options = {'radius_init': 1.0, 'radius_final': 1e-6, 'npt': npt_f(n)}
        interpolation = Interpolation(pb, options)
        for seed in range(100):
            rng = np.random.default_rng(seed)
            values = rng.standard_normal(options['npt'])
            quadratic = Quadratic(interpolation, values)

            # Check basic properties.
            assert quadratic.n == n
            assert quadratic.npt == options['npt']

            # Check the interpolation conditions.
            tol = 10.0 * np.sqrt(np.finfo(float).eps) * options['npt']
            for k in range(options['npt']):
                assert abs(quadratic(interpolation.point(k), interpolation) - values[k]) < tol

            # Check the Hessian matrix.
            hess = quadratic.hess(interpolation)
            for i in range(n):
                coord_vec = np.atleast_1d(np.squeeze(np.eye(1, n, i)))
                assert np.linalg.norm(quadratic.hess_prod(coord_vec, interpolation) - hess[:, i]) < tol

            # Check the curvature.
            v = rng.standard_normal(n)
            assert abs(quadratic.curv(v, interpolation) - v @ quadratic.hess_prod(v, interpolation)) < tol


def _problem(x0):
    obj = ObjectiveFunction(rosen, False, False, 0)
    bounds = BoundConstraints(np.zeros(x0.size), np.ones(x0.size))
    linear_ub = LinearConstraints(np.ones((2, x0.size)), np.ones(2), False)
    linear_eq = LinearConstraints(np.ones((2, x0.size)), np.ones(2), True)
    nonlinear_ub = NonlinearConstraints(None, False, False, False, 0)
    nonlinear_eq = NonlinearConstraints(None, True, False, False, 0)
    return Problem(obj, x0, bounds, linear_ub, linear_eq, nonlinear_ub, nonlinear_eq)