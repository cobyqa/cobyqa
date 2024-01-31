#!/usr/bin/env python3
r"""
Solve the feasibility problem

.. math::

    (y + x^2)^2 + 0.1 y^2 \le 1,\\
    y \le e^{-x} - 3,\\
    y \le x - 4.

"""
import numpy as np
from cobyqa import minimize
from scipy.optimize import LinearConstraint, NonlinearConstraint


def cub(x):
    return [(x[1] + x[0] ** 2) ** 2 + 0.1 * x[1] ** 2, x[1] - np.exp(-x[0])]


if __name__ == "__main__":
    x0 = [0.0, 0.0]
    constraints = [
        LinearConstraint([[-1.0, 1.0]], -np.inf, [-4.0]),
        NonlinearConstraint(cub, -np.inf, [1.0, -3.0]),
    ]
    options = {
        "disp": True,
        "feasibility_tol": np.finfo(float).eps,
    }
    minimize(None, x0, constraints=constraints, options=options)
