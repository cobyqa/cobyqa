#!/usr/bin/env python3
"""
Minimize the Rosenbrock function subject to simple bounds and randomly generated
linear inequality and equality constraints.
"""
import numpy as np
from scipy.optimize import rosen

from cobyqa import minimize


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    n, m_linear_ub, m_linear_eq = 10, 3, 2

    # Generate an initial guess satisfying the bound constraints.
    xl = -2.048 * np.ones(n)
    xu = 2.048 * np.ones(n)
    x0 = rng.uniform(-3.0, 3.0, n)
    x0 = np.maximum(xl, np.minimum(xu, x0))

    # Generate feasible linear inequality and equality constraints.
    x_rand = rng.uniform(xl, xu)
    aub = rng.standard_normal((m_linear_ub, n))
    bub = np.dot(aub, x_rand) + rng.uniform(0.0, 1.0, m_linear_ub)
    aeq = rng.standard_normal((m_linear_eq, n))
    beq = np.dot(aeq, x_rand)

    res = minimize(rosen, x0, xl=xl, xu=xu, aeq=aeq, beq=beq)  # , aub=aub, bub=bub
    print(res)
