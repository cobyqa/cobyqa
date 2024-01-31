#!/usr/bin/env python3
"""
Minimize the Rosenbrock function subject to simple bounds and randomly
generated linear inequality and equality constraints.
"""
import numpy as np
from cobyqa import minimize
from scipy.optimize import Bounds, LinearConstraint, rosen


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, m_linear_ub, m_linear_eq = 10, 3, 2

    # Generate an initial guess satisfying the bound constraints.
    bounds = Bounds(-2.048 * np.ones(n), 2.048 * np.ones(n))
    x0 = rng.uniform(-3.0, 3.0, n)
    x0 = np.maximum(bounds.lb, np.minimum(bounds.ub, x0))

    # Generate feasible linear inequality and equality constraints.
    x_rand = rng.uniform(bounds.lb, bounds.ub)
    aub = rng.standard_normal((m_linear_ub, n))
    bub = np.dot(aub, x_rand) + rng.uniform(0.0, 1.0, m_linear_ub)
    aeq = rng.standard_normal((m_linear_eq, n))
    beq = np.dot(aeq, x_rand)
    constraints = [
        LinearConstraint(aub, -np.inf, bub),
        LinearConstraint(aeq, beq, beq),
    ]
    res = minimize(rosen, x0, bounds=bounds, constraints=constraints)
    print(res)
