#!/usr/bin/env python3
import numpy as np
from scipy.optimize import rosen

from cobyqa import minimize

np.set_printoptions(precision=4, linewidth=np.inf, suppress=True)


def ball(x, radius=1.0, order=None):
    """
    The ball constraint function.
    """
    return np.linalg.norm(x, order) - radius


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    n, mlub, mleq = 10, 3, 0

    # Generate a initial guess satisfying the bound constraints.
    xl = -2.048 * np.ones(n)
    xu = 2.048 * np.ones(n)
    x0 = rng.uniform(-3.0, 3.0, n)
    x0 = np.maximum(xl, np.minimum(xu, x0))

    # Generate feasible linear inequality and equality constraints.
    x_alt = rng.uniform(xl, xu)
    Aub = rng.standard_normal((mlub, n))
    bub = np.dot(Aub, x_alt) + rng.uniform(0.0, 1.0, mlub)
    Aeq = rng.standard_normal((mleq, n))
    beq = np.dot(Aeq, x_alt)

    res = minimize(rosen, x0, xl=xl, xu=xu, Aub=Aub, bub=bub, Aeq=Aeq, beq=beq)
    print(res)
