#!/usr/bin/env python3
import numpy as np

from cobyqa import minimize

np.set_printoptions(precision=4, suppress=True)


def q(x):
    """
    A quadratic objective function.
    """
    return (x[0] - 1.0) ** 2.0 + (x[1] - 2.5) ** 2.0


if __name__ == '__main__':
    # Set the constraints of the problem.
    x0 = [2.0, 0.0]
    xl = [0.0, 0.0]
    Aub = [[-1.0, 2.0], [1.0, 2.0], [1.0, -2.0]]
    bub = [2.0, 6.0, 2.0]

    res = minimize(q, x0, xl=xl, Aub=Aub, bub=bub)
    print(res)
