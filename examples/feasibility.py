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


def cub(x):
    return [(x[1] + x[0] ** 2) ** 2 + 0.1 * x[1] ** 2 - 1, x[1] - np.exp(-x[0]) + 3]


if __name__ == '__main__':
    aub = [[-1.0, 1.0]]
    bub = [-4.0]

    options = {'verbose': True}
    minimize(None, np.zeros(2), aub=aub, bub=bub, cub=cub, options=options)
