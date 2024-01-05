#!/usr/bin/env python3
"""
Solve Example 16.4 of [1]_.

References
----------
.. [1] J. Nocedal and S. J. Wright. Numerical Optimization. Springer Series in
   Operations Research and Financial Engineering. Springer, New York, NY, USA,
   second edition, 2006.
"""
import numpy as np
from cobyqa import minimize
from scipy.optimize import Bounds


def quad(x):
    return (x[0] - 1.0) ** 2.0 + (x[1] - 2.5) ** 2.0


if __name__ == '__main__':
    x0 = [2.0, 0.0]
    bounds = Bounds([0.0, 0.0], np.inf)
    aub = [[-1.0, 2.0], [1.0, 2.0], [1.0, -2.0]]
    bub = [2.0, 6.0, 2.0]

    res = minimize(quad, x0, bounds=bounds, aub=aub, bub=bub)
    print(res)
