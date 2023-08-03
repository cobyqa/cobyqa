#!/usr/bin/env python3
"""
Solve Example 16.4 of [1]_.

References
----------
.. [1] J. Nocedal and S. J. Wright. Numerical Optimization. Springer Series in
   Operations Research and Financial Engineering. Springer, New York, NY, USA,
   second edition, 2006.
"""
from cobyqa import minimize


def fun(x):
    return (x[0] - 1.0) ** 2.0 + (x[1] - 2.5) ** 2.0


if __name__ == '__main__':
    x0 = [2.0, 0.0]
    xl = [0.0, 0.0]
    aub = [[-1.0, 2.0], [1.0, 2.0], [1.0, -2.0]]
    bub = [2.0, 6.0, 2.0]

    res = minimize(fun, x0, xl=xl, aub=aub, bub=bub)
    print(res)
