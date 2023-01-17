#!/usr/bin/env python3
"""
Solve Example 16.4 of [1]_.

References
----------
.. [1] J. Nocedal and S. J. Wright. Numerical Optimization. Second. Springer
   Ser. Oper. Res. Financ. Eng. New York, NY, US: Springer, 2006.
"""
import numpy as np

from cobyqa import minimize

np.set_printoptions(precision=4, suppress=True)


def fun(x):
    return (x[0] - 1.0) ** 2.0 + (x[1] - 2.5) ** 2.0


if __name__ == "__main__":
    x0 = [2.0, 0.0]
    xl = [0.0, 0.0]
    aub = [[-1.0, 2.0], [1.0, 2.0], [1.0, -2.0]]
    bub = [2.0, 6.0, 2.0]

    res = minimize(fun, x0, xl=xl, aub=aub, bub=bub)
    print(res)
