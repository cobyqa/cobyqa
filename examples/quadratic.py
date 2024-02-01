#!/usr/bin/env python3
"""
Solve Example 16.4 of [1]_.

References
----------
.. [1] J. Nocedal and S. J. Wright. *Numerical Optimization*. Springer Ser.
   Oper. Res. Financ. Eng. Springer, New York, NY, USA, second edition, 2006.
   `doi:10.1007/978-0-387-40065-5
   <https://doi.org/10.1007/978-0-387-40065-5>`_.
"""
import numpy as np
from cobyqa import minimize
from scipy.optimize import Bounds, LinearConstraint


def quad(x):
    return (x[0] - 1.0) ** 2.0 + (x[1] - 2.5) ** 2.0


if __name__ == "__main__":
    x0 = [2.0, 0.0]
    bounds = Bounds([0.0, 0.0], np.inf)
    constraints = LinearConstraint(
        [[-1.0, 2.0], [1.0, 2.0], [1.0, -2.0]],
        -np.inf,
        [2.0, 6.0, 2.0],
    )
    res = minimize(quad, x0, bounds=bounds, constraints=constraints)
    print(res)
