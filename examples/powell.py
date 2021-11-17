#!/usr/bin/env python3
"""
Solve Examples A-G of [1]_.

References
----------
.. [1] M. J. D. Powell. "A direct search optimization method that models the
   objective and constraint functions by linear interpolation." In: Advances in
   Optimization and Numerical Analysis. Ed. by S. Gomez and J. P. Hennart.
   Dordrecht, NL: Springer, 1994, pp. 51--67.
"""

import numpy as np

from cobyqa import minimize


def fun(x, no):
    if no == 'A':
        fx = 10.0 * (x[0] + 1.0) ** 2.0 + x[1] ** 2.0
    elif no == 'B':
        fx = x[0] * x[1]
    elif no == 'C':
        fx = x[0] * x[1] * x[2]
    elif no == 'D':
        fx = (x[0] ** 2.0 - x[1]) ** 2.0 + (1.0 + x[0]) ** 2.0
    elif no == 'E':
        fx = 10.0 * (x[0] ** 2.0 - x[1]) ** 2.0 + (1.0 + x[0]) ** 2.0
    elif no == 'F':
        fx = -x[0] - x[1]
    elif no == 'G':
        fx = x[2]
    else:
        raise NotImplementedError
    return fx


def x0(no):
    if no in 'ABDEF':
        n = 2
    elif no in 'CG':
        n = 3
    else:
        raise NotImplementedError
    return np.ones(n)


def cub(x, no):
    if no in 'ADE':
        cx = []
    elif no == 'B':
        cx = x[0] ** 2.0 + x[1] ** 2.0 - 1.0
    elif no == 'C':
        cx = x[0] ** 2.0 + 2.0 * x[1] ** 2.0 + 3.0 * x[2] ** 2.0 - 1.0
    elif no == 'F':
        cx = [x[0] ** 2.0 + x[1] ** 2.0 - 1.0, x[0] ** 2.0 - x[1]]
    elif no == 'G':
        cx = [
            -5.0 * x[0] + x[1] - x[2],
            5.0 * x[0] + x[1] - x[2],
            x[0] ** 2.0 + x[1] ** 2.0 + 4.0 * x[1] - x[2],
        ]
    else:
        raise NotImplementedError
    return np.array(cx)


def solution(no):
    if no == 'A':
        s = ([-1.0, 0.0],)
    elif no == 'B':
        e = np.sqrt(2.0) / 2.0
        s = ([-e, e], [e, -e])
    elif no == 'C':
        e1 = 1.0 / np.sqrt(3.0)
        e2 = 1.0 / np.sqrt(6.0)
        e3 = 1.0 / 3.0
        s = ([-e1, e2, e3], [e1, -e2, e3], [e1, e2, -e3], [-e1, -e2, -e3])
    elif no in 'DE':
        s = ([-1.0, 1.0],)
    elif no == 'F':
        e = 1.0 / np.sqrt(2.0)
        s = ([e, e],)
    elif no == 'G':
        s = ([0.0, -3.0, -3.0],)
    else:
        raise NotImplementedError
    return (np.array(xs) for xs in s)


def _maxcv(x, no):
    cx = cub(x, no)
    return np.max(-cx, initial=0.0)


def _distance(x, no):
    return min(np.linalg.norm(x - xs) for xs in solution(no))


if __name__ == '__main__':
    for problem in 'ABCDEFG':
        print(f'Problem {problem}')
        print('---------')
        res = minimize(fun, x0(problem), problem, cub=cub)
        print(f'Function values      : {res.nfev}')
        print(f'Objective function   : {res.fun:.4e}')
        print(f'Constraint violation : {_maxcv(res.x, problem):.4e}')
        print(f'Distance to solution : {_distance(res.x, problem):.4e}')
        print()
