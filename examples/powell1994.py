#!/usr/bin/env python3
"""
Solve Examples A-J of [1]_.

References
----------
.. [1] M. J. D. Powell. "A direct search optimization method that models the
   objective and constraint functions by linear interpolation." In: Advances in
   Optimization and Numerical Analysis. Edited by S. Gomez and J. P. Hennart.
   Dordrecht, NL: Springer, 1994, pages 51–67.
"""
import numpy as np

from cobyqa import minimize


def fun(x, no):
    if no == "A":
        return 10.0 * (x[0] + 1.0) ** 2.0 + x[1] ** 2.0
    elif no == "B":
        return x[0] * x[1]
    elif no == "C":
        return x[0] * x[1] * x[2]
    elif no == "D":
        return (x[0] ** 2.0 - x[1]) ** 2.0 + (1.0 + x[0]) ** 2.0
    elif no == "E":
        return 10.0 * (x[0] ** 2.0 - x[1]) ** 2.0 + (1.0 + x[0]) ** 2.0
    elif no == "F":
        return -x[0] - x[1]
    elif no == "G":
        return x[2]
    elif no == "H":
        return x[0] ** 2.0 + x[1] ** 2.0 + 2.0 * x[2] ** 2.0 + x[3] ** 2.0 - 5.0 * x[0] - 5.0 * x[1] - 21.0 * x[2] + 7.0 * x[3]
    elif no == "I":
        return (x[0] - 10.0) ** 2.0 + 5.0 * (x[1] - 12.0) ** 2.0 + x[2] ** 4.0 + 3.0 * (x[3] - 11.0) ** 2.0 + 10.0 * x[4] ** 6.0 + 7.0 * x[5] ** 2.0 + x[6] ** 4.0 - 4.0 * x[5] * x[6] - 10.0 * x[5] - 8.0 * x[6]
    elif no == "J":
        return -0.5 * (x[0] * x[3] - x[1] * x[2] + x[2] * x[8] - x[4] * x[8] + x[4] * x[7] - x[5] * x[6])
    else:
        raise NotImplementedError


def x0(no):
    if no in "ABDEF":
        x = [1.0, 1.0]
    elif no in "CG":
        x = [1.0, 1.0, 1.0]
    elif no == "H":
        x = [0.0, 0.0, 0.0, 0.0]
    elif no == "I":
        x = [1.0, 2.0, 0.0, 4.0, 0.0, 1.0, 1.0]
    elif no == "J":
        x = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    else:
        raise NotImplementedError
    return np.array(x)


def xl(no):
    if no in "ABCDEFGHI":
        return None
    if no == "J":
        x = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.0]
    else:
        raise NotImplementedError
    return np.array(x)


def aub(no):
    if no in "ABCDEFHIJ":
        return None
    if no == "G":
        a = [[-5.0, 1.0, -1.0], [5.0, 1.0, -1.0]]
    else:
        raise NotImplementedError
    return np.array(a)


def bub(no):
    if no in "ABCDEFHIJ":
        return None
    if no == "G":
        b = [0.0, 0.0]
    else:
        raise NotImplementedError
    return np.array(b)


def cub(x, no):
    if no in "ADE":
        c = []
    elif no == "B":
        c = [x[0] ** 2.0 + x[1] ** 2.0 - 1.0]
    elif no == "C":
        c = [x[0] ** 2.0 + 2.0 * x[1] ** 2.0 + 3.0 * x[2] ** 2.0 - 1.0]
    elif no == "F":
        c = [
            x[0] ** 2.0 + x[1] ** 2.0 - 1.0,
            x[0] ** 2.0 - x[1],
        ]
    elif no == "G":
        c = [x[0] ** 2.0 + x[1] ** 2.0 + 4.0 * x[1] - x[2]]
    elif no == "H":
        c = [
            x[0] ** 2.0 + x[1] ** 2.0 + x[2] ** 2.0 + x[3] ** 2.0 + x[0] - x[1] + x[2] - x[3] - 8.0,
            x[0] ** 2.0 + 2.0 * x[1] ** 2.0 + x[2] ** 2.0 + 2.0 * x[3] ** 2.0 - x[0] - x[3] - 10.0,
            2.0 * x[0] ** 2.0 + x[1] ** 2.0 + x[2] ** 2.0 + 2.0 * x[0] - x[1] - x[3] - 5.0,
        ]
    elif no == "I":
        c = [
            2.0 * x[0] ** 2.0 + 3.0 * x[1] ** 4.0 + x[2] + 4.0 * x[3] ** 2.0 + 5.0 * x[4] - 127.0,
            7.0 * x[0] + 3.0 * x[1] + 10.0 * x[2] ** 2.0 + x[3] - x[4] - 282.0,
            23.0 * x[0] + x[1] ** 2.0 + 6.0 * x[5] ** 2.0 - 8.0 * x[6] - 196.0,
            4.0 * x[0] ** 2.0 + x[1] ** 2.0 - 3.0 * x[0] * x[1] + 2.0 * x[2] ** 2.0 + 5.0 * x[5] - 11.0 * x[6],
        ]
    elif no == "J":
        c = [
            x[2] ** 2.0 + x[3] ** 2.0 - 1.0,
            x[4] ** 2.0 + x[5] ** 2.0 - 1.0,
            (x[0] - x[4]) ** 2.0 + (x[1] - x[5]) ** 2.0 - 1.0,
            (x[0] - x[6]) ** 2.0 + (x[1] - x[7]) ** 2.0 - 1.0,
            (x[2] - x[4]) ** 2.0 + (x[3] - x[5]) ** 2.0 - 1.0,
            (x[2] - x[6]) ** 2.0 + (x[3] - x[7]) ** 2.0 - 1.0,
            x[6] ** 2.0 + (x[7] - x[8]) ** 2.0 - 1.0,
            x[8] ** 2.0 - 1.0,
            x[0] ** 2.0 + (x[1] - x[8]) ** 2.0 - 1.0,
            -x[2] * x[8],
            x[5] * x[6] - x[4] * x[7],
            x[1] * x[2] - x[0] * x[3],
            x[4] * x[8],
        ]
    else:
        raise NotImplementedError
    return np.array(c)


def solution(no):
    if no == "A":
        s = ([-1.0, 0.0],)
    elif no == "B":
        e = np.sqrt(2.0) / 2.0
        s = ([-e, e], [e, -e])
    elif no == "C":
        e1 = 1.0 / np.sqrt(3.0)
        e2 = 1.0 / np.sqrt(6.0)
        e3 = 1.0 / 3.0
        s = ([-e1, e2, e3], [e1, -e2, e3], [e1, e2, -e3], [-e1, -e2, -e3])
    elif no in "DE":
        s = ([-1.0, 1.0],)
    elif no == "F":
        e = 1.0 / np.sqrt(2.0)
        s = ([e, e],)
    elif no == "G":
        s = ([0.0, -3.0, -3.0],)
    elif no == "H":
        s = ([0.0, 1.0, 2.0, -1.0],)
    elif no == "I":
        s = ([2.330499, 1.951372, -0.4775414, 4.365726, -0.6244870, 1.038131, 1.594227],)
    elif no == "J":
        s = ([0.8841292, 0.4672425, 0.03742076, 0.9992996, 0.8841292, 0.4672424, 0.03742076, 0.9992996, 0.0],)
    else:
        raise NotImplementedError
    return (np.array(xs) for xs in s)


def _max_cv(x, no):
    resid = np.max(cub(x, no), initial=0.0)
    if xl(no) is not None:
        resid = max(resid, np.max(xl(no) - x, initial=0.0))
    if aub(no) is not None:
        resid = max(resid, np.max(np.dot(aub(no), x) - bub(no), initial=0.0))
    return resid


def _distance(x, no):
    return min(np.linalg.norm(x - xs) for xs in solution(no))


if __name__ == "__main__":
    print("+---------+-------------+--------------------+----------------------+----------------------+")
    print("| Problem | Evaluations | Objective function | Constraint violation | Distance to solution |")
    print("+---------+-------------+--------------------+----------------------+----------------------+")
    for problem in "ABCDEFGHIJ":
        res = minimize(fun, x0(problem), problem, xl(problem), aub=aub(problem), bub=bub(problem), cub=cub)
        print(f"|   ({problem})   |{res.nfev:^13}|{res.fun:^20.4e}|{_max_cv(res.x, problem):^22.4e}|{_distance(res.x, problem):^22.4e}|")
    print("+---------+-------------+--------------------+----------------------+----------------------+")
