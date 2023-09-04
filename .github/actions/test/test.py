import sys

import numpy as np

from cobyqa import minimize


def arwhead(x):
    return np.sum((x[:-1] ** 2.0 + x[-1] ** 2.0) ** 2.0 - 4.0 * x[:-1] + 3.0)


np.set_printoptions(
    precision=3,
    linewidth=sys.maxsize,
    sign=' ',
)

if __name__ == '__main__':
    n = 5
    x0 = np.zeros(n)
    xl = -5.12 * np.ones(n)
    xu = 5.12 * np.ones(n)
    aeq = np.c_[np.ones((1, n - 1)), 0.0]
    beq = np.ones(1)

    options = {'verbose': True, 'debug': True}
    minimize(arwhead, x0, aeq=aeq, beq=beq, options=options)
