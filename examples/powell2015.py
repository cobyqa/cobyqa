#!/usr/bin/env python3
"""
Solve Example (6.7)--(6.8) of [1]_.

References
----------
.. [1] M. J. D. Powell. On fast trust region methods for quadratic models with
   linear constraints. Math. Program. Comput., 7(3):237--267, 2015.
"""
import numpy as np
from matplotlib import pyplot as plt

from cobyqa import minimize

np.set_printoptions(
    precision=3,
    threshold=10,
    suppress=True,
)


def fun(x):
    f = 0.0
    for i in range(1, x.size // 2):
        for j in range(i):
            norm = np.hypot(x[2 * i] - x[2 * j], x[2 * i + 1] - x[2 * j + 1])
            f += min(1.0 / norm, 1e3) if norm > 1e-3 else 1e3
    f /= x.size ** 2.0
    return f


def _plot_points(x, title=None):
    fig, ax = plt.subplots(dpi=300)
    ax.plot([0.0, 0.0, 2.0, 0.0], [0.0, 2.0, 0.0, 0.0], color='black')
    ax.scatter(x[::2], x[1::2], s=25, color='black')
    ax.set_aspect('equal', 'box')
    ax.axis('off')
    if title is not None:
        ax.set_title(f'{title.strip()} ($n = {x.size}$)', fontsize=20)
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    n = 80  # must be even

    xl = np.zeros(n, dtype=float)
    aub = np.zeros((n // 2, n), dtype=float)
    bub = 2.0 * np.ones(n // 2, dtype=float)
    x0 = np.empty(n, dtype=float)
    for i in range(n // 2):
        aub[i, [2 * i, 2 * i + 1]] = 1.0
        x0_even = rng.uniform(0.0, 2.0)
        x0_odd = rng.uniform(0.0, 2.0)
        if x0_even + x0_odd > 2.0:
            x0_even = 2.0 - x0_even
            x0_odd = 2.0 - x0_odd
        x0[2 * i] = x0_even
        x0[2 * i + 1] = x0_odd
    _plot_points(x0, 'Initial points')

    options = {'verbose': True}
    res = minimize(fun, x0, xl=xl, aub=aub, bub=bub, options=options)
    _plot_points(res.x, 'Final points')
