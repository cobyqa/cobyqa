#!/usr/bin/env python3
"""
Solve Example (6.7)--(6.8) of [1]_.

References
----------
.. [1] M. J. D. Powell. On fast trust region methods for quadratic models with
   linear constraints. *Math. Program. Comput.*, 7(3):237--267, 2015.
   `doi:10.1007/s12532-015-0084-4
   <https://doi.org/10.1007/s12532-015-0084-4>`_.
"""
from contextlib import suppress

import numpy as np
from cobyqa import minimize
from scipy.optimize import Bounds, LinearConstraint


def fun(x):
    f = 0.0
    for i in range(1, x.size // 2):
        for j in range(i):
            norm = np.hypot(x[2 * i] - x[2 * j], x[2 * i + 1] - x[2 * j + 1])
            f += min(1.0 / norm, 1e3) if norm > 1e-3 else 1e3
    f /= x.size**2.0
    return f


def _plot_points(x, title=None):
    with suppress(ImportError):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(dpi=300)
        ax.plot([0.0, 0.0, 2.0, 0.0], [0.0, 2.0, 0.0, 0.0], color="black")
        ax.scatter(x[::2], x[1::2], s=25, color="black")
        ax.set_aspect("equal", "box")
        ax.axis("off")
        if title is not None:
            ax.set_title(f"{title.strip()} ($n = {x.size}$)", fontsize=20)
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 80  # must be even

    aub = np.zeros((n // 2, n))
    bub = 2.0 * np.ones(n // 2)
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
    _plot_points(x0, "Initial points")

    res = minimize(
        fun,
        x0,
        bounds=Bounds(np.zeros(n), np.inf),
        constraints=LinearConstraint(aub, -np.inf, bub),
        options={"disp": True},
    )
    print(res)
    _plot_points(res.x, "Final points")
