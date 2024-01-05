.. _usage:

Usage
=====

.. currentmodule:: cobyqa

We provide below basic usage information on how to use COBYQA.
For more details on the signature of the `minimize` function, please refer to the :ref:`API documentation <api>`.

How to use COBYQA
-----------------

COBYQA provides a `minimize` function.
This is the entry point to the solver.
It solves unconstrained, bound-constrained, linearly constrained, and nonlinearly constrained optimization problems.

We provide below simple examples on how to use COBYQA.

Examples
--------

Example of unconstrained optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let us first minimize the Rosenbrock function implemented in `scipy.optimize`, defined as

.. math::

    f(x) = \sum_{i = 1}^{n - 1} 100 (x_{i + 1} - x_i^2)^2 + (x_i - 1)^2

for :math:`x \in \R^n`.
To solve the problem using COBYQA, run:

.. code-block:: python

    from cobyqa import minimize
    from scipy.optimize import rosen

    x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    res = minimize(rosen, x0)
    print(res.x)

This should display the desired output ``[1. 1. 1. 1. 1.]``.

Example of linear constrained optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To see how bound and linear constraints are handled using `minimize`, let us solve Example 16.4 of :cite:`ua-Nocedal_Wright_2006`, defined as

.. math::

    \begin{aligned}
        \min_{x \in \R^2}   & \quad (x_1 - 1)^2 + (x_2 - 2.5)^2\\
        \text{s.t.}         & \quad -x_1 + 2x_2 \le 2,\\
                            & \quad x_1 + 2x_2 \le 6,\\
                            & \quad x_1 - 2x_2 \le 2,\\
                            & \quad x_1 \ge 0,\\
                            & \quad x_2 \ge 0.
    \end{aligned}

To solve the problem using COBYQA, run:

.. code-block:: python

    import numpy np
    from cobyqa import minimize
    from scipy.optimize import Bounds

    def fun(x):
        return (x[0] - 1.0) ** 2.0 + (x[1] - 2.5) ** 2.0

    x0 = [2.0, 0.0]
    bounds = Bounds([0.0, 0.0], np.inf)
    aub = [[-1.0, 2.0], [1.0, 2.0], [1.0, -2.0]]
    bub = [2.0, 6.0, 2.0]
    res = minimize(fun, x0, bounds=bounds, aub=aub, bub=bub)
    print(res.x)

This should display the desired output ``[1.4 1.7]``.

Example of nonlinear constrained optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To see how nonlinear constraints are handled, we solve Problem (F) of :cite:`ua-Powell_1994`, defined as

.. math::

    \begin{aligned}
        \min_{x \in \R^2}   & \quad -x_1 - x_2\\
        \text{s.t.}         & \quad x_1^2 - x_2 \le 0,\\
                            & \quad x_1^2 + x_2^2 \le 1.
    \end{aligned}

To solve the problem using COBYQA, run:

.. code-block:: python

    from cobyqa import minimize

    def fun(x):
        return -x[0] - x[1]

    def cub(x):
        return [x[0] ** 2.0 - x[1], x[0] ** 2.0 + x[1] ** 2.0 - 1.0]

    x0 = [1.0, 1.0]
    res = minimize(fun, x0, cub=cub)
    print(res.x)

This should display the desired output ``[0.7071 0.7071]``.

.. bibliography::
    :labelprefix: UA
    :keyprefix: ua-
