.. _linalg.bvtcg:

.. currentmodule:: cobyqa.linalg

Bound constrained truncated conjugate gradient
==============================================

When no linear nor nonlinear constraints are provided to COBYQA (that is when the considered problem is bound constrained), the trust-region subproblem to solve at each iteration is of the form

.. math::
    :label: bvtcg

    \begin{array}{ll}
        \min        & \quad q(x) = \inner{x, g} + \frac{1}{2} \inner{x, H x}\\
        \text{s.t.} & \quad l \le x \le u,\\
                    & \quad \norm{x} \le \Delta,\\
                    & \quad x \in \R^n,
    \end{array}

where :math:`g \in \R^n`, :math:`H \in \R^{n \times n}` is a symmetric matrix, :math:`l \in \R^n` and :math:`u \in \R^n` are the lower and upper bounds of the problem (with :math:`l < u`), :math:`\Delta > 0` is the given trust-region radius, and :math:`\norm{\cdot}` denotes the Euclidean norm.

.. _tcg_base:

The unconstrained case
----------------------

We assume in this section that the lower and upper bounds in :eq:`bvtcg` are respectively set to :math:`-\infty` and :math:`+\infty`.
Powell showed in :cite:`bvtcg-Powell_1975` that a trust-region method is convergent if the trust-region step :math:`x^{\ast} \in \R^n` satisfies

.. math::

    q(x^0) - q(x^{\ast}) \ge \gamma \norm{g} \min \set{\Delta, \norm{g} / \norm{H}},

for some :math:`\gamma > 0`.
It is easy to see that a Cauchy step satisfies such a condition (with :math:`\gamma = 1/2`).
Therefore, to preserve the computational efficiency of a trust-region method, it is usual to solve inexactly problem :eq:`bvtcg` using the truncated conjugate gradient method of Steihaug :cite:`bvtcg-Steihaug_1983` and Toint :cite:`bvtcg-Toint_1981`.
Given the initial values :math:`x^0 = 0` and :math:`d^0 = -g`, it generates the sequence of iterates

.. math::

    \left\{
    \begin{array}{l}
        \alpha_k = -\inner{d^k, g^k} / \inner{d^k, Hd^k},\\
        \beta_k = \norm{g^{k + 1}}^2 / \norm{g^k}^2,\\
        x^{k + 1} = x^k + \alpha_k d^k,\\
        d^{k + 1} = -g^k + \beta_k d^k,
    \end{array}
    \right.

where :math:`g^k = \nabla q(x^k) = g + Hx^k`.
The computations are stopped if either

#. :math:`g^k = 0` and :math:`\inner{d^k, Hd^k} \ge 0`, in which case the global minimizer is found; or
#. :math:`\norm{x^{k + 1}} \ge \Delta` or :math:`\inner{d^k, Hd^k} < 0`, in which case :math:`x^k + \alpha_{\Delta} d^k` is returned, where :math:`\alpha_{\Delta} > 0` is chosen so that :math:`\norm{x^k + \alpha_{\Delta} d^k} = \Delta`.

It is known that the truncated conjugate gradient method terminates after at most :math:`n` iterations, and that the reduction in the objective function obtained with the truncated conjugate gradient method is at least half of the reduction obtained by the global minimizer.

The constrained case
--------------------

We assume in this section that at least some values of the lower and upper bounds in :eq:`bvtcg` are finite.
When no linear nor nonlinear constraints are provided, COBYQA solves the trust-region subproblem using the TRSBOX algorithm :cite:`bvtcg-Powell_2009`, which is presented below.

The bound constrained truncated conjugate gradient procedure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The strategy employed by `bvtcg` to solve inexactly problem :eq:`bvtcg` is an active-set variation of the truncated conjugate gradient procedure.
At each iteration of the method, a truncate conjugate gradient step is performed on the coordinates that are not fixed by the working set.
If a new bound is hit during such iteration, the bound is added to the working set, and the procedure is restarted.
The working set is only enlarged through the iterations, which then ensures the termination of the method.

The initial working set is a subset of the active bounds at the origin.
Clearly, an active bound should not be included in the working set if a Cauchy step (a positive step along :math:`-g`) would depart from the bound, as the bound is never removed from the working set.
Therefore, the :math:`i`-th bound should be included in the working set if either :math:`l_i = x_i^k` and :math:`\nabla q (x^k) \ge 0` or :math:`u_i = x_i^k` and :math:`\nabla q (x^k) \le 0`.
The complete framework of `bvtcg` is described below.
For sake of clarity, we denote :math:`\mathcal{I}^k` the :math:`k`-th working set and :math:`\Pi_k(v)` the vector whose :math:`i`-th coordinate is :math:`v_i` if :math:`i \notin \mathcal{I}^k`, and zero otherwise.

#. Set :math:`x^0 = 0`, :math:`g^0 = g`, the working set :math:`\mathcal{I}^0`, and :math:`k = 0`.
#. Set :math:`d^k = -\Pi_k(g^k)` and stop the computations if :math:`\norm{d^k} = 0`.
#. Set :math:`\alpha_{\Delta, k} = \argmax \set{\alpha \ge 0 : \norm{x^k + \alpha d^k} \le \Delta}`.
#. Set :math:`\alpha_{Q, k}` to :math:`-\inner{d^k, g^k} / \inner{d^k, Hd^k}` if :math:`\inner{d^k, Hd^k} > 0` and :math:`+\infty` otherwise.
#. Set :math:`\alpha_{B, k} = \argmax \set{\alpha \ge 0 : l \le x^k + \alpha d^k \le u}` and :math:`\alpha_k = \min \set{\alpha_{\Delta, k}, \alpha_{Q, k}, \alpha_{B, k}}`.
#. Update :math:`x^{k + 1} = x^k + \alpha_k d^k` and :math:`g^{k + 1} = g^k + \alpha_k H d^k`.
#. If :math:`\alpha_k = \alpha_{B, k}`, add a new active coordinate to :math:`\mathcal{I}^k` to obtain :math:`\mathcal{I}^{k + 1}`, increment :math:`k`, and go to step 2.
#. If :math:`\alpha_k = \alpha_{\Delta, k}`, stop the computations.
#. Set :math:`\beta_k = \norm{\Pi_k(g^{k + 1})}^2 / \norm{\Pi_k(g^k)}^2`
#. Update :math:`\mathcal{I}^{k + 1} = \mathcal{I}^k`, :math:`d^{k + 1} = -\Pi_k(g^{k + 1}) + \beta_k d^k`, increment :math:`k`, and go to step 3.

Further refinement of the trial step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the step :math:`x^k` returned by the constrained truncated conjugate gradient procedure satisfies :math:`\norm{x^k} = \Delta`, it is likely that the objective function in :eq:`bvtcg` can be further decreased by moving this point round the trust-region boundary.
In fact, the global solution of problem :eq:`bvtcg` is on the trust-region boundary.
The method `bvtcg` then may further reduce the function evaluation by returning in this case an approximate solution to

.. math::

    \begin{array}{ll}
        \min        & \quad q(x) = \inner{x, g} + \frac{1}{2} \inner{x, H x}\\
        \text{s.t.} & \quad l \le x \le u,\\
                    & \quad \norm{x} = \Delta,\\
                    & \quad x \in \vspan \set{\Pi_k(x^k), \Pi_k(g^k)} \subseteq \R^n.
    \end{array}

To do so, the method builds an orthogonal basis :math:`\set{\Pi_k(x^k), s}` of :math:`\vspan \set{\Pi_k(x^k), \Pi_k(g^k)}` by selecting the vector :math:`s \in \R^n` such that :math:`\inner{s, \Pi_k(x^k)} = 0`, :math:`\inner{s, \Pi_k(g^k)} < 0`, and :math:`\norm{s} = \norm{\Pi_k(x^k)}`.
Further, the method considers the function :math:`x(\theta) = x^k + (\cos \theta - 1) \Pi_k(x^k) + \sin \theta s` with :math:`0 \le \theta \le \pi / 4` and solves approximately

.. math::

    \begin{array}{ll}
        \min        & \quad q(x(\theta))\\
        \text{s.t.} & \quad l \le x(\theta) \le u,\\
                    & \quad 0 \le \theta \le \pi / 4,
    \end{array}

the trust-region condition being automatically ensured by the choice of :math:`s`.
To solve approximately such a problem, `bvtcg` seeks for the greatest reduction in the objective function for a range of equally spaced values of :math:`\theta`, chosen to ensure the feasibility of the iterates.
If the value of the approximate solution is restricted by a bound, it is added to the working set :math:`\mathcal{I}^k`, and the refinement procedure is restarted.
Since the working set is only increased, this procedure terminates in at most :math:`n - \abs{\mathcal{I}^k}`, where :math:`\abs{\mathcal{I}^k}` denotes the number of active bounds at the end of the constrained truncated conjugate gradient procedure.

.. bibliography::
    :labelprefix: B
    :keyprefix: bvtcg-