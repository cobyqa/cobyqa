.. _linalg.bvtcg:

.. currentmodule:: cobyqa.linalg

Bound constrained truncated conjugate gradient
==============================================

When no linear nor nonlinear constraints are provided to COBYQA (that is when
the considered problem is bound constrained), the trust-region subproblem to
solve at each iteration is of the form

.. math::
    :label: bvtcg

    \begin{array}{ll}
        \min        & \quad f(x) = \inner{x, g} + \frac{1}{2} \inner{x, H x}\\
        \text{s.t.} & \quad l \le x \le u,\\
                    & \quad \norm{x} \le \Delta,\\
                    & \quad x \in \R^n,
    \end{array}

where :math:`g \in \R^n` approximates the gradient of the nonlinear objective
function at the origin, :math:`H \in \R^{n \times n}` is a symmetric matrix
that approximates the Hessian matrix of the nonlinear objective function at the
origin, :math:`l \in \R^n` and :math:`u \in \R^n` are the lower and upper
bounds of the problems (with :math:`l < u`), :math:`\Delta > 0` is the current
trust-region radius, and :math:`\norm{\cdot}` denotes the Euclidean norm.

.. _tcg_base:

The unconstrained case
----------------------

We assume in this section that the lower and upper bounds in :eq:`bvtcg` are
respectively set to :math:`-\infty` and :math:`+\infty`. Powell showed in
:cite:`bvtcg-Powell_1975` that a trust-region method is convergent if the
trust-region step :math:`x^{\ast} \in \R^n` satisfies

.. math::

    f(x^0) - f(x^{\ast}) \ge \gamma \norm{g} \min \set{\Delta, \norm{g} / \norm{H}},

for some :math:`\gamma > 0`, where :math:`\norm{\cdot}` is the Euclidean norm.
It is easy to see that a Cauchy step satisfies such a condition
(with :math:`\gamma = 1/2`). Therefore, to preserve the computational
efficiency of a trust-region method, it is usual to solve inexactly problem
:eq:`bvtcg` using the truncated conjugate gradient method of Steihaug
:cite:`bvtcg-Steihaug_1983` and Toint :cite:`bvtcg-Toint_1981`. Given the
initial values :math:`x^0 = 0` and :math:`d^0 = -g`, it generates the sequence
of iterates

.. math::

    \left\{
    \begin{array}{l}
        \alpha_k = -\inner{d^k, g^k} / \inner{d^k, Hd^k},\\
        \beta_k = \norm{g^{k + 1}}^2 / \norm{g^k}^2,\\
        x^{k + 1} = x^k + \alpha_k d^k,\\
        d^{k + 1} = -g^k + \beta_k d^k,
    \end{array}
    \right.

where :math:`g^k = \nabla f(x^k) = g + Hx^k`. The computations are stopped if
either

#. :math:`g^k = 0` and :math:`\inner{d^k, Hd^k} \ge 0`, in which case the
   global minimizer is found; or
#. :math:`\norm{x^{k + 1}} \ge \Delta` or :math:`\inner{d^k, Hd^k} < 0`, in
   which case :math:`x^k + \alpha_{\Delta} d^k` is returned, where
   :math:`\alpha_{\Delta} > 0` is chosen so that
   :math:`\norm{x^k + \alpha_{\Delta} d^k} = \Delta`.

It is known that the truncated conjugate gradient method terminates after at
most :math:`n` iterations, and that the reduction in the objective function
obtained with the truncated conjugate gradient method is at least half of the
reduction obtained by the global minimizer.

The constrained case
--------------------

We assume in this section that at least some values of the lower and upper
bounds in :eq:`bvtcg` are finite. When no linear nor nonlinear constraints are
provided, COBYQA solves the trust-region subproblem using the TRSBOX algorithm
:cite:`bvtcg-Powell_2009`, which is presented below.

The bound constrained truncated conjugate gradient procedure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The strategy employed by `bvtcg` to tackle the bound constraints in the
truncated conjugate gradient procedure is the use of an active set. At each
iteration of the method, a truncate conjugate gradient step is performed on the
coordinates that are not fixed by the active set. If a new bound is hit during
such iteration, the bound is added to the active set, and the procedure is
restarted. The active set is only enlarged through the iterations, which then
ensures the termination of the method.

The initial active set is a subset of the active bounds at the origin. Clearly,
a active bound should not be included in the active set if a Cauchy step (a
positive step along :math:`-g`) would depart from the bound, as the bound is
never removed from the active set. The complete framework of `bvtcg` is
described below. For sake of clarity, we denote :math:`\mathcal{I}` the active
set and :math:`\Pi(v)` the vector whose :math:`i`-th coordinate is :math:`v_i`
if :math:`i \notin \mathcal{I}`, and zero otherwise.

#. Set :math:`x^0 = 0` and the active set :math:`\mathcal{I}` to the indices
   for which either :math:`l_i = 0` and :math:`g_i \ge 0` or :math:`u_i = 0`
   and :math:`g_i \le 0`.
#. Set :math:`g^0 = \nabla f(x^0)`, :math:`d^0 = -\Pi(g^0)`, and :math:`k = 0`.
#. Let :math:`\alpha_{\Delta, k}` be the largest number such that
   :math:`\norm{x^k + \alpha_{\Delta, k} d^k} = \Delta`.
#. Let :math:`\alpha_{Q, k}` be :math:`-\inner{d^k, g^k} / \inner{d^k, Hd^k}`
   if :math:`\inner{d^k, Hd^k} > 0` and :math:`+\infty` otherwise.
#. Let :math:`\alpha_{B, k}` be the largest number such that
   :math:`l \le x^k + \alpha_{B, k} d^k \le u` and
   :math:`\alpha_k = \min \set{\alpha_{\Delta, k}, \alpha_{Q, k}, \alpha_{B, k}}`.
#. Update :math:`x^{k + 1} = x^k + \alpha_k d^k` and
   :math:`g^{k + 1} = g^k + \alpha_k H d^k`.
#. If :math:`\alpha_k = \alpha_{\Delta, k}` or :math:`g^{k + 1} = 0`, stop the
   computations.
#. If :math:`\alpha_k = \alpha_{B, k}`, add a new active coordinate to
   :math:`\mathcal{I}`, set :math:`x^0 = x^{k + 1}`, and go to step 2.
#. Set :math:`\beta_k = \norm{\Pi(g^{k + 1})}^2 / \norm{\Pi(g^k)}^2`, update
   :math:`d^{k + 1} = -\Pi(g^k) + \beta_k d^k`, increment :math:`k`, and go to
   step 3.

Further refinement of the trial step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the step :math:`x^k` returned by the constrained truncated conjugate
gradient procedure satisfies :math:`\norm{x^k} = \Delta`, it is likely that
the objective function in :eq:`bvtcg` can be further decreased by moving this
point round the trust-region boundary. In fact, the global solution of problem
:eq:`bvtcg` is on the trust-region boundary. The method `bvtcg` then may
further reduce the function evaluation by returning in this case an approximate
solution to

.. math::

    \begin{array}{ll}
        \min        & \quad f(x) = \inner{x, g} + \frac{1}{2} \inner{x, H x}\\
        \text{s.t.} & \quad l \le x \le u,\\
                    & \quad \norm{x} = \Delta,\\
                    & \quad x \in \vspan \set{\Pi(x^k), \Pi(g^k)} \subseteq \R^n.
    \end{array}

To do so, the method builds an orthogonal basis :math:`\set{\Pi(x^k), s}` of
:math:`\vspan \set{\Pi(x^k), \Pi(g^k)}` by selecting the vector
:math:`s \in \R^n` such that :math:`\inner{s, \Pi(x^k)} = 0`,
:math:`\inner{s, \Pi(g^k)} < 0`, and :math:`\norm{s} = \norm{\Pi(x^k)}`.
Further, the method considers the function
:math:`x(\theta) = x^k + (\cos \theta - 1) \Pi(x^k) + \sin \theta s` with
:math:`0 \le \theta \le \pi / 4` and solves approximately

.. math::

    \begin{array}{ll}
        \min        & \quad f(x(\theta))\\
        \text{s.t.} & \quad l \le x(\theta) \le u,\\
                    & \quad 0 \le \theta \le \pi / 4,
    \end{array}

the trust-region condition being automatically ensured by the choice of
:math:`s`. If the value of :math:`\theta` is restricted by a bound, it is added
to the active set :math:`\mathcal{I}`, and the refinement procedure is
restarted. Since the active set is very reduced, this procedure terminates in
at most :math:`n - \abs{\mathcal{I}}`, where :math:`\abs{\mathcal{I}}` denotes
the number of active bounds at the end of the constrained truncated conjugate
gradient procedure.

.. bibliography::
    :labelprefix: B
    :keyprefix: bvtcg-