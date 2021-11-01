.. _linalg.cpqp:

.. currentmodule:: cobyqa.linalg

Convex piecewise quadratic programming
======================================

In general (that is when some linear and/or nonlinear constraints are
provided), to determine a trust-region normal step, COBYQA must solve a problem
of the form

.. math::
    :label: cpqp

    \begin{array}{ll}
        \min        & \quad f(x) = \frac{1}{2} \big(\norm{[Ax - b]_+}^2 + \norm{Cx - d}^2\big)\\
        \text{s.t.} & \quad l \le x \le u,\\
                    & \quad \norm{x} \le \Delta,\\
                    & \quad x \in \R^n,
    \end{array}

where :math:`A \in \R^{m_1 \times n}` and :math:`C \in \R^{m_2 \times n}` are
the Jacobian matrices of the inequality and equality constraints (or some
approximations) of the original nonlinear optimization problem,
:math:`b \in \R^{m_1}` and :math:`d \in \R^{m_2}` are the corresponding
right-hand sides, :math:`l \in \R^n` and :math:`u \in \R^n` are the lower an
upper bounds of the problems (with :math:`l < u`), :math:`\Delta > 0` is the
current trust-region radius, :math:`[\cdot]_+` denotes the componentwise
positive part operator, and :math:`\norm{\cdot}` denotes the Euclidean norm.

Reformulation of the problem
----------------------------

The main difficulty in solving (approximately) problem :eq:`cpqp` is the
piecewise quadratic term :math:`x \mapsto \norm{[Ax - b]_+}^2` in the objective
function. However, such a problem can clearly be reformulated by introducing a
slack variable :math:`y \in \R^{m_1}` as

.. math::
    :label: cpqp-slack

    \begin{array}{ll}
        \min        & \quad f(x, y) = \frac{1}{2} \big(\norm{y}^2 + \norm{Cx - d}^2\big)\\
        \text{s.t.} & \quad l \le x \le u, ~ 0 \le y,\\
                    & \quad Ax - y \le b,\\
                    & \quad \norm{x} \le \Delta,\\
                    & \quad x \in \R^n, ~ y \in \R^{m_1}.
    \end{array}

In doing so, we observe that problem :eq:`cpqp-slack` is almost of the right
form to be solved approximately by the algorithm `lctcg`, as detailed in the
:ref:`corresponding section <linalg.lctcg>`. The only difference is in the
trust-region constraint, which does not apply to all the variables of the
reformulated problem. From a geometrical point of view, the feasible set
engendered by the only nonlinear constraint is a ball for the problems tackled
by `lctcg`, while being a cylinder in problem :eq:`cpqp-slack`. Thus, method
described below is a variable of the `lctcg` algorithm.

Description of the method
-------------------------

The general framework employed by `cpqp` to solve problem :eq:`cpqp-slack`, and
thus, problem :eq:`nnls` is presented below.

.. What is \Pi_x and \Pi_y?
.. Extra constraint on y?

#. Set :math:`x^0 = 0` and :math:`y^0 = [-b]_+`.
#. Set :math:`g_x^0 = \nabla_x f(x^0, y^0)`, :math:`d_x^0 = \Pi_x(g_x^0)`,
   :math:`d_y^0 = \Pi_y(y^0)`, the active set
   :math:`\mathcal{I} \subseteq \mathcal{J}(x^0, y^0)`, and :math:`k = 0`.
#. Let :math:`\alpha_{\Delta, k}` be the largest number such that
   :math:`\norm{x^k + \alpha_{\Delta, k} d_x^k} = \Delta`.
#. Let :math:`\alpha_{Q, k}` be
   :math:`-(\inner{d_x^k, g_x^k} + \inner{d_y^k, y^k}) / (\norm{Cd_x^k}^2 + \norm{d_y^k}^2)`
   if :math:`\norm{Cd_x^k}^2 + \norm{d_y^k}^2 > 0` and :math:`+\infty` otherwise.
#. Let :math:`\alpha_{L, k}` be the largest number such that
   :math:`A (x^k + \alpha_{B, k} d_x^k) - y^k - \alpha_{B, k} d_y^k \le b` and
   :math:`\alpha_k = \min \set{\alpha_{\Delta, k}, \alpha_{Q, k}, \alpha_{L, k}}`.
#. Update :math:`x^{k + 1} = x^k + \alpha_k d_x^k`,
   :math:`y^{k + 1} = y^k + \alpha_k d_y^k` and
   :math:`g_x^{k + 1} = g_x^k + \alpha_k C^{\T} C d_x^k`.
#. If :math:`\alpha_k = \alpha_{\Delta, k}` or :math:`g_x^{k + 1} = 0` and
   :math:`y^{k + 1} = 0`, stop the computations.
#. If :math:`\alpha_k = \alpha_{L, k}`, set :math:`x^0 = x^{k + 1}`,
   :math:`y^0 = y^{k + 1}`, and go to step 2.
#. Set
   :math:`\beta_k = (\inner{\Pi_x(g_x^{k + 1}), C^{\T} C d_x^k} + \inner{\Pi_y(y^{k + 1}), d_y^k}) / (\norm{Cd_x^k}^2 + \norm{d_y^k}^2)`,
   update :math:`d_x^{k + 1} = -\Pi_x(g_x^{k + 1}) + \beta_k d_x^k`,
   :math:`d_y^{k + 1} = -\Pi_y(y^{k + 1}) + \beta_k d_y^k`, increment
   :math:`k`, and go to step 3.

.. bibliography::
    :labelprefix: C
    :keyprefix: cpqp-
