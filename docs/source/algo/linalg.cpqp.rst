.. _linalg.cpqp:

.. currentmodule:: cobyqa.linalg

Convex piecewise quadratic programming
======================================

In general (that is when some linear and/or nonlinear constraints are provided), to determine a trust-region normal step, COBYQA must solve a problem of the form

.. math::
    :label: cpqp

    \begin{array}{ll}
        \min        & \quad q(x) = \frac{1}{2} \big(\norm{[Ax - b]_+}^2 + \norm{Cx - d}^2\big)\\
        \text{s.t.} & \quad l \le x \le u,\\
                    & \quad \norm{x} \le \Delta,\\
                    & \quad x \in \R^n,
    \end{array}

where :math:`A \in \R^{m_1 \times n}`, :math:`C \in \R^{m_2 \times n}`, :math:`b \in \R^{m_1}`, :math:`d \in \R^{m_2}`, :math:`l \in \R^n` and :math:`u \in \R^n` are the lower an upper bounds of the problem (with :math:`l < u`), :math:`\Delta > 0` is the given trust-region radius, :math:`[\cdot]_+` denotes the componentwise positive part operator, and :math:`\norm{\cdot}` denotes the Euclidean norm.

Reformulation of the problem
----------------------------

The main difficulty in solving problem :eq:`cpqp`, even approximately, is the piecewise quadratic term :math:`\norm{[Ax - b]_+}^2` in the objective function.
However, problem :eq:`cpqp` can clearly be reformulated by introducing a slack variable :math:`y \in \R^{m_1}` as

.. math::
    :label: cpqp-slack

    \begin{array}{ll}
        \min        & \quad q(x, y) = \frac{1}{2} \big(\norm{y}^2 + \norm{Cx - d}^2\big)\\
        \text{s.t.} & \quad l \le x \le u, ~ 0 \le y,\\
                    & \quad Ax - y \le b,\\
                    & \quad \norm{x} \le \Delta,\\
                    & \quad x \in \R^n, ~ y \in \R^{m_1}.
    \end{array}

The reformulated problem :eq:`cpqp-slack` is simpler to solve than problem :eq:`cpqp` because its objective function is quadratic, although

#. the dimension of the reformulated problem is higher, and
#. linear inequality constraints appear in the reformulated problem.

We observe nonetheless that problem :eq:`cpqp-slack` is almost of the right form to be solved approximately by the algorithm `lctcg`, as detailed in the :ref:`presentation of the method <linalg.lctcg>`.
Only the trust-region constraint differs, and is not applied to all variables of the reformulated problem.
From a geometrical point of view, the feasible set engendered by the only nonlinear constraint is a ball for the problems tackled by `lctcg`, while it is a cylinder in problem :eq:`cpqp-slack`.
To solve problem :eq:`cpqp-slack`, and hence :eq:`cpqp`, we design below a variation of the `lctcg` algorithm.

Description of the method
-------------------------

Recall that the `lctcg` algorithm is an active-set variation of the truncated conjugate gradient algorithm.
In this algorithm, the trust-region constraint intervenes solely in the corresponding termination condition, as detailed in the :ref:`presentation of the method <linalg.lctcg>`.
Therefore, to solve approximately problem :eq:`cpqp-slack`, and hence :eq:`cpqp`, it is easy to adapt the `lctcg` algorithm, as presented below.

The working set
^^^^^^^^^^^^^^^

As for the `lctcg` algorithm, we must consider a constraint as active whenever its residual becomes small.
More specifically, for some feasible :math:`(x, y) \in \R^n \times \R^{m_1}`, we let

.. math::

    \mathcal{J}(x, y) = \set[\bigg]{j \le m_1 : b_j - \inner{a_j, x} - \inner{e_j, y} \le \eta \Delta \sqrt{\norm{a_j}^2 + 1}},

where :math:`e_j \in \R^{m_1}` denotes the :math:`j`-th standard unit vector, and :math:`\eta` is some positive constant (set to :math:`\eta = 0.2` in `cpqp`), and the working set if a subset of :math:`\mathcal{J}(x^0, y^0)`, with :math:`x^0 = 0` and :math:`y^0 = [-b]_+`.
Similarly, the initial search directions are set as follows.
We denote :math:`\big(\Pi_x(u, v), \Pi_y(u, v)\big) \in \R^n \times \R^{m_1}` the unique solution of

.. math::

    \begin{array}{ll}
        \min        & \quad \frac{1}{2} \big(\norm{d_x - u}^2 + \norm{d_y - v}^2\big)\\
        \text{s.t.} & \quad \inner{a_j, d_x} - \inner{e_j, d_y} \le 0, ~ j \in \mathcal{J}(x^0, y^0),\\
                    & \quad d_x \in \R^n, ~ d_y \in \R^{m_1},
    \end{array}

where :math:`u \in \R^n` and :math:`v \in \R^{m_1}`.
Then the initial search directions :math:`d_x^0 \in \R^n` and :math:`d_y^0 \in \R^{m_1}` are set to :math:`d_x^0 = -\Pi_x(g_x^0, y^0)` and :math:`d_y^0 = -\Pi_y(g_x^0, y^0)`, where :math:`g_x^k = \nabla_x q(x^k, y^k) = C^{\T} (C x^k - d)`.
The solution of such a problem is calculated using the Goldfarb and Idnani method for quadratic programming :cite:`lctcg-Goldfarb_Idnani_1983`

The linearly constrained truncated conjugate gradient-like procedure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The general framework employed by `cpqp` to solve problem :eq:`cpqp-slack`, and thus, problem :eq:`nnls` is presented below.

#. Set :math:`x^0 = 0` and :math:`y^0 = [-b]_+`.
#. Set :math:`g_x^0 = C^{\T} (C x^0 - d)`, :math:`d_x^0 = -\Pi_x(g_x^0, y^0)`, :math:`d_y^0 = -\Pi_y(g_x^0, y^0)`, and :math:`k = 0`.
#. Let :math:`\alpha_{\Delta, k} = \argmax \set{\alpha \ge 0 : \norm{x^k + \alpha d_x^k} \le \Delta}`.
#. Let :math:`\alpha_{Q, k}` be :math:`-(\inner{d_x^k, g_x^k} + \inner{d_y^k, y^k}) / (\norm{Cd_x^k}^2 + \norm{d_y^k}^2)` if :math:`\norm{Cd_x^k}^2 + \norm{d_y^k}^2 > 0` and :math:`+\infty` otherwise.
#. Let :math:`\alpha_{L, k} = \argmax \set{\alpha \ge 0 : A (x^k + \alpha d_x^k) - y^k - \alpha d_y^k \le b}` and :math:`\alpha_k = \min \set{\alpha_{\Delta, k}, \alpha_{Q, k}, \alpha_{L, k}}`.
#. Update :math:`x^{k + 1} = x^k + \alpha_k d_x^k`, :math:`y^{k + 1} = y^k + \alpha_k d_y^k`, and :math:`g_x^{k + 1} = g_x^k + \alpha_k C^{\T} C d_x^k`.
#. If :math:`\alpha_k = \alpha_{\Delta, k}` or :math:`g_x^{k + 1} = 0` and :math:`y^{k + 1} = 0`, stop the computations.
#. If :math:`\alpha_k = \alpha_{L, k}`, set :math:`x^0 = x^{k + 1}`, :math:`y^0 = y^{k + 1}`, and go to step 2.
#. Set :math:`\beta_k = (\inner{\Pi_x(g_x^{k + 1}), C^{\T} C d_x^k} + \inner{\Pi_y(y^{k + 1}), d_y^k}) / (\norm{Cd_x^k}^2 + \norm{d_y^k}^2)`, update :math:`d_x^{k + 1} = -\Pi_x(g_x^{k + 1}) + \beta_k d_x^k`, :math:`d_y^{k + 1} = -\Pi_y(y^{k + 1}) + \beta_k d_y^k`, increment :math:`k`, and go to step 3.

The operators :math:`\Pi_x` and :math:`\Pi_y` are evaluated using the Goldfarb and Idnani method for quadratic programming :cite:`lctcg-Goldfarb_Idnani_1983` (and hence, `lctcg` and `cpqp` share a Python function that determines the active set).

Details of the implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Similarly to `lctcg`, after calculating the initial search directions :math:`d_x^0` and :math:`d_y^0` at step 2, the term :math:`b_j - \inner{a_j, x^0 + d_x^0} + y^0 + d_y^0` may be substantial.
In such a case, the method will make a first step towards the boundaries of the active constraints.
Moreover, to prevent defect from computer rounding error, an additional trust-region constraint on :math:`y` is added, namely

.. math::

    \norm{y} \le \sqrt{\norm{y^0}^2 + \norm{C x^0 - d}^2} = \sqrt{\norm{[-b]_+]}^2 + \norm{d}^2}.

In doing so, all the variables of the reformulated problem are bounded.

.. TODO: Termination analysis

.. bibliography::
    :labelprefix: C
    :keyprefix: cpqp-
