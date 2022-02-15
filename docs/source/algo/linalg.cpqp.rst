.. _linalg.cpqp:

.. currentmodule:: cobyqa.linalg

Convex piecewise quadratic programming
======================================

In general (that is when some linear and/or nonlinear constraints are provided), to determine a trust-region normal step, |project| must solve a problem of the form

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
However, problem :eq:`cpqp` can clearly be reformulated by introducing a variable :math:`y \in \R^{m_1}` as

.. math::
    :label: cpqp-slack

    \begin{array}{ll}
        \min        & \quad q(x, y) = \frac{1}{2} \big(\norm{y}^2 + \norm{Cx - d}^2\big)\\
        \text{s.t.} & \quad Ex + Fy \le g,\\
                    & \quad \norm{x} \le \Delta,\\
                    & \quad x \in \R^n, ~ y \in \R^{m_1},
    \end{array}

where

.. math::

    E =
    \begin{bmatrix}
        A\\
        I\\
        -I\\
        0
    \end{bmatrix}
    \in \R^{(2n + 2m_1) \times n} \quad, F =
    \begin{bmatrix}
        -I\\
        0\\
        0\\
        -I
    \end{bmatrix}
    \in \R^{(2n + 2m_1) \times m_1} \quad, \text{and} \quad g =
    \begin{bmatrix}
        b\\
        u\\
        -l\\
        0
    \end{bmatrix}
    \in \R^{2n + 2m_1}

Compared to the original problem :eq:`cpqp`, the advantage of the reformulated problem :eq:`cpqp-slack` is that its objective function is quadratic, although

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

For convenience, we denote :math:`e_j` and :math:`f_j`, for :math:`j \in \set{1, 2, \dots, 2n + 2m_1}`, respectively the rows of the matrices :math:`E` and :math:`F`.
As for the `lctcg` algorithm, we must consider a constraint as active whenever its residual becomes small.
More specifically, for some feasible :math:`(x, y) \in \R^n \times \R^{m_1}`, we let

.. math::

    \mathcal{J}(x, y) = \set[\bigg]{j \le 2n + 2m_1 : g_j - \inner{e_j, x} - \inner{f_j, y} \le \eta \Delta \sqrt{\norm{e_j}^2 + \norm{f_j}^2}},

where :math:`\eta` is some positive constant (set to :math:`\eta = 0.2` in `cpqp`), and the initial working set if a subset of :math:`\mathcal{J}(x^0, y^0)`, with :math:`x^0 = 0` and :math:`y^0 = [-b]_+`.
Similarly, the initial search directions are set as follows.
We denote :math:`\big(\Pi_{x, k}(u, v), \Pi_{y, k}(u, v)\big) \in \R^n \times \R^{m_1}` the unique solution of

.. math::

    \begin{array}{ll}
        \min        & \quad \frac{1}{2} \big(\norm{s_x - u}^2 + \norm{s_y - v}^2\big)\\
        \text{s.t.} & \quad \inner{e_j, s_x} + \inner{f_j, s_y} \le 0, ~ j \in \mathcal{J}(x^k, y^k),\\
                    & \quad s_x \in \R^n, ~ s_y \in \R^{m_1},
    \end{array}

where :math:`u \in \R^n` and :math:`v \in \R^{m_1}`.
Then the initial search directions :math:`s_x^0 \in \R^n` and :math:`s_y^0 \in \R^{m_1}` are set to :math:`s_x^0 = -\Pi_{x, 0}(g_x^0, y^0)` and :math:`s_y^0 = -\Pi_{y, 0}(g_x^0, y^0)`, where :math:`g_x^k = \nabla_x q(x^k, y^k) = C^{\T} (C x^k - d)`.
The solution of such a problem is calculated using the Goldfarb and Idnani method for quadratic programming :cite:`lctcg-Goldfarb_Idnani_1983`

The linearly constrained truncated conjugate gradient-like procedure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The general framework employed by `cpqp` to solve problem :eq:`cpqp-slack`, and thus, problem :eq:`nnls` is presented below.

#. Set :math:`x^0 = 0`, :math:`y^0 = [-b]_+`, :math:`g_x^0 = -C^{\T} C d`, and :math:`k = 0`.
#. Set :math:`\hat{k} = k`, :math:`s_x^k = -\Pi_{x, \hat{k}}(g_x^k, y^k)`, :math:`s_y^k = -\Pi_{y, \hat{k}}(g_x^k, y^k)`, and stop the computations if :math:`\norm{s_x^k} + \norm{s_y^k} = 0`.
#. Let :math:`\alpha_{\Delta, k} = \argmax \set{\alpha \ge 0 : \norm{x^k + \alpha s_x^k} \le \Delta}`.
#. Let :math:`\alpha_{Q, k}` be :math:`-(\inner{s_x^k, g_x^k} + \inner{s_y^k, y^k}) / (\norm{Cs_x^k}^2 + \norm{s_y^k}^2)` if :math:`\norm{Cs_x^k}^2 + \norm{s_y^k}^2 > 0` and :math:`+\infty` otherwise.
#. Let :math:`\alpha_{L, k} = \argmax \set{\alpha \ge 0 : E (x^k + \alpha s_x^k) + F (y^k + \alpha s_y^k) \le g}`.
#. Set :math:`\alpha_k = \min \set{\alpha_{\Delta, k}, \alpha_{Q, k}, \alpha_{L, k}}`.
#. Update :math:`x^{k + 1} = x^k + \alpha_k s_x^k`, :math:`y^{k + 1} = y^k + \alpha_k s_y^k`, and :math:`g_x^{k + 1} = g_x^k + \alpha_k C^{\T} C s_x^k`.
#. If :math:`\alpha_k = \alpha_{L, k}`, increment :math:`k`, and go to step 2.
#. If :math:`\alpha_k = \alpha_{\Delta, k}`, stop the computations.
#. Set :math:`\beta_k = (\inner{\Pi_{x, \hat{k}}(g_x^{k + 1}), C^{\T} C s_x^k} + \inner{\Pi_{y, \hat{k}}(y^{k + 1}), s_y^k}) / (\norm{Cs_x^k}^2 + \norm{s_y^k}^2)`.
#. Update :math:`s_x^{k + 1} = -\Pi_{x, \hat{k}}(g_x^{k + 1}) + \beta_k s_x^k`, :math:`s_y^{k + 1} = -\Pi_{y, \hat{k}}(y^{k + 1}) + \beta_k s_y^k`, increment :math:`k`, and go to step 3.

The operators :math:`\Pi_{x, \hat{k}}` and :math:`\Pi_{y, \hat{k}}` are evaluated using the Goldfarb and Idnani method for quadratic programming :cite:`lctcg-Goldfarb_Idnani_1983` (and hence, `lctcg` and `cpqp` share a function that determines the active set).

Similarly to `lctcg`, after calculating the search directions :math:`s_x^k` and :math:`s_y^k` at step 2, the term :math:`g_j - \inner{e_j, x^k + s_x^k} - \inner{f_j, y^k + s_y^k}` may be substantial.
In such a case, the method will make a first step towards the boundaries of the active constraints.

.. TODO: Termination analysis

.. bibliography::
    :labelprefix: C
    :keyprefix: cpqp-
