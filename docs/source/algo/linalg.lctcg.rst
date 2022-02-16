.. _linalg.lctcg:

.. currentmodule:: cobyqa.linalg

Linear constrained truncated conjugate gradient
===============================================

In general (that is when some linear and/or nonlinear constraints are provided), to determine a trust-region tangential step, |project| must solve a problem of the form

.. math::
    :label: lctcg

    \begin{array}{ll}
        \min        & \quad q(x) = \inner{x, g} + \frac{1}{2} \inner{x, H x}\\
        \text{s.t.} & \quad Ax \le b,\\
                    & \quad Cx = d,\\
                    & \quad \norm{x} \le \Delta,\\
                    & \quad x \in \R^n,
    \end{array}

where :math:`g \in \R^n`, :math:`H \in \R^{n \times n}` is a symmetric matrix, :math:`A \in \R^{m_1 \times n}` and :math:`C \in \R^{m_2 \times n}` are the Jacobian matrices of the linear inequality and equality constraints, :math:`b \in \R^{m_1}` and :math:`d \in \R^{m_2}` are the corresponding right-hand sides, :math:`\Delta > 0` is the given trust-region radius, and :math:`\norm{\cdot}` denotes the Euclidean norm.
For simplicity, we assume throughout this section that the matrix :math:`C` is full row rank.

The unconstrained case
----------------------

As mentioned in the section :ref:`tcg_base` of the description of the `bvtcg` method, it is usual to solve inexactly problem :eq:`lctcg` using a truncated conjugate gradient method of Steihaug :cite:`lctcg-Steihaug_1983` and Toint :cite:`lctcg-Toint_1981` in the unconstrained case.
Given the initial values :math:`x^0 = 0` and :math:`s^0 = -g`, it generates the sequence of iterates

.. math::

    \left\{
    \begin{array}{l}
        \alpha_k = -\inner{s^k, g^k} / \inner{s^k, Hs^k},\\
        \beta_k = \norm{g^{k + 1}}^2 / \norm{g^k}^2,\\
        x^{k + 1} = x^k + \alpha_k s^k,\\
        s^{k + 1} = -g^k + \beta_k s^k,
    \end{array}
    \right.

where :math:`g^k = \nabla q(x^k) = g + Hx^k`. The computations are stopped if either

#. :math:`\norm{g^k} = 0`; or
#. :math:`\norm{x^{k + 1}} \ge \Delta` or :math:`\inner{s^k, Hs^k} < 0`, in which case :math:`x^k + \alpha_{\Delta} s^k` is returned, where :math:`\alpha_{\Delta} > 0` is chosen so that :math:`\norm{x^k + \alpha_{\Delta} s^k} = \Delta`.

The constrained case
--------------------

When linear and/or nonlinear constraints are provided, |project| solves its trust-region tangential subproblems using a modified TRSTEP algorithm :cite:`lctcg-Powell_2015`.
It is an active-set variation of the truncated conjugate gradient algorithm, which maintains the QR factorization of the matrix whose columns are the gradients of the active constraints.
As for `bvtcg`, if a new constraint is added to the working set, the procedure is restarted.
However, we allow constraints to be removed from the working set in this method.

The working set
^^^^^^^^^^^^^^^

For convenience, we denote :math:`a_j`, for :math:`j \in \set{1, 2, \dots, m_1}`, the rows of the matrix :math:`A`.
We assume that the initial guess :math:`x^0` is feasible and that an inequality constraint :math:`b_j - \inner{a_j, x}` is positive and tiny for some :math:`j \le m_1`. If :math:`j` do not belong to the working set and if :math:`\inner{a_j, g} < 0`, then it is likely that :math:`\norm{x^1 - x^0}` is small, as a step along the search direction :math:`s^0 = -g` quickly exits the feasible set.
Therefore, we must consider a constraint as active whenever its residual becomes small.
More precisely, for some feasible :math:`x \in \R^n`, we let

.. math::

    \mathcal{J}(x) = \set[\big]{j \le m_1 : b_j - \inner{a_j, x} \le \eta \Delta \norm{a_j}},

where :math:`\eta` is some positive constant (set to :math:`\eta = 0.2` in `lctcg`), and the initial working set is a subset of :math:`\mathcal{J}(x^0)`.
Moreover, the initial search direction :math:`s^0` should be close to :math:`-\nabla q(x^0)` and prevent the point :math:`x^1` to be close from :math:`x^0`.
We denote :math:`\Pi_k(v)` the unique solution of

.. math::
    :label: init-search

    \begin{array}{ll}
        \min        & \quad \frac{1}{2} \norm{s - v}^2\\
        \text{s.t.} & \quad \inner{a_j, s} \le 0, ~ j \in \mathcal{J}(x^k),\\
                    & \quad \inner{c_j, s} = 0, ~ j \in \set{1, 2, \dots, m_2}\\
                    & \quad s \in \R^n.
    \end{array}

where :math:`v \in \R^n`. Then, the initial search direction :math:`s^0` is set to :math:`-\Pi_0\big(\nabla q(x^0)\big)`.
If :math:`\inner{a_j, s^0} < 0` for some :math:`j`, then the point :math:`x^1` will be further from this constraint than the initial guess.
Therefore, the working set :math:`\mathcal{I}^0` is chosen to be :math:`\set{j \in \mathcal{J}(x^0) : \inner{a_j, s^0} = 0}` (or a subset of it, so that :math:`\set{a_j : j \in \mathcal{I}}` is a basis of :math:`\vspan \set{a_j : j \in \mathcal{J}(x^0), ~ \inner{a_j, s^0} = 0}`).
The solution of such a problem is calculated using the Goldfarb and Idnani method for quadratic programming :cite:`lctcg-Goldfarb_Idnani_1983`.

The linearly constrained truncated conjugate gradient procedure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The general framework employed by `lctcg` is presented below.
The restarting mechanism is initiated whenever the current step reaches the boundary of a constraint that is not yet included in the working set if the distance between this step and the trust-region boundary is larger than :math:`\eta \Delta`.

#. Set :math:`x^0 = 0`, :math:`g^0 = g`, and :math:`k = 0`.
#. Set :math:`\hat{k} = k`, :math:`s^k = -\Pi_{\hat{k}}(g^k)`, and stop the computations if :math:`\norm{s^k} = 0`.
#. Set :math:`\alpha_{\Delta, k} = \argmax \set{\alpha \ge 0 : \norm{x^k + \alpha s^k} \le \Delta}`.
#. Set :math:`\alpha_{Q, k}` to :math:`-\inner{s^k, g^k} / \inner{s^k, Hs^k}` if :math:`\inner{s^k, Hs^k} > 0` and :math:`+\infty` otherwise.
#. Set :math:`\alpha_{L, k} = \argmax \set{\alpha \ge 0 : A (x^k + \alpha s^k) \le b}`.
#. Set :math:`\alpha_k = \min \set{\alpha_{\Delta, k}, \alpha_{Q, k}, \alpha_{L, k}}`.
#. Update :math:`x^{k + 1} = x^k + \alpha_k s^k` and :math:`g^{k + 1} = g^k + \alpha_k H s^k`.
#. If :math:`\alpha_k = \alpha_{L, k}` and :math:`\norm{x^{k + 1}} \le (1 - \eta) \Delta`, increment :math:`k`, and go to step 2.
#. If :math:`\alpha_k = \alpha_{\Delta, k}` or :math:`\alpha_k = \alpha_{L, k}` and :math:`\norm{x^{k + 1}} > (1 - \eta) \Delta`, stop the computations.
#. Set :math:`\beta_k = \inner{\Pi_{\hat{k}}(g^{k + 1}), Hs^k} / \inner{s^k, Hs^k}`.
#. Update :math:`s^{k + 1} = -\Pi_{\hat{k}}(g^{k + 1}) + \beta_k s^k`, increment :math:`k`, and go to step 3.

As mentioned earlier, the operator :math:`\Pi_{\hat{k}}` is evaluated using the Goldfarb and Idnani method for quadratic programming :cite:`lctcg-Goldfarb_Idnani_1983`.
It builds the QR factorization of the matrix whose columns are the gradients of the active constraints (namely the linear equality constraints and the linear inequality constraints indexed by :math:`\mathcal{I}^{\hat{k}}`).
We denote :math:`\hat{Q} R` such a factorization, with :math:`\hat{Q} \in \R^{n \times (m_2 + \abs{\mathcal{I}^{\hat{k}}})}` and :math:`R \in \R^{(m_2 + \abs{\mathcal{I}^{\hat{k}}}) \times (m_2 + \abs{\mathcal{I}^{\hat{k}}})}`, where :math:`\abs{\mathcal{I}^{\hat{k}}}` denotes the cardinal number of the working set.
We clearly have :math:`\abs{\mathcal{I}^{\hat{k}}} \le n - m_2`.
Let :math:`\check{Q} \in \R^{n \times (n - m_2 - \abs{\mathcal{I}^{\hat{k}}})}` be any matrix such that :math:`\begin{bmatrix} \hat{Q}& \check{Q} \end{bmatrix}` is orthogonal.
Powell showed in :cite:`lctcg-Powell_2015` that the solution of problem :eq:`init-search` is given by

.. math::

    \Pi_{\hat{k}}(v) = \check{Q} \check{Q}^{\T} v, \quad v \in \R^n.

Therefore, the term :math:`\Pi_{\hat{k}}(g^{k + 1})` in steps 10 and 11 can be easily computed at each iteration.
Moreover, after calculating the search direction :math:`s^k` at step 2, the term :math:`b_j - \inner{a_j, x^k + s^k}` may be substantial.
In such a case, the method will make a first step towards the boundaries of the active constraints.

Additional stopping criteria
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If :math:`\nabla q(x^k)` is small along the current search direction, namely if

.. math::

    \alpha_{\Delta, k} \abs{\inner{s^k, \nabla q(x^k)}} \le \nu \big(q(x^0) - q(x^k)\big),

where :math:`\nu` is some positive constant (set to :math:`\nu = 0.01` in `lctcg`), then the method considers that the iterations should terminate.
Thus, the iterations are also stopped if the reduction provided by the current search direction is tiny compared to the reduction so far, that is if

.. math::

    q(x^k) - q(x^{k + 1}) \le \nu \big(q(x^0) - q(x^{k + 1})\big).

.. TODO: Termination analysis

.. bibliography::
    :labelprefix: L
    :keyprefix: lctcg-
