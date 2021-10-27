.. _linalg.lctcg:

.. currentmodule:: cobyqa.linalg

Linear constrained truncated conjugate gradient
===============================================

In general (that is when some linear and/or nonlinear constraints are
provided), to determine a trust-region tangential step, COBYQA must solve a
problem of the form

.. math::
    :label: lctcg

    \begin{array}{ll}
        \min        & \quad f(x) = \inner{x, g} + \frac{1}{2} \inner{x, H x}\\
        \text{s.t.} & \quad Ax \le b,\\
                    & \quad Cx = d,\\
                    & \quad \norm{x} \le \Delta,\\
                    & \quad x \in \R^n,
    \end{array}

where :math:`g \in \R^n` approximates the gradient of the nonlinear objective
function at the origin, :math:`H \in \R^{n \times n}` is a symmetric matrix
that approximates the Hessian matrix of the nonlinear objective function at the
origin, :math:`A \in \R^{m_1 \times n}` and :math:`C \in \R^{m_2 \times n}` are
the Jacobian matrices of the linear inequality and equality constraints,
:math:`b \in \R^{m_1}` and :math:`d \in \R^{m_2}` are the corresponding
right-hand sides, :math:`\Delta > 0` is the current trust-region radius, and
:math:`\norm{\cdot}` is the Euclidean norm.

The unconstrained case
----------------------

As mentioned in the section :ref:`tcg_base` of the description of the `bvtcg`
method, it is usual to solve inexactly problem :eq:`lctcg` using a truncated
conjugate gradient method of Steihaug :cite:`lctcg-Steihaug_1983` and Toint
:cite:`lctcg-Toint_1981` in the unconstrained case. Given the initial values
:math:`x^0 = 0` and :math:`d^0 = -g`, it generates the sequence of iterates

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

The constrained case
--------------------

When linear and/or nonlinear constraints are provided, COBYQA solves its
trust-region tangential subproblems using a modified TRSTEP algorithm
:cite:`lctcg-Powell_2015`. It is an active-set variation of the truncated
conjugate gradient algorithm, which maintains the QR factorization of the
matrix whose columns are the gradients of the active constraints. As for
`bvtcg`, if a new constraint is added to the active set, the procedure is
restarted. However, we allow constraints to be removed from the active set in
this method.

Management of the active set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For convenience, we denote
:math:`a_j`, for :math:`j \in \set{1, 2, \dots, m_1}` and
:math:`c_j`, for :math:`j \in \set{1, 2, \dots, m_2}` the rows of the matrices
:math:`A` and :math:`C`. We assume that the initial guess :math:`x^0` is
feasible and that an inequality constraint :math:`b_j - \inner{a_j, x}` is
positive and tiny for some :math:`j \le m_1`. If :math:`j` do not belong to the
active set and if :math:`\inner{a_j, g} < 0`, then it is likely that the
:math:`\norm{x^1 - x^0}` is small, as a step along the search direction
:math:`d^0 = -g` quickly exits the feasible set. Therefore, we must consider a
constraint active whenever its residual becomes small. More precisely, for some
feasible :math:`x \in \R^n`, we let

.. math::

    \mathcal{J}(x) = \set[\big]{j \le m_1 : b_j - \inner{a_j, x} \le \eta \Delta \norm{a_j}},

where :math:`\eta` is some positive constant
(set to :math:`\eta = 0.2` in `lctcg`). At each iteration, the active set is a
subset of :math:`\mathcal{J}(x^k)`. Moreover, the initial search direction
:math:`d^0` should be close to :math:`-g` and prevent the point :math:`x^1` to
be close from :math:`x^0`. Therefore, the initial search direction :math:`d^0`
is the unique solution of

.. math::

    \begin{array}{ll}
        \min        & \quad \frac{1}{2} \norm{g + d}^2\\
        \text{s.t.} & \quad \inner{a_j, d} \le 0, ~ j \in \mathcal{J}(x^k),\\
                    & \quad \inner{c_j, d} = 0, ~ j \in \set{1, 2, \dots, m_2}\\
                    & \quad d \in \R^n.
    \end{array}

If :math:`\inner{a_j, d^0} < 0` for some :math:`j`, then the point :math:`x^1`
will be further from this constraint than the initial guess. Therefore, the
active set :math:`\mathcal{I}` is chosen to be
:math:`\set{j \in \mathcal{J}(x^0) : \inner{a_j, d^0} = 0}` (or a subset of it,
chosen so that :math:`\set{a_j : j \in \mathcal{I}}` is a basis of
:math:`\vspan \set{a_j : j \in \mathcal{J}(x^0), ~ \inner{a_j, d^0} = 0}`.
bibtex spohinx
The solution of such a problem is calculated using the Goldfarb and Idnani
method for quadratic programming :cite:`lctcg-Goldfarb_Idnani_1983`.

.. bibliography::
    :labelprefix: L
    :keyprefix: lctcg-
