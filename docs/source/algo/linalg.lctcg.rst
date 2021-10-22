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
right-hand sides, and :math:`\Delta > 0` is the current trust-region radius.

As mentioned in the section :ref:`tcg_base` of the description of the `bvtcg`
method, it is usual to solve inexactly problem :eq:`lctcg` using a truncated
conjugate gradient method.

Description of the method
-------------------------
