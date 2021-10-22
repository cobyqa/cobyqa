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
        \min        & \quad f(x) = \frac{1}{2} \norm{[Ax - b]_+}^2 + \frac{1}{2} \norm{Cx - d}^2\\
        \text{s.t.} & \quad l \le x \le u,\\
                    & \quad \norm{x} \le \Delta,\\
                    & \quad x \in \R^n,
    \end{array}

where :math:`A \in \R^{m_1 \times n}`, :math:`b \in \R^{m_1}`,
:math:`C \in \R^{m_2 \times n}`, :math:`d \in \R^{m_2}`, :math:`l \in \R^n` and
:math:`u \in \R^n` are the lower and upper bounds of the problems
(with :math:`l < u`), and :math:`\Delta > 0` is the current trust-region
radius.
