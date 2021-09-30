.. _optimize:

*******************
Optimization solver
*******************

Statement of the problem
************************

We are interested in nonlinear constrained optimization problems of the form

.. math::
    :label: nlc

    \begin{array}{ll}
        \min        & \quad \obj(\dv)\\
        \text{s.t.} & \quad \bl \le \dv \le \bu,\\
                    & \quad \con(\dv) \le 0, ~ \icon \in \sub,\\
                    & \quad \con(\dv) = 0, ~ \icon \in \seq.
    \end{array}

where the *objective function* :math:`\obj` and the *constraint functions*
:math:`\con`, with :math:`\icon \in \sub \cup \seq`, are real-valued functions
on :math:`\R^{\nv}`, and :math:`\bl \in \R^{\nv}` and :math:`\bu \in \R^{\nv}`
with :math:`\bl < \bu` are respectively referred to as the *upper bound* and
*lower bound* on the *decision variable* :math:`\dv \in \R^{\nv}`.
We denote the cardinal numbers of :math:`\sub` and :math:`\seq` respectively
:math:`\mub` and :math:`\meq`.

The Lagrangian function :math:`\lag \colon \R^n \times \R^{\mub + \meq} \to \R`
for problem :eq:`nlc` is given by

.. math::
    :label: lag

    \lag(\dv, \lmv) = \obj(\dv) + \sum_{\icon \in \sub \cup \seq} \lmv_{\icon}
    \con(\dv),

where :math:`\lmv \in \R^{\mub + \meq}` is referred to as the *dual variable*.