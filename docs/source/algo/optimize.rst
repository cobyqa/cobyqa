.. _optimize:

.. currentmodule:: cobyqa

*********************************
Optimization framework (`cobyqa`)
*********************************

.. seealso::
    `minimize`,
    `OptimizeResult`

Statement of the problem
========================

The optimization solver |project| is designed to solve the nonlinearly-constrained optimization problem

.. math::
    :label: nlcp

    \begin{array}{ll}
        \min        & \quad f(x)\\
        \text{s.t.} & \quad c_i(x) \le 0, ~ i \in \mathcal{I},\\
                    & \quad c_i(x) = 0, ~ i \in \mathcal{E},\\
                    & \quad l \le x \le u,\\
                    & \quad x \in \R^n,
    \end{array}

where the objective and constraint functions :math:`f` and :math:`c_i`, with :math:`i ∈ \mathcal{I} \cup \mathcal{E}`, are real-valued functions on :math:`\R^n`, and the bound vectors :math:`l \in (\R \cup \set{-\infty})^n` and :math:`u \in (\R \cup \set{+\infty})^n` satisfy :math:`l \le u`.
In opposition to the general nonlinear constraints in :eq:`nlcp`, |project| never violates the bound constraints, as they often represent inalienable physical or theoretical constraints of industrial or research applications.
The solver does not use any derivative of :math:`f` or :math:`c_i`, with :math:`i ∈ \mathcal{I} \cup \mathcal{E}`.
Therefore, it should be employed only when derivative information is unavailable or prohibitively expensive to evaluate.
Such a problem could already be solved using existing solvers, such as COBYLA :cite:`opti-Powell_1994` available through the library `PDFO <https://www.pdfo.net>`_, but

#. it uses linear models, so we expect better performance from |project|; and
#. it may violate the bound constraints in :eq:`nlcp`.

Outline of the method
=====================

The optimization solver |project| is a derivative-free trust-region SQP method, described below.
We emphasize once again that the method always respect the bound constraints, at each step of the algorithm.

Sequential quadratic programming (SQP) framework
------------------------------------------------

- Description of the SQP framework in a derivative-based setting.
- The SQP framework can be used in a DFO setting by replacing the derivatives by models.

We denote :math:`\mathcal{L}` the Lagrangian function for problem :eq:`nlcp`, defined by

.. math::

    \mathcal{L}(x, \lambda) = f(x) + \sum_{i \in \mathcal{I} \cup \mathcal{E}} \lambda_i c_i(x),

where :math:`\lambda_i`, for :math:`i \in \mathcal{I} \cup \mathcal{E}`, denote the Lagrange multipliers.
We do not include the bound constraints in this definition, as they are always respected.
When derivatives of :math:`f` and :math:`c_i`, for :math:`i \in \mathcal{I} \cup \mathcal{E}`, are available, the SQP framework :cite:`opti-Wilson_1963,opti-Han_1976,opti-Han_1977,opti-Powell_1978a,opti-Powell_1978b` generates a step :math:`d^k` from a given iterate :math:`x^k` by solving

.. math::
    :label: sqp

    \begin{array}{ll}
        \min        & \quad f(x^k) + \inner{\nabla f(x^k), d} + \frac{1}{2} \inner{d, B^k d}\\
        \text{s.t.} & \quad c_i(x^k) + \inner{\nabla c_i(x^k), d} \le 0, ~ i \in \mathcal{I},\\
                    & \quad c_i(x^k) + \inner{\nabla c_i(x^k), d} = 0, ~ i \in \mathcal{E},\\
                    & \quad l \le x^k + d \le u,\\
                    & \quad d \in \R^n,
    \end{array}

where :math:`B^k \approx \nabla_{x, x}^2 \mathcal{L} (x^k, \lambda^k)`, for some Lagrange multiplier approximations :math:`\lambda^k`, and sets :math:`x^{k + 1} = x^k + d^k`.

Trust-region framework
----------------------

- The SQP framework is not globally convergent, hence the need of a trust-region embedment (line-search strategies are usually slow in practice).
- Description of the trust-region framework in the unconstrained case.
- Powell's trust-region radii.
- Given a merit function that satisfy some properties, the trust-region framework can be easily adapted to the constrained case.

Quadratic models based on underdetermined interpolation
-------------------------------------------------------

- Given the SQP framework, it is natural to use quadratic models of the objective and constraint functions.
- Models obtained by fully-determined interpolation and regression are prohibitively expensive to initialize.
- Default models minimize the update in the Hessian in Frobenius norm and alternative models minimizes the Hessian itself in Frobenius norm.
- The alternative models replace the default models whenever 3 successive trust-region trial steps provided a very low trust-region ratio (rare).
- At each iteration, at most one point of the interpolation set is modified, resulting in an at-most rank-2 update of the matrix of the KKT system.
- Quick description of the update of the interpolation set and the models.

Merit function and penalty coefficient
--------------------------------------

- The choice of the merit function needs to satisfy the above-mentioned properties.

The merit function
^^^^^^^^^^^^^^^^^^

- Description of the l-2 merit function.
- To satisfy the properties, the trial steps need to satisfy some properties.

Increasing the penalty coefficient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- The lower bound on the penalty coefficient is the maximum between the norm of the Lagrange multipliers and the the value making the trust-region ratio meaningful.
- The penalty coefficient is increased if its current value is too close from the threshold.
- The optimal point in the interpolation set may change when the penalty coefficient is increased, and the method may then need a restart.

Decreasing the penalty coefficient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- To prevent the penalty coefficient from being excessively big, it is reduced when the bound on the trust-region radius is reduced.
- Description of the new value of the penalty coefficient.

Estimation of the Lagrange multipliers
--------------------------------------

.. currentmodule:: cobyqa.linalg

- The estimation of the Lagrange multipliers is based on the KKT conditions on the modelled subproblem.
- Description of the Lagrange multipliers estimates.
- The complementary slackness condition is "relaxed" to prevent most of the multipliers related to the inequality constraints from being zero.
- The subproblem is numerically solved by `nnls`.

Byrd-Omojokun-like trust-region composite steps
-----------------------------------------------

- The trust-region SQP subproblem may be infeasible, and hence cannot be solved as-is.
- Composite step approaches split the trust-region subproblems into two subproblems, each being feasible.
- Other composite step approaches exist such as Vardi and CDT.
- Description of the two subproblems.
- They need to be solved only approximately (done by `bvtcg`, `lctcg`, and `cpqp`).
- The trust-region radius of the tangential subproblem is estimated smartly.
- The trial step satisfy the properties required by the merit function.

Geometry-improvement steps
--------------------------

- This mechanism attempts to maintain an acceptable geometry of the interpolation set.
- Description of the conditions indicating that a geometry-improvement step is required.
- The absolute value of the Lagrange polynomial bound the denominator of the updating formula from below.
- Two steps are calculated, and the one leading to the greatest denominator of the updating formula is chosen.

Maratos effect and second-order correction steps
------------------------------------------------

- Description of the Maratos effect.
- It can be handled by making a second-order correction step in the trust-region framework.
- It is attempted when the current trial point does not provide any decrease in the objective function and the normal step is not too large.
- Description of the second-order correction subproblem.
- It needs to be solved only approximately (done by `cpqp`).

Exit statuses of the method
---------------------------

.. currentmodule:: cobyqa

- Description of all the possible exit statuses of |project|.
- A detailed documentation of the structure of the outputs can be found at `OptimizeResult`,

Computational and numerical efficiencies
========================================

- This section presents some implementation details aiming at improving the computational and numerical efficiencies of |project|.

Shift of the origin in the calculations
---------------------------------------

- The origin in the calculations is shifted to prevent defects from computer rounding errors.
- The models need to be updated when the origin in the calculations is shifted.
- The mechanism is triggered whenever the current optimal best point is far from the current origin, and the origin in the calculations becomes the current optimal best point.

Storage of the quadratic models
-------------------------------

- For numerical efficiency, the second-order term of the quadratic models is stored using an implicit/explicit scheme.
- The matrix of the KKT system is not solved directly, as is NEWUOA.

.. bibliography::
    :labelprefix: O
    :keyprefix: opti-
