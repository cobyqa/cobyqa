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

We denote :math:`\mathcal{L}` the Lagrangian function for problem :eq:`nlcp`, defined by

.. math::

    \mathcal{L}(x, \lambda) = f(x) + \sum_{i \in \mathcal{I} \cup \mathcal{E}} \lambda_i c_i(x),

where :math:`\lambda_i`, for :math:`i \in \mathcal{I} \cup \mathcal{E}`, denote the Lagrange multipliers.
We do not include the bound constraints in this definition, as they are always respected.
When derivatives of :math:`f` and :math:`c_i`, for :math:`i \in \mathcal{I} \cup \mathcal{E}`, are available, the SQP framework :cite:`opti-Wilson_1963,opti-Han_1976,opti-Han_1977,opti-Powell_1978a,opti-Powell_1978b` generates a step :math:`d^k` from a given iterate :math:`x^k` by solving

.. math::

    \begin{array}{ll}
        \min        & \quad \inner{\nabla f(x^k), d} + \frac{1}{2} \inner{d, B^k d}\\
        \text{s.t.} & \quad c_i(x^k) + \inner{\nabla c_i(x^k), d} \le 0, ~ i \in \mathcal{I},\\
                    & \quad c_i(x^k) + \inner{\nabla c_i(x^k), d} = 0, ~ i \in \mathcal{E},\\
                    & \quad l \le x^k + d \le u,\\
                    & \quad d \in \R^n,
    \end{array}

where :math:`B^k \approx \nabla_{x, x}^2 \mathcal{L}(x^k, \lambda^k)` and :math:`\inner{\cdot, \cdot}` denotes the Euclidean inner product, for some Lagrange multiplier approximations :math:`\lambda^k`, and sets :math:`x^{k + 1} = x^k + d^k`.
The SQP framework can be used in derivative-free settings by replacing the gradients :math:`\nabla f(x^k)` and :math:`\nabla c_i(x^k)` for :math:`i \in \mathcal{I} \cup \mathcal{E}` by some approximations.
In |project|, we build some quadratic approximations :math:`\widehat{f}_k` and :math:`\widehat{c}_{k, i}` of :math:`f` and :math:`c_i` for :math:`i \in \mathcal{I} \cup \mathcal{E}` around :math:`x^k` (more details are given below), and we hence consider the SQP subproblem

.. math::
    :label: dfsqp

    \begin{array}{ll}
        \min        & \quad \inner{\nabla \widehat{f}_k(x^k), d} + \frac{1}{2} \inner{d, \nabla_{x, x}^2 \widehat{\mathcal{L}}_k(x^k, \lambda^k) d}\\
        \text{s.t.} & \quad \widehat{c}_{k, i}(x^k) + \inner{\nabla \widehat{c}_{k, i}(x^k), d} \le 0, ~ i \in \mathcal{I},\\
                    & \quad \widehat{c}_{k, i}(x^k) + \inner{\nabla \widehat{c}_{k, i}(x^k), d} = 0, ~ i \in \mathcal{E},\\
                    & \quad l \le x^k + d \le u,\\
                    & \quad d \in \R^n,
    \end{array}

where :math:`\widehat{\mathcal{L}}_k` is defined by

.. math::

    \widehat{\mathcal{L}}_k(x, \lambda) = \widehat{f}_k(x) + \sum_{i \in \mathcal{I} \cup \mathcal{E}} \lambda_i \widehat{c}_{k, i}(x).

The main flaw of the SQP method is that its convergence requires the initial guess :math:`x^0` to be close enough from a true solution :math:`x^{\ast}` of :eq:`nlcp`.
Hence, in |project|, we embedded the SQP subproblem :eq:`dfsqp` in a trust-region framework as explained below, to globalize its convergence properties.

Trust-region framework
----------------------

The trust-region SQP subproblem (see, e.g., :cite:`opti-Conn_Gould_Toint_2009`) considered by |project| can be written as

.. math::
    :label: trsqp

    \begin{array}{ll}
        \min        & \quad \inner{\nabla \widehat{f}_k(x^k), d} + \frac{1}{2} \inner{d, \nabla_{x, x}^2 \widehat{\mathcal{L}}_k(x^k, \lambda^k) d}\\
        \text{s.t.} & \quad \widehat{c}_{k, i}(x^k) + \inner{\nabla \widehat{c}_{k, i}(x^k), d} \le 0, ~ i \in \mathcal{I},\\
                    & \quad \widehat{c}_{k, i}(x^k) + \inner{\nabla \widehat{c}_{k, i}(x^k), d} = 0, ~ i \in \mathcal{E},\\
                    & \quad l \le x^k + d \le u,\\
                    & \quad \norm{d} \le \Delta_k,\\
                    & \quad d \in \R^n,
    \end{array}

were :math:`\Delta_k` is the current trust-region radius, automatically adjusted by the method, and :math:`\norm{\cdot}` denotes the :math:`\ell_2`-norm.
Given a merit function :math:`\varphi_k` on the original problem :eq:`nlcp` and its modelled counterpart :math:`\widehat{\varphi}_k`, we define the trust-region ratio

.. math::
    :label: ratio

    \rho_k = \frac{\varphi_k(x^k) - \varphi_k(x^k + d^k)}{\widehat{\varphi}_k(x^k) - \widehat{\varphi}_k(x^k + d^k)},

were :math:`d^k` is an approximate solution to subproblem :eq:`trsqp`.
A crucial property we have to ensure is :math:`\widehat{\varphi}_k(x^k + d^k) \le \widehat{\varphi}_k(x^k)` for the ratio :eq:`ratio` to be meaningful.
Given constants :math:`\eta_1`, :math:`\eta_2`, :math:`\eta_3`, :math:`\gamma_1`, and :math:`\gamma_2` (set respectively to :math:`0`, :math:`0.1`, :math:`0.7`, :math:`0.5`, and :math:`\sqrt{2}` in |project|), the next iterate :math:`x^{k + 1}` is set to :math:`x^k + d^k` if :math:`\rho_k > \eta_1` and :math:`x^k` otherwise.
Moreover, the next trust-region radius :math:`\Delta_{k + 1}` is set to

.. math::

    \left\{
    \begin{array}{ll}
        \gamma_1 \Delta_k                                                                       & \text{if} ~ \rho_k \le \eta_2,\\
        \max \set{\gamma_1 \Delta_k, \norm{d^k}}                                                & \text{if} ~ \eta_2 < \rho_k \le \eta_3,\\
        \min \set{\gamma_2 \Delta_k, \max \set{\gamma_1 \Delta_k, \gamma_1^{-1} \norm{d^k}}}    & \text{otherwise}.
    \end{array}
    \right.

A lower bound on :math:`\Delta_k` is also maintained by |project|, as in :cite:`opti-Powell_1994` or :cite:`opti-Powell_2006`.

Byrd-Omojokun-like trust-region composite steps
-----------------------------------------------

.. currentmodule:: cobyqa.linalg

The main difficulty in solving the trust-region subproblem :eq:`trsqp` is that it may be infeasible.
To cope with this difficulty, |project| employs a composite-step approach.
It consists in solving it as :math:`d^k = n^k + t^k`, were

#. The normal step :math:`n^k` aims at improving the infeasibility of :math:`x^k` (and is hence zero if :math:`x^k` is feasible); and
#. The tangential step :math:`t^k` aims at reducing the objective function in :eq:`trsqp` without worsening the constraint violations.

We employ a Byrd-Omojokun-like approach :cite:`opti-Byrd_1987,opti-Omojokun_1989`, for which the normal step :math:`n^k` solves

.. math::

    \begin{array}{ll}
        \min        & \quad \sum_{i \in \mathcal{I}} \big[\widehat{c}_{k, i}(x^k) + \inner{\nabla \widehat{c}_{k, i}(x^k), d}\big]_+^2 + \sum_{i \in \mathcal{E}} \big[\widehat{c}_{k, i}(x^k) + \inner{\nabla \widehat{c}_{k, i}(x^k), d}\big]^2\\
        \text{s.t.} & \quad l \le x^k + d \le u,\\
                    & \quad \norm{d} \le \zeta \Delta_k,\\
                    & \quad d \in \R^n,
    \end{array}

where :math:`[\cdot]_+` denotes the positive-part operator, and :math:`\zeta \in (0, 1)`.
Such a problem is feasible, and is solved in |project| using a modified truncated conjugate gradient method (see `cpqp` for details). The tangential step :math:`t^k` then solves

.. math::
    :label: tgsp

    \begin{array}{ll}
        \min        & \quad \inner{\nabla \widehat{f}_k(x^k) + \nabla_{x, x}^2 \widehat{\mathcal{L}}_k(x^k, \lambda^k) n^k, d} + \frac{1}{2} \inner{d, \nabla_{x, x}^2 \widehat{\mathcal{L}}_k(x^k, \lambda^k) d}\\
        \text{s.t.} & \quad \min \set{0, \widehat{c}_{k, i}(x^k) + \inner{\nabla \widehat{c}_{k, i}(x^k), n^k}} + \inner{\nabla \widehat{c}_{k, i}(x^k), d} \le 0, ~ i \in \mathcal{I},\\
                    & \quad \inner{\nabla \widehat{c}_{k, i}(x^k), d} = 0, ~ i \in \mathcal{E},\\
                    & \quad l \le x^k + n^k + d \le u,\\
                    & \quad \norm{n^k + d} \le \Delta_k,\\
                    & \quad d \in \R^n,
    \end{array}

Such a problem is also feasible, and it is not clear that the constant :math:`\zeta` is designed to prevent the tangential step to be zero if the current point is far from being feasible.
However the trust-region constraint of the tangential subproblem :eq:`tgsp` is not centered, and is then replace for convenience by

.. math::
    :label: center

    \norm{d} \le \sqrt{\Delta_k^2 - \norm{n^k}^2}.

In doing so, it is easy to verify that

.. math::

    \norm{n^k + t^k} \le \max_{d \in \R^n, ~ \norm{d} \le \Delta_k} \norm{d} + \sqrt{\Delta_k^2 - \norm{d}^2} \le \sqrt{2} \Delta_k.

This problem is then solved in |project| using a modified truncated conjugate gradient method (see `bvtcg` and `lctcg` for details).

Quadratic models based on underdetermined interpolation
-------------------------------------------------------

It is natural to use quadratic models of the objective and constraint functions when using an SQP framework.
However, models obtained by fully-determined interpolation and regression are prohibitively expensive to initialize, as they require at least :math:`(n + 1) (n + 2) / 2` function evaluations to build the initial models.
Therefore, |project| uses quadratic models obtained by underdetermined interpolation.
Given a finite interpolation set :math:`\mathcal{Y}_k \subseteq \R^n` with :math:`x^k \in \mathcal{Y}_k`, the quadratic function :math:`\widehat{f}_k` (and similarly :math:`\widehat{c}_{k, i}` for :math:`i \in \mathcal{I} \cup \mathcal{E}`) satisfy

.. math::
    :label: interp

    \widehat{f}_k(y) = f(y), ~ y \in \mathcal{Y}_k.

If the number of element in :math:`\mathcal{Y}_k` is strictly less than :math:`(n + 1) (n + 2) / 2`, the quadratic function :math:`\widehat{f}_k` is not entirely defined by :eq:`interp`.
The remaining freedom in building :math:`\mathcal{f}_{k + 1}` is bequeathed by satisfying the variational problem

.. math::
    :label: model

    \begin{array}{ll}
        \min        & \quad \norm{\nabla^2 \widehat{f}_{k + 1} - \nabla^2 \widehat{f}_k}_{\mathsf{F}}\\
        \text{s.t.} & \quad \widehat{f}_{k + 1}(y) = f(y), ~ y \in \mathcal{Y}_k,\\
                    & \quad \widehat{f}_{k + 1} \in \R^2 [X^n],
    \end{array}

where :math:`\norm{\cdot}_{\mathsf{F}}` denotes the Frobenius norm.
Interested reader may refer to :cite:`opti-Powell_2004a` for more information.
Rarely (when three consecutive trust-region step provided a very low trust-region ratio :eq:`ratio`), |project| considers that the model :math:`\widehat{f}_k` do not represent accurately enough the true objective :math:`f` and builds the next model as

.. math::

    \begin{array}{ll}
        \min        & \quad \norm{\nabla^2 \widehat{f}_{k + 1}}_{\mathsf{F}}\\
        \text{s.t.} & \quad \widehat{f}_{k + 1}(y) = f(y), ~ y \in \mathcal{Y}_k,\\
                    & \quad \widehat{f}_{k + 1} \in \R^2 [X^n].
    \end{array}

At each iteration, |project| replaces a point of :math:`\mathcal{Y}_k` by :math:`x^k + d^k`.
This results in an at-most rank-2 update of the matrix of the KKT system of the variational problem :eq:`model`.
Therefore, this matrix is maintained by |project|, and is updated iteratively.
The updating formula can be found in :cite:`opti-Powell_2004b` and :cite:`opti-Powell_2006`.
This updating formula has a denominator, and |project| must therefore ensure that it always remains nonzero.

Geometry-improvement steps
--------------------------

The interpolation conditions :eq:`interp` must not contradict each others, and the conditioning of the corresponding system :eq:`model` has to be maintained.
To do this, |project| entertains geometry-improvement iterations :cite:`opti-Conn_Scheinberg_Vicente_2008a,opti-Conn_Scheinberg_Vicente_2008b` in place of the trust-region iterations to prevent the geometry of the interpolation set :math:`\mathcal{Y}_k` from deteriorating.
|project| assumes that the geometry of the interpolation set is acceptable whenever :math:`\norm{x^k - y} \le \Delta_k` for all :math:`y \in \mathcal{Y_k}`.
A geometry-improvement step is then entertained if either :math:`d^k` is low compared to :math:`\Delta_k` or if :math:`\rho_k < \eta_1`.
Note that a geometry-improvement step necessarily followed by a usual trust-region step.

The general idea used by |project| to maintain the geometry of the interpolation set is to prevent the denominator of the updating formula of the KKT matrix of interpolation (see :cite:`opti-Powell_2004b` and :cite:`opti-Powell_2006`) from being too small.
Powell :cite:`opti-Powell_2001` showed that this denominator is lower bounded by :math:`\abs{\ell_k(x^k + d)}`, were :math:`\ell_k` denotes the :math:`k`-th Lagrange polynomial of the interpolation set.
It is the quadratic function whose value is zero at each interpolation point, except at the :math:`k`-th one, whose value is one.
The freedom bequeathed by these interpolation conditions is taken up by minimizing the Hessian matrix of the quadratic function is Frobenius norm.

The strategy of |project| to estimate the optimal geometry-improvement step is as follows.
A first step :math:`s^{k, 1} = \alpha_k x^k + (1 - \alpha_k) y^k` is calculated (see `bvlag`), were :math:`\alpha_k` and :math:`y^k` solve the problem

.. math::

    \begin{array}{ll}
        \min        & \quad \abs{\ell_k(\alpha x^k + (1 - \alpha) y)}\\
        \text{s.t.} & \quad l \le \alpha x^k + (1 - \alpha) y \le u,\\
                    & \quad \alpha \in (0, 1), ~ y \in \mathcal{Y}_k \setminus \set{x^k}.
    \end{array}

A second estimation :math:`s^{k, 2}` is made by simply performing a bound constrained Cauchy step on :math:`\abs{\ell_k(x^k + d)}` (see `bvcs`).
|project| then select the geometry-improvement step among :math:`\set{s^{k, 1}, s^{k, 2}}`, selecting the one leading to the greatest denominator in the updating formula.
This mechanism is taken from :cite:`opti-Powell_2009`.

Merit function and penalty coefficient
--------------------------------------

Recall that the chosen merit functions :math:`\varphi_k` and :math:`\widehat{\varphi}_k` must satisfy the crucial property :math:`\widehat{\varphi}_k(x^k + d^k) \le \widehat{\varphi}_k(x^k)` when :math:`d^k` is obtained during a trust-region iteration.

The merit function
^^^^^^^^^^^^^^^^^^

The merit functions we employ in |project| are the :math:`\ell_2`-merit functions

.. math::

    \varphi_k(x) = f(x) + \sigma_k \sqrt{\sum_{i \in \mathcal{I}} [c_i(x)]_+^2 + \sum_{i \in \mathcal{E}} c_i(x)^2},

and

.. math::

    \widehat{\varphi}_k(x) = \widehat{f}_k(x) + \sigma_k \sqrt{\sum_{i \in \mathcal{I}} [\widehat{c}_i(x^k) + \inner{\nabla \widehat{c}_i(x^k), x - x^k}]_+^2 + \sum_{i \in \mathcal{E}} [\widehat{c}_i(x^k) + \inner{\nabla \widehat{c}_i(x^k), x - x^k}]^2},

where :math:`\sigma_k \ge 0` is a penalty parameter automatically maintained by |project|.
Using the above-mentioned merit function and Byrd-Omojokun-like composite step, we easily have the following theorem.

    **Theorem 1.** If the :math:`k`-th iteration is a trust-region step for which the trial step :math:`d^k` is provided by the above-mentioned Byrd-Omojokun-like composite step, then there exists :math:`\overline{\sigma}_k > 0` such that for all :math:`\sigma_k \ge \overline{\sigma}_k`, we have :math:`\widehat{\varphi}_k(x^k + d^k) \le \widehat{\varphi}(x^k)`.

Thus, the necessary property of the merit function is satisfied by choosing wisely its penalty parameter at each iteration.
Moreover, |project| considers that the optimal point so far is given by

.. math::

    \argmin_{y \in \mathcal{Y}_k} \varphi_k(y).

Increasing the penalty coefficient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If we wish to obtain second-order criticality conditions on the penalty function, we need to have :math:`\liminf_{k \to \infty} \sigma_k > \norm{\lambda^{\ast}}`, where the :math:`\lambda^{\ast}` is the optimal Lagrange multiplier. (see, e.g., Theorem 14.5.1 of :cite:`opti-Conn_Gould_Toint_2009`).
Therefore, |project| ensures that the penalty parameter satisfies :math:`\sigma_k > \max\set{\overline{\sigma}^k, \norm{\lambda^k}}`.
The strategy of |project| to increase the penalty parameter is as follows.
If :math:`\sigma_k \le \nu_1 \max\set{\overline{\sigma}^{k + 1}, \norm{\lambda^{k + 1}}}`, then :math:`\sigma_{k + 1} = \nu_2 \max\set{\overline{\sigma}^{k + 1}, \norm{\lambda^{k + 1}}}`, where :math:`\nu_1 > 1` and :math:`\nu_2 > \nu_1` are constants (respectively set to :math:`1.5` and :math:`2`).
Otherwise, :math:`\sigma_{k + 1} = \sigma_k`.

When increasing the penalty parameter, we may have :math:`\varphi_{k + 1}(x^k) > \varphi_{k + 1}(y)` for some :math:`y \in \mathcal{Y}_k`.
In such a case, a trust-region iteration is entertained after modifying the optimal point so far.

Decreasing the penalty coefficient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To prevent the penalty coefficient from being intensively big, |project| attempts to reduce the penalty parameter as in :cite:`opti-Powell_1994`.
We define for convenience the operators :math:`\xi_{i, 1}` and :math:`\xi_{i, 1}` as

.. math::

    \xi_{i, 1}(x) = \left\{
    \begin{array}{ll}
        c_i(x)          & \text{if} ~ i \in \mathcal{I},\\
        \abs{c_i(x)}    & \text{if} ~ i \in \mathcal{E},\\
    \end{array}
    \right.

and

.. math::

    \xi_{i, 2}(x) = \left\{
    \begin{array}{ll}
        c_i(x)          & \text{if} ~ i \in \mathcal{I},\\
        -\abs{c_i(x)}   & \text{if} ~ i \in \mathcal{E},\\
    \end{array}
    \right.

When the lower bound on the trust-region radius is reduced, the penalty parameter is tentatively set to

.. math::
    :label: ppred

    \sigma_{k + 1} = \frac{\max_{y \in \mathcal{Y}_k} f(y) - \min_{y \in \mathcal{Y}_k} f(y)}{\min_{i \in \mathcal{I} \cup \mathcal{E}} \set[\big]{\max_{y \in \mathcal{Y}_k} \xi_{i, 1}(y) - \big[\min_{y \in \mathcal{Y}_k} \xi_{i, 2}(y)\big]_-}}

where :math:`[\cdot]_-` denotes the negative-part operator, provided that this value provides an actual decrease in the penalty parameter.
See :cite:`opti-Powell_1994` for an explanation of the ratio :eq:`ppred`.

Estimation of the Lagrange multipliers
--------------------------------------

The Lagrange multiplier :math:`\lambda^k` is estimated as follows.
Assuming that the original problem is smooth, and given some constraint qualification condition, the KKT conditions provide that given a solution :math:`x^{\ast}` to the original problem :eq:`nlcp`, there exists a Lagrange multiplier :math:`\lambda^{\ast}` such that

.. math::

    \left\{
    \begin{array}{l}
        \nabla_x \mathcal{L}(x^{\ast}, \lambda^{\ast}) = 0,\\
        \lambda_i^{\ast} c_i(x^{\ast}) = 0, ~ i \in \mathcal{I},\\
        c_i(x^{\ast}) \le 0, ~ i \in \mathcal{I},\\
        c_i(x^{\ast}) = 0, ~ i \in \mathcal{E},\\
        \lambda_i^{\ast} \ge 0, ~ i \in \mathcal{I}.
    \end{array}
    \right.

Therefore, to attempts to solve this KKT system as much as possible using the information available so far, |project| sets :math:`\lambda^k` to the least-squares multipliers, i.e., a solution to

.. math::

    \begin{array}{ll}
        \min        & \quad \norm[\big]{\nabla_x \widehat{\mathcal{L}}_k(x^k, \lambda)}\\
        \text{s.t.} & \quad \lambda_i [\widehat{c}_{k, i}(x^k)]_- = 0, ~ i \in \mathcal{I},\\
                    & \quad \lambda_i \ge 0, ~ i \in \mathcal{I}.
    \end{array}

This subproblem is a nonnegatively constrained linear least-squares problem (see `nnls` for details on how to solve such a problem).

Maratos effect and second-order correction steps
------------------------------------------------

It is well-known that the Maratos effect may prevent Newton-type methods from achieving superlinear convergence :cite:`opti-Maratos_1978`.
Such an effect occurs when the linear models of the constraints in :eq:`trsqp` does not represent accurately enough the curvature of the actual constraints and is caused by discontinuities of the derivatives.
In a trust-region framework, this defect is usually coped using a second-order correction (see, e.g., :cite:`opti-Conn_Gould_Toint_2009`), the simplest of which is the minimum :math:`\ell_2`-norm.
It is attempted when the current trial point does not provide any decrease in the objective function and the normal step is not too large.
In such a case, the trust-region step :math:`d^k` is replaced by :math:`d^k + s^k` where :math:`s^k` solves

.. math::
    :label: soc

    \begin{array}{ll}
        \min        & \quad \sum_{i \in \mathcal{I}} \big[\widehat{c}_{k, i}(x^k + d^k) + \inner{\nabla \widehat{c}_{k, i}(x^k + d^k), d}\big]_+^2 + \sum_{i \in \mathcal{E}} \big[\widehat{c}_{k, i}(x^k + d^k) + \inner{\nabla \widehat{c}_{k, i}(x^k + d^k), d}\big]^2\\
        \text{s.t.} & \quad l \le x^k + d^k + d \le u,\\
                    & \quad \norm{d} \le \norm{d^k},\\
                    & \quad d \in \R^n,
    \end{array}

Since only an approximate second-order correction step is required, problem :eq:`soc` can clearly be solved numerically using `cpqp`.

Exit statuses of the method
---------------------------

.. currentmodule:: cobyqa

We provide here for convenience details on the possibles exit statuses of |project|.
A detailed documentation of the structure of the outputs can be found at `OptimizeResult`.

===========  ===================================================================
Exit status  Explanation
===========  ===================================================================
0            The lower bound for the trust-region radius has been reached, the
             usual stopping criterion of trust-region method. If such an exit
             status is achieved, the optimization terminated successfully.
-----------  -------------------------------------------------------------------
1            The user can provide a target function value to |project|, which
             stops the computation is this value has been achieved by a feasible
             point. If such an exit status is achieved, the optimization
             terminated successfully.
-----------  -------------------------------------------------------------------
2            The user can provide an absolute tolerance on the objective
             function values to |project|, which stops the computations if two
             consecutive function values are within such a tolerance. If such an
             exit status is achieved, the optimization terminated successfully.
-----------  -------------------------------------------------------------------
3            The user can provide a relative tolerance on the objective function
             values to |project|, which stops the computations if two
             consecutive function values are within such a tolerance. If such an
             exit status is achieved, the optimization terminated successfully.
-----------  -------------------------------------------------------------------
4            The user can provide an absolute tolerance on the decision variable
             (that is, the current iterate) to |project|, which stops the
             computations if two consecutive iterates are within such a
             tolerance in :math:`\ell_2`-norm. If such an exit status is
             achieved, the optimization terminated successfully.
-----------  -------------------------------------------------------------------
5            The user can provide a relative tolerance on the decision variable
             (that is, the current iterate) to |project|, which stops the
             computations if two consecutive iterates are within such a
             tolerance in :math:`\ell_2`-norm. If such an exit status is
             achieved, the optimization terminated successfully.
-----------  -------------------------------------------------------------------
6            The user can provide a maximum number of function evaluations to
             |project|, which stops the computation if such an amount of
             function evaluations has been performed.
-----------  -------------------------------------------------------------------
7            The user can provide a maximum number of iterations to |project|,
             which stops the computation if such an amount of iterations has
             been performed.
-----------  -------------------------------------------------------------------
8            Occasionally, due to computer rounding errors, the denominator of
             the updating formula of the KKT matrix of interpolation is zero.
             It means that the above-mentioned mechanisms deployed to prevent
             this from happening failed. This occurs very rarely.
-----------  -------------------------------------------------------------------
9            All variables are fixed by the constraints. If such an exit status
             is achieved and if this point is feasible, the optimization
             terminated successfully.
-----------  -------------------------------------------------------------------
-1           The bound constraints are infeasible.
===========  ===================================================================

Computational and numerical efficiencies
========================================

We prevent in this section some implementation details aiming at improving the computational and numerical efficiencies of |project|.

Shift of the origin in the calculations
---------------------------------------

The origin in the calculations is shifted to prevent defects from computer rounding errors.
This means that all the calculations are done relatively to a vector maintained by |project|.
It is initially set to the best point in the initial interpolation set, and it is updated whenever the current optimal best point is far from the current origin, and the origin in the calculations becomes the current optimal best point.

Storage of the quadratic models
-------------------------------

For numerical efficiency, the second-order term of the quadratic models is not stored as-is.
Rather, the implementation technique suggested in section 3 of :cite:`opti-Powell_2004a` is used by |project|.
The general idea is to write the the Hessian matrix of the models as

.. math::

    \nabla^2 \widehat{f}_k(x) = \Omega_k + \sum_{y \in \mathcal{Y}_k} \omega_{k, y} (y - x^k) (y - x^k)^{\mathsf{T}},

where :math:`\Omega_k \in \R^{n \times n}` is referred to as the explicit part of the Hessian matrix, and :math:`\set{\omega_{k, y}}_{y \in \mathcal{Y}_k}` is referred to as the implicit part of the Hessian matrix.
Of course, this matrix is never built, as only matrix-vector products are needed by |project|.
Such a decomposition of the Hessian matrices of the quadratic models reduces the numerical complexity the updates of the models when a point of :math:`\mathcal{Y}_k` is modified.

|project| also maintains the inverse matrix of the KKT system of interpolation (see :cite:`opti-Powell_2004b`).
Since only its inverse is required in the computations, the original matrix is not stored.
From a theoretical standpoint, the leading :math:`\abs{\mathcal{Y_k}} \times \abs{\mathcal{Y_k}}` submatrix :math:`\Lambda` of the inverse matrix has rank :math:`\abs{\mathcal{Y}_k} - n - 1` and is positive definite.
However, as mentioned in :cite:`opti-Powell_2006`, these properties may be lost in practice.
Therefore, a decomposition :math:`\Lambda = Z D Z^{\mathsf{T}}` is used instead, where :math:`Z` is a general matrix of size :math:`\abs{\mathcal{Y}_k} \times (\abs{\mathcal{Y}_k} - n - 1)` (which preserves the rank), and :math:`D` is a diagonal matrix with entries :math:`\pm 1`.
Theoretically, :math:`D` could always be the identity matrix, but negative values are allowed to tackle numerical difficulties.
Nevertheless, it has been observed that this matrix remains the identity matrix in most of the tested applications.

.. only:: html or text

    .. rubric:: References

.. bibliography::
    :labelprefix: O
    :keyprefix: opti-
