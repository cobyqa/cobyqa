.. _algorithms:

Description of the method
=========================

This page provides detailed mathematical descriptions of the algorithms underneath COBYQA.
It is a derivative-free trust-region SQP method designed to tackle nonlinearly constrained optimization problems that include equality and inequality constraints.
A particular feature of COBYQA is that it visits only points that respect the bound constraints, if any.
This is useful because the objective functions of applications that admit bound constraints are often undefined when the bounds are violated.

.. seealso::

    For a complete description of COBYQA, we refer to Chapters 5 to 7 of the following Ph.D. thesis:

    - T.\  M.\  Ragonneau. "`Model-Based Derivative-Free Optimization Methods and Software <https://tomragonneau.com/documents/thesis.pdf>`_." Ph.D.\  thesis. Hong Kong: Department of Applied Mathematics, The Hong Kong Polytechnic University, 2022.

Statement of the problem
------------------------

The problems we consider are of the form

.. math::

    \min_{x \in \R^n} f(x) \quad \text{s.t.} \quad \left\{ \begin{array}{l}
        g(x) \le 0,\\
        h(x) = 0,\\
        l \le x \le u,
    \end{array} \right.

where :math:`f`, :math:`g`, and :math:`h` represent the objective and constraint functions.
The lower bounds :math:`l \in (\R \cup \set{-\infty})^n` and the upper bounds :math:`u \in (\R \cup \set{\infty})^n` satisfy :math:`l \le u`.
Note that the bound constraints are not included in the inequality constraints, because they will be handled separately.

The derivative-free trust-region SQP method
-------------------------------------------

We present in this section the general framework of COBYQA, named after *Constrained Optimization BY Quadratic Approximations*.
It does not use derivatives of the objective function or the nonlinear constraint functions, but models them using underdetermined interpolation based on the derivative-free symmetric Broyden update.

Interpolation-based quadratic models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recall that the basic trust-region SQP method models the objective and constraint functions based on their gradients and Hessian matrices.
Since we do not have access to such information, we use interpolation-based quadratic models of these functions.
Specifically, we use quadratic models obtained by underdetermined interpolation based on the derivative-free symmetric Broyden update.

At the :math:`k`-th iteration, the objective function :math:`f` is approximated by the quadratic model :math:`\hat{f}_k` that solves

.. math::

    \min_{Q \in \mathcal{P}_{2, n}} \norm{\nabla^2 Q - \nabla^2 \hat{f}_{k - 1}}_{\mathsf{F}} \quad \text{s.t.} \quad Q(y) = f(y), ~ y \in \mathcal{Y}_k,

where :math:`\mathcal{P}_{2, n}` is the set of quadratic polynomials on :math:`\R^n`, :math:`\norm{\cdot}_{\mathsf{F}}` denotes the Frobenius norm, and :math:`\mathcal{Y}_k \subseteq \R^n` is a finite interpolation set updated along the iterations.
The interpolation must satisfy

.. math::

    n + 2 \le \card (\mathcal{Y}_k) \le \frac{1}{2} (n + 1) (n + 2),

so that the quadratic model :math:`\hat{f}_k` is well-defined (note that COBYQA ensures that :math:`\mathcal{Y}_k` is always sufficiently well-poised).
The quadratic models :math:`\hat{g}_k` and :math:`\hat{h}_k` of  :math:`g` and :math:`h` are defined similarly.

The derivative-free trust-region SQP framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We now present the derivative-free trust-region SQP framework employed by COBYQA.
We denote for convenience by :math:`\hat{\mathcal{L}}_k` the Lagrangian function

.. math::

    \hat{\mathcal{L}}_k(x, \lambda, \mu) = \hat{f}_k(x) + \lambda^{\T} \hat{g}_k(x) + \mu^{\T} \hat{h}_k(x).

Also, given a penalty parameter :math:`\gamma_k \ge 0`, we denote by :math:`\varphi_k` the :math:`\ell_2`-merit function

.. math::

    \varphi_k(x) = f(x) + \gamma_k \sqrt{\norm{[g(x)]^+}_2^2 + \norm{h(x)}_2^2},

where :math:`[\cdot]^+` is the componentwise positive-part operator.
Finally, we denote by :math:`\hat{\varphi}_k` the :math:`\ell_2`-merit function

.. math::

    \hat{\varphi}_k(d) = \hat{f}_k(x_k) + \nabla \hat{f}_k(x_k)^{\T} d + \frac{1}{2} d^{\T} \nabla^2 \hat{\mathcal{L}}_k(x_k, \lambda_k, \mu_k) d + \gamma_k \sqrt{\norm{[\hat{g}_k(x_k) + \nabla \hat{g}_k(x_k) d]^+}_2^2 + \norm{\hat{h}_k(x_k) + \nabla \hat{h}_k(x_k) d}_2^2},

where :math:`x_k \in \mathcal{Y}_k` is the best interpolation point according to the merit function :math:`\varphi_{k - 1}`, and :math:`\lambda_k` and :math:`\mu_k` are approximations of the Lagrange multipliers associated with the inequality and equality constraints, respectively.

We provide below a simplified framework of the COBYQA algorithm.

.. admonition:: Simplified framework of COBYQA

    **Data**: Initial trust-region radius :math:`\Delta^0 > 0`.

    - Set the penalty parameter :math:`\gamma_{-1} \gets 0`
    - Build the initial interpolation set :math:`\mathcal{Y}_0 \subseteq \R^n`
    - Define :math:`x_0` to a solution to :math:`\min_{y \in \mathcal{Y}_0} \varphi_0(y)`
    - Estimate the Lagrange multipliers :math:`\lambda_0` and :math:`\mu_0`
    - For :math:`k = 0, 1, \dots` until convergence, do

      - Compute the models :math:`\hat{f}_k`, :math:`\hat{g}_k`, and :math:`\hat{h}_k`
      - Set the trial step :math:`d_k` to an approximate solution to

      .. math::

          \min_{d \in \R^n} \hat{f}_k(x_k) + \nabla \hat{f}_k(x_k)^{\T} d + \frac{1}{2} d^{\T} \nabla^2 \hat{\mathcal{L}}_k(x_k, \lambda_k, \mu_k) d \quad \text{s.t.} \quad \left\{ \begin{array}{l}
              \hat{g}_k(x_k) + \nabla \hat{g}_k(x_k) d \le 0,\\
              \hat{h}_k(x_k) + \nabla \hat{h}_k(x_k) d = 0,\\
              l \le x_k + d \le u,\\
              \norm{d} \le \Delta^k
          \end{array} \right.

      - Pick a penalty parameter :math:`\gamma_k \ge \max \set[\big]{\gamma_{k - 1}, \sqrt{\norm{\lambda_k}^2 + \norm{\mu_k}^2}}` providing :math:`\hat{\varphi}_k(d_k) < \hat{\varphi}_k(0)`
      - Evaluate the reduction ratio

      .. math::

          \rho_k \gets \frac{\varphi_k(x_k) - \varphi_k(x_k + d_k)}{\hat{\varphi}_k(0) - \hat{\varphi}_k(d_k)}

      - If :math:`\rho_k > 0` then

        - Choose a point :math:`\bar{y} \in \mathcal{Y}_k` to remove from :math:`\mathcal{Y}_k`

      - Else

        - Choose a point :math:`\bar{y} \in \mathcal{Y}_k \setminus \set{x_k}` to remove from :math:`\mathcal{Y}_k`

      - Update the interpolation set :math:`\mathcal{Y}_{k + 1} \gets (\mathcal{Y}_k \setminus \set{\bar{y}}) \cup \set{x_k + d_k}`
      - Update the current iterate :math:`x_{k + 1}` to a solution to :math:`\min_{y \in \mathcal{Y}_{k + 1}} \varphi_k(y)`
      - Estimate the Lagrange multipliers :math:`\lambda_{k + 1}` and :math:`\mu_{k + 1}`
      - Update the trust-region radius :math:`\Delta_{k + 1}`
      - Improve the geometry of :math:`\mathcal{Y}_{k + 1}` if necessary

A lot of questions need to be answered
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The framework above is a simplified version of the COBYQA algorithm.
Maybe questions need to be answered to completely define the algorithm.
We provide below some examples.

#. How to calculate the trial step? What if the trust-region subproblem is infeasible?
#. What are the approximate Lagrange multipliers? How to estimate them?
#. How to update the penalty parameter?
#. How to update the trust-region radius?
#. What if the interpolation set :math:`\mathcal{Y}_k` is almost nonpoised?

The answers to these questions (and more) are provided in Ph.D. thesis mentioned at the beginning of this page.
