.. _linalg.nnls:

.. currentmodule:: cobyqa.linalg

Nonnegative least squares
=========================

The update mechanism of the Lagrange multipliers in COBYQA is based on a
constrained least-squares problem, where some variables must remain nonnegative
(in order to satisfy some complementary slackness conditions). The problem we
solve is of the form

.. math::
    :label: nnls

    \begin{array}{ll}
        \min        & \quad f(x) = \frac{1}{2} \norm{Ax - b}^2\\
        \text{s.t.} & \quad x_i \ge 0, ~ i = 1, 2, \dots, n_0,\\
                    & \quad x \in \R^n,
    \end{array}

where :math:`A \in \R^{m \times n}`, :math:`b \in \R^m`, :math:`n_0` is a
nonnegative integer with :math:`n_0 \le n`, and :math:`\norm{\cdot}` is the
Euclidean norm. We observe that if :math:`n_0 = 0`, problem :eq:`nnls` is a
simple unconstrained least-squares problem, which can be solved using
traditional methods (see, e.g., `numpy.linalg.lstsq`).

Description of the method
-------------------------

In order to solve problem :eq:`nnls` when :math:`n_0 \ge 1`, we construct an
active-set method based on Algorithm 23.10 of [LaHa74]_, referred to as `nnls`.
The framework of the method is described below.

#. Set the active set :math:`\mathcal{I}^0` initially to
   :math:`\set{1, 2, \dots, n_0}`, the initial guess :math:`x^0` to the
   origin, and :math:`k = 0`.
#. Evaluate the gradient of the objective function of :eq:`nnls` at
   :math:`x^k`, namely :math:`\nabla f(x^k) = A^{\T} (Ax^k - b)`.
#. If the KKT conditions for problem :eq:`nnls` hold at :math:`x^k`, stop the
   computations.
#. Remove from the active set :math:`\mathcal{I}^k` an index yielding
   :math:`\min \set{\partial_i f(x) : i \in \mathcal{I}^k}` to build
   :math:`\mathcal{I}^{k + 1}`.
#. Let :math:`A_{\scriptscriptstyle\mathcal{I}}^{k + 1}` be the matrix whose
   :math:`i`-th column if the :math:`i`-th column of :math:`A` if
   :math:`i \notin \mathcal{I}^{k + 1}`, and zero otherwise.
#. Let :math:`z^{k + 1}` be a solution of the least squares
   :math:`\min \norm{A_{\scriptscriptstyle\mathcal{I}}^{k + 1} z - b}` with
   :math:`z_i^{k + 1} = 0` for :math:`i \in \mathcal{I}^{k + 1}`, and increment
   :math:`k`.
#. If :math:`z_i^k > 0` for all :math:`i \notin \mathcal{I}^k` with
   :math:`i \le n_0`, update :math:`x^k = z^k` and go to step 2. Set otherwise
   :math:`x^k = x^{k - 1}`.
#. Set :math:`\alpha_k = \min \set{x_i^k / (x_i^k - z_i^k) : z_i^k \le 0, ~ i \notin \mathcal{I}^k, ~ i \le n_0}`.
#. Update :math:`x^{k + 1} = x^k + \alpha_k (z^k - x^k)`,
   :math:`\mathcal{I}^{k + 1} = \mathcal{I}^k \cup \set{i \le n_0 : x_i^{k + 1} = 0}`,
   increment :math:`k`, and go to step 5.

The unconstrained least-squares subproblem of `nnls` at step 6 is solved using
`numpy.linalg.lstsq`. Several refinements of this framework have been made in
the implementation.

#. The number of past iterations is maintained to stop the computations if a
   given threshold is exceeded.
#. Numerical difficulties may arise at step 8 of the method whenever the
   denominator comes close to zero (although it remains theoretically nonzero).
   The division is then safeguarded.
#. Computer rounding errors may engender infinite cycling if the solution has
   been found when checking the KKT conditions. Therefore, the computations are
   stopped if no progress is made from an iteration to another in terms of
   objective function reduction. The theoretical analysis below show that
   strict function decrease must occur from an iteration to another.

Convergence of the method
-------------------------

To study the theoretical properties of the framework, we regard it as
consisting of two nested loops. The inner loop (steps 5--9) has unique entry
and exit points at steps 5 and 7, and only broadens the active set. The outer
loop (steps 2--9) also has unique entry and exit points at steps 2 and 3, and
only narrows the active set. Since the termination criteria of the outer loop
are the KKT conditions for problem :eq:`nnls`, it is clear that the termination
of the algorithm implies its convergence. In order to prove the termination of
the algorithm, we necessitate the following result whose proof is established
in Lemma 23.17 of [LaHa74]_.

**Lemma 1.** Assume that a matrix :math:`A \in \R^{m \times n}` is a full
column rank matrix (with :math:`n \le m`) and that a vector :math:`b \in \R^m`
satisfies :math:`\inner{b, Ae_i} = 0` for
:math:`i \in \set{1, 2, \dots, n} \setminus \set{j}` and
:math:`\inner{b, Ae_j} > 0` with :math:`j \in \set{1, 2, \dots, n}`. Then
the solution vector :math:`\bar{x}` of the least-squares problem
:math:`\min \norm{Ax - b}` satisfies :math:`\bar{x}_j > 0`.

Assume that the KKT conditions for problem :eq:`nnls` do not hold at the
origin. At the first iteration, the algorithm selects the index
(:math:`j`, say) of the most negative component of
:math:`\nabla f(0) = -A^{\T} b` and remove it from the active set
:math:`\mathcal{I}^0`. According to Lemma 1, the solution of the least-squares
problem at step 6 satisfies :math:`z_j^1 > 0`. It is then easy to see that at
each iteration, the solution satisfies :math:`z_j^{k + 1} > 0`, where :math:`j`
is the index selected at step 4. Such a solution is then modified by the inner
loop to ensure that :math:`x_i^{k + 1} \ge 0` for any
:math:`i \in \set{1, 2, \dots, n_0}`. To do so, it will select the closest
point to :math:`z^{k + 1}` on the line joining :math:`x^k` to :math:`z^{k + 1}`
that is feasible.

Termination of the inner loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is clear at this step that all operations made in the inner loop are
well-defined. Moreover, at each inner loop iteration, the cardinal number of
the active set :math:`\mathcal{I}^k` is incremented. If :math:`\mathcal{I}^k`
is maximal, that is :math:`\mathcal{I}^k = \set{1, 2, \dots, n_0}`, then the
condition at step 7 is always true, and the loop terminates. Therefore, the
inner loop must terminate in at most :math:`n_0 - \abs{\mathcal{I}^k} + 1`
iterations, where :math:`\abs{\mathcal{I}^k}` denotes the cardinal number of
the active set when the first inner iteration started.

Termination of the outer loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Termination of the outer loop can be directly inferred by showing that the
value of :math:`f` strictly decreases at each outer loop iteration. Such a
condition ensures that the active set :math:`\mathcal{I}^k` at a given outer
loop iteration is different from all its previous instances, as the feasibility
of each iterate of the outer loop has already been shown (it is specifically
the purpose of the inner loop).

When an inner loop iteration finishes, the value of :math:`x^{k + 1}` is solves

.. math::

    \begin{array}{ll}
        \min        & \quad f(x) = \frac{1}{2} \norm{Ax - b}^2\\
        \text{s.t.} & \quad x_i = 0, ~ i \in \mathcal{I}^{k + 1},\\
                    & \quad x_i > 0, ~ i \notin \mathcal{I}^{k + 1}, ~ i \le n_0,\\
                    & \quad x \in \R^n.
    \end{array}

If the KKT conditions for problem :eq:`nnls` hold at :math:`x^{k + 1}`, then
termination occurs. Otherwise, after incrementing :math:`k`, the index yielding
the least value of :math:`\nabla f(x^k)` is removed from the active set
:math:`\mathcal{I}^k`, and the tentative solution vector :math:`z^{k + 1}`
clearly provides :math:`f(z^{k + 1}) < f(x^k)`. The result is then proven if no
inner loop is entertained (that is, if condition at step 7 holds). Otherwise,
at the end of each inner loop, we have

.. math::

    \begin{aligned}
        \sqrt{2 f\big(x^{k + 1}\big)}
            &= \norm[\big]{A (x^k + \alpha_k (z^k - x^k)) - b}\\
            &= \norm[\big]{(1 - \alpha_k) (Ax^k - b) + \alpha_k (Az^k - b)}\\
            &< \norm{Ax^k - b} = \sqrt{2 f(x^k)},
    \end{aligned}

since :math:`\alpha_k \in (0, 1)`. The termination of the outer loop is then
proven, as well as the convergence of the method.

.. rubric:: References

.. [LaHa74] C. L. Lawson and R. J. Hanson. Solving Least Squares Problems.
   Classics Appl. Math. Philadelphia, PA, US: SIAM, 1974.
