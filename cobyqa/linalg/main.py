import numpy as np
from numpy.testing import assert_

from ._bvcs import bvcs as _bvcs
from ._bvlag import bvlag as _bvlag
from ._bvtcg import bvtcg as _bvtcg
from ._cpqp import cpqp as _cpqp
from ._lctcg import lctcg as _lctcg
from ._nnls import nnls as _nnls


def bvcs(xpt, kopt, gq, curv, xl, xu, delta, *args, **kwargs):
    """
    Evaluate Cauchy step on the absolute value of a Lagrange polynomial, subject
    to bound constraints on its coordinates and its length.

    Parameters
    ----------
    xpt : numpy.ndarray, shape (npt, n)
        Set of points. Each row of `xpt` stores the coordinates of a point.
    kopt : int
        Index of the point from which the Cauchy step is evaluated.
    gq : array_like, shape (n,)
        Gradient of the Lagrange polynomial of the points in `xpt` (not
        necessarily the `kopt`-th one) at ``xpt[kopt, :]``.
    curv : callable
        Function providing the curvature of the Lagrange polynomial.

            ``curv(x, *args) -> float``

        where ``x`` is an array with shape (n,) and ``args`` is the tuple of
        fixed parameters needed to specify the function.
    xl : array_like, shape (n,)
        Lower-bound constraints on the decision variables. Use ``-numpy.inf`` to
        disable the bounds on some variables.
    xu : array_like, shape (n,)
        Upper-bound constraints on the decision variables. Use ``numpy.inf`` to
        disable the bounds on some variables.
    delta : float
        Upper bound on the length of the Cauchy step.
    *args : tuple, optional
        Parameters to forward to the curvature function.

    Returns
    -------
    step : numpy.ndarray, shape (n,)
        Cauchy step.
    cauchy : float
        Square of the Lagrange polynomial evaluation at the Cauchy point.

    Other Parameters
    ----------------
    debug : bool, optional
        Whether to make debugging tests during the execution, which is
        not recommended in production (the default is False).

    Raises
    ------
    AssertionError
        The vector ``xpt[kopt, :]`` is not feasible (only in debug mode).

    See Also
    --------
    bvlag : Bounded variable absolute Lagrange polynomial maximization

    Notes
    -----
    The method is adapted from the ALTMOV algorithm [1]_, and the vector
    ``xpt[kopt, :]`` must be feasible.

    References
    ----------
    .. [1] M. J. D. Powell. The BOBYQA algorithm for bound constrained
       optimization without derivatives. Tech. rep. DAMTP 2009/NA06. Cambridge,
       UK: Department of Applied Mathematics and Theoretical Physics, University
       of Cambridge, 2009.
    """
    xpt = np.atleast_2d(xpt)
    if xpt.dtype.kind in np.typecodes['AllInteger']:
        xpt = np.asarray(xpt, dtype=float)
    xpt = np.asfortranarray(xpt)
    gq = np.atleast_1d(gq).astype(float)
    xl = np.atleast_1d(xl).astype(float)
    xu = np.atleast_1d(xu).astype(float)

    # Check the sizes of the inputs.
    assert_(xpt.ndim == 2)
    assert_(gq.ndim == 1)
    assert_(xl.ndim == 1)
    assert_(xu.ndim == 1)
    assert_(gq.size == xpt.shape[1])
    assert_(xl.size == xpt.shape[1])
    assert_(xu.size == xpt.shape[1])

    def curv_safe(x):
        cx = np.float64(curv(x, *args))
        return cx

    debug = kwargs.get('debug', False)
    step, cauchy = _bvcs(xpt, kopt, gq, curv_safe, xl, xu, delta, debug)  # noqa
    return np.array(step, dtype=float), cauchy


def bvlag(xpt, kopt, klag, gq, xl, xu, delta, alpha, **kwargs):
    """
    Estimate a point that maximizes a lower bound on the denominator of the
    updating formula, subject to bound constraints on its coordinates and its
    length.

    Parameters
    ----------
    xpt : numpy.ndarray, shape (npt, n)
        Set of points. Each row of `xpt` stores the coordinates of a point.
    kopt : int
        Index of a point in `xpt`. The estimated point will lie on a line
        joining ``xpt[kopt, :]`` to another point in `xpt`.
    klag : int
        Index of the point in `xpt`.
    gq : array_like, shape (n,)
        Gradient of the `klag`-th Lagrange polynomial at ``xpt[kopt, :]``.
    xl : array_like, shape (n,)
        Lower-bound constraints on the decision variables. Use ``-numpy.inf`` to
        disable the bounds on some variables.
    xu : array_like, shape (n,)
        Upper-bound constraints on the decision variables. Use ``numpy.inf`` to
        disable the bounds on some variables.
    delta : float
        Upper bound on the length of the step.
    alpha : float
        Real parameter.

    Returns
    -------
    step : numpy.ndarray, shape (n,)
        Step from ``xpt[kopt, :]`` towards the estimated point.

    Other Parameters
    ----------------
    debug : bool, optional
        Whether to make debugging tests during the execution, which is
        not recommended in production (the default is False).

    Raises
    ------
    AssertionError
        The vector ``xpt[kopt, :]`` is not feasible (only in debug mode).

    See Also
    --------
    bvcs : Bounded variable Cauchy step

    Notes
    -----
    The denominator of the updating formula is given in Equation (3.9) of [2]_,
    and the parameter `alpha` is the referred in Equation (4.12) of [1]_.

    References
    ----------
    .. [1] M. J. D. Powell. "The NEWUOA software for unconstrained optimization
       without derivatives." In: Large-Scale Nonlinear Optimization. Ed. by G.
       Di Pillo and M. Roma. New York, NY, US: Springer, 2006, pp. 255-–297.
    .. [2] M. J. D. Powell. The BOBYQA algorithm for bound constrained
       optimization without derivatives. Tech. rep. DAMTP 2009/NA06. Cambridge,
       UK: Department of Applied Mathematics and Theoretical Physics, University
       of Cambridge, 2009.
    """
    xpt = np.atleast_2d(xpt).astype(float)
    xpt = np.asfortranarray(xpt)
    gq = np.atleast_1d(gq)
    if gq.dtype.kind in np.typecodes['AllInteger']:
        gq = np.asarray(gq, dtype=float)
    xl = np.atleast_1d(xl).astype(float)
    xu = np.atleast_1d(xu).astype(float)

    # Check the sizes of the inputs.
    assert_(xpt.ndim == 2)
    assert_(gq.ndim == 1)
    assert_(xl.ndim == 1)
    assert_(xu.ndim == 1)
    assert_(gq.size == xpt.shape[1])
    assert_(xl.size == xpt.shape[1])
    assert_(xu.size == xpt.shape[1])

    debug = kwargs.get('debug', False)
    step = _bvlag(xpt, kopt, klag, gq, xl, xu, delta, alpha, debug)  # noqa
    return np.array(step, dtype=float)


def bvtcg(xopt, gq, hessp, xl, xu, delta, *args, **kwargs):
    """
    Minimize approximately a quadratic function subject to bound and
    trust-region constraints using a truncated conjugate gradient.

    Parameters
    ----------
    xopt : numpy.ndarray, shape (n,)
        Point around which the Taylor expansions of the quadratic function is
        defined.
    gq : array_like, shape (n,)
        Gradient of the quadratic function at `xopt`.
    hessp : callable
        Function providing the product of the Hessian matrix of the quadratic
        function with any vector.

            ``hessp(x, *args) -> array_like, shape(n,)``

        where ``x`` is an array with shape (n,) and `args` is a tuple of
        parameters to forward to the objective function. It is assumed that the
        Hessian matrix implicitly defined by `hessp` is symmetric, but not
        necessarily positive semidefinite.
    xl : array_like, shape (n,)
        Lower-bound constraints on the decision variables. Use ``-numpy.inf`` to
        disable the bounds on some variables.
    xu : array_like, shape (n,)
        Upper-bound constraints on the decision variables. Use ``numpy.inf`` to
        disable the bounds on some variables.
    delta : float
        Upper bound on the length of the step from `xopt`.
    *args : tuple, optional
        Parameters to forward to the Hessian product function.

    Returns
    -------
    step : numpy.ndarray, shape (n,)
        Step from `xopt` towards the estimated point.

    Other Parameters
    ----------------
    debug : bool, optional
        Whether to make debugging tests during the execution, which is
        not recommended in production (the default is False).
    improve_tcg : bool, optional
        Whether to improve the truncated conjugate gradient step round the
        trust-region boundary (the default is True).

    Raises
    ------
    ValueError
        The vector `xopt` is not feasible (only in debug mode).

    See Also
    --------
    cpqp : Convex piecewise quadratic programming
    lctcg : Linear constrained truncated conjugate gradient
    nnls : Nonnegative least squares

    Notes
    -----
    The method is adapted from the TRSBOX algorithm [1]_.

    References
    ----------
    .. [1] M. J. D. Powell. The BOBYQA algorithm for bound constrained
       optimization without derivatives. Tech. rep. DAMTP 2009/NA06. Cambridge,
       UK: Department of Applied Mathematics and Theoretical Physics, University
       of Cambridge, 2009.
    """
    xopt = np.atleast_1d(xopt)
    if xopt.dtype.kind in np.typecodes['AllInteger']:
        xopt = np.asarray(xopt, dtype=float)
    gq = np.atleast_1d(gq).astype(float)
    xl = np.atleast_1d(xl).astype(float)
    xu = np.atleast_1d(xu).astype(float)

    # Check the sizes of the inputs.
    assert_(xopt.ndim == 1)
    assert_(gq.ndim == 1)
    assert_(xl.ndim == 1)
    assert_(xu.ndim == 1)
    assert_(gq.size == xopt.size)
    assert_(xl.size == xopt.size)
    assert_(xu.size == xopt.size)

    def hessp_safe(x):
        hx = np.atleast_1d(hessp(x, *args))
        if hx.dtype.kind in np.typecodes['AllInteger']:
            hx = np.asarray(hx, dtype=np.float64)
        return hx

    debug = kwargs.get('debug', False)
    improve_tcg = kwargs.get('improve_tcg', True)
    step = _bvtcg(xopt, gq, hessp_safe, xl, xu, delta, debug, improve_tcg)  # noqa
    return np.array(step, dtype=float)


def cpqp(xopt, Aub, bub, Aeq, beq, xl, xu, delta, **kwargs):
    r"""
    Minimize approximately a convex piecewise quadratic function subject to
    bound and trust-region constraints using a truncated conjugate gradient.

    The method minimizes the function

    .. math::

        \frac{1}{2} ( \| [ \mathtt{Aub} \times x - \mathtt{bub} ]_+\|_2^2 +
        \| \mathtt{Aeq} \times x - \mathtt{beq} \|_2^2 ),

    where :math:`[ \cdot ]_+` denotes the componentwise positive part operator.

    Parameters
    ----------
    xopt : numpy.ndarray, shape (n,)
        Center of the trust-region constraint.
    Aub : array_like, shape (mlub, n)
        Matrix `Aub` as shown above.
    bub : array_like, shape (mlub,)
        Vector `bub` as shown above.
    Aeq : array_like, shape (mleq, n)
        Matrix `Aeq` as shown above.
    beq : array_like, shape (meq,)
        Vector `beq` as shown above.
    xl : array_like, shape (n,)
        Lower-bound constraints on the decision variables. Use ``-numpy.inf`` to
        disable the bounds on some variables.
    xu : array_like, shape (n,)
        Upper-bound constraints on the decision variables. Use ``numpy.inf`` to
        disable the bounds on some variables.
    delta : float
        Upper bound on the length of the step from `xopt`.

    Returns
    -------
    step : numpy.ndarray, shape (n,)
        Step from `xopt` towards the estimated point.

    Other Parameters
    ----------------
    debug : bool, optional
        Whether to make debugging tests during the execution, which is
        not recommended in production (the default is False).

    Raises
    ------
    AssertionError
        The vector `xopt` is not feasible (only in debug mode).

    See Also
    --------
    bvtcg : Bounded variable truncated conjugate gradient
    lctcg : Linear constrained truncated conjugate gradient
    nnls : Nonnegative least squares

    Notes
    -----
    The method is adapted from the TRSTEP algorithm [1]_. To cope with the
    convex piecewise quadratic objective function, the method minimizes

    .. math::

        \frac{1}{2} ( \| \mathtt{Aeq} \times x - \mathtt{beq} \|_2^2 +
        \| y \|_2^2 )

    subject to the original constraints, where the slack variable :math:`y` is
    lower bounded by zero and :math:`\mathtt{Aub} \times x - \mathtt{bub}`.

    References
    ----------
    .. [1] M. J. D. Powell. "On fast trust region methods for quadratic models
       with linear constraints." In: Math. Program. Comput. 7 (2015), pp.
       237–-267.
    """
    xopt = np.atleast_1d(xopt)
    if xopt.dtype.kind in np.typecodes['AllInteger']:
        xopt = np.asarray(xopt, dtype=float)
    Aub = np.atleast_2d(Aub).astype(float)
    Aub = np.asfortranarray(Aub)
    bub = np.atleast_1d(bub).astype(float)
    Aeq = np.atleast_2d(Aeq).astype(float)
    Aeq = np.asfortranarray(Aeq)
    beq = np.atleast_1d(beq).astype(float)
    xl = np.atleast_1d(xl).astype(float)
    xu = np.atleast_1d(xu).astype(float)

    # Check the sizes of the inputs.
    assert_(xopt.ndim == 1)
    assert_(Aub.ndim == 2)
    assert_(bub.ndim == 1)
    assert_(Aeq.ndim == 2)
    assert_(beq.ndim == 1)
    assert_(xl.ndim == 1)
    assert_(xu.ndim == 1)
    assert_(Aub.shape[0] == bub.size)
    assert_(Aub.shape[1] == xopt.size)
    assert_(Aeq.shape[0] == beq.size)
    assert_(Aeq.shape[1] == xopt.size)
    assert_(xl.size == xopt.size)
    assert_(xu.size == xopt.size)

    debug = kwargs.get('debug', False)
    mu = kwargs.get('constraint_activation_factor', 0.2)
    step = _cpqp(xopt, Aub, bub, Aeq, beq, xl, xu, delta, mu, debug)  # noqa
    return np.array(step, dtype=float)


def lctcg(xopt, gq, hessp, Aub, bub, Aeq, beq, xl, xu, delta, *args, **kwargs):
    """
    Minimize approximately a quadratic function subject to bound, linear, and
    trust-region constraints using a truncated conjugate gradient.

    Parameters
    ----------
    xopt : numpy.ndarray, shape (n,)
        Point around which the Taylor expansions of the quadratic function is
        defined.
    gq : array_like, shape (n,)
        Gradient of the quadratic function at `xopt`.
    hessp : callable
        Function providing the product of the Hessian matrix of the quadratic
        function with any vector.

            ``hessp(x, *args) -> array_like, shape(n,)``

        where ``x`` is an array with shape (n,) and `args` is a tuple of
        parameters to forward to the objective function. It is assumed that the
        Hessian matrix implicitly defined by `hessp` is symmetric, but not
        necessarily positive semidefinite.
    Aub : array_like, shape (mlub, n), optional
        Jacobian matrix of the linear inequality constraints. Each row of `Aub`
        stores the gradient of a linear inequality constraint.
    bub : array_like, shape (mlub,), optional
        Right-hand side vector of the linear inequality constraints
        ``Aub @ x <= bub``, where ``x`` has the same size than `xopt`.
    Aeq : array_like, shape (mleq, n), optional
        Jacobian matrix of the linear equality constraints. Each row of `Aeq`
        stores the gradient of a linear equality constraint.
    beq : array_like, shape (mleq,), optional
        Right-hand side vector of the linear equality constraints
        `Aeq @ x = beq`, where ``x`` has the same size than `xopt`.
    xl : array_like, shape (n,)
        Lower-bound constraints on the decision variables. Use ``-numpy.inf`` to
        disable the bounds on some variables.
    xu : array_like, shape (n,)
        Upper-bound constraints on the decision variables. Use ``numpy.inf`` to
        disable the bounds on some variables.
    delta : float
        Upper bound on the length of the step from `xopt`.
    *args : tuple, optional
        Parameters to forward to the Hessian product function.

    Returns
    -------
    step : numpy.ndarray, shape (n,)
        Step from `xopt` towards the estimated point.

    Other Parameters
    ----------------
    debug : bool, optional
        Whether to make debugging tests during the execution, which is
        not recommended in production (the default is False).

    Raises
    ------
    AssertionError
        The vector `xopt` is not feasible (only in debug mode).

    See Also
    --------
    bvtcg : Bounded variable truncated conjugate gradient
    cpqp : Convex piecewise quadratic programming
    nnls : Nonnegative least squares

    Notes
    -----
    The method is adapted from the TRSTEP algorithm [1]_. It is an active-set
    variation of the truncated conjugate gradient method, which maintains the QR
    factorization of the matrix whose columns are the gradients of the active
    constraints. The linear equality constraints are then handled by considering
    them as always active.

    References
    ----------
    .. [1] M. J. D. Powell. "On fast trust region methods for quadratic models
       with linear constraints." In: Math. Program. Comput. 7 (2015), pp.
       237--267.
    """
    xopt = np.atleast_1d(xopt)
    if xopt.dtype.kind in np.typecodes['AllInteger']:
        xopt = np.asarray(xopt, dtype=float)
    gq = np.atleast_1d(gq).astype(float)
    Aub = np.atleast_2d(Aub).astype(float)
    Aub = np.asfortranarray(Aub)
    bub = np.atleast_1d(bub).astype(float)
    Aeq = np.atleast_2d(Aeq).astype(float)
    Aeq = np.asfortranarray(Aeq)
    beq = np.atleast_1d(beq).astype(float)
    xl = np.atleast_1d(xl).astype(float)
    xu = np.atleast_1d(xu).astype(float)

    # Check the sizes of the inputs.
    assert_(xopt.ndim == 1)
    assert_(gq.ndim == 1)
    assert_(Aub.ndim == 2)
    assert_(bub.ndim == 1)
    assert_(Aeq.ndim == 2)
    assert_(beq.ndim == 1)
    assert_(xl.ndim == 1)
    assert_(xu.ndim == 1)
    assert_(gq.size == xopt.size)
    assert_(Aub.shape[0] == bub.size)
    assert_(Aub.shape[1] == xopt.size)
    assert_(Aeq.shape[0] == beq.size)
    assert_(Aeq.shape[1] == xopt.size)
    assert_(xl.size == xopt.size)
    assert_(xu.size == xopt.size)

    def hessp_safe(x):
        hx = np.atleast_1d(hessp(x, *args))
        if hx.dtype.kind in np.typecodes['AllInteger']:
            hx = np.asarray(hx, dtype=np.float64)
        return hx

    debug = kwargs.get('debug', False)
    mu = kwargs.get('constraint_activation_factor', 0.2)
    step = _lctcg(xopt, gq, hessp_safe, Aub, bub, Aeq, beq, xl, xu, delta, debug, mu)  # noqa
    return np.array(step, dtype=float)


def nnls(A, b, k=None, maxiter=None):
    """
    Compute the least-squares solution to the equation ``A @ x = b`` subject to
    the nonnegativity constraints ``x[:k] >= 0``.

    Parameters
    ----------
    A : array_like, shape (m, n)
        Matrix `A` as shown above.
    b : array_like, shape (m,)
        Right-hand side vector `b` as shown above.
    k : int, optional
        Number of nonnegativity constraints. The first `k` components of the
        solution vector are nonnegative (the default is ``A.shape[1]``).
    maxiter : int, optional
        Maximum number of inner iterations (the default is ``3 * A.shape[1]``).

    Returns
    -------
    x : numpy.ndarray, shape (n,)
        Solution vector ``x`` as shown above.

    See Also
    --------
    bvtcg : Bounded variable truncated conjugate gradient
    cpqp : Convex piecewise quadratic programming
    lctcg : Linear constrained truncated conjugate gradient

    Notes
    -----
    The method is adapted from the NNLS algorithm [1]_.

    References
    ----------
    .. [1] C. L. Lawson and R. J. Hanson. Solving Least Squares Problems.
       Classics Appl. Math. Philadelphia, PA, US: SIAM, 1974.
    """
    A = np.atleast_2d(A)
    if A.dtype.kind in np.typecodes['AllInteger']:
        A = np.asarray(A, dtype=float)
    A = np.asfortranarray(A)
    b = np.atleast_1d(b)
    if b.dtype.kind in np.typecodes['AllInteger']:
        b = np.asarray(b, dtype=float)
    n = A.shape[1]
    if k is None:
        k = n
    if k < 0 or k > n:
        raise ValueError('Number of nonnegative constraints is invalid')
    if maxiter is None:
        maxiter = 3 * n

    # Check the sizes of the inputs.
    assert_(A.ndim == 2)
    assert_(b.ndim == 1)
    assert_(A.shape[0] == b.size)

    x = _nnls(A, b, k, maxiter)  # noqa
    return np.array(x, dtype=float)
