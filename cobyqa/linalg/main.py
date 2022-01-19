import numpy as np

from ._bvcs import bvcs as _bvcs
from ._bvlag import bvlag as _bvlag
from ._bvtcg import bvtcg as _bvtcg
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

    def hessp_safe(x):
        hx = np.atleast_1d(hessp(x, *args))
        if hx.dtype.kind in np.typecodes['AllInteger']:
            hx = np.asarray(hx, dtype=np.float64)
        return hx

    debug = kwargs.get('debug', False)
    step = _bvtcg(xopt, gq, hessp_safe, xl, xu, delta, debug)  # noqa
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

    x = _nnls(A, b, k, maxiter)  # noqa
    return np.array(x, dtype=float)
