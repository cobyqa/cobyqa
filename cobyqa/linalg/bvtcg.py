import numpy as np

from ._bvtcg import bvtcg as _bvtcg


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
    bdtol : float, optional
        Tolerance for comparisons on the bound constraints (the default is
        ``10 * eps * n * max(1, max(abs(xl)), max(abs(xu)))``.
    debug : bool, optional
        Whether to make debugging tests during the execution, which is
        not recommended in production (the default is False).

    Raises
    ------
    AssertionError
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
            hx = np.asarray(hx, dtype=float)
        return hx

    debug = kwargs.get('debug', False)
    step = _bvtcg(xopt, gq, hessp_safe, xl, xu, delta, debug)  # noqa
    return np.array(step, dtype=float)
