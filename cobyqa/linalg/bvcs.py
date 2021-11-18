import numpy as np
from numpy.testing import assert_

from .utils import get_bdtol


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
    bdtol : float, optional
        Tolerance for comparisons on the bound constraints (the default is
        ``10 * eps * n * max(1, max(abs(xl)), max(abs(xu)))``.
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
    gq = np.atleast_1d(gq).astype(float)
    xl = np.atleast_1d(xl).astype(float)
    xu = np.atleast_1d(xu).astype(float)

    # Define the tolerances to compare floating-point numbers with zero.
    eps = np.finfo(float).eps
    tiny = np.finfo(float).tiny
    npt, n = xpt.shape
    tol = 10.0 * eps * npt
    bdtol = get_bdtol(xl, xu, **kwargs)

    # Shift the bounds to carry out all calculations at the origin.
    xl -= xpt[kopt, :]
    xu -= xpt[kopt, :]

    # Ensure the feasibility of the initial guess.
    if kwargs.get('debug', False):
        assert_(np.max(xl) < bdtol)
        assert_(np.min(xu) > -bdtol)
        assert_(np.isfinite(delta))
        assert_(delta > 0.0)

    # Start the procedure.
    step = np.zeros_like(gq)
    stpsav = np.empty_like(step)
    bigstp = 2.0 * delta
    csav = 0.0
    cauchy = 0.0
    for isign in range(2):
        # Initialize the computations of the Cauchy step. The free components of
        # the Cauchy step are set to bigstp and the computations stop
        # immediately if every free component of the gradient is zero.
        ifree = np.minimum(-xl, gq) > 0.0
        ifree = ifree | (np.maximum(-xu, gq) < 0.0)
        cc = np.zeros_like(step)
        cc[ifree] = bigstp
        gqsq = np.inner(gq[ifree], gq[ifree])
        if np.sqrt(gqsq) < tol * max(1.0, bigstp):
            break

        # Fix the remaining components of the Cauchy step to the lower and
        # upper bounds as the trust-region constraint allows.
        ccsq = 0.0
        ccsqsav = -1.0
        stplen = 0.0
        delsq = delta ** 2.0
        while ccsq > ccsqsav and np.sqrt(gqsq) >= tol * bigstp:
            ccsqsav = ccsq
            stplen = np.sqrt(delsq / gqsq)
            ifree = np.abs(cc - bigstp) < tol * bigstp
            temp = np.full_like(cc, np.inf)
            temp[ifree] = -stplen * gq[ifree]
            ixl = temp <= xl
            cc[ixl] = xl[ixl]
            ccsq += np.inner(cc[ixl], cc[ixl])
            temp[np.logical_not(ifree)] = -np.inf
            ixu = temp >= xu
            cc[ixu] = xu[ixu]
            ccsq += np.inner(cc[ixu], cc[ixu])
            ifree[ixl | ixu] = False
            gqsq = np.inner(gq[ifree], gq[ifree])
            delsq = delta ** 2.0 - ccsq

        # Set the free components of the Cauchy step and all components of the
        # trial step. The Cauchy step may be scaled hereinafter.
        ifree = np.abs(cc - bigstp) < tol * bigstp
        cc[ifree] = -stplen * gq[ifree]
        step[ifree] = np.maximum(xl[ifree], np.minimum(xu[ifree], cc[ifree]))
        iopt = np.logical_not(ifree) & (np.abs(cc) < tol * bigstp)
        step[iopt] = 0.0
        ifixed = np.logical_not(ifree | iopt)
        ixl = ifixed & (gq > 0.0)
        step[ixl] = xl[ixl]
        ixu = ifixed & np.logical_not(ixl)
        step[ixu] = xu[ixu]
        gqcc = np.inner(gq, cc)

        # Set the curvature of the knew-th Lagrange function along the Cauchy
        # step. Scale the Cauchy step by a factor less than one if that can
        # reduce the modulus of the Lagrange function.
        crv = curv(cc, *args)
        ubf = 1.0 + np.sqrt(2.0)
        if isign == 1:
            crv *= -1.0
        if -gqcc < crv < -ubf * gqcc and abs(crv) > tiny * abs(gqcc):
            scale = -gqcc / crv
            step = np.maximum(xl, np.minimum(xu, scale * cc))
            cauchy = (0.5 * gqcc * scale) ** 2.0
        else:
            cauchy = (gqcc + 0.5 * crv) ** 2.0

        # If isign is zero, then the step is calculated as before after
        # reversing the sign of the gradient. Thus two step vectors become
        # available. The chosen one gives the largest value of cauchy.
        if isign == 0:
            gq *= -1.0
            np.copyto(stpsav, step)
            csav = cauchy
            continue
        if csav > cauchy:
            np.copyto(step, stpsav)
            cauchy = csav
        break

    return step, cauchy
