import numpy as np
from numpy.testing import assert_


def bvcs(xpt, kopt, gq, curv, args, xl, xu, delta, **kwargs):
    r"""
    Evaluate a constrained Cauchy step of

    .. math::

        \min_x | \Lambda_k ( x ) |,

    subject to the bound constraints :math:`\mathtt{xl} \le x \le \mathtt{xu}`
    and the trust region :math:`\| x - \mathtt{xopt} \|_2 \le \mathtt{delta}`,
    where :math:`\Lambda_k` refers to the `k`-th Lagrange polynomial of the
    interpolation points for a certain `k`, and where ``xopt`` is
    ``xpt[kopt, :]``. The vector ``xopt`` must be feasible.

    Parameters
    ----------
    xpt : array_like, shape (m,n)
        Point coordinates as shown above. Each row represents a point.
    kopt : int
        Index ``kopt`` as shown above.
    gq : array_like, shape (n,)
        Gradient of a Lagrange polynomial of the interpolation points (not
        necessarily the ``kopt``-th one) at ``xpt[kopt, :]``.
    curv : callable
        Function providing the curvature of the Lagrange polynomial.

            ``curv(x, *args) -> float``

        where ``x`` is an array with shape (n,) and ``args`` is the tuple of
        fixed parameters needed to specify the function.
    args : tuple
        Extra arguments to pass to the curvature function.
    xl : array_like, shape (n,)
        Lower-bound constraints ``xl`` as shown above.
    xu : array_like, shape (n,)
        Upper-bound constraints ``xu`` as shown above.
    delta : float
        Trust-region radius.

    Returns
    -------
    step : numpy.ndarray, shape (n,)
        Cauchy step, namely :math:`x - \mathtt{xopt}` as shown above.
    cauchy : float
        Square of the Lagrange polynomial evaluation at the Cauchy point.

    Other Parameters
    ----------------
    bdtol : float, optional
        Tolerance for comparisons.
        Default is ``10 * eps * n * max(1, max(abs(xl)), max(abs(xu)))``.

    Raises
    ------
    AssertionError
        The vector ``xopt`` is not feasible.

    Notes
    -----
    The method is adapted from the ALTMOV algorithm [1]_. The ``klag``-th
    Lagrange polynomial :math:`\Lambda_{ \mathtt{klag} }`, defined by

    .. math::

        \Lambda_{ \mathtt{klag} } ( \mathtt{xpt}[k, :] ) =
        \delta_{k, \mathtt{klag}}

    The freedom bequeathed by the interpolation conditions is taken up by
    minimizing :math:`\| \nabla^2 \Lambda_{ \mathtt{klag} } \|_{\mathsf{F}}`.

    References
    ----------
    .. [1] M. J. D. Powell. The BOBYQA algorithm for bound constrained
       optimization without derivatives. Tech. rep. DAMTP 2009/NA06. Cambridge,
       UK: Department of Applied Mathematics and Theoretical Physics, University
       of Cambridge, 2009.
    """
    # Format the inputs. Copies of the gradient in GQ, the lower-bound
    # constraints in XL, and the upper-bound constraints in XU are made to
    # prevent the changes made in this function to affect the original vector.
    xpt = np.asarray(xpt)
    if xpt.dtype.kind in np.typecodes['AllInteger']:
        xpt = np.asarray(xpt, dtype=float)
    gq = np.array(gq, dtype=float)
    xl = np.array(xl, dtype=float)
    xu = np.array(xu, dtype=float)
    xopt = xpt[kopt, :]

    # Define the tolerances to compare floating-point numbers with zero.
    eps = np.finfo(float).eps
    tiny = np.finfo(float).tiny
    npt, n = xpt.shape
    tol = 1e1 * eps * npt
    tolbd = tol * np.max(np.abs(np.r_[xl, xu]), initial=1.)
    tolbd = kwargs.get('bdtol', tolbd)

    # Shift the bounds to carry out all calculations at the origin.
    xl -= xopt
    xu -= xopt

    # Ensure the feasibility of the initial guess.
    assert_(np.max(xl) < tolbd)
    assert_(np.min(xu) > -tolbd)
    assert_(np.isfinite(delta))
    assert_(delta > 0.)

    # Initialize the procedure. The method sets
    # 1. STEP       trial step from XOPT;
    # 2. STPSAV     trial step from XOPT at the first iteration;
    # 3. BIGSTP     diameter of the trust region;
    # 4. CSAV       square root of the KNEW-th Lagrange polynomial at XOPT+STEP
    #               for the first iteration, or zero.
    step = np.zeros_like(xopt)
    stpsav = np.empty_like(step)
    bigstp = 2. * delta
    csav = 0.
    cauchy = 0.
    for isign in range(2):
        # Initialize the computations of the Cauchy step. The free components of
        # the Cauchy step are set to BIGSTP and the computations stop
        # immediately if every free component of the gradient is zero.
        ifree = np.minimum(-xl, gq) > 0.
        ifree = np.logical_or(ifree, np.maximum(-xu, gq) < 0.)
        cc = np.zeros_like(step)
        cc[ifree] = bigstp
        gqsq = np.inner(gq[ifree], gq[ifree])
        if gqsq < tol * bigstp:
            break

        # Fix the remaining components of the Cauchy step in CC to the lower and
        # upper bounds in XL and XU as the trust-region constraint allows.
        ccsq = 0.
        ccsqsav = -1.
        stplen = 0.
        delsq = delta ** 2.
        while ccsq > ccsqsav and gqsq >= tol * bigstp and delsq >= tol * delta:
            ccsqsav = ccsq
            stplen = np.sqrt(delsq / gqsq)
            ifree = np.less(np.abs(cc - bigstp), tol * bigstp)
            temp = np.full_like(cc, np.inf)
            temp[ifree] = -stplen * gq[ifree]
            ixl = np.less_equal(temp, xl)
            cc[ixl] = xl[ixl]
            ccsq += np.inner(cc[ixl], cc[ixl])
            temp[np.logical_not(ifree)] = -np.inf
            ixu = np.greater_equal(temp, xu)
            cc[ixu] = xu[ixu]
            ccsq += np.inner(cc[ixu], cc[ixu])
            ifree[np.logical_or(ixl, ixu)] = False
            gqsq = np.inner(gq[ifree], gq[ifree])
            delsq = delta ** 2. - ccsq

        # Set the free components of the Cauchy step in CC and all components of
        # the trial step in STEP. The Cauchy step may be scaled later.
        ifree = np.less(cc - bigstp, tol * bigstp)
        cc[ifree] = -stplen * gq[ifree]
        step[ifree] = np.maximum(xl[ifree], np.minimum(xu[ifree], cc[ifree]))
        iopt = np.logical_and(np.logical_not(ifree), np.abs(cc) < tol * bigstp)
        step[iopt] = 0.
        ifixed = np.logical_not(np.logical_or(ifree, iopt))
        ixl = np.logical_and(ifixed, gq > 0.)
        step[ixl] = xl[ixl]
        ixu = np.logical_and(ifixed, np.logical_not(ixl))
        step[ixu] = xu[ixu]
        gqcc = np.inner(gq, cc)

        # Set CRV to the curvature of the KNEW-th Lagrange function along the
        # Cauchy step. Scale CC by a factor less than one if that can reduce the
        # modulus of the Lagrange function at XOPT+CC. Set CAUCHY to the final
        # value of the square of this function.
        crv = curv(cc, *args)
        ubf = 1. + np.sqrt(2.)
        if isign == 1:
            crv *= -1.
        if -gqcc < crv < -ubf * gqcc and abs(crv) > tiny * abs(gqcc):
            scale = -gqcc / crv
            step = np.maximum(xl, np.minimum(xu, scale * cc))
            cauchy = (.5 * gqcc * scale) ** 2.
        else:
            cauchy = (gqcc + .5 * crv) ** 2.

        # If ISIGN is zero, then STEP is calculated as before after reversing
        # the sign of GLAG. Thus two STEP vectors become available. The one that
        # is chosen is the one that gives the larger value of CAUCHY.
        if isign == 0:
            gq *= -1.
            np.copyto(stpsav, step)
            csav = cauchy
            continue
        if csav > cauchy:
            np.copyto(step, stpsav)
            cauchy = csav
        break

    return step, cauchy
