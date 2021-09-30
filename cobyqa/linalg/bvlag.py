import numpy as np
from numpy.testing import assert_


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
    bdtol : float, optional
        Tolerance for comparisons on the bound constraints (the default is
        ``10 * eps * n * max(1, max(abs(xl)), max(abs(xu)))``.

    Raises
    ------
    AssertionError
        The vector ``xpt[kopt, :]`` is not feasible.

    Notes
    -----
    The denominator of the updating formula is given in Equation (3.9) of [1]_,
    and the parameter `alpha` is the referred in Equation (4.12) of [2]_.

    References
    ----------
    .. [1] M. J. D. Powell. The BOBYQA algorithm for bound constrained
       optimization without derivatives. Tech. rep. DAMTP 2009/NA06. Cambridge,
       UK: Department of Applied Mathematics and Theoretical Physics, University
       of Cambridge, 2009.
    .. [2] M. J. D. Powell. "The NEWUOA software for unconstrained optimization
       without derivatives." In: Large-Scale Nonlinear Optimization. Ed. by G.
       Di Pillo and M. Roma. New York, NY, US: Springer, 2006, pp. 255-–297.
    """
    # Format the inputs. Copies of the lower-bound constraints in XL and the
    # upper-bound constraints in XU are made to prevent the changes made in this
    # function to affect the original vector.
    xpt = np.asarray(xpt)
    if xpt.dtype.kind in np.typecodes['AllInteger']:
        xpt = np.asarray(xpt, dtype=float)
    gq = np.asarray(gq)
    if gq.dtype.kind in np.typecodes['AllInteger']:
        gq = np.asarray(gq, dtype=float)
    xl = np.array(xl, dtype=float)
    xu = np.array(xu, dtype=float)
    xopt = xpt[kopt, :]
    xpt = xpt - xopt[np.newaxis, :]

    # Define the tolerances to compare floating-point numbers with zero.
    eps = np.finfo(float).eps
    tiny = np.finfo(float).tiny
    npt, n = xpt.shape
    tol = 1e1 * eps * max(npt, n)
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

    # Start the iterative procedure. The method sets
    # 1. SIGSAV     largest admissible value of the real parameter SIGMA so far;
    # 2. STPSAV     length of the best step so far;
    # 3. IBDSAV     index of the simple bound restraining the computations;
    # 4. KSAV       index of the interpolation point defining the above line.
    sigsav = 0.
    stpsav = 0.
    ibdsav = 0.
    ksav = -1
    for k in range(npt):
        # Search for a point on the line between XOPT and XPT[K,:]. By
        # considering only the trust-region constraint and ignoring the simple
        # bounds, the method sets
        # 1. XLBD       lower bound on the step along the line;
        # 2. XUBD       upper bound on the step along the line.
        if k == kopt:
            continue
        xgq = np.inner(xpt[k, :], gq)
        distsq = np.inner(xpt[k, :], xpt[k, :])
        dist = np.sqrt(distsq)
        if dist > tiny * delta:
            xubd = delta / dist
        else:
            xubd = 0.
        xlbd = -xubd
        ilbd = 0
        iubd = 0
        xumin = min(1., xubd)

        # Update the lower and upper bound un XLBD and XUBD to take into account
        # the simple bounds in XL and XU along the current line.
        ipos = np.greater(xpt[k, :], tiny * np.maximum(np.abs(xl), np.abs(xu)))
        pxl = np.full_like(xopt, -np.inf)
        pxl[ipos] = np.divide(xl[ipos], xpt[k, ipos])
        pxl[pxl <= xlbd] = -np.inf
        if np.any(np.isfinite(pxl)):
            ipxl = np.argmax(pxl)
            xlbd = pxl[ipxl]
            ilbd = -ipxl - 1
        pxl = np.full_like(xopt, np.inf)
        pxl[ipos] = np.divide(xu[ipos], xpt[k, ipos])
        pxl[pxl >= xubd] = np.inf
        if np.any(np.isfinite(pxl)):
            ipxl = np.argmin(pxl)
            xubd = max(xumin, pxl[ipxl])
            iubd = ipxl + 1
        ineg = np.less(xpt[k, :], -tiny * np.maximum(np.abs(xl), np.abs(xu)))
        nxl = np.full_like(xopt, -np.inf)
        nxl[ineg] = np.divide(xu[ineg], xpt[k, ineg])
        nxl[nxl <= xlbd] = -np.inf
        if np.any(np.isfinite(nxl)):
            inxl = np.argmax(nxl)
            xlbd = nxl[inxl]
            ilbd = inxl + 1
        nxl = np.full_like(xopt, np.inf)
        nxl[ineg] = np.divide(xl[ineg], xpt[k, ineg])
        nxl[nxl >= xubd] = np.inf
        if np.any(np.isfinite(nxl)):
            inxl = np.argmin(nxl)
            xubd = max(xumin, nxl[inxl])
            iubd = -inxl - 1

        # Compute the best point along the line joining XOPT and XPT[K,:] that
        # respects the trust-region constraint and the simple bounds.
        if k == klag:
            diff = xgq - 1.
            stplen = xlbd
            vlag = xlbd * (xgq - diff * xlbd)
            isbd = ilbd
            temp = xubd * (xgq - diff * xubd)
            if abs(temp) > abs(vlag):
                stplen = xubd
                vlag = temp
                isbd = iubd
            txl = .5 * xgq - diff * xlbd
            txu = .5 * xgq - diff * xubd
            if txl * txu < 0. and abs(diff) > .5 * tiny * abs(xgq):
                temp = .25 * xgq ** 2. / diff
                if abs(temp) > abs(vlag):
                    stplen = .5 * xgq / diff
                    vlag = temp
                    isbd = 0
        else:
            stplen = xlbd
            vlag = xlbd * (1. - xlbd)
            isbd = ilbd
            temp = xubd * (1. - xubd)
            if abs(temp) > abs(vlag):
                stplen = xubd
                vlag = temp
                isbd = iubd
            if xubd > .5:
                if abs(vlag) < .25:
                    stplen = .5
                    vlag = .25
                    isbd = 0
            vlag *= xgq

        # Calculate the parameter given in Equation (3.9) of Powell (2009) for
        # the current line-search and maintain the optimal values so far.
        temp = stplen * (1. - stplen) * distsq
        sigsq = vlag ** 2. * (vlag ** 2. + .5 * alpha * temp ** 2.)
        if sigsq > sigsav:
            sigsav = sigsq
            stpsav = stplen
            ibdsav = isbd
            ksav = k

    # Construct the vector XNEW as described above.
    step = np.maximum(xl, np.minimum(xu, stpsav * xpt[ksav, :]))
    if ibdsav < 0:
        step[-ibdsav - 1] = xl[-ibdsav - 1]
    elif ibdsav > 0:
        step[ibdsav - 1] = xu[ibdsav - 1]

    return step
