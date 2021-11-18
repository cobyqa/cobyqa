import numpy as np
from numpy.testing import assert_

from .utils import get_bdtol


def bvtcg(xopt, gq, hessp, args, xl, xu, delta, **kwargs):
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
    args : tuple
        Parameters to forward to the Hessian product function.
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

    # Define the tolerances to compare floating-point numbers with zero.
    tiny = np.finfo(float).tiny
    n = gq.size
    tol = 10.0 * np.finfo(float).eps * n
    bdtol = get_bdtol(xl, xu, **kwargs)

    # Shift the bounds to carry out all calculations at the origin.
    xl -= xopt
    xu -= xopt

    # Ensure the feasibility of the initial guess.
    if kwargs.get('debug', False):
        assert_(np.max(xl) < bdtol)
        assert_(np.min(xu) > -bdtol)
        assert_(np.isfinite(delta))
        assert_(delta > 0.0)

    # Initialize the working sets and the trial step. The vector xbdi stores the
    # working sets related to the bounds, where the value 0 indicates that the
    # component is not restricted by the bounds, the value -1 indicates that the
    # component is fixed by the lower bound, and the value 1 indicates that the
    # component is fixed by the upper bound.
    step = np.zeros_like(gq)
    xbdi = np.zeros(n, dtype=int)
    xbdi[(step <= xl) & (gq >= 0.0)] = -1
    xbdi[(step >= xu) & (gq <= 0.0)] = 1
    ifree = np.equal(xbdi, 0)
    ifixed = np.not_equal(xbdi, 0)
    nact = np.count_nonzero(np.abs(xbdi))

    # Start the iterative procedure. It uses the least curvature of the Hessian
    # of the objective function that occurs in the conjugate gradient searches
    # that are not restricted by any constraints. It is set to 0 if the current
    # trial step reaches the boundary of the trust region, and to -1 if all the
    # search directions are constrained.
    sd = np.zeros_like(step)
    delsq = delta ** 2.0
    tolsd = tol
    beta = 0.0
    gqsq = 0.0
    gqsqsav = 0.0
    qred = 0.0
    crvmin = -1.0
    iterc = 0
    maxiter = n
    while abs(crvmin) > tolsd:
        # Set the next search direction of the conjugate gradient method to the
        # steepest descent direction initially and when the iterations are
        # restarted because a variable has just been fixed by a bound. The
        # maximum number of iterations is set to the theoretical upper bound on
        # the iteration number initially and at a restart. The computations are
        # stopped if no further progress is possible.
        sd[ifree] = beta * sd[ifree] - gq[ifree]
        sd[ifixed] = 0.0
        sdsq = np.inner(sd, sd)
        tolsd = tol * np.max(np.abs(sd), initial=1.)
        if np.sqrt(sdsq) < tolsd:
            break
        if beta < tolsd:
            gqsq = sdsq
            maxiter = iterc + n - nact

        # Set the steplength of the current search direction allowed by the
        # trust-region constraint. The calculations are stopped if no further
        # progress is possible in the current search direction.
        rhs = delsq - np.inner(step[ifree], step[ifree])
        if rhs <= 0.0:
            crvmin = 0.0
            continue
        sdstep = np.inner(sd[ifree], step[ifree])
        sqrd = np.sqrt(sdsq * rhs + sdstep ** 2.0)
        if sdstep < 0.0 and sdsq > tiny * abs(sqrd - sdstep):
            alpht = (sqrd - sdstep) / sdsq
        elif abs(sqrd + sdstep) > tiny * rhs:
            alpht = rhs / (sqrd + sdstep)
        else:
            break
        alpha = alpht

        # Reduce the steplength if necessary to the value that minimizes the
        # quadratic function. The method do not require the objective function
        # to be positive semidefinite, so that the curvature of the model at the
        # current search direction may be negative, in which case the model is
        # not lower bounded.
        hsd = np.atleast_1d(hessp(sd, *args))
        if hsd.dtype.kind in np.typecodes['AllInteger']:
            hsd = np.asarray(hsd, dtype=float)
        curv = np.inner(sd[ifree], hsd[ifree])
        if curv > tiny * abs(gqsq):
            alphm = gqsq / curv
        else:
            alphm = np.inf
        alpha = min(alpha, alphm)

        # Reduce the steplength if necessary in order to preserve the simple
        # bounds, setting the index of the new constrained variable if any.
        ipos = sd > tiny * np.abs(xu - step)
        ineg = sd < -tiny * np.abs(step - xl)
        distbd = np.full_like(step, np.inf)
        distbd[ipos] = (xu[ipos] - step[ipos]) / sd[ipos]
        distbd[ineg] = (xl[ineg] - step[ineg]) / sd[ineg]
        inew = -1
        if np.any(np.isfinite(distbd)):
            inew = np.argmin(distbd)
            alphf = distbd[inew]
            if alphf >= alpha:
                inew = -1
            else:
                alpha = alphf

        # Make the actual conjugate gradient iteration. The max operator below
        # is crucial as it prevents numerical difficulties engendered by
        # computer rounding errors.
        if alpha > 0.0:
            iterc += 1
            crv = curv / sdsq
            if inew == -1 and crv > 0.0:
                crvmin = min(crvmin, crv) if abs(crvmin + 1.0) > tolsd else crv
            gqsqsav = gqsq
            gq += alpha * hsd
            gqsq = np.inner(gq[ifree], gq[ifree])
            step += alpha * sd
            sdred = max(0.0, alpha * (gqsqsav - 0.5 * alpha * curv))
            qred += sdred

        # Restart the conjugate gradient method if it has hit a new bound. If
        # the step is on or outside the trust region, further improvement
        # attempts round the trust-region boundary are done below.
        if inew >= 0:
            nact += 1
            if sd[inew] < 0.0:
                xbdi[inew] = -1
            else:
                xbdi[inew] = 1
            ifree[inew] = False
            ifixed[inew] = True
            delsq -= step[inew] * step[inew]
            if delsq <= 0.0:
                crvmin = 0.0
                continue
            beta = 0.0
            continue

        # If the step did not reach the trust-region boundary, apply another
        # conjugate gradient iteration or return if the maximum number of
        # iterations is exceeded.
        if alpha < alpht:
            if iterc >= maxiter:
                break
            beta = gqsq / gqsqsav
            continue

        # The trust-region boundary has been reached by the step. End the
        # truncated conjugate gradient procedure and attempt to improve the
        # current step round the trust-region boundary.
        crvmin = 0.0
        continue
    else:
        # Whenever the truncated conjugate gradient computations stopped because
        # the current trial step hit the trust-region boundary, a search is
        # performed to attempt improving the solution round the trust-region
        # boundary on the two dimensional space spanned by the free components
        # of the step and the gradient.
        stepsq = np.inner(step[ifree], step[ifree])
        gqsq = np.inner(gq[ifree], gq[ifree])
        gdstep = np.inner(gq[ifree], step[ifree])
        sd[ifree] = step[ifree]
        sd[ifixed] = 0.0
        hsd = np.atleast_1d(hessp(sd, *args))
        if hsd.dtype.kind in np.typecodes['AllInteger']:
            hsd = np.asarray(hsd, dtype=float)
        hred = np.copy(hsd)
        inew = -1
        while nact < n - 1:
            # Whenever the previous iteration has hit a bound, the computation
            # round the trust-region boundary are restarted.
            if inew >= 0:
                sd[ifree] = step[ifree]
                sd[ifixed] = 0.0
                hsd = np.atleast_1d(hessp(sd, *args))
                if hsd.dtype.kind in np.typecodes['AllInteger']:
                    hsd = np.asarray(hsd, dtype=float)
                hred = np.copy(hsd)
            tolsd = tol * np.max(np.abs(sd), initial=1.0)

            # Let the search direction be a linear combination of the reduced
            # step and the reduced gradient that is orthogonal to the step.
            iterc += 1
            disc = gqsq * stepsq - gdstep ** 2.0
            sqrd = np.sqrt(max(disc, 0.0))
            if sqrd < tolsd:
                break
            sd[ifree] = (gdstep * step[ifree] - stepsq * gq[ifree]) / sqrd
            sd[ifixed] = 0.0
            gdsd = -sqrd

            # By considering the simple bounds on the variables, calculate an
            # upper bound on the tangent of half the angle of the alternative
            # iteration and restart the alternative iterations if a free
            # variable has reached a new bound.
            angbd = 1.0
            inew = -1
            sl = np.full_like(step, np.inf)
            su = np.full_like(step, np.inf)
            sl[ifree] = step[ifree] - xl[ifree]
            su[ifree] = xu[ifree] - step[ifree]
            if np.any(sl <= 0.0):
                nact += 1
                inew = np.argmax(sl <= 0.0)
                xbdi[inew] = -1
                ifree[inew] = False
                ifixed[inew] = True
                stepsq -= step[inew] ** 2.0
                gqsq -= gq[inew] ** 2.0
                gdstep -= gq[inew] * step[inew]
                continue
            elif np.any(su <= 0.0):
                nact += 1
                inew = np.argmax(su <= 0.0)
                xbdi[inew] = 1
                ifree[inew] = False
                ifixed[inew] = True
                stepsq -= step[inew] ** 2.0
                gqsq -= gq[inew] ** 2.0
                gdstep -= gq[inew] * step[inew]
                continue
            ssq = np.square(step[ifree]) + np.square(sd[ifree])
            temp = np.full_like(step, -np.inf)
            temp[ifree] = ssq - np.square(xl[ifree])
            itemp = temp > 0.0
            temp[itemp] = np.sqrt(temp[itemp]) - sd[itemp]
            temp[np.logical_not(itemp)] = -np.inf
            isl = angbd * temp - sl > tiny * temp
            xbdisav = 0
            if np.any(isl):
                ratio = np.full_like(step, np.inf)
                ratio[isl] = sl[isl] / temp[isl]
                inew = np.argmin(ratio)
                angbd = ratio[inew]
                xbdisav = -1
            temp[ifree] = ssq - np.square(xu[ifree])
            itemp = temp > 0.0
            temp[itemp] = np.sqrt(temp[itemp]) + sd[itemp]
            temp[np.logical_not(itemp)] = -np.inf
            isu = angbd * temp - su > tiny * temp
            if np.any(isu):
                ratio = np.full_like(step, np.inf)
                ratio[isu] = su[isu] / temp[isu]
                inew = np.argmin(ratio)
                angbd = ratio[inew]
                xbdisav = 1

            # Calculate the necessary curvatures for the alternative iteration.
            hsd = np.atleast_1d(hessp(sd, *args))
            if hsd.dtype.kind in np.typecodes['AllInteger']:
                hsd = np.asarray(hsd, dtype=float)
            sdhsd = np.inner(sd[ifree], hsd[ifree])
            stephsd = np.inner(step[ifree], hsd[ifree])
            stephred = np.inner(step[ifree], hred[ifree])

            # Seek the greatest reduction in the objective function for a range
            # of equally spaced values in [0, angbd], corresponding to the
            # tangent of half the angle of the alternative iteration. For the
            # computations to hold, the interval should be split into at least
            # three parts, and nalt represents the number of intervals in the
            # unconstrained case.
            nalt = 20
            iu = int((nalt - 3) * angbd + 3.1)
            angt = angbd * np.arange(1, iu + 1, dtype=float) / float(iu)
            sth = 2.0 * angt / (1.0 + np.square(angt))
            temp = sdhsd + angt * (stephred * angt - 2.0 * stephsd)
            rednew = sth * (gdstep * angt - gdsd)
            rednew -= sth * (0.5 * sth * temp)
            isav = np.argmax(rednew)
            redmax = rednew[isav]
            rednew[isav] = -np.inf
            redprev = rednew[isav - 1] if isav > 0 else 0.0
            rednext = rednew[isav + 1] if isav < iu - 1 else 0.0
            if redmax <= 0.0:
                isav = -1
                redmax = 0.0

            # Set the sine and cosine of the angle of the alternative iteration,
            # and return if no reduction is possible. The computations are
            # stopped if either no further reduction can be obtained on the
            # sampling of the trust-region boundary or if no further progress on
            # the step is possible.
            if isav == -1:
                break
            angc = angbd
            if isav < iu - 1:
                temp = (rednext - redprev) / (2.0 * redmax - redprev - rednext)
                angc *= (isav + 1.0 + 0.5 * temp) / float(iu)
            cth = (1.0 - angc * angc) / (1.0 + angc ** 2.0)
            sth = 2.0 * angc / (1.0 + angc ** 2.0)
            temp = sdhsd + angc * (angc * stephred - 2.0 * stephsd)
            sdred = sth * (angc * gdstep - gdsd - 0.5 * sth * temp)
            if sdred <= 0.0:
                break

            # Update the step with the current search direction. If the angle of
            # the alternative iteration is restricted by a bound on a free
            # variable, that variable is fixed at the bound and the computations
            # of the alternative iterations are restarted.
            step[ifree] = cth * step[ifree] + sth * sd[ifree]
            gq += (cth - 1.0) * hred + sth * hsd
            stepsq = np.inner(step[ifree], step[ifree])
            gdstep = np.inner(gq[ifree], step[ifree])
            gqsq = np.inner(gq[ifree], gq[ifree])
            hred = cth * hred + sth * hsd
            qred += sdred
            if inew >= 0 and isav == iu - 1:
                nact += 1
                xbdi[inew] = xbdisav
                ifree[inew] = False
                ifixed[inew] = True
                stepsq -= step[inew] ** 2.0
                gqsq -= gq[inew] ** 2.0
                gdstep -= gq[inew] * step[inew]
                continue
            if sdred > 0.0:
                inew = -1
                continue

            # End the alternative iteration computations since no new bound has
            # been hit by the new step and the reduction of the current step is
            # low compared to the reduction in the objective function so far.
            break

    # Ensure that the bound constraints are respected and that the components
    # fixed by the working sets are set to their respective bounds.
    step = np.minimum(xu, np.maximum(xl, step))
    ixl = np.equal(xbdi, -1)
    ixu = np.equal(xbdi, 1)
    step[ixl] = xl[ixl]
    step[ixu] = xu[ixu]

    return step
