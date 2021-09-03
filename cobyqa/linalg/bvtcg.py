import numpy as np
from numpy.testing import assert_


def bvtcg(xopt, gq, hessp, args, xl, xu, delta, **kwargs):
    r"""
    Minimize approximately the quadratic function

    .. math::

        \mathtt{gq}^{\mathsf{T}} ( x - \mathtt{xopt} ) + \frac{1}{2}
        ( x - \mathtt{xopt} )^{\mathsf{T}} \mathtt{Hq} ( x - \mathtt{xopt} ),

    subject to the bound constraints :math:`\mathtt{xl} \le x \le \mathtt{xu}`
    and the trust region :math:`\| x - \mathtt{xopt} \|_2 \le \mathtt{delta}`.
    It is essential to ensure the symmetricity of the matrix ``Hq``, and the
    vector ``xopt`` must be feasible.

    Parameters
    ----------
    xopt : array_like, shape (n,)
        Array ``xopt`` as shown above.
    gq : array_like, shape (n,)
        Array ``gq`` as shown above.
    hessp : callable
        Function providing the product :math`\mathtt{Hq} x` as shown above.

            ``hessp(x, *args) -> array_like, shape(n,)``

        where ``x`` is an array with shape (n,) and ``args`` is the tuple of
        fixed parameters needed to specify the function. It is assumed that the
        implicit Hessian matrix ``Hq`` in the function ``hessp`` is symmetric,
        but not necessarily positive semidefinite.
    args : tuple
        Extra arguments to pass to the Hessian function.
    xl : array_like, shape (n,)
        Lower-bound constraints ``xl`` as shown above.
    xu : array_like, shape (n,)
        Upper-bound constraints ``xu`` as shown above.
    delta : float
        Trust-region radius.

    Returns
    -------
    step : numpy.ndarray, shape (n,)
        Step from ``xopt`` towards the solution, namely
        :math:`x - \mathtt{xopt}` as shown above.

    Other Parameters
    ----------------
    bdtol : float, optional
        Tolerance for comparisons.
        Default is ``10 * eps * n * max(1, max(abs(xl)), max(abs(xu)))``.

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
    # Format the inputs. Copies of the lower-bound constraints in XL and the
    # upper-bound constraints in XU are made to prevent the changes made in this
    # function to affect the original vector.
    xopt = np.asarray(xopt)
    if xopt.dtype.kind in np.typecodes['AllInteger']:
        xopt = np.asarray(xopt, dtype=float)
    gq = np.array(gq, dtype=float)
    xl = np.array(xl, dtype=float)
    xu = np.array(xu, dtype=float)

    # Define the tolerances to compare floating-point numbers with zero.
    eps = np.finfo(float).eps
    tiny = np.finfo(float).tiny
    n = gq.size
    tol = 1e1 * eps * n
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

    # Initialize the working sets and the trial step. The method sets
    # 1. STEP       trial step;
    # 2. XBDI       working sets related to the bounds, where the value 0
    #               indicates that the component is not restricted by the
    #               bounds, the value -1 indicates that the component is fixed
    #               by the lower bound, and the value 1 indicates that the
    #               component is fixed by the upper bound;
    # 3. IFREE      inactivity flag of the variables;
    # 4. IFIXED     activity flag of the variables;
    # 5. NACT       number of active constraints.
    step = np.zeros_like(gq)
    xbdi = np.zeros(step.size, dtype=int)
    xbdi[(step <= xl) & (gq >= 0.)] = -1
    xbdi[(step >= xu) & (gq <= 0.)] = 1
    ifree = np.equal(xbdi, 0)
    ifixed = np.not_equal(xbdi, 0)
    nact = np.count_nonzero(np.abs(xbdi))

    # Start the iterative procedure. It uses the least curvature of the Hessian
    # of the objective function that occurs in the conjugate gradient searches
    # that are not restricted by any constraints. It is set to 0 if the current
    # trial step reaches the boundary of the trust region, and to -1 if all the
    # search directions are constrained.
    sd = np.empty_like(step)
    delsq = delta ** 2.
    tolsd = tol
    beta = 0.
    gqsq = 0.
    gqsqsav = 0.
    qred = 0.
    crvmin = -1.
    iterc = 0
    maxiter = n
    while abs(crvmin) > tolsd:
        # Set the next search direction of the conjugate gradient method in SD
        # to the steepest descent direction initially and when the iterations
        # are restarted because a variable has just been fixed by a bound. The
        # maximum number of iterations in MAXITER is set to the theoretical
        # upper bound on the iteration number initially and at a restart. The
        # computations are stopped if no further progress is possible.
        sd[ifree] = beta * sd[ifree] - gq[ifree]
        sd[ifixed] = 0.
        sdsq = np.inner(sd, sd)
        tolsd = tol * np.max(np.abs(sd), initial=1.)
        if sdsq < tolsd:
            break
        if beta < tolsd:
            gqsq = sdsq
            maxiter = iterc + n - nact

        # Determine a bound on the steplength, ignoring the simple bounds, by
        # setting the length of the step to the trust-region boundary in ALPHAD
        # and the length of the step such that the objective function decreases
        # monotonically in ALPHAQ. The temporary steplength is set to the least
        # of them, and the length to the bounds will be taken into account
        # hereunder. If the step is on or outside the trust region, further
        # improvement attempts round the trust-region boundary are done.
        hsd = np.asarray(hessp(sd, *args))
        if hsd.dtype.kind in np.typecodes['AllInteger']:
            hsd = np.asarray(hsd, dtype=float)
        resid = delsq - np.inner(step[ifree], step[ifree])
        sdstep = np.inner(sd[ifree], step[ifree])
        sdhsd = np.inner(sd[ifree], hsd[ifree])
        if resid <= 0.:
            crvmin = 0.
            continue
        sqrd = np.sqrt(sdsq * resid + sdstep ** 2.)
        if sdstep < 0.:
            alphad = (sqrd - sdstep) / sdsq
        else:
            alphad = resid / (sqrd + sdstep)
        alphaq = gqsq / sdhsd if sdhsd > tiny * abs(gqsq) else np.inf
        alpha = min(alphad, alphaq)

        # Reduce ALPHA if necessary in order to preserve the simple bounds,
        # letting INEW be the index of the new constrained variable.
        ipos = np.greater(sd, tiny * np.abs(xu - step))
        ineg = np.less(sd, -tiny * np.abs(step - xl))
        distbd = np.full_like(step, np.inf)
        distbd[ipos] = np.divide(xu[ipos] - step[ipos], sd[ipos])
        distbd[ineg] = np.divide(xl[ineg] - step[ineg], sd[ineg])
        distbd[distbd >= alpha] = np.inf
        inew = -1
        if np.any(np.isfinite(distbd)):
            inew = np.argmin(distbd)
            alpha = distbd[inew]

        # Apply the conjugate gradient step and set SDRED to the decrease that
        # occurs in the objective function. The least curvature of the objective
        # function so far in CRVMIN and the gradient of the objective function
        # at STEP in GQ are adequately updated.
        if alpha > 0.:
            iterc += 1
            crv = sdhsd / sdsq
            if inew == -1 and crv > 0.:
                crvmin = min(crvmin, crv) if abs(crvmin + 1.) > tolsd else crv
            gqsqsav = gqsq
            gq += alpha * hsd
            gqsq = np.inner(gq[ifree], gq[ifree])
            step += alpha * sd
            sdred = max(0., alpha * (gqsqsav - .5 * alpha * sdhsd))
            qred += sdred

        # Restart the conjugate gradient method if it has hit a new bound. If
        # the step is on or outside the trust region, further improvement
        # attempts round the trust-region boundary are done below.
        if inew >= 0:
            nact += 1
            if sd[inew] < 0.:
                xbdi[inew] = -1
            else:
                xbdi[inew] = 1
            ifree[inew] = False
            ifixed[inew] = True
            delsq -= step[inew] * step[inew]
            if delsq <= 0.:
                crvmin = 0.
                continue
            beta = 0.
            continue

        # If the step did not reach the trust-region boundary, apply another
        # conjugate gradient iteration or return if the maximum number of
        # iterations is exceeded.
        if alpha < alphad:
            if iterc >= maxiter:
                break
            beta = gqsq / gqsqsav
            continue

        # The trust-region boundary has been reached by the step. End the
        # truncated conjugate gradient procedure and attempt to improve the
        # current step round the trust-region boundary.
        crvmin = 0.
        continue
    else:
        # Whenever the truncated conjugate gradient computations stopped because
        # the current trial step hit the trust-region boundary, a search is
        # performed to attempt improving the solution round the trust-region
        # boundary on the two dimensional space spanned by the free components
        # of STEP and GD.
        stepsq = np.inner(step[ifree], step[ifree])
        gqsq = np.inner(gq[ifree], gq[ifree])
        gdstep = np.inner(gq[ifree], step[ifree])
        sd[ifree] = step[ifree]
        sd[ifixed] = 0.
        hsd = np.asarray(hessp(sd, *args))
        if hsd.dtype.kind in np.typecodes['AllInteger']:
            hsd = np.asarray(hsd, dtype=float)
        hred = np.copy(hsd)
        inew = -1
        while nact < n - 1:
            # Whenever the previous iteration has hit a bound, the computation
            # round the trust-region boundary are restarted.
            if inew >= 0:
                sd[ifree] = step[ifree]
                sd[ifixed] = 0.
                hsd = np.asarray(hessp(sd, *args))
                if hsd.dtype.kind in np.typecodes['AllInteger']:
                    hsd = np.asarray(hsd, dtype=float)
                hred = np.copy(hsd)
            tolsd = tol * np.max(np.abs(sd), initial=1.)

            # Let the search direction SD be a linear combination of the reduced
            # step and the reduced gradient that is orthogonal to STEP.
            iterc += 1
            disc = gqsq * stepsq - gdstep ** 2.
            if disc < tolsd:
                break
            sqrd = np.sqrt(disc)
            sd[ifree] = (gdstep * step[ifree] - stepsq * gq[ifree]) / sqrd
            sd[ifixed] = 0.
            gdsd = -sqrd

            # By considering the simple bounds on the variables, calculate an
            # upper bound on the tangent of half the angle of the alternative
            # iteration in ANGBD and restart the iterations if a free variable
            # has reached a bound.
            angbd = 1.
            inew = -1
            sl = np.full_like(step, np.inf)
            su = np.full_like(step, np.inf)
            sl[ifree] = step[ifree] - xl[ifree]
            su[ifree] = xu[ifree] - step[ifree]
            if np.any(sl <= 0.):
                nact += 1
                inew = np.argmax(sl <= 0.)
                xbdi[inew] = -1
                ifree[inew] = False
                ifixed[inew] = True
                stepsq -= step[inew] ** 2.
                gqsq -= gq[inew] ** 2.
                gdstep -= gq[inew] * step[inew]
                continue
            elif np.any(su <= 0.):
                nact += 1
                inew = np.argmax(su <= 0.)
                xbdi[inew] = 1
                ifree[inew] = False
                ifixed[inew] = True
                stepsq -= step[inew] ** 2.
                gqsq -= gq[inew] ** 2.
                gdstep -= gq[inew] * step[inew]
                continue
            ssq = np.square(step[ifree]) + np.square(sd[ifree])
            temp = np.full_like(step, -np.inf)
            temp[ifree] = ssq - np.square(xl[ifree])
            itemp = np.greater(temp, 0.)
            temp[itemp] = np.sqrt(temp[itemp]) - sd[itemp]
            temp[np.logical_not(itemp)] = -np.inf
            isl = np.greater(angbd * temp - sl, tiny * temp)
            xbdisav = 0
            if np.any(isl):
                ratio = np.full_like(step, np.inf)
                ratio[isl] = np.divide(sl[isl], temp[isl])
                inew = np.argmin(ratio)
                angbd = ratio[inew]
                xbdisav = -1
            temp[ifree] = ssq - np.square(xu[ifree])
            itemp = np.greater(temp, 0.)
            temp[itemp] = np.sqrt(temp[itemp]) + sd[itemp]
            temp[np.logical_not(itemp)] = -np.inf
            isu = np.greater(angbd * temp - su, tiny * temp)
            if np.any(isu):
                ratio = np.full_like(step, np.inf)
                ratio[isu] = np.divide(su[isu], temp[isu])
                inew = np.argmin(ratio)
                angbd = ratio[inew]
                xbdisav = 1

            # Calculate the necessary curvatures for the alternative iteration.
            hsd = np.asarray(hessp(sd, *args))
            if hsd.dtype.kind in np.typecodes['AllInteger']:
                hsd = np.asarray(hsd, dtype=float)
            sdhsd = np.inner(sd[ifree], hsd[ifree])
            stephsd = np.inner(step[ifree], hsd[ifree])
            stephred = np.inner(step[ifree], hred[ifree])

            # Seek the greatest reduction in the objective function for a range
            # of equally spaced values of ANGT in [0,ANGBD], where ANGT is the
            # tangent of half the angle of the alternative iteration. For the
            # computations to hold, the interval should be split into at least
            # three parts, and NALT represents the number of intervals in the
            # unconstrained case.
            nalt = 20
            iu = int(float(nalt - 3) * angbd + 3.1)
            angt = angbd * np.arange(1, iu + 1, dtype=float) / float(iu)
            sth = np.divide(2. * angt, 1. + np.square(angt))
            temp = sdhsd + np.multiply(angt, stephred * angt - 2. * stephsd)
            rednew = np.multiply(sth, gdstep * angt - gdsd)
            rednew -= np.multiply(sth, .5 * np.multiply(sth, temp))
            isav = np.argmax(rednew)
            redmax = rednew[isav]
            rednew[isav] = -np.inf
            redprev = rednew[isav - 1] if isav > 0 else 0.
            rednext = rednew[isav + 1] if isav < iu - 1 else 0.
            if redmax <= 0.:
                isav = -1
                redmax = 0.

            # Set the sine and cosine of the angle of the alternative iteration,
            # calculate SDRED and return if no reduction is possible. The
            # computations are stopped if either no further reduction can be
            # obtained on the sampling of the trust-region boundary or if no
            # further progress on the step is possible.
            if isav == -1:
                break
            angc = angbd
            if isav < iu - 1:
                temp = (rednext - redprev) / (2. * redmax - redprev - rednext)
                angc *= (float(isav + 1) + .5 * temp) / float(iu)
            cth = (1. - angc * angc) / (1. + angc ** 2.)
            sth = 2. * angc / (1. + angc ** 2.)
            temp = sdhsd + angc * (angc * stephred - 2. * stephsd)
            sdred = sth * (angc * gdstep - gdsd - .5 * sth * temp)
            if sdred <= 0.:
                break

            # Update GOPT, STEP and HRED. If the angle of the alternative
            # iteration is restricted by a bound on a free variable, that
            # variable is fixed at the bound and the computation is restarted.
            step[ifree] = cth * step[ifree] + sth * sd[ifree]
            gq += (cth - 1.) * hred + sth * hsd
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
                stepsq -= step[inew] ** 2.
                gqsq -= gq[inew] ** 2.
                gdstep -= gq[inew] * step[inew]
                continue
            if sdred > 0.:
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
