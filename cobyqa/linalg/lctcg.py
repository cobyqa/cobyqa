import numpy as np
from numpy.testing import assert_

from .utils import givens


def lctcg(xopt, gq, hessp, args, Aub, bub, Aeq, beq, xl, xu, delta, **kwargs):
    r"""
    Minimize approximately the quadratic function

    .. math::

        \mathtt{gq}^{\mathsf{T}} ( x - \mathtt{xopt} ) + \frac{1}{2}
        ( x - \mathtt{xopt} )^{\mathsf{T}} \mathtt{Hq} ( x - \mathtt{xopt} ),

    subject to the linear constraints :math:`\mathtt{Aub} x \le \mathtt{bub}`
    and :math:`\mathtt{Aeq} x = \mathtt{beq}`, the bound constraints
    :math:`\mathtt{xl} \le x \le \mathtt{xu}`, and the trust region
    :math:`\| x - \mathtt{xopt} \|_2 \le \mathtt{delta}`. This procedure assumes
    that the matrix ``Hq`` is symmetric and the vector ``xopt`` is feasible.

    Parameters
    ----------
    xopt : array_like, shape (n,)
        Array ``xopt`` as shown above.
    gq : array_like, shape (n,)
        Array ``gq`` as shown above.
    hessp : callable
        Function providing the product :math:`\mathtt{Hq} x` as shown above.

            ``hessp(x, *args) -> array_like, shape(n,)``

        where ``x`` is an array with shape (n,) and ``args`` is the tuple of
        fixed parameters needed to specify the function. It is assumed that the
        implicit Hessian matrix ``Hq`` in the function ``hessp`` is symmetric,
        but not necessarily positive semidefinite.
    args : tuple
        Extra arguments to pass to the Hessian function.
    Aub : array_like, shape (mub, n)
        Matrix ``Aub`` as shown above.
    bub : array_like, shape (mub,)
        Right-hand side vector ``bub`` as shown above.
    Aeq : array_like, shape (meq, n)
        Matrix ``Aeq`` as shown above.
    beq : array_like, shape (meq,)
        Right-hand side vector ``beq`` as shown above.
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
    actf : float, optional
        Factor of proximity to the linear constraints.
        Default is 0.2.
    bdtol : float, optional
        Tolerance for comparisons on the bound constraints.
        Default is ``10 * eps * n * max(1, max(abs(xl)), max(abs(xu)))``.
    lctol : float, optional
        Tolerance for comparisons on the linear constraints.
        Default is ``10 * eps * n * max(1, max(abs(bub)))``.

    Raises
    ------
    AssertionError
        The vector ``xopt`` is not feasible.

    Notes
    -----
    The method is adapted from the TRSTEP algorithm [1]_. It is an active-set
    variation of the truncated conjugate gradient method, which maintains the QR
    factorization of the matrix whose columns are the gradients of the active
    constraints. The linear equality constraints are then handled by considering
    that they are always active.

    References
    ----------
    .. [1] M. J. D. Powell. "On fast trust region methods for quadratic models
       with linear constraints." In: Math. Program. Comput. 7 (2015), pp.
       237--267.
    """
    # Format the inputs. Copies of the gradient in GQ, the right-hand side of
    # the linear inequality constraints in BUB, the right-hand side of the
    # linear equality constraints in BEQ, the lower-bound constraints in XL, and
    # the upper-bound constraints in XU are made to prevent the changes made in
    # this function to affect the original vector.
    xopt = np.asarray(xopt)
    if xopt.dtype.kind in np.typecodes['AllInteger']:
        xopt = np.asarray(xopt, dtype=float)
    gq = np.array(gq, dtype=float)
    Aub = np.asarray(Aub)
    if Aub.dtype.kind in np.typecodes['AllInteger']:
        Aub = np.asarray(Aub, dtype=float)
    bub = np.array(bub, dtype=float)
    Aeq = np.asarray(Aeq)
    if Aeq.dtype.kind in np.typecodes['AllInteger']:
        Aeq = np.asarray(Aeq, dtype=float)
    beq = np.array(beq, dtype=float)
    xl = np.array(xl, dtype=float)
    xu = np.array(xu, dtype=float)
    n = gq.size
    mub = bub.size
    meq = beq.size

    # Define the tolerances to compare floating-point numbers with zero.
    eps = np.finfo(float).eps
    tiny = np.finfo(float).tiny
    tol = 1e1 * eps * n
    tolbd = tol * np.max(np.abs(np.r_[xl, xu]), initial=1.)
    tolbd = kwargs.get('bdtol', tolbd)
    tollc = tol * np.max(np.abs(bub), initial=1.)
    tollc = kwargs.get('lctol', tollc)

    # Shift the constraints to carry out all calculations at the origin.
    bub -= np.dot(Aub, xopt)
    beq -= np.dot(Aeq, xopt)
    xl -= xopt
    xu -= xopt

    # Ensure that the initial guess respects the linear inequality constraints
    # and remove the inequality constraints whose gradients are zero.
    if mub > 0:
        assert_(np.min(bub) > -tollc)
    slc = np.sqrt(np.sum(np.square(Aub), axis=1))
    izero = np.less_equal(slc, tol * np.abs(bub))
    if np.any(izero):
        ikeep = np.logical_not(izero)
        Aub = Aub[ikeep, :]
        bub = bub[ikeep]
        slc = slc[ikeep]
        mub -= np.count_nonzero(izero)

    # Set the working arrays of the active constraints. The method sets
    # 1. NACT       number of active constraints among the linear inequality
    #               constraints and the bounds;
    # 2. IACT       indices of the active constraints, where values between 0
    #               and MUB-1 refer to the indices of linear inequality
    #               constraints, and values whose indices are above or equal MUB
    #               represent bound constraints, starting by the lower bounds;
    # 3. QFAC       orthogonal part of the QR factorization of the matrix whose
    #               columns are the gradients of the active constraints,
    #               starting with the linear equality constraints, followed by
    #               the others, in accordance with IACT, and when MEQ+NACT < N,
    #               (N-NACT-MEQ) columns are added to QFAC to complete an
    #               orthogonal matrix;
    # 4. RFAC       upper triangular part of the QR factorization.
    # The warm start process of the Powell's method TRSTEP is not entertained,
    # since it requires the Jacobian matrix of the constraints to be constant.
    # It could however be easily implemented by adding NACT, IACT, QFAC, and
    # RFAC to the arguments of the function.
    # TODO: Remove the rank condition on AEQ and adapt the comments.
    nact = 0
    iact = np.empty(n, dtype=int)
    rfac = np.zeros((n, n))
    qfac, rfac[:, :meq] = np.linalg.qr(Aeq.T, 'complete')

    # Ensure the full row rankness of the Jacobian matrix of the linear equality
    # constraints and the feasibility of the initial guess for the bounds and
    # the linear equality constraints.
    if meq > 0:
        rdiag = np.diag(rfac[:meq, :meq])
        assert_(np.min(np.abs(rdiag)) > tol * np.max(rdiag, initial=1.))
        assert_(np.max(np.abs(beq)) < tollc)
    assert_(np.max(xl) < tolbd)
    assert_(np.min(xu) > -tolbd)
    assert_(np.isfinite(delta))
    assert_(delta > 0.)

    # Normalize the linear inequality constraints. It is done so that the
    # residual values can be compared with the trust-region radius. The
    # residuals of the bound constraints are already normalized, as their
    # Jacobian matrix is plus or minus identity.
    Aub = np.divide(Aub, slc[:, np.newaxis])
    bub = np.divide(bub, slc)

    # Initialize the working sets and the trial step. The method sets
    # 1. STEP       trial step;
    # 2. RESID      constraint violations of the inequality constraints, whose
    #               order is consistent with QFAC and RFAC;
    # 3. INACT      inactivity flag of the linear inequality constraints;
    # 4. IFREE      indices of the inactive constraints.
    # The general initialization below is kept if a warm start is implemented in
    # the future, as NACT may be nonzero. The constraints whose residuals are
    # greater than DELTA are not considered in INACT since they cannot become
    # active during the calculations due to the Cauchy-Schwarz inequality.
    step = np.zeros_like(gq)
    resid = np.maximum(0., np.r_[bub, -xl, xu])
    inact = np.full(mub + 2 * n, True)
    inact[iact[:nact]] = False
    inact[resid > delta] = False

    # Start the iterative procedure.
    sd = np.zeros_like(step)
    delsq = delta ** 2.
    actf = kwargs.get('actf', .2)
    stepsq = 0.
    reduct = 0.
    alpbd = 1.
    jsav = 0
    iterc = 0
    while iterc < n - meq - nact:
        gamma = 0.
        if jsav >= 0:
            # Pick the active set for the current trial step, and set SDD to the
            # vector closest to -GQ that is orthogonal to the normals of the
            # active constraints. SDD is then scaled to have length ACTF*DELTA,
            # as then a move of SDD from the current trial step is allowed by
            # the linear constraints. When None is returned, the projected
            # direction is null, and the current trial step leads to the optimal
            # solution, which stops the calculations.
            res = getact(gq, Aub, Aeq, nact, iact, qfac, rfac, delta, resid,
                         inact, **kwargs)
            if res is None:
                break
            sdd, sddsq, nact = res
            snorm = np.sqrt(sddsq)
            if snorm <= tiny * delta:
                break
            scale = actf * delta / snorm
            sdd = scale * sdd

            # The trial step is set to the shortest move to the boundaries of
            # the active constraints, including the linear equality constraints.
            if np.max(resid[iact[:nact]], initial=0.) > 0.:
                resd = np.empty(nact, dtype=float)
                for k in range(nact):
                    resd[k] = resid[iact[k]]
                    resd[k] -= np.inner(rfac[meq:meq + k, meq + k], resd[:k])
                    resd[k] /= rfac[meq + k, meq + k]
                sd = np.dot(qfac[:, meq:meq + nact], resd)

                # The direction previously calculated in SD is also the shortest
                # move from STEP+SDD to the boundaries of the active
                # constraints. Set the scalar GAMMA to the greatest steplength
                # of this move that satisfies the trust-region bound.
                lstep = step + sdd
                rhs = delsq - np.inner(lstep, lstep)
                sdstep = np.inner(sd, lstep)
                sdsq = np.inner(sd, sd)
                if rhs > 0.:
                    sqrd = np.sqrt(sdsq * rhs + sdstep ** 2.)
                    if sdstep <= 0. and sdsq > tiny * abs(sqrd - sdstep):
                        gamma = (sqrd - sdstep) / sdsq
                    elif abs(sqrd + sdstep) > tiny * rhs:
                        gamma = rhs / (sqrd + sdstep)
                    else:
                        gamma = 1.

                # Reduce GAMMA if necessary so that the move STEP+SDD+GAMMA*SD
                # also satisfies the nonactive constraints. The active
                # constraints are satisfied since the search direction belongs
                # to their range space. The for loop below could be removed by
                # using the function numpy.dot, but numerical results have shown
                # that this may reduce the precisions of the computations.
                asd = np.zeros_like(resid)
                asdd = np.zeros_like(resid)
                for k in range(mub + 2 * n):
                    if inact[k] and k < mub:
                        asd[k] = np.inner(Aub[k, :], sd)
                        asdd[k] = np.inner(Aub[k, :], sdd)
                    elif inact[k] and k < mub + n:
                        asd[k] = -sd[k - mub]
                        asdd[k] = -sdd[k - mub]
                    elif inact[k]:
                        asd[k] = sd[k - mub - n]
                        asdd[k] = sdd[k - mub - n]
                idiv = np.greater(asd, tiny * np.abs(resid - asdd))
                if np.any(idiv):
                    dfeas = np.divide(resid[idiv] - asdd[idiv], asd[idiv])
                    gamma = min(gamma, max(np.min(dfeas), 0.))
                gamma = min(gamma, 1.)

            # Set the next direction SD for seeking a reduction in the objective
            # function subject to the trust region and the linear constraints.
            sd = sdd + gamma * sd
            iterc = 0 if gamma <= 0. else -1
            alpbd = 1.

        # Set ALPHA to the steplength from the current trial step along SD to
        # the trust-region boundary. The calculations are stopped if the first
        # derivative of this step is sufficiently small or if no further
        # progress is possible in the current search direction.
        iterc += 1
        rhs = delsq - stepsq
        if rhs <= 0.:
            break
        sdgq = np.inner(sd, gq)
        sdstep = np.inner(sd, step)
        sdsq = np.inner(sd, sd)
        sqrd = np.sqrt(rhs * sdsq + sdstep ** 2.)
        if sdgq >= 0.:
            break
        if sdstep <= 0. and sdsq > tiny * abs(sqrd - sdstep):
            alpha = (sqrd - sdstep) / sdsq
        elif abs(sqrd + sdstep) > tiny * rhs:
            alpha = rhs / (sqrd + sdstep)
        else:
            break

        # Set SDD to the change in gradient along the search direction SD.
        sdd = np.asarray(hessp(sd, *args))
        if sdd.dtype.kind in np.typecodes['AllInteger']:
            sdd = np.asarray(sdd, dtype=float)

        # Set SDGSD to the curvature of the model along SD and reduce ALPHA if
        # necessary to the value that minimizes the model.
        sdgsd = np.inner(sd, sdd)
        alpht = alpha
        if sdgq + alpha * sdgsd > 0.:
            alpha = -sdgq / sdgsd

        # Make a further reduction in ALPHA if necessary to preserve the
        # feasibility of the current step. The evaluations of this function with
        # the TRSTEP Fortran code of Powell may differ because the code below is
        # sensitive to computer rounding error. First, the for loop could be
        # removed by using the function numpy.dot, but numerical results have
        # shown that this may reduce the precisions of the computations when a
        # component of the product is below the machine epsilon in absolute
        # value. Secondly, the detection of a new bound whenever the components
        # in ASD and in RESID are close to zero is sensible to rounding errors.
        asd = np.zeros_like(resid)
        for k in range(mub + 2 * n):
            if inact[k] and k < mub:
                asd[k] = np.inner(Aub[k, :], sd)
            elif inact[k] and k < mub + n:
                asd[k] = -sd[k - mub]
            elif inact[k]:
                asd[k] = sd[k - mub - n]
        izero = np.less_equal(np.abs(asd), tiny * np.abs(resid))
        izero = np.logical_and(izero, np.abs(resid) < tollc)
        izero = np.logical_and(izero, inact)
        jsav = -1
        alphm = alpha
        if np.any(izero):
            jsav = np.argmax(izero)
            alpha = 0.
        else:
            iclose = np.greater(np.abs(asd), tiny * np.abs(resid))
            iclose = np.logical_and(iclose, alpha * asd > resid)
            iclose = np.logical_and(iclose, inact)
            if np.any(iclose):
                afeas = np.full_like(asd, np.inf)
                afeas[iclose] = np.divide(resid[iclose], asd[iclose])
                jsav = np.argmin(afeas)
                alpha = afeas[jsav]
        alpha = min(alphm, max(alpha, alpbd))
        if gamma > 0.:
            alpha = min(alpha, 1.)

        # Make the actual conjugate gradient iteration. The positive part
        # operator applied to RESID is crucial, as it prevents numerical
        # difficulties engendered when negative values are created due to
        # computer rounding errors.
        step += alpha * sd
        stepsq = np.inner(step, step)
        gq += alpha * sdd
        resid[inact] = np.maximum(0., resid[inact] - alpha * asd[inact])
        if nact > 0:
            resid[iact[:nact]] *= max(0., 1. - gamma)
        reduct -= alpha * (sdgq + .5 * alpha * sdgsd)

        # Conjugate gradient tests for termination. When the distance from the
        # current step to the trust-region boundary is greater than ACTF*DELTA
        # and a new constraint has been hit, the calculations are restarted.
        if abs(alpha - alpht) <= tol * max(alpha, 1.):
            break
        elif jsav >= 0:
            if stepsq <= (1. - actf) ** 2. * delsq:
                iterc = 0
                continue
            break

        # Calculate the next search direction, which is conjugate to the
        # previous one except in the case ITERC=MEQ+NACT.
        sdu = gq
        if meq + nact > 0:
            temp = np.dot(qfac[:, meq + nact:].T, gq)
            sdu = np.dot(qfac[:, meq + nact:], temp)
        if gamma > 0.:
            beta = 0.
        else:
            gqsdd = np.inner(sdu, sdd)
            beta = gqsdd / sdgsd
        sd = beta * sd - sdu
        alpbd = 0.

    return step


def getact(gq, Aub, Aeq, nact, iact, qfac, rfac, delta, resid, inact, **kwargs):
    r"""
    Pick the actual active set. It is defined by the property that the
    projection of -GQ into the space orthogonal to the normals of the active
    constraints is as large as possible, subject to this projected steepest
    descent direction moving no closer to the boundary of every constraint whose
    current residual is at most ACTF*DELTA.
    """
    # Initialize the initial Lagrange multipliers of the calculation in VLAM.
    tdel = kwargs.get('actf', .2) * delta
    vlam = np.zeros_like(gq)
    cgsqsav = 2. * np.inner(gq, gq)
    n = gq.size
    mub = Aub.shape[0]
    meq = Aeq.shape[0]

    # Define the tolerances to compare floating-point numbers with zero.
    eps = np.finfo(float).eps
    tiny = np.finfo(float).tiny
    tol = 1e1 * eps * n
    tolgd = tol * np.max(np.abs(gq), initial=1.)

    # Remove the constraints from the initial active set IACT that are not
    # active anymore, and those whose Lagrange multipliers are nonnegative.
    nactc = nact
    for ic in range(nactc - 1, -1, -1):
        if resid[iact[ic]] > tdel:
            nact = rmact(meq, nact, iact, ic, qfac, rfac, inact, vlam)
    ic = nact - 1
    while ic >= 0:
        icm = ic + meq
        lmi = np.inner(qfac[:, icm], gq)
        lmi -= np.inner(rfac[icm, icm + 1:meq + nact], vlam[ic + 1:nact])
        if lmi >= 0.:
            nact = rmact(meq, nact, iact, ic, qfac, rfac, inact, vlam)
            ic = nact - 1
        else:
            vlam[ic] = lmi / rfac[icm, icm]
            ic -= 1

    # Start the iterative procedure.
    while nact < n - meq:
        # Set the new search direction in SD, and terminate if no further
        # calculation is possible. The situation MEQ+NACT=N occurs for
        # sufficiently large DELTA if the origin is in the convex hull of the
        # constraint gradients.
        sd = -np.dot(qfac[:, meq + nact:], np.dot(qfac[:, meq + nact:].T, gq))
        cgsq = np.inner(sd, sd)
        if cgsq >= cgsqsav or cgsq <= tolgd:
            return
        cgsqsav = cgsq
        sdnorm = np.sqrt(cgsq)

        # Pick the next integer INEXT corresponding to the index of the most
        # violated constraint, and terminate is no such constraint is found. The
        # purpose of CTOL below is to estimate whether a positive value of
        # VIOLMX may be due to computer rounding errors.
        scale = sdnorm / delta
        iclose = np.logical_and(inact, resid <= tdel)
        resall = np.full_like(resid, -np.inf)
        for k in range(mub + 2 * n):
            if iclose[k] and k < mub:
                resall[k] = np.inner(Aub[k, :], sd)
            elif iclose[k] and k < mub + n:
                resall[k] = -sd[k - mub]
            elif iclose[k]:
                resall[k] = sd[k - mub - n]
        iviolmx = np.greater(resall, scale * resid)
        inext = -1
        violmx = 0.
        if np.any(iviolmx):
            resall[np.logical_not(iviolmx)] = -np.inf
            inext = np.argmax(resall)
            violmx = resall[inext]
        ctol = 0.
        if violmx > 0. and nact > 0:
            asd = np.zeros_like(resid)
            for k in range(nact):
                if iact[k] < mub:
                    asd[iact[k]] = np.inner(Aub[iact[k], :], sd)
                elif iact[k] < mub + n:
                    asd[iact[k]] = -sd[iact[k] - mub]
                else:
                    asd[iact[k]] = sd[iact[k] - mub - n]
            ctol = np.max(np.r_[np.abs(asd), ctol])
        if inext == -1 or violmx <= 1e1 * ctol:
            return sd, cgsq, nact

        # Apply Givens rotations to the last (N-NACT-MEQ) columns of QFAC so
        # that the first (MEQ+NACT+1) columns are the ones required for the
        # addition of the INEXT-th constraint. The corresponding appropriate
        # columns to RFAC is then added.
        rdiag = 0.
        for j in range(n - 1, -1, -1):
            if inext < mub:
                sprod = np.inner(qfac[:, j], Aub[inext, :])
            elif inext < mub + n:
                sprod = -qfac[inext - mub, j]
            else:
                sprod = qfac[inext - mub - n, j]
            if j < meq + nact:
                rfac[j, meq + nact] = sprod
            elif abs(rdiag) <= eps * max(1., abs(sprod)):
                rdiag = sprod
            else:
                rdiag = givens(qfac, sprod, rdiag, j + 1, j, 1)
        if rdiag < 0.:
            qfac[:, meq + nact] *= -1.
        rfac[meq + nact, meq + nact] = abs(rdiag)
        iact[nact] = inext
        vlam[nact] = 0.
        nact += 1
        inact[inext] = False

        while violmx > ctol:
            # Update the Lagrange multipliers of the active linear inequality
            # constraints and the active bound constraints in VLAM.
            vmu = np.empty(nact)
            vmu[-1] = 1. / rfac[meq + nact - 1, meq + nact - 1] ** 2.
            for i in range(nact - 2, -1, -1):
                imeq = meq + i
                vmu[i] = -np.inner(rfac[imeq, imeq + 1:meq + nact], vmu[i + 1:])
                vmu[i] /= rfac[imeq, imeq]
            imult = np.greater(np.abs(vmu), tiny * np.abs(vlam[:nact]))
            imult = np.logical_and(imult, vlam[:nact] >= violmx * vmu)
            ic = -1
            vmult = violmx
            if np.any(imult):
                multr = np.full_like(vmu, np.inf)
                iimult = np.flatnonzero(imult)
                multr[imult] = np.divide(vlam[iimult], vmu[imult])
                ic = np.argmin(multr)
                vmult = multr[ic]
            vlam[:nact] -= vmult * vmu
            if ic >= 0:
                vlam[ic] = 0.
            violmx = max(violmx - vmult, 0.)
            if ic == -1:
                violmx = 0.

            # Reduce the active set if necessary, so that all components of the
            # new VLAM are negative, and reset the residuals of the constraints
            # that become inactive during the process.
            for ic in range(nact - 1, -1, -1):
                if vlam[ic] >= 0.:
                    nact = rmact(meq, nact, iact, ic, qfac, rfac, inact, vlam)

    return


def rmact(meq, nact, iact, ic, qfac, rfac, inact, vlam):
    r"""
    Rearrange the active constraints in IACT so that the new value of
    IACT(NACT-1) is the old value of IACT(IC) by applying a sequence of Givens
    rotations to the matrices QFAC and RFAC. All arrays are modified in place,
    and the value of NACT is then reduced by one.
    """
    inact[iact[ic]] = True
    for jc in range(meq + ic, meq + nact - 1):
        # Perform a Givens rotations on the rows JC and JC+1 of RFAC. Only the
        # meaningful components, those in RFAC[:NACTM,:NACTM], are updated.
        cval = rfac[jc + 1, jc + 1]
        sval = rfac[jc, jc + 1]
        hval = givens(rfac, cval, sval, jc, jc + 1, 0, slice(jc, meq + nact))
        rfac[[jc, jc + 1], jc:meq + nact] = rfac[[jc + 1, jc], jc:meq + nact]
        rfac[:jc + 2, [jc, jc + 1]] = rfac[:jc + 2, [jc + 1, jc]]
        rfac[jc, jc] = hval
        rfac[jc + 1, jc] = 0.

        # Perform a Givens rotations on the columns JC and JC+1 of QFAC.
        givens(qfac, cval, sval, jc, jc + 1, 1)
        qfac[:, [jc, jc + 1]] = qfac[:, [jc + 1, jc]]
    iact[ic:nact - 1] = iact[ic + 1:nact]
    vlam[ic:nact - 1] = vlam[ic + 1:nact]
    nact -= 1

    return nact
