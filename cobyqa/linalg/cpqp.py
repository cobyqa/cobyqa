import numpy as np
from numpy.testing import assert_

from .utils import givens


def cpqp(xopt, Aub, bub, Aeq, beq, xl, xu, delta, **kwargs):
    r"""
    Minimize approximately the convex piecewise quadratic function

    .. math::

        \frac{1}{2} ( \| [ \mathtt{Aub} x - \mathtt{bub} ]_+\|_2^2 +
        \| \mathtt{Aeq} x - \mathtt{beq} \|_2^2 ),

    subject to the bound constraints :math:`\mathtt{xl} \le x \le \mathtt{xu}`
    and the trust region :math:`\| x - \mathtt{xopt} \|_2 \le \mathtt{delta}`.
    This procedure assumes that the vector ``xopt`` is feasible.

    Parameters
    ----------
    xopt : array_like, shape (n,)
        Array ``xopt`` as shown above.
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
        Default is ``10 * eps * N * max(1, max(abs(xl)), max(abs(xu)))``, where
        ``N`` is the size of the reformulated problem, defined by
        :math:`\mathtt{N} = \mathtt{mub} + \mathtt{n}`.

    Raises
    ------
    AssertionError
        The vector ``xopt`` is not feasible.

    Notes
    -----
    The method is adapted from the TRSTEP algorithm [1]_. To cope with the
    convex piecewise quadratic objective function, the method minimizes

    .. math::

        \frac{1}{2} ( \| \mathtt{Aeq} x - \mathtt{beq} \|_2^2 + \| y \|_2^2 )

    subject to the bound constraints :math:`\mathtt{xl} \le x \le \mathtt{xu}`
    and :math:`0 \le y`, the linear inequality constraints
    :math:`\mathtt{Aub} x - y \le \mathtt{bub}`, and the trust region
    :math:`\| x - \mathtt{xopt} \|_2 \le \mathtt{delta}`.

    References
    ----------
    .. [1] M. J. D. Powell. "On fast trust region methods for quadratic models
       with linear constraints." In: Math. Program. Comput. 7 (2015), pp.
       237–-267.
    """
    # Format the inputs. Copies of the right-hand side of the linear inequality
    # constraints in BUB, the right-hand side of the linear equality constraints
    # in BEQ, the lower-bound constraints in XL, and the upper-bound constraints
    # in XU are made to prevent the changes made in this function to affect the
    # original vector.
    xopt = np.asarray(xopt)
    if xopt.dtype.kind in np.typecodes['AllInteger']:
        xopt = np.asarray(xopt, dtype=float)
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
    n = xopt.size
    mub = bub.size

    # Set the working arrays of the active constraints. The method sets
    # 1. NACT       number of active constraints among the linear inequality
    #               constraints and the bounds;
    # 2. IACT       indices of the active constraints, where values between 0
    #               and MUB-1 refer to the indices of linear inequality
    #               constraints, and values whose indices are above or equal MUB
    #               represent bound constraints, starting by the lower bounds;
    # 3. QFAC       orthogonal part of the QR factorization of the matrix whose
    #               columns are the gradients of the active constraints, ordered
    #               in accordance with IACT, and when NACT < N, (N-NACT) columns
    #               are added to QFAC to complete an orthogonal matrix;
    # 4. RFAC       upper triangular part of the QR factorization in RFAC.
    nact = 0
    iact = np.empty(mub + n, dtype=int)
    rfac = np.zeros((mub + n, mub + n))
    qfac = np.eye(mub + n)

    # Define the tolerances to compare floating-point numbers with zero.
    eps = np.finfo(float).eps
    tiny = np.finfo(float).tiny
    tol = 1e1 * eps * (mub + n)
    tolbd = tol * np.max(np.abs(np.r_[xl, xu]), initial=1.)
    tolbd = kwargs.get('bdtol', tolbd)

    # Evaluate shifts to carry out all calculations at the origin.
    bub -= np.dot(Aub, xopt)
    beq -= np.dot(Aeq, xopt)
    xl -= xopt
    xu -= xopt

    # Normalize the linear inequality constraints. It is done so that the
    # residual values can be compared with the trust-region radius. The Jacobian
    # matrix of the linear constraints of the reformulated problem is [Aub -I],
    # and the terms associated with the negative identity matrix will be
    # normalized on the fly to improve the computational efficiency.
    slc = np.sqrt(np.sum(np.square(Aub), axis=1) + 1.)
    Aub = np.divide(Aub, slc[:, np.newaxis])
    bubsav = np.copy(bub)
    bub = np.divide(bub, slc)

    # Ensure the feasibility of the initial guess, that is the origin.
    assert_(np.max(xl) < tolbd)
    assert_(np.min(xu) > -tolbd)
    assert_(np.isfinite(delta))
    assert_(delta > 0.)

    # Initialize the working sets and the trial step. The method sets
    # 1. GQ         gradient of the objective function;
    # 2. STEP       trial step;
    # 3. RESID      constraint violations of the inequality constraints,
    #               including the bound constraints;
    # 4. SLSQBD     bound on the norm squared of the slack variable vector;
    # 5. INACT      inactivity flag of the linear inequality constraints;
    # 6. IFREE      indices of the inactive constraints.
    # The values of the slack variables are not stored explicitly, as they equal
    # the last MUB coefficients of GQ. The constraints or the reformulated
    # problem whose normalized residuals are greater than DELTA are not
    # considered in INACT since they cannot become active during the
    # calculations due to the Cauchy-Schwarz inequality.
    gq = np.r_[np.dot(Aeq.T, -beq), np.maximum(0., -bubsav)]
    step = np.zeros(n)
    resid = np.r_[bub + np.divide(gq[n:], slc), -xl, gq[n:], xu]
    resid = np.maximum(0., resid)
    slsqbd = np.inner(beq, beq) + np.inner(gq[n:], gq[n:])
    inact = np.full(2 * (mub + n), True)
    inact[iact[:nact]] = False
    inact[resid > delta] = False

    # Start the iterative procedure.
    sd = np.zeros_like(gq)
    delsq = delta ** 2.
    actf = kwargs.get('actf', .2)
    stepsq = 0.
    reduct = 0.
    alpbd = 1.
    jsav = 0
    iterc = 0
    while iterc < mub + n - nact:
        gamma = 0.
        if jsav >= 0:
            # Pick the active set for the current trial step, and set SDD to the
            # vector closest to -GQ that is orthogonal to the normals of the
            # active constraints. SDD is then scaled to have length ACTF*DELTA,
            # as then a move of SDD from the current trial step is allowed by
            # the linear constraints. When None is returned, the projected
            # direction is null, and the current trial step leads to the optimal
            # solution, which stops the calculations.
            res = getact(gq, Aub, slc, nact, iact, qfac, rfac, delta, resid,
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
            # the active constraints of the reformulated problem.
            if np.max(resid[iact[:nact]], initial=0.) > 0.:
                resd = np.zeros(nact, dtype=float)
                for k in range(nact):
                    resd[k] = resid[iact[k]]
                    resd[k] -= np.inner(rfac[:k, k], resd[:k])
                    resd[k] /= rfac[k, k]
                sd = np.dot(qfac[:, :nact], resd)

                # The direction previously calculated in SD is also the shortest
                # move from STEP+SDD to the boundaries of the active
                # constraints. Set the scalar GAMMA to the greatest steplength
                # of this move that satisfies the trust-region bound.
                lstep = step + sdd[:n]
                rhs = delsq - np.inner(lstep, lstep)
                sdstep = np.inner(sd[:n], lstep)
                sdsq = np.inner(sd[:n], sd[:n])
                if rhs > 0.:
                    sqrd = np.sqrt(sdsq * rhs + sdstep ** 2.)
                    if sdstep <= 0. and sdsq > tiny * abs(sqrd - sdstep):
                        gamma = (sqrd - sdstep) / sdsq
                    elif abs(sqrd + sdstep) > tiny * rhs:
                        gamma = rhs / (sqrd + sdstep)
                    else:
                        gamma = 1.

                # Reduce GAMMA if necessary so that the move
                # STEP+SDD[:N]+GAMMA*SD[:N] also satisfies the inactive
                # constraints. The active constraints are satisfied since the
                # search direction belongs to their range space. The for loop
                # below could be removed by using the function numpy.dot, but
                # numerical results have shown that this may reduce the
                # precisions of the computations.
                asd = np.zeros_like(resid)
                asdd = np.zeros_like(resid)
                for k in range(2 * (mub + n)):
                    if inact[k] and k < mub:
                        asd[k] = np.inner(Aub[k, :], sd[:n])
                        asd[k] -= sd[k + n] / slc[k]
                        asdd[k] = np.inner(Aub[k, :], sdd[:n])
                        asdd[k] -= sdd[k + n] / slc[k]
                    elif inact[k] and k < 2 * mub + n:
                        asd[k] = -sd[k - mub]
                        asdd[k] = -sdd[k - mub]
                    elif inact[k]:
                        asd[k] = sd[k - 2 * mub - n]
                        asdd[k] = sdd[k - 2 * mub - n]
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
        # the boundary implicitly implied by the slack variable. The
        # calculations are stopped if the first derivative of this step is
        # sufficiently small or if no further progress is possible in the
        # current search direction.
        iterc += 1
        rhs = slsqbd - np.inner(gq[n:], gq[n:])
        sdgq = np.inner(sd, gq)
        sdsl = np.inner(sd[n:], gq[n:])
        sdsqsl = np.inner(sd[n:], sd[n:])
        sqrd = np.sqrt(rhs * sdsqsl + sdsl ** 2.)
        if sdgq >= 0.:
            break
        if sdsl <= 0. and sdsqsl > tiny * (sqrd - sdsl):
            alpha = (sqrd - sdsl) / sdsqsl
        elif abs(sqrd + sdsl) > tiny * abs(rhs):
            alpha = rhs / (sqrd + sdsl)
        else:
            alpha = np.inf

        # Reduce the steplength ALPHA so that the first N variables satisfy the
        # trust-region constraint. If the norm of SD is zero, stop the
        # computations, so that ALPHA will always be defined.
        rhs = delsq - stepsq
        if rhs <= 0.:
            break
        sdstep = np.inner(sd[:n], step)
        sdsq = np.inner(sd[:n], sd[:n])
        sqrd = np.sqrt(rhs * sdsq + sdstep ** 2.)
        if sdstep <= 0. and sdsq > tiny * (sqrd - sdstep):
            alpha = min(alpha, (sqrd - sdstep) / sdsq)
        elif abs(sqrd + sdstep) > tiny * abs(rhs):
            alpha = min(alpha, rhs / (sqrd + sdstep))
        elif np.isinf(alpha):
            break

        # Set SDD to the change in gradient along the search direction SD.
        sdd = np.r_[np.dot(Aeq.T, np.dot(Aeq, sd[:n])), sd[n:]]

        # Set SDGSD to the curvature of the model along SD and reduce ALPHA if
        # necessary to the value that minimizes the model. The objective
        # function is convex, so that SDGSD is nonnegative.
        sdgsd = np.inner(sd, sdd)
        alpht = alpha
        if sdgq + alpha * sdgsd > 0.:
            alpha = -sdgq / sdgsd

        # Make a further reduction in ALPHA if necessary to preserve
        # feasibility of the current step. The evaluations of this function with
        # the TRSTEP Fortran code of Powell may differ because the code below is
        # sensitive to computer rounding error.
        asd = np.zeros_like(resid)
        for k in range(2 * (mub + n)):
            if inact[k] and k < mub:
                asd[k] = np.inner(Aub[k, :], sd[:n])
                asd[k] -= sd[k + n] / slc[k]
            elif inact[k] and k < 2 * mub + n:
                asd[k] = -sd[k - mub]
            elif inact[k]:
                asd[k] = sd[k - 2 * mub - n]
        izero = np.less_equal(np.abs(asd), tiny * np.abs(resid))
        izero = izero & (np.abs(resid) < tolbd) & inact
        jsav = -1
        alphm = alpha
        if np.any(izero):
            jsav = np.argmax(izero)
            alpha = 0.
        else:
            iclose = np.greater(np.abs(asd), tiny * np.abs(resid))
            iclose = iclose & (alpha * asd > resid) & inact
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
        step += alpha * sd[:n]
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
        # previous one except in the case ITERC=NACT.
        sdu = gq
        if nact > 0:
            sdu = np.dot(qfac[:, nact:], np.dot(qfac[:, nact:].T, sdu))
        if gamma > 0.:
            beta = 0.
        else:
            sdusdd = np.inner(sdu, sdd)
            beta = sdusdd / sdgsd
        sd = beta * sd - sdu
        alpbd = 0.

    return step


def getact(gq, Aub, slc, nact, iact, qfac, rfac, delta, resid, inact, **kwargs):
    r"""
    Pick the current actual active set. It is defined by the property that the
    projection of -GQ into the space orthogonal to the normals of the active
    constraints is as large as possible, subject to this projected steepest
    descent direction moving no closer to the boundary of every constraint whose
    current residual is at most ACTF*DELTA.
    """
    # Initialize the initial Lagrange multipliers of the calculation in VLAM.
    tdel = kwargs.get('actf', .2) * delta
    vlam = np.zeros_like(gq)
    cgsqsav = 2. * np.inner(gq, gq)
    mub, n = Aub.shape

    # Define the tolerances to compare floating-point numbers with zero.
    eps = np.finfo(float).eps
    tiny = np.finfo(float).tiny
    tol = 1e1 * eps * (mub + n)
    tolgd = tol * np.max(np.abs(gq), initial=1.)

    # Remove the constraints from the initial active set IACT that are not
    # active anymore, and those whose Lagrange multipliers are nonnegative.
    nactc = nact
    for ic in range(nactc - 1, -1, -1):
        if resid[iact[ic]] > tdel:
            nact = rmact(nact, iact, ic, qfac, rfac, inact, vlam)
    ic = nact - 1
    while ic >= 0:
        lmi = np.inner(qfac[:, ic], gq)
        lmi -= np.inner(rfac[ic, ic + 1:nact], vlam[ic + 1:nact])
        if lmi >= 0.:
            nact = rmact(nact, iact, ic, qfac, rfac, inact, vlam)
            ic = nact - 1
        else:
            vlam[ic] = lmi / rfac[ic, ic]
            ic -= 1

    # Start the iterative procedure.
    while nact < mub + n:
        # Set the new search direction in SD, and terminate if no further
        # calculation is possible, namely if SD is the origin. The situation
        # NACT=MUB+N occurs for sufficiently large DELTA if the origin is in the
        # convex hull of the constraint gradients.
        sd = -np.dot(qfac[:, nact:], np.dot(qfac[:, nact:].T, gq))
        cgsqa = np.inner(sd[:n], sd[:n])
        cgsqb = np.inner(sd[n:], sd[n:])
        cgsq = cgsqa + cgsqb
        if cgsq >= cgsqsav or cgsq <= tolgd:
            return
        cgsqsav = cgsq
        sdnorm = np.sqrt(cgsqa)

        # Pick the next integer INEXT corresponding to the index of the most
        # violated constraint, and terminate is no such constraint is found. The
        # purpose of CTOL below is to estimate whether a positive value of
        # VIOLMX may be due to computer rounding errors.
        scale = sdnorm / delta
        iclose = inact & (resid <= tdel)
        resall = np.full_like(resid, -np.inf)
        for k in range(2 * (mub + n)):
            if iclose[k] and k < mub:
                resall[k] = np.inner(Aub[k, :], sd[:n])
                resall[k] -= np.divide(sd[k + n], slc[k])
            elif iclose[k] and k < 2 * mub + n:
                resall[k] = -sd[k - mub]
            elif iclose[k]:
                resall[k] = sd[k - 2 * mub - n]
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
                    asd[iact[k]] = np.inner(Aub[iact[k], :], sd[:n])
                    asd[iact[k]] -= sd[iact[k] + n] / slc[iact[k]]
                elif iact[k] < 2 * mub + n:
                    asd[iact[k]] = -sd[iact[k] - mub]
                else:
                    asd[iact[k]] = sd[iact[k] - 2 * mub - n]
            ctol = np.max(np.r_[np.abs(asd), ctol])
        if inext == -1 or violmx <= 1e1 * ctol:
            return sd, cgsq, nact

        # Apply Givens rotations to the last (N-NACT) columns of QFAC so that
        # the first (NACT+1) columns are the ones required for the addition of
        # the INEXT-th constraint. The corresponding appropriate columns to RFAC
        # is then added.
        rdiag = 0.
        for j in range(mub + n - 1, -1, -1):
            if inext < mub:
                sprod = np.inner(qfac[:n, j], Aub[inext, :])
                sprod -= qfac[n + inext, j] / slc[inext]
            elif inext < 2 * mub + n:
                sprod = -qfac[inext - mub, j]
            else:
                sprod = qfac[inext - 2 * mub - n, j]
            if j < nact:
                rfac[j, nact] = sprod
            elif abs(rdiag) <= eps * max(1., abs(sprod)):
                rdiag = sprod
            else:
                rdiag = givens(qfac, sprod, rdiag, j + 1, j, 1)
        if rdiag < 0.:
            qfac[:, nact] *= -1.
        rfac[nact, nact] = abs(rdiag)
        iact[nact] = inext
        vlam[nact] = 0.
        nact += 1
        inact[inext] = False

        while violmx > ctol:
            # Update the Lagrange multipliers of the active linear inequality
            # constraints and the active bound constraints in VLAM.
            vmu = np.empty(nact)
            vmu[-1] = 1. / rfac[nact - 1, nact - 1] ** 2.
            for i in range(nact - 2, -1, -1):
                vmu[i] = -np.inner(rfac[i, i + 1:nact], vmu[i + 1:])
                vmu[i] /= rfac[i, i]
            imult = np.greater(np.abs(vmu), tiny * np.abs(vlam[:nact]))
            imult = imult & (vlam[:nact] >= violmx * vmu)
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
                    nact = rmact(nact, iact, ic, qfac, rfac, inact, vlam)

    return


def rmact(nact, iact, ic, qfac, rfac, inact, vlam):
    r"""
    Rearrange the active constraints in IACT so that the new value of
    IACT(NACT-1) is the old value of IACT(IC) by applying a sequence of Givens
    rotations to the matrices QFAC and RFAC. All arrays are modified in place,
    and the value of NACT is then reduced by one.
    """
    inact[iact[ic]] = True
    for jc in range(ic, nact - 1):
        # Perform a Givens rotations on the rows JC and JC+1 of RFAC. Only the
        # meaningful components, those in RFAC[:NACT,:NACT], are updated.
        cval = rfac[jc + 1, jc + 1]
        sval = rfac[jc, jc + 1]
        hval = givens(rfac, cval, sval, jc, jc + 1, 0, slice(jc, nact))
        rfac[[jc, jc + 1], jc:nact] = rfac[[jc + 1, jc], jc:nact]
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
