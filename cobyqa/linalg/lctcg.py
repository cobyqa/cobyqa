import numpy as np
from numpy.testing import assert_

from .utils import getact, qr


def lctcg(xopt, gq, hessp, args, Aub, bub, Aeq, beq, xl, xu, delta, **kwargs):
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
    args : tuple
        Parameters to forward to the Hessian product function.
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

    Returns
    -------
    step : numpy.ndarray, shape (n,)
        Step from `xopt` towards the estimated point.

    Other Parameters
    ----------------
    bdtol : float, optional
        Tolerance for comparisons on the bound constraints (the default is
        ``10 * eps * n * max(1, max(abs(xl)), max(abs(xu)))``.
    lctol : float, optional
        Tolerance for comparisons on the linear constraints (the default is
        ``10 * eps * n * max(1, max(abs(bub)))``).

    Raises
    ------
    AssertionError
        The vector `xopt` is not feasible.

    Notes
    -----
    The method is adapted from the TRSTEP algorithm [1]_. It is an active-set
    variation of the truncated conjugate gradient method, which maintains the QR
    factorization of the matrix whose columns are the gradients of the active
    constraints. The linear equality constraints are then handled by considering
    them are always active.

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
    Aub = np.atleast_2d(Aub)
    if Aub.dtype.kind in np.typecodes['AllInteger']:
        Aub = np.asarray(Aub, dtype=float)
    bub = np.atleast_1d(bub).astype(float)
    Aeq = np.atleast_2d(Aeq)
    if Aeq.dtype.kind in np.typecodes['AllInteger']:
        Aeq = np.asarray(Aeq, dtype=float)
    beq = np.atleast_1d(beq).astype(float)
    xl = np.atleast_1d(xl).astype(float)
    xu = np.atleast_1d(xu).astype(float)

    # Define the tolerances to compare floating-point numbers with zero.
    eps = np.finfo(float).eps
    tiny = np.finfo(float).tiny
    mlub, n = Aub.shape
    tol = 10. * eps * n
    lctol = tol * np.max(np.abs(bub), initial=1.)
    lctol = kwargs.get('lctol', lctol)
    bdtol = tol * np.max(np.abs(np.r_[xl, xu]), initial=1.)
    bdtol = kwargs.get('bdtol', bdtol)

    # Shift the constraints to carry out all calculations at the origin.
    bub -= np.dot(Aub, xopt)
    beq -= np.dot(Aeq, xopt)
    xl -= xopt
    xu -= xopt

    # Ensure the feasibility of the initial guess.
    if mlub > 0:
        assert_(np.min(bub) > -lctol)
    if beq.size > 0:
        assert_(np.max(np.abs(beq)) < lctol)
    assert_(np.max(xl) < bdtol)
    assert_(np.min(xu) > -bdtol)
    assert_(np.isfinite(delta))
    assert_(delta > 0.)

    # Remove the linear constraints whose gradients are zero, and normalize the
    # remaining constraints. The bound constraints are already normalized.
    temp = np.linalg.norm(Aub, axis=1)
    izero = np.less_equal(np.abs(temp), tol * np.maximum(1., np.abs(bub)))
    if np.any(izero):
        ikeep = np.logical_not(izero)
        Aub = Aub[ikeep, :]
        bub = bub[ikeep]
        temp = temp[ikeep]
        mlub -= np.count_nonzero(izero)
    Aub = np.divide(Aub, temp[:, np.newaxis])
    bub = np.divide(bub, temp)

    # Set the initial active set to the empty set, and calculate the normalized
    # residuals of the constraints at the origin. The residuals of the linear
    # equality constraints are not maintained as the method ensures that the
    # search directions lies in the linear space they span.
    nact = np.array(0, dtype=int)
    iact = np.empty(n, dtype=int)
    rfac = np.zeros((n, n), dtype=float)
    qfac, req, _ = qr(Aeq.T, pivoting=True)
    temp = np.maximum(1., np.linalg.norm(req[:, :np.min(req.shape)], axis=0))
    mleq = np.count_nonzero(np.abs(np.diag(req)) >= tol * temp)
    rfac[:, :mleq] = req[:, :mleq]
    resid = np.maximum(0., np.r_[bub, -xl, xu])

    # Start the iterative calculations. The truncated conjugate gradient method
    # should be stopped after n - mleq - nact iterations, except if a new
    # constraint has been hit in the last iteration, in which case the method is
    # restarted and the iteration counter reinitialized.
    step = np.zeros_like(gq)
    sd = np.zeros_like(step)
    reduct = 0.
    stepsq = 0.
    alpbd = 1.
    inext = 0
    iterc = 0
    gamma = 0.
    while iterc < n - mleq - nact or inext >= 0:
        # A new constraints has been hit in the last iteration, or it is the
        # initial iteration. The method must be restarted.
        if inext >= 0:
            # Pick the active set for the current trial step. The step provided
            # by the Goldfarb and Idnani algorithm is scaled to have length
            # 0.2 * delta, so that it is allowed by the linear constraints.
            sdd = getact(gq, evalc, (Aub,), resid, iact, mleq, nact, qfac, rfac,
                         delta)
            snorm = np.linalg.norm(sdd)
            if snorm <= .2 * tiny * delta:
                break
            sdd *= .2 * delta / snorm

            # If the modulus of the residual of an active constraint is
            # substantial, the search direction is the move towards the
            # boundaries of the active constraints.
            gamma = 0.
            if np.max(resid[iact[:nact]], initial=0.) > 1e-4 * delta:
                # Calculate the projection towards the boundaries of the active
                # constraints. The length of this step is computed hereinafter.
                temp = resid[iact[:nact]]
                for k in range(nact):
                    klec = mleq + k
                    temp[k] -= np.inner(rfac[mleq:klec, klec], temp[:k])
                    temp[k] /= rfac[klec, klec]
                sd = np.dot(qfac[:, mleq:mleq + nact], temp)

                # Determine the greatest steplength along the previously
                # calculated direction allowed by the trust-region constraint.
                rhs = delta ** 2. - np.inner(step + sdd, step + sdd)
                temp = np.inner(sd, step + sdd)
                sdsq = np.inner(sd, sd)
                if rhs > 0.:
                    sqrd = np.sqrt(sdsq * rhs + temp ** 2.)
                    if temp <= 0. and sdsq > tiny * abs(sqrd - temp):
                        gamma = max((sqrd - temp) / sdsq, 0.)
                    elif abs(sqrd + temp) > tiny * rhs:
                        gamma = max(rhs / (sqrd + temp), 0.)
                    else:
                        gamma = 1.

                # Reduce the steplength so that the move satisfies the nonactive
                # constraints. The active constraints are already satisfied.
                if gamma > 0.:
                    for i in range(mlub + 2 * n):
                        if i not in iact[:nact]:
                            asd = evalc(i, sd, Aub)
                            asdd = evalc(i, sdd, Aub)
                            if asd > tiny * abs(resid[i] - asdd):
                                temp = max((resid[i] - asdd) / asd, 0.)
                                gamma = min(gamma, temp)
                    gamma = min(gamma, 1.)

            # Set the new search direction. If the modulus of the residual of an
            # active constraint was substantial, an additional iteration must be
            # entertained as this direction is not determined by the quadratic
            # objective function to be minimized.
            sd = sdd + gamma * sd
            iterc = 0 if gamma <= 0. else -1
            alpbd = 1.

        # Set the steplength of the current search direction allowed by the
        # trust-region constraint. The calculations are stopped if no further
        # progress is possible in the current search direction, or if the
        # derivative term of the step is sufficiently small.
        iterc += 1
        rhs = delta ** 2. - stepsq
        if rhs <= 0.:
            break
        sdgq = np.inner(sd, gq)
        if sdgq >= 0.:
            break
        sdstep = np.inner(sd, step)
        sdsq = np.inner(sd, sd)
        sqrd = np.sqrt(sdsq * rhs + sdstep ** 2.)
        if sdstep <= 0. and sdsq > tiny * abs(sqrd - sdstep):
            alpht = max((sqrd - sdstep) / sdsq, 0.)
        elif abs(sqrd + sdstep) > tiny * rhs:
            alpht = max(rhs / (sqrd + sdstep), 0.)
        else:
            break
        alpha = alpht
        if -alpha * sdgq <= 1e-2 * reduct:
            break

        # Reduce the steplength if necessary to the value that minimizes the
        # quadratic function. The method do not require the objective function
        # to be positive semidefinite, so that the curvature of the model at the
        # current search direction may be negative, in which case the model is
        # not lower bounded.
        hsd = np.asarray(hessp(sd, *args))
        if hsd.dtype.kind in np.typecodes['AllInteger']:
            hsd = np.asarray(hsd, dtype=float)
        curv = np.inner(sd, hsd)
        if curv > tiny * abs(sdgq):
            alphm = max(-sdgq / curv, 0.)
        else:
            alphm = np.inf
        alpha = min(alpha, alphm)

        # Reduce the steplength if necessary to preserve feasibility.
        inext = -1
        asd = np.zeros_like(resid)
        alphf = np.inf
        for i in range(mlub + 2 * n):
            if i not in iact[:nact]:
                asd[i] = evalc(i, sd, Aub)
                if abs(asd[i]) > tiny * abs(resid[i]):
                    if alphf * asd[i] > resid[i]:
                        alphf = max(resid[i] / asd[i], 0.)
                        inext = i
        alpha = min(alpha, alphf)
        alpha = max(alpha, alpbd)
        alpha = min((alpha, alphm, alpht))
        if iterc == 0:
            alpha = min(alpha, 1.)

        # Make the actual conjugate gradient iteration. The max operators below
        # are crucial as they prevent numerical difficulties engendered by
        # computer rounding errors.
        step += alpha * sd
        stepsq = np.inner(step, step)
        gq += alpha * hsd
        for i in range(mlub + 2 * n):
            if i not in iact[:nact]:
                resid[i] = max(0., resid[i] - alpha * asd[i])
        if iterc == 0:
            resid[iact[:nact]] *= max(0., 1. - gamma)
        reduct -= alpha * (sdgq + .5 * alpha * curv)

        # If the step reached the boundary of the trust region or if the step
        # that would be obtained in the unconstrained case is insubstantial.,
        # the truncated conjugate gradient method must be stopped.
        if alpha >= alpht:
            break
        alphs = min(alphm, alpht)
        if -alphs * (sdgq + .5 * alphs * curv) <= 1e-2 * reduct:
            break

        # Restart the calculations if a new constraint has been hit.
        if inext >= 0:
            if stepsq <= .64 * delta ** 2.:
                continue
            break

        # Calculate next search direction, which is conjugate to the previous
        # one, except if iterc is zero, which occurs if the previous search
        # direction was not determined by the quadratic objective function to be
        # minimized but by the active constraints.
        sdu = gq
        if mleq + nact > 0:
            temp = np.dot(qfac[:, mleq + nact:].T, gq)
            sdu = np.dot(qfac[:, mleq + nact:], temp)
        if iterc == 0:
            beta = 0.
        else:
            beta = np.inner(sdu, hsd) / curv
        sd = beta * sd - sdu
        alpbd = 0.
    if reduct <= 0.:
        return np.zeros_like(step)
    return step


def evalc(i, x, Aub):
    """
    Evaluation of the left-hand side of a constraint.

    Parameters
    ----------
    i : int
        Index of the constraint to be evaluated.
    x : numpy.ndarray, shape (n,)
        Point at which the constraint is to be evaluated.
    Aub : numpy.ndarray, shape (mlub, n)
        Jacobian matrix of the linear inequality constraints. Each row of `Aub`
        stores the gradient of a linear inequality constraint.

    Returns
    -------
    float
        Value of the `i`-th constraint at `x`.
    """
    mlub, n = Aub.shape
    if i < mlub:
        return np.inner(Aub[i, :], x)
    elif i < mlub + n:
        return -x[i - mlub]
    else:
        return x[i - mlub - n]
