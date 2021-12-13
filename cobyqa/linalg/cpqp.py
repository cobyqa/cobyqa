import numpy as np
from numpy.testing import assert_

from .utils import getact, get_bdtol


def cpqp(xopt, Aub, bub, Aeq, beq, xl, xu, delta, **kwargs):
    r"""
    Minimize approximately a convex piecewise quadratic function subject to
    bound and trust-region constraints using a truncated conjugate gradient.

    The method minimizes the function

    .. math::

        \frac{1}{2} ( \| [ \mathtt{Aub} \times x - \mathtt{bub} ]_+\|_2^2 +
        \| \mathtt{Aeq} \times x - \mathtt{beq} \|_2^2 ),

    where :math:`[ \cdot ]_+` denotes the componentwise positive part operator.

    Parameters
    ----------
    xopt : numpy.ndarray, shape (n,)
        Center of the trust-region constraint.
    Aub : array_like, shape (mlub, n)
        Matrix `Aub` as shown above.
    bub : array_like, shape (mlub,)
        Vector `bub` as shown above.
    Aeq : array_like, shape (mleq, n)
        Matrix `Aeq` as shown above.
    beq : array_like, shape (meq,)
        Vector `beq` as shown above.
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
    bvtcg : Bounded variable truncated conjugate gradient
    lctcg : Linear constrained truncated conjugate gradient
    nnls : Nonnegative least squares

    Notes
    -----
    The method is adapted from the TRSTEP algorithm [1]_. To cope with the
    convex piecewise quadratic objective function, the method minimizes

    .. math::

        \frac{1}{2} ( \| \mathtt{Aeq} \times x - \mathtt{beq} \|_2^2 +
        \| y \|_2^2 )

    subject to the original constraints, where the slack variable :math:`y` is
    lower bounded by zero and :math:`\mathtt{Aub} \times x - \mathtt{bub}`.

    References
    ----------
    .. [1] M. J. D. Powell. "On fast trust region methods for quadratic models
       with linear constraints." In: Math. Program. Comput. 7 (2015), pp.
       237–-267.
    """
    xopt = np.atleast_1d(xopt)
    if xopt.dtype.kind in np.typecodes['AllInteger']:
        xopt = np.asarray(xopt, dtype=float)
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
    tiny = np.finfo(float).tiny
    mlub, n = Aub.shape
    bdtol = get_bdtol(xl, xu, **kwargs)

    # Shift the constraints to carry out all calculations at the origin.
    bub -= np.dot(Aub, xopt)
    beq -= np.dot(Aeq, xopt)
    xl -= xopt
    xu -= xopt

    # Ensure the feasibility of the initial guess.
    if kwargs.get('debug', False):
        assert_(np.max(xl) < bdtol)
        assert_(np.min(xu) > -bdtol)
        assert_(np.isfinite(delta))
        assert_(delta > 0.0)

    # Normalize the linear constraints of the reformulated problem. The
    # right-hand side of the linear inequality constraints is normalized
    # hereinafter as the original values are required to evaluate the gradient
    # of the objective function of the reformulated problem. The bound
    # constraints are already normalized.
    lcn = np.sqrt(np.sum(np.square(Aub), axis=1) + 1.0)
    Aub = Aub / lcn[:, np.newaxis]

    # Set the initial active set to the empty set.
    nact = np.array(0, dtype=int)
    iact = np.empty(mlub + n, dtype=int)
    rfac = np.zeros((mlub + n, mlub + n), dtype=float)
    qfac = np.eye(mlub + n, dtype=float)

    # Calculate the gradient of the objective function of the reformulated
    # problem and the normalized the right-hand side of the linear inequality
    # constraints and the residuals of the constraints at the initial guess.
    gq = np.r_[np.dot(Aeq.T, -beq), np.maximum(0.0, -bub)]
    bub /= lcn
    resid = np.r_[bub + gq[n:] / lcn, -xl, gq[n:], xu]
    resid = np.maximum(0.0, resid)

    # Start the iterative calculations. The truncated conjugate gradient method
    # should be stopped after n - nact iterations, except if a new constraint
    # has been hit in the last iteration, in which case the method is restarted
    # and the iteration counter reinitialized.
    step = np.zeros_like(xopt)
    sd = np.zeros_like(gq)
    mu1 = kwargs.get('mu1', 0.2)
    reduct = 0.0
    stepsq = 0.0
    alpbd = 1.0
    inext = 0
    ncall = 0
    iterc = 0
    gamma = 0.0
    while iterc < mlub + n - nact or inext >= 0:
        # A new constraints has been hit in the last iteration, or it is the
        # initial iteration. The method must be restarted.
        if inext >= 0:
            # Pick the active set for the current trial step. The step provided
            # by the Goldfarb and Idnani algorithm is scaled to have length
            # 0.2 * delta, so that it is allowed by the linear constraints.
            sdd = getact(gq, evalc, resid, iact, 0, nact, qfac, rfac,
                         delta, Aub, lcn, **kwargs)
            snorm = np.linalg.norm(sdd)
            ncall += 1
            if snorm <= mu1 * tiny * delta:
                break
            sdd *= mu1 * delta / snorm

            # If the modulus of the residual of an active constraint is
            # substantial, the search direction is the move towards the
            # boundaries of the active constraints.
            gamma = 0.
            if np.max(resid[iact[:nact]], initial=0.0) > 0.0:
                # Calculate the projection towards the boundaries of the active
                # constraints. The length of this step is computed hereinafter.
                temp = resid[iact[:nact]]
                for k in range(nact):
                    temp[k] -= np.inner(rfac[:k, k], temp[:k])
                    temp[k] /= rfac[k, k]
                sd = np.dot(qfac[:, :nact], temp)

                # Determine the greatest steplength along the previously
                # calculated direction allowed by the trust-region constraints.
                rhs = delta ** 2.0 - np.inner(step + sdd[:n], step + sdd[:n])
                temp = np.inner(sd[:n], step + sdd[:n])
                sdsq = np.inner(sd[:n], sd[:n])
                if rhs > 0.0:
                    sqrd = np.sqrt(sdsq * rhs + temp ** 2.0)
                    if temp <= 0.0 and sdsq > tiny * abs(sqrd - temp):
                        gamma = (sqrd - temp) / sdsq
                    elif abs(sqrd + temp) > tiny * rhs:
                        gamma = rhs / (sqrd + temp)
                    else:
                        gamma = 1.0

                # Reduce the steplength so that the move satisfies the nonactive
                # constraints. The active constraints are already satisfied.
                if gamma > 0.0:
                    for i in range(2 * (mlub + n)):
                        if i not in iact[:nact]:
                            asd = evalc(i, sd, Aub, lcn)
                            asdd = evalc(i, sdd, Aub, lcn)
                            if asd > max(tiny * abs(resid[i] - asdd), bdtol):
                                temp = max((resid[i] - asdd) / asd, 0.0)
                                gamma = min(gamma, temp)
                    gamma = min(gamma, 1.0)

            # Set the new search direction. If the modulus of the residual of an
            # active constraint was substantial, an additional iteration must be
            # entertained as this direction is not determined by the quadratic
            # objective function to be minimized.
            sd = sdd + gamma * sd
            iterc = 0 if gamma <= 0.0 else -1
            alpbd = 1.0

        # Set the steplength of the current search direction allowed by the
        # trust-region constraints. The calculations are stopped if no further
        # progress is possible in the current search direction or if the
        # derivative term of the step is sufficiently small.
        iterc += 1
        rhs = delta ** 2.0 - stepsq
        if rhs <= 0.0:
            break
        sdgq = np.inner(sd, gq)
        if sdgq >= 0.0:
            break
        sdstep = np.inner(sd[:n], step)
        sdsq = np.inner(sd[:n], sd[:n])
        sqrd = np.sqrt(sdsq * rhs + sdstep ** 2.0)
        if sdstep <= 0.0 and sdsq > tiny * abs(sqrd - sdstep):
            alpht = (sqrd - sdstep) / sdsq
        elif abs(sqrd + sdstep) > tiny * rhs:
            alpht = rhs / (sqrd + sdstep)
        else:
            alpht = np.inf
        alpha = alpht
        if -alpha * sdgq <= 1e-2 * reduct:
            break

        # Reduce the steplength if necessary to the value that minimizes the
        # quadratic function. The method do not require the objective function
        # to be positive semidefinite, so that the curvature of the model at the
        # current search direction may be negative, in which case the model is
        # not lower bounded.
        hsd = np.r_[np.dot(Aeq.T, np.dot(Aeq, sd[:n])), sd[n:]]
        curv = np.inner(sd, hsd)
        if curv == np.inf:
            alphm = 0.0
        elif curv > tiny * abs(sdgq):
            alphm = max(-sdgq / curv, 0.0)
        else:
            alphm = np.inf

        # Reduce the steplength if necessary to preserve feasibility.
        inext = -1
        asd = np.zeros_like(resid)
        alphf = np.inf
        for i in range(2 * (mlub + n)):
            if i not in iact[:nact]:
                asd[i] = evalc(i, sd, Aub, lcn)
                if abs(asd[i]) > tiny * abs(resid[i]):
                    if alphf * asd[i] > resid[i] and asd[i] > bdtol:
                        alphf = resid[i] / asd[i]
                        inext = i
        if alphf < alpha:
            alpha = alphf
        else:
            inext = -1
        alpha = max(alpha, alpbd)
        alpha = min((alpha, alphm, alpht))
        if iterc == 0:
            alpha = min(alpha, 1.0)
        if alpha == np.inf:
            break

        # Make the actual conjugate gradient iteration. The max operators below
        # are crucial as they prevent numerical difficulties engendered by
        # computer rounding errors.
        if alpha > 0.0:
            step += alpha * sd[:n]
            stepsq = np.inner(step, step)
            gq += alpha * hsd
            for i in range(2 * (mlub + n)):
                if i not in iact[:nact]:
                    resid[i] = max(0.0, resid[i] - alpha * asd[i])
            reduct -= alpha * (sdgq + 0.5 * alpha * curv)
        if iterc == 0:
            resid[iact[:nact]] *= max(0.0, 1.0 - gamma)

        # If the step that would be obtained in the unconstrained case is
        # insubstantial, the truncated conjugate gradient method is stopped.
        alphs = min(alphm, alpht)
        if -alphs * (sdgq + 0.5 * alphs * curv) <= 1e-2 * reduct:
            break

        # Prevent infinite cycling due to computer rounding errors.
        if ncall > min(10000, 200 * (mlub + n) ** 2):
            break

        # Restart the calculations if a new constraint has been hit.
        if inext >= 0:
            continue

        # If the step reached the boundary of a trust region or if the step that
        # would be obtained in the unconstrained case is insubstantial, the
        # truncated conjugate gradient method is stopped.
        if alpha >= alpht:
            break

        # Calculate next search direction, which is conjugate to the previous
        # one, except if iterc is zero, which occurs if the previous search
        # direction was not determined by the quadratic objective function to be
        # minimized but by the active constraints.
        sdu = gq
        if nact > 0:
            temp = np.dot(qfac[:, nact:].T, gq)
            sdu = np.dot(qfac[:, nact:], temp)
        if iterc == 0:
            beta = 0.0
        else:
            beta = np.inner(sdu, hsd) / curv
        sd = beta * sd - sdu
        alpbd = 0.0

    # To prevent numerical difficulties emerging from computer rounding errors
    # on ill-conditioned problems, the reduction is computed from scratch.
    resid_ub = np.maximum(-bub, 0.0)
    reduct = np.inner(resid_ub, resid_ub) + np.inner(beq, beq)
    resid_ub = np.maximum(np.dot(Aub, step) - bub, 0.0)
    resid_eq = np.dot(Aeq, step) - beq
    reduct -= np.inner(resid_ub, resid_ub) + np.inner(resid_eq, resid_eq)
    reduct *= 0.5
    if reduct <= 0.0:
        return np.zeros_like(step)
    return step


def evalc(i, x, Aub, lcn):
    """
    Evaluation of the left-hand side of a constraint.

    Parameters
    ----------
    i : int
        Index of the constraint to be evaluated.
    x : numpy.ndarray, shape (mlub + n,)
        Point at which the constraint is to be evaluated.
    Aub : numpy.ndarray, shape (mlub, n)
        Jacobian matrix of the linear inequality constraints. Each row of `Aub`
        stores the gradient of a linear inequality constraint.
    lcn : numpy.ndarray, shape (mlub,)
        Normalization factors of the linear inequality constraints.

    Returns
    -------
    float
        Value of the `i`-th constraint at `x`.
    """
    mlub, n = Aub.shape
    if i < mlub:
        return np.inner(Aub[i, :], x[:n]) - x[n + i] / lcn[i]
    elif i < 2 * mlub + n:
        return -x[i - mlub]
    else:
        return x[i - 2 * mlub - n]
