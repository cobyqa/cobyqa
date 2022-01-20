# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

from libc.stdlib cimport free, malloc
from libc.math cimport fabs, fmax, fmin, isfinite, sqrt

import numpy as np
cimport numpy as np
np.import_array()

# Avoid namespace lookup for NumPy types and array creation methods
from numpy import empty as np_empty
from numpy import zeros as np_zeros
from numpy import int32 as np_int32
from numpy import float64 as np_float64

from ._utils cimport dot, getact, inner, isact, absmax_array, max_array, min_array

def cpqp(double[:] xopt, double[::1, :] aub, double[:] bub, double[::1, :] aeq, double[:] beq, double[:] xl, double[:] xu, double delta, double mu, bint debug):
    """
    Minimize approximately a convex piecewise quadratic function subject to
    bound and trust-region constraints using a truncated conjugate gradient.
    """
    cdef int n = xopt.shape[0]
    cdef int mlub = aub.shape[0]
    cdef int mleq = aeq.shape[0]

    # Shift the constraints to carry out all calculations at the origin.
    dot(aub, xopt, bub, 'n', -1.0, 1.0)  # noqa
    dot(aeq, xopt, beq, 'n', -1.0, 1.0)  # noqa
    cdef Py_ssize_t i
    for i in range(n):
        xl[i] -= xopt[i]
        xu[i] -= xopt[i]

    # Ensure the feasibility of the initial guess.
    cdef double eps = np.finfo(np_float64).eps
    cdef double tiny = np.finfo(np_float64).tiny
    cdef double huge = np.finfo(np_float64).max
    cdef double tol = 10.0 * eps * float(n)
    cdef double bdtol = tol * fmax(absmax_array(xl, 1.0), absmax_array(xu, 1.0))
    if debug:
        if max_array(xl) > bdtol:
            raise ValueError('Constraint xl <= xopt fails initially.')
        if min_array(xu) < -bdtol:
            raise ValueError('Constraint xopt <= xu fails initially.')
        if not isfinite(delta) or delta <= 0.0:
            raise ValueError('Constraint delta > 0 fails initially.')

    # Normalize the linear constraints of the reformulated problem. The bound
    # constraints are already normalized.
    cdef double[:] lcn = np.empty(mlub, dtype=np_float64)
    cdef Py_ssize_t j
    for i in range(mlub):
        lcn[i] = inner(aub[i, :], aub[i, :]) + 1.0
        lcn[i] = sqrt(lcn[i])
        for j in range(n):
            aub[i, j] /= lcn[i]
        bub[i] /= lcn[i]

    # Set the initial active set to the empty set, and calculate the normalized
    # residuals of the constraints at the origin. The residuals of the linear
    # equality constraints are not maintained as the method ensures that the
    # search directions lies in the linear space they span.
    cdef int* nact = <int *> malloc(sizeof(int))
    nact[0] = 0
    cdef int[:] iact = np_empty(mlub + n, dtype=np_int32)
    cdef double[::1, :] qfac = np_zeros((mlub + n, mlub + n), dtype=np_float64, order='F')
    cdef double[::1, :] rfac = np_empty((mlub + n, mlub + n), dtype=np_float64, order='F')
    for i in range(mlub + n):
        qfac[i, i] = 1.0

    # Calculate the gradient of the objective function of the reformulated
    # problem and the normalized the right-hand side of the linear inequality
    # constraints and the residuals of the constraints at the initial guess.
    cdef double[:] gq = np.empty(mlub + n)
    cdef double[:] resid = np_empty(2 * (mlub + n), dtype=np_float64)
    for i in range(mlub + n):
        if i < n:
            gq[i] = -inner(aeq[:, i], beq)
        else:
            gq[i] = fmax(0.0, -lcn[i - n] * bub[i - n])
    for i in range(2 * (mlub + n)):
        if i < mlub:
            resid[i] = fmax(0.0, bub[i] + gq[n + i] / lcn[i])
        elif i < mlub + n:
            resid[i] = fmax(0.0, -xl[i - mlub])
        elif i < 2 * mlub + n:
            resid[i] = fmax(0.0, gq[i - mlub])
        else:
            resid[i] = fmax(0.0, xu[i - 2 * mlub - n])

    # Start the iterative calculations. The truncated conjugate gradient method
    #     # should be stopped after n - nact iterations, except if a new constraint
    #     # has been hit in the last iteration, in which case the method is restarted
    #     # and the iteration counter reinitialized.
    cdef double[:] step = np_zeros(n, dtype=np_float64)
    cdef double[:] sd = np_zeros(mlub + n, dtype=np_float64)
    cdef double[:] sdd = np_zeros(mlub + n, dtype=np_float64)
    cdef double[:] vecta = np_empty(max(mlub + n, mleq), dtype=np_float64)
    cdef double[:] vectb = np_empty(mlub + n, dtype=np_float64)
    cdef double[:] hsd = np_empty(mlub + n, dtype=np_float64)
    cdef double[:] asd = np_empty(2 * (mlub + n), dtype=np_float64)
    cdef double[:] asdd = np_empty(2 * (mlub + n), dtype=np_float64)
    cdef double reduct = 0.0
    cdef double stepsq = 0.0
    cdef double alpbd = 1.0
    cdef double gamma = 0.0
    cdef int inext = 0
    cdef int ncall = 0
    cdef int iterc = 0
    cdef double alpha, alphm, alpht
    cdef double curv, resmax, rhs, sdgq, sdsq, sdstep, snorm, sqrd, temp
    cdef tuple args = (aub, lcn)
    while iterc < mlub + n - nact[0] or inext >= 0:
        # A new constraints has been hit in the last iteration, or it is the
        # initial iteration. The method must be restarted.
        if inext >= 0:
            # Pick the active set for the current trial step. The step provided
            # by the Goldfarb and Idnani algorithm is scaled to have length
            # 0.2 * delta, so that it is allowed by the linear constraints.
            getact(gq, evalc, resid, iact, 0, nact, qfac, rfac, delta, mu, args, sdd)
            ncall += 1
            snorm = inner(sdd, sdd)
            snorm = sqrt(snorm)
            if snorm <= mu * tiny * delta:
                break
            for i in range(mlub + n):
                sdd[i] *= mu * delta / snorm

            # If the modulus of the residual of an active constraint is
            # substantial, the search direction is the move towards the
            # boundaries of the active constraints.
            gamma = 0.0
            resmax = 0.0
            for k in range(nact[0]):
                resmax = fmax(resmax, resid[iact[k]])
            if resmax > 1e-4 * delta:  # TODO: Change to 0
                # Calculate the projection towards the boundaries of the active
                # constraints. The length of this step is computed hereinafter.
                for k in range(nact[0]):
                    vecta[k] = resid[iact[k]]
                    vecta[k] -= inner(rfac[:k, k], vecta[:k])
                    vecta[k] /= rfac[k, k]
                dot(qfac[:, :nact[0]], vecta[:nact[0]], sd, 'n', 1.0, 0.0)  # noqa

                # Determine the greatest steplength along the previously
                # calculated direction allowed by the trust-region constraint.
                rhs = delta ** 2.0
                temp = 0.0
                sdsq = 0.0
                for i in range(n):
                    rhs -= (step[i] + sdd[i]) ** 2.0
                    temp += sd[i] * (step[i] + sdd[i])
                    sdsq += sd[i] ** 2.0
                if rhs > 0.0:
                    sqrd = sqrt(sdsq * rhs + temp ** 2.0)
                    if temp <= 0.0 and sdsq > tiny * fabs(sqrd - temp):
                        gamma = fmax((sqrd - temp) / sdsq, 0.0)
                    elif fabs(sqrd + temp) > tiny * rhs:
                        gamma = fmax(rhs / (sqrd + temp), 0.0)
                    else:
                        gamma = 1.0

                # Reduce the steplength so that the move satisfies the nonactive
                # constraints. The active constraints are already satisfied.
                if gamma > 0.0:
                    for i in range(2 * (mlub + n)):
                        if not isact(i, iact, nact):
                            asd[i] = evalc(i, sd, args)
                            asdd[i] = evalc(i, sdd, args)
                            if asd[i] > fmax(tiny * fabs(resid[i] - asdd[i]), bdtol):
                                temp = fmax((resid[i] - asdd[i]) / asd[i], 0.0)
                                gamma = fmin(gamma, temp)
                    gamma = fmin(gamma, 1.0)

            # Set the new search direction. If the modulus of the residual of an
            # active constraint was substantial, an additional iteration must be
            # entertained as this direction is not determined by the quadratic
            # objective function to be minimized.
            for i in range(mlub + n):
                sd[i] = sdd[i] + gamma * sd[i]
            if gamma <= 0.0:
                iterc = 0
            else:
                iterc = -1
            alpbd = 1.0

        # Set the steplength of the current search direction allowed by the
        # trust-region constraint. The calculations are stopped if no further
        # progress is possible in the current search direction, or if the
        # derivative term of the step is sufficiently small.
        iterc += 1
        rhs = delta ** 2.0 - stepsq
        if rhs <= 0.0:
            break
        sdgq = inner(sd, gq)
        if sdgq >= -bdtol:
            break
        sdstep = inner(sd[:n], step)
        sdsq = inner(sd[:n], sd[:n])
        sqrd = sqrt(sdsq * rhs + sdstep ** 2.0)
        if sdstep <= 0.0 and sdsq > tiny * fabs(sqrd - sdstep):
            alpht = fmax((sqrd - sdstep) / sdsq, 0.0)
        elif fabs(sqrd + sdstep) > tiny * rhs:
            alpht = fmax(rhs / (sqrd + sdstep), 0.0)
        else:
            alpht = huge
        alpha = alpht
        if -alpha * sdgq <= 1e-2 * reduct:
            break

        # Reduce the steplength if necessary to the value that minimizes the
        # quadratic function. The method do not require the objective function
        # to be positive semidefinite, so that the curvature of the model at the
        # current search direction may be negative, in which case the model is
        # not lower bounded.
        dot(aeq, sd[:n], vecta[:mleq], 'n', 1.0, 0.0)
        dot(aeq, vecta[:mleq], hsd[:n], 't', 1.0, 0.0)
        hsd[n:] = sd[n:]
        curv = inner(sd, hsd)
        if curv >= huge:
            alphm = 0.0
        elif curv > tiny * fabs(sdgq):
            alphm = fmax(-sdgq / curv, 0.0)
        else:
            alphm = huge
        alpha = fmin(alpha, alphm)

        # Reduce the steplength if necessary to preserve feasibility.
        inext = -1
        alphf = huge
        for i in range(2 * (mlub + n)):
            if not isact(i, iact, nact):
                asd[i] = evalc(i, sd, args)
                if fabs(asd[i]) > tiny * fabs(resid[i]):
                    if alphf * asd[i] > resid[i] and asd[i] > bdtol:
                        alphf = fmax(resid[i] / asd[i], 0.0)
                        inext = i
            else:
                asd[i] = 0.0
        if alphf < alpha:
            alpha = alphf
        else:
            inext = -1
        alpha = fmax(alpha, alpbd)
        alpha = fmin(alpha, fmin(alphm, alpht))
        if iterc == 0:
            alpha = fmin(alpha, 1.0)
        if alpha >= huge:
            break

        # Make the actual conjugate gradient iteration. The max operators below
        # are crucial as they prevent numerical difficulties engendered by
        # computer rounding errors.
        if alpha > 0.0:
            stepsq = 0.0
            for i in range(mlub + n):
                if i < n:
                    step[i] += alpha * sd[i]
                    stepsq += step[i] ** 2.0
                gq[i] += alpha * hsd[i]
            for i in range(2 * (mlub + n)):
                if not isact(i, iact, nact):
                    resid[i] = fmax(0.0, resid[i] - alpha * asd[i])
            reduct -= alpha * (sdgq + 0.5 * alpha * curv)
        if iterc == 0:
            for k in range(nact[0]):
                resid[iact[k]] *= fmax(0.0, 1.0 - gamma)

        # If the step that would be obtained in the unconstrained case is
        # insubstantial, the truncated conjugate gradient method is stopped.
        alphs = fmin(alphm, alpht)
        if -alphs * (sdgq + 0.5 * alphs * curv) <= 1e-2 * reduct:
            break

        # Prevent infinite cycling due to computer rounding errors.
        if ncall > min(10000, 100 * (mlub + n) ** 2):
            break

        # Restart the calculations if a new constraint has been hit.
        if inext >= 0:
            continue

        # If the step reached the boundary of the trust region, the truncated
        # conjugate gradient method is stopped.
        if alpha >= alpht:
            break

        # Calculate next search direction, which is conjugate to the previous
        # one, except if iterc is zero, which occurs if the previous search
        # direction was not determined by the quadratic objective function to be
        # minimized but by the active constraints.
        vecta[:mlub + n] = gq
        if nact[0] > 0:
            dot(qfac[:, nact[0]:], vecta[:mlub + n], vectb[:mlub + n - nact[0]], 't', 1.0, 0.0)  # noqa
            dot(qfac[:, nact[0]:], vectb[:mlub + n - nact[0]], vecta[:mlub + n], 'n', 1.0, 0.0)  # noqa
        if iterc == 0:
            beta = 0.0
        else:
            beta = inner(vecta[:mlub + n], hsd) / curv
        for i in range(mlub + n):
            sd[i] = beta * sd[i] - vecta[i]
        alpbd = 0.0
    free(nact)

    # To prevent numerical difficulties emerging from computer rounding errors
    # on ill-conditioned problems, the reduction is computed from scratch.
    reduct = 0.0
    for i in range(mlub):
        temp = -bub[i]
        reduct += fmax(0.0, temp) ** 2.0
        temp += inner(aub[i, :], step)
        reduct -= fmax(0.0, temp) ** 2.0
    for i in range(mleq):
        temp = -beq[i]
        reduct += temp ** 2.0
        temp += inner(aeq[i, :], step)
        reduct -= temp ** 2.0
    reduct *= 0.5
    if reduct <= 0.0:
        step[:] = 0.0
    return step


cdef double evalc(int i, double[:] x, tuple args):
    """
    Evaluate the `i`-th constraint of the problem at `x`.
    """
    cdef double[::1, :] aub = args[0]
    cdef double[:] lcn = args[1]
    cdef int mlub = aub.shape[0]
    cdef int n = aub.shape[1]
    cdef double cx
    if i < mlub:
        cx = inner(aub[i, :], x[:n]) - x[n + i] / lcn[i]
    elif i < 2 * mlub + n:
        cx = -x[i - mlub]
    else:
        cx = x[i - 2 * mlub - n]
    return cx
