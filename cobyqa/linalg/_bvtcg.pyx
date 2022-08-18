# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

from libc.math cimport fabs, fmax, fmin, isfinite, sqrt

import numpy as np
cimport numpy as np
np.import_array()

# Avoid namespace lookup for NumPy types and array creation methods
from numpy import empty as np_empty
from numpy import zeros as np_zeros
from numpy import int32 as np_int32
from numpy import float64 as np_float64

from ._utils cimport inner, absmax_array, max_array, min_array


def bvtcg(double[:] xopt, double[:] gq, object hessp, double[:] xl, double[:] xu, double delta, bint debug, bint improve_tcg):
    """
    Minimize approximately a quadratic function subject to bound and
    trust-region constraints using a truncated conjugate gradient.
    """
    cdef int n = gq.shape[0]

    # Shift the bounds to carry out all calculations at the origin.
    cdef Py_ssize_t i
    for i in range(n):
        xl[i] -= xopt[i]
        xu[i] -= xopt[i]

    # Ensure the feasibility of the initial guess.
    cdef double eps = np.finfo(np_float64).eps
    cdef double tiny = np.finfo(np_float64).tiny
    cdef double huge = np.finfo(np_float64).max
    cdef double tol = 10.0 * eps * float(n)
    cdef double bdtol
    if debug:
        bdtol = tol * fmax(absmax_array(xl, 1.0), absmax_array(xu, 1.0))
        if max_array(xl) > bdtol:
            raise ValueError('Constraint xl <= xopt fails initially.')
        if min_array(xu) < -bdtol:
            raise ValueError('Constraint xopt <= xu fails initially.')
        if not isfinite(delta) or delta <= 0.0:
            raise ValueError('Constraint delta > 0 fails initially.')
    for i in range(n):
        xl[i] = fmin(xl[i], 0.0)
        xu[i] = fmax(xu[i], 0.0)

    # Initialize the working sets and the trial step. The vector xbdi stores the
    # working sets related to the bounds, where the value 0 indicates that the
    # component is not restricted by the bounds, the value -1 indicates that the
    # component is fixed by the lower bound, and the value 1 indicates that the
    # component is fixed by the upper bound.
    cdef double[:] step = np_zeros(n, dtype=np_float64)
    cdef int[:] xbdi = np_zeros(n, dtype=np_int32)
    cdef int nact = 0
    for i in range(n):
        if step[i] <= xl[i] and gq[i] >= 0.0:
            xbdi[i] = -1
            nact += 1
        elif step[i] >= xu[i] and gq[i] <= 0.0:
            xbdi[i] = 1
            nact += 1

    # Start the iterative procedure.
    cdef double[:] sd = np_zeros(n, dtype=np_float64)
    cdef double[:] hsd = np_empty(n, dtype=np_float64)
    cdef double[:] hst = np_empty(n, dtype=np_float64)
    cdef double stepsq = 0.0
    cdef double beta = 0.0
    cdef double reduct = 0.0
    cdef int iterc = 0
    cdef int maxiter = n
    cdef double alpha, alpht, sdgq, sdsq, sdstep, rhs, temp, tempa, tempb
    cdef double angt, cosv, gqsq, gqstep, sinv
    cdef double curv, sdhsd, ssq, sthsd, sthst
    cdef double redmax, redsav, rdnext, rdprev
    cdef int xsav, ialt, isav
    while delta ** 2.0 - stepsq > tol * absmax_array(sd, 1.0):
        # Set the next search direction of the conjugate gradient method to the
        # steepest descent direction initially and when the iterations are
        # restarted because a variable has just been fixed by a bound. The
        # maximum number of iterations is set to the theoretical upper bound on
        # the iteration number initially and at a restart. The computations are
        # stopped if no further progress is possible.
        sdsq = 0.0
        gqsq = 0.0
        sdgq = 0.0
        sdstep = 0.0
        for i in range(n):
            if xbdi[i] == 0:
                sd[i] = beta * sd[i] - gq[i]
                sdsq += sd[i] ** 2.0
                gqsq += gq[i] ** 2.0
                sdgq += sd[i] * gq[i]
                sdstep += sd[i] * step[i]
            else:
                sd[i] = 0.0
        if sqrt(sdsq) < tol * absmax_array(sd, 1.0):
            break
        if beta <= 0.0:
            maxiter = iterc + n - nact
        if gqsq * delta ** 2.0 <= 1e-4 * reduct ** 2.0:
            break

        # Set the steplength of the current search direction allowed by the
        # trust-region constraint. The calculations are stopped if no further
        # progress is possible in the current search direction.
        iterc += 1
        rhs = delta ** 2.0 - stepsq
        if rhs <= 0.0:
            continue
        if sdgq >= 0.0:
            break
        sqrd = sqrt(sdsq * rhs + sdstep ** 2.0)
        if sdstep <= 0.0 and sdsq > tiny * fabs(sqrd - sdstep):
            alpht = fmax((sqrd - sdstep) / sdsq, 0.0)
        elif fabs(sqrd + sdstep) > tiny * rhs:
            alpht = fmax(rhs / (sqrd + sdstep), 0.0)
        else:
            break
        alpha = alpht

        # Reduce the steplength if necessary to the value that minimizes the
        # quadratic function. The method do not require the objective function
        # to be positive semidefinite, so that the curvature of the model at the
        # current search direction may be negative, in which case the model is
        # not lower bounded.
        hsd = hessp(sd)
        curv = inner(sd, hsd)
        if curv >= huge:
            alpha = 0.0
        elif curv > tiny * fabs(sdgq):
            alpha = fmin(alpha, fmax(-sdgq / curv, 0.0))

        # Reduce the steplength if necessary in order to preserve the simple
        # bounds, setting the index of the new constrained variable if any.
        inew = -1
        for i in range(n):
            if sd[i] > tiny * fabs(xu[i] - step[i]):
                temp = (xu[i] - step[i]) / sd[i]
            elif sd[i] < -tiny * fabs(step[i] - xl[i]):
                temp = (xl[i] - step[i]) / sd[i]
            else:
                temp = huge
            if temp < alpha:
                alpha = temp
                inew = i

        # Make the actual conjugate gradient iteration.
        if alpha > 0.0:
            stepsq = 0.0
            for i in range(n):
                step[i] += alpha * sd[i]
                gq[i] += alpha * hsd[i]
                stepsq += step[i] ** 2.0
            reduct -= alpha * (sdgq + 0.5 * alpha * curv)

        # If the step reached the boundary of the trust region, the truncated
        # conjugate gradient method must be stopped.
        if alpha >= alpht:
            stepsq = delta ** 2.0
            continue

        # Restart the conjugate gradient method if it has hit a new bound. If
        # the step is on or outside the trust region, further improvement
        # attempts round the trust-region boundary are done below.
        if inew >= 0:
            if sd[inew] < 0.0:
                xbdi[inew] = -1
            else:
                xbdi[inew] = 1
            nact += 1
            beta = 0.0
            continue

        # If the step did not reach the trust-region boundary, apply another
        # conjugate gradient iteration or return if the maximum number of
        # iterations is exceeded.
        if iterc >= maxiter:
            break
        beta = 0.0
        for i in range(n):
            if xbdi[i] == 0:
                beta += gq[i] * hsd[i]
        beta /= curv
        continue
    else:
        # Whenever the truncated conjugate gradient computations stopped because
        # the current trial step hit the trust-region boundary, a search is
        # performed to attempt improving the solution round the trust-region
        # boundary on the two-dimensional space spanned by the free components
        # of the step and the gradient.
        iterc = 0
        inew = -1
        while improve_tcg and nact < n - 1:
            # Whenever the previous iteration has hit a bound, the computation
            # round the trust-region boundary are restarted.
            stepsq = 0.0
            gqstep = 0.0
            gqsq = 0.0
            for i in range(n):
                if xbdi[i] == 0:
                    stepsq += step[i] ** 2.0
                    gqstep += gq[i] * step[i]
                    gqsq += gq[i] ** 2.0
                if iterc == 0 or inew >= 0:
                    if xbdi[i] == 0:
                        sd[i] = step[i]
                    else:
                        sd[i] = 0.0
            if iterc == 0 or inew >= 0:
                hsd = hessp(sd)
                for i in range(n):
                    hst[i] = hsd[i]

            # Let the search direction be a linear combination of the reduced
            # step and the reduced gradient that is orthogonal to the step.
            iterc += 1
            temp = gqsq * stepsq - gqstep ** 2.0
            if temp <= 1e-4 * reduct ** 2.0:
                break
            sdgq = -sqrt(temp)
            for i in range(n):
                if xbdi[i] == 0:
                    sd[i] = (stepsq * gq[i] - gqstep * step[i]) / sdgq
                else:
                    sd[i] = 0.0

            # By considering the simple bounds on the variables, calculate an
            # upper bound on the tangent of half the angle of the alternative
            # iteration and restart the alternative iterations if a free
            # variable has reached a new bound.
            angbd = 1.0
            inew = -1
            xsav = 0
            for i in range(n):
                if xbdi[i] == 0:
                    tempa = step[i] - xl[i]
                    if tempa <= 0.0:
                        nact += 1
                        inew = i
                        xbdi[inew] = -1
                        continue
                    tempb = xu[i] - step[i]
                    if tempb <= 0.0:
                        nact += 1
                        inew = i
                        xbdi[inew] = 1
                        continue
                    ssq = step[i] ** 2.0 + sd[i] ** 2.0
                    temp = ssq - xl[i] ** 2.0
                    if temp > 0.0:
                        temp = sqrt(temp) - sd[i]
                        if angbd * temp > tempa:
                            angbd = tempa / temp
                            inew = i
                            xsav = -1
                    temp = ssq - xu[i] ** 2.0
                    if temp > 0.0:
                        temp = sqrt(temp) + sd[i]
                        if angbd * temp > tempb:
                            angbd = tempb / temp
                            inew = i
                            xsav = 1

            # Calculate the necessary curvatures for the alternative iteration.
            hsd = hessp(sd)
            sdhsd = 0.0
            sthsd = 0.0
            sthst = 0.0
            for i in range(n):
                if xbdi[i] == 0:
                    sdhsd += sd[i] * hsd[i]
                    sthsd += step[i] * hsd[i]
                    sthst += step[i] * hst[i]

            # Seek the greatest reduction in the objective function for a range
            # of equally spaced values in [0, angbd], corresponding to the
            # tangent of half the angle of the alternative iteration. For the
            # computations to hold, the interval should be split into at least
            # three parts, and nalt represents the number of intervals in the
            # unconstrained case.
            redmax = 0.0
            redsav = 0.0
            rdprev = 0.0
            rdnext = 0.0
            isav = -1
            ialt = int(17 * angbd + 3.1)
            for i in range(ialt):
                angt = angbd * float(i + 1) / float(ialt)
                sinv = 2.0 * angt / (1.0 + angt ** 2.0)
                temp = sdhsd + angt * (sthst * angt - 2.0 * sthsd)
                rednew = sinv * (gqstep * angt - sdgq - 0.5 * sinv * temp)
                if rednew > redmax:
                    redmax = rednew
                    rdprev = redsav
                    isav = i
                elif i == isav + 1:
                    rdnext = rednew
                redsav = rednew

            # Set the sine and cosine of the angle of the alternative iteration,
            # and return if no reduction is possible. The computations are
            # stopped if either no further reduction can be obtained on the
            # sampling of the trust-region boundary or if no further progress on
            # the step is possible.
            if isav == -1:
                break
            angt = angbd
            if isav < ialt - 1:
                temp = (rdnext - rdprev) / (2.0 * redmax - rdprev - rdnext)
                angt *= (float(isav + 1) + 0.5 * temp) / float(ialt)
            cosv = (1.0 - angt ** 2.0) / (1.0 + angt ** 2.0)
            sinv = 2.0 * angt / (1.0 + angt ** 2.0)
            temp = sdhsd + angt * (angt * sthst - 2.0 * sthsd)
            temp = sinv * (angt * gqstep - sdgq - 0.5 * sinv * temp)
            if temp <= 0.0:
                break

            # Update the step with the current search direction. If the angle of
            # the alternative iteration is restricted by a bound on a free
            # variable, that variable is fixed at the bound and the computations
            # of the alternative iterations are restarted.
            for i in range(n):
                gq[i] += (cosv - 1.0) * hst[i] + sinv * hsd[i]
                hst[i] = cosv * hst[i] + sinv * hsd[i]
                if xbdi[i] == 0:
                    step[i] = cosv * step[i] + sinv * sd[i]
            reduct += temp
            if inew >= 0 and isav == ialt - 1:
                nact += 1
                xbdi[inew] = xsav
                continue
            if temp >= 1e-2 * reduct:
                inew = -1
                continue

            # End the alternative iteration computations since no new bound has
            # been hit by the new step and the reduction of the current step is
            # low compared to the reduction in the objective function so far.
            break

    # Ensure that the bound constraints are respected and that the components
    # fixed by the working sets are set to their respective bounds.
    for i in range(n):
        step[i] = fmin(xu[i], fmax(xl[i], step[i]))
        if xbdi[i] == -1:
            step[i] = xl[i]
        elif xbdi[i] == 1:
            step[i] = xu[i]
    return step
