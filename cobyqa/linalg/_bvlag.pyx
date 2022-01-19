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
from numpy import float64 as np_float64

from ._utils cimport max_array, max_abs_array, min_array


def bvlag(double[::1, :] xpt, int kopt, int klag, double[::1] gq, double[::1] xl, double[::1] xu, double delta, double alpha, bint debug):
    cdef int npt = xpt.shape[0]
    cdef int n = xpt.shape[1]

    # Shift the bounds to carry out all calculations at the origin.
    cdef double temp, tempa, tempb
    cdef Py_ssize_t i, k
    for i in range(n):
        temp = xpt[kopt, i]
        xl[i] -= temp
        xu[i] -= temp
        for k in range(npt):
            xpt[k, i] -= temp

    # Ensure the feasibility of the initial guess.
    cdef double eps = np.finfo(np_float64).eps
    cdef double tiny = np.finfo(np_float64).tiny
    cdef double tol = 10.0 * eps * float(n)
    cdef double bdtol = tol * fmax(max_abs_array(xl, 1.0), max_abs_array(xu, 1.0))
    if debug:
        if max_array(xl) > bdtol:
            raise ValueError('Constraint xl <= xopt fails initially.')
        if min_array(xu) < -bdtol:
            raise ValueError('Constraint xopt <= xu fails initially.')
        if not isfinite(delta) or delta <= 0.0:
            raise ValueError('Constraint delta > 0 fails initially.')

    # Start the iterative procedure. The method sets the largest admissible
    # value of the real parameter sigma so far in sigsav, the length of the best
    # step so far in stpsav, the index of the simple bound restraining the
    # computations in ibdsav, and index of the interpolation point defining the
    # above line in ksav.
    cdef double sigsav = 0.0
    cdef double stpsav = 0.0
    cdef int ibdsav = 0
    cdef int ksav = -1
    cdef double diff, dist, distsq, sigsq, stplen, vlag, xgq, xlbd, xubd, xumin
    cdef int ilbd, isbd, iubd
    for k in range(npt):
        # Search for a point on the line between xopt and xpt[k, :], by
        # considering only the trust-region constraint and ignoring the simple
        # bounds for the moment.
        if k == kopt:
            continue
        xgq = 0.0
        distsq = 0.0
        for i in range(n):
            xgq += xpt[k, i] * gq[i]
            distsq += xpt[k, i] ** 2.0
        dist = sqrt(distsq)
        if dist > tiny * delta:
            xubd = delta / dist
        else:
            xubd = 0.0
        xlbd = -xubd
        ilbd = 0
        iubd = 0
        xumin = fmin(1.0, xubd)

        # Update the lower and upper bounds to take into account the simple
        # bounds along the current line.
        for i in range(n):
            if xpt[k, i] > bdtol:
                if xlbd * xpt[k, i] < xl[i]:
                    xlbd = xl[i] / xpt[k, i]
                    ilbd = -i - 1
                if xubd * xpt[k, i] > xu[i]:
                    xubd = fmax(xumin, xu[i] / xpt[k, i])
                    iubd = i + 1
            elif xpt[k, i] < -bdtol:
                if xlbd * xpt[k, i] > xu[i]:
                    xlbd = xu[i] / xpt[k, i]
                    ilbd = i + 1
                if xubd * xpt[k, i] < xl[i]:
                    xubd = fmax(xumin, xl[i] / xpt[k, i])
                    iubd = -i - 1

        # Compute the best point along the line joining xopt and xpt[k, :] that
        # respects the trust-region constraint and the simple bounds.
        if k == klag:
            diff = xgq - 1.0
            stplen = xlbd
            vlag = xlbd * (xgq - diff * xlbd)
            isbd = ilbd
            temp = xubd * (xgq - diff * xubd)
            if fabs(temp) > fabs(vlag):
                stplen = xubd
                vlag = temp
                isbd = iubd
            tempa = 0.5 * xgq - diff * xlbd
            tempb = 0.5 * xgq - diff * xubd
            if tempa * tempb < 0.0 and fabs(diff) > 0.5 * tiny * fabs(xgq):
                temp = 0.25 * xgq ** 2.0 / diff
                if fabs(temp) > fabs(vlag):
                    stplen = 0.5 * xgq / diff
                    vlag = temp
                    isbd = 0
        else:
            stplen = xlbd
            vlag = xlbd * (1.0 - xlbd)
            isbd = ilbd
            temp = xubd * (1.0 - xubd)
            if fabs(temp) > fabs(vlag):
                stplen = xubd
                vlag = temp
                isbd = iubd
            if xubd > 0.5:
                if fabs(vlag) < 0.25:
                    stplen = 0.5
                    vlag = 0.25
                    isbd = 0
            vlag *= xgq

        # Calculate the parameter given in Equation (3.9) of Powell (2009) for
        # the current line-search and maintain the optimal values so far.
        temp = stplen * (1.0 - stplen) * distsq
        sigsq = vlag ** 2.0 * (vlag ** 2.0 + 0.5 * alpha * temp ** 2.0)
        if sigsq > sigsav:
            sigsav = sigsq
            stpsav = stplen
            ibdsav = isbd
            ksav = k

    # Construct the returned step.
    cdef double[::1] step = np_empty(n, dtype=np_float64)
    for i in range(n):
        step[i] = fmax(xl[i], fmin(xu[i], stpsav * xpt[ksav, i]))
    if ibdsav < 0:
        step[-ibdsav - 1] = xl[-ibdsav - 1]
    elif ibdsav > 0:
        step[ibdsav - 1] = xu[ibdsav - 1]
    return step
