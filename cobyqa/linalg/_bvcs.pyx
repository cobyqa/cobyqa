# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

from libc.math cimport INFINITY, fabs, fmax, fmin, isfinite, sqrt

import numpy as np
cimport numpy as np
np.import_array()

# Avoid namespace lookup for NumPy types and array creation methods
from numpy import empty as np_empty
from numpy import zeros as np_zeros
from numpy import float64 as np_float64

from ._utils cimport max_array, absmax_array, min_array


def bvcs(double[::1, :] xpt, int kopt, double[:] gq, object curv, double[:] xl, double[:] xu, double delta, bint debug):
    """
    Evaluate Cauchy step on the absolute value of a Lagrange polynomial, subject
    to bound constraints on its coordinates and its length.
    """
    cdef int npt = xpt.shape[0]
    cdef int n = xpt.shape[1]

    # Shift the bounds to carry out all calculations at the origin.
    cdef Py_ssize_t i
    for i in range(n):
        xl[i] -= xpt[kopt, i]
        xu[i] -= xpt[kopt, i]

    # Ensure the feasibility of the initial guess.
    cdef double eps = np.finfo(np_float64).eps
    cdef double tiny = np.finfo(np_float64).tiny
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


    # Start the procedure.
    cdef double[:] step = np_zeros(n, dtype=np_float64)
    cdef double[:] cstep = np_empty(n, dtype=np_float64)
    cdef double[:] stpsav = np_empty(n, dtype=np_float64)
    cdef double bigstp = 2.0 * delta
    cdef double csav = 0.0
    cdef double cauchy = 0.0
    cdef double crv, cssq, csqsav, delsq, gqcc, gqsq, scale, stplen, temp
    cdef Py_ssize_t isign
    for isign in range(2):
        # Initialize the computations of the Cauchy step. The free components of
        # the Cauchy step are set to bigstp and the computations stop
        # immediately if every free component of the gradient is zero.
        gqsq = 0.0
        for i in range(n):
            if fmin(-xl[i], gq[i]) > 0.0 or fmax(-xu[i], gq[i]) < 0.0:
                cstep[i] = bigstp
                gqsq += gq[i] ** 2.0
            else:
                cstep[i] = 0.0
        if sqrt(gqsq) < tol * fmax(1.0, bigstp):
            break

        # Fix the remaining components of the Cauchy step to the lower and
        # upper bounds as the trust-region constraint allows.
        cssq = 0.0
        csqsav = -1.0
        stplen = 0.0
        while cssq > csqsav and sqrt(gqsq) > tol * fmax(1.0, bigstp):
            delsq = delta ** 2.0 - cssq
            if delsq > 0.0:
                csqsav = cssq
                stplen = sqrt(delsq / gqsq)
                gqsq = 0.0
                for i in range(n):
                    if fabs(cstep[i] - bigstp) < tol * fmax(1.0, bigstp):
                        temp = -stplen * gq[i]
                        if temp <= xl[i]:
                            cstep[i] = xl[i]
                            cssq += cstep[i] ** 2.0
                        elif temp >= xu[i]:
                            cstep[i] = xu[i]
                            cssq += cstep[i] ** 2.0
                        else:
                            gqsq += gq[i] ** 2.0
            else:
                break

        # Set the free components of the Cauchy step and all components of the
        # trial step. The Cauchy step may be scaled hereinafter.
        gqcc = 0.0
        for i in range(n):
            if fabs(cstep[i] - bigstp) < tol * fmax(1.0, bigstp):
                cstep[i] = -stplen * gq[i]
                step[i] = fmax(xl[i], fmin(xu[i], cstep[i]))
            elif fabs(cstep[i]) < tol * fmax(1.0, bigstp):
                step[i] = 0.0
            elif (xl[i] > -INFINITY and gq[i] > 0.0) or xu[i] == INFINITY:
                step[i] = xl[i]
            else:
                step[i] = xu[i]
            gqcc += gq[i] * cstep[i]

        # Set the curvature of the knew-th Lagrange function along the Cauchy
        # step. Scale the Cauchy step by a factor less than one if that can
        # reduce the modulus of the Lagrange function.
        crv = curv(cstep)
        if isign == 1:
            crv *= -1.0
        if -gqcc < crv < -(1.0 + sqrt(2.0)) * gqcc and fabs(crv) > tiny * fabs(gqcc):
            scale = -gqcc / crv
            for i in range(n):
                step[i] = fmax(xl[i], fmin(xu[i], scale * cstep[i]))
            cauchy = (0.5 * gqcc * scale) ** 2.0
        else:
            cauchy = (gqcc + 0.5 * crv) ** 2.0

        # If isign is zero, then the step is calculated as before after
        # reversing the sign of the gradient. Thus, two steps become available.
        # The chosen one gives the largest value of cauchy.
        if isign == 0:
            for i in range(n):
                gq[i] = -gq[i]
                stpsav[i] = step[i]
            csav = cauchy
            continue
        if csav > cauchy:
            step[:] = stpsav
            cauchy = csav

    return step, cauchy
