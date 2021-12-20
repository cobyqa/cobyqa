# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

from libc.math cimport fabs, isfinite

import numpy as np
cimport numpy as np
np.import_array()

cdef double get_bdtol(double[::1] xl, double[::1] xu):
    cdef int n = xl.shape[0]
    cdef double eps = np.finfo(np.float64).eps
    cdef double tol = 10.0 * eps * n
    cdef double bd_max = 1.0
    cdef Py_ssize_t i
    for i in range(n):
        if isfinite(xl[i]) and fabs(xl[i]) > bd_max:
            bd_max = fabs(xl[i])
        if isfinite(xu[i]) and fabs(xu[i]) > bd_max:
            bd_max = fabs(xu[i])
    return tol * bd_max

cdef double get_lctol(double[::1, :] a, double[::1] b):
    cdef int m = a.shape[0]
    cdef int n = a.shape[1]
    cdef double eps = np.finfo(np.float64).eps
    cdef double tol = 10.0 * eps * max(m, n)
    cdef double b_max = 1.0
    cdef Py_ssize_t i
    for i in range(m):
        if isfinite(b[i]) and fabs(b[i]) > b_max:
            b_max = fabs(b[i])
    return tol * b_max