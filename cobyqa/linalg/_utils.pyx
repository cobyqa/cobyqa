# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

from libc.math cimport fabs, isfinite

import numpy as np
cimport numpy as np
np.import_array()

# Avoid namespace lookup for NumPy types
from numpy import float64 as np_float64

cdef double get_tol(int n):
    return 10.0 * np.finfo(np_float64).eps * n

cdef double get_bdtol(double[::1] xl, double[::1] xu):
    cdef int n = xl.shape[0]
    if xu.shape[0] != n:
        raise ValueError('Bound shapes are inconsistent')
    cdef double bd_max = 1.0
    cdef Py_ssize_t i
    for i in range(n):
        if isfinite(xl[i]):
            bd_max = max(bd_max, fabs(xl[i]))
        if isfinite(xu[i]):
            bd_max = max(bd_max, fabs(xu[i]))
    return get_tol(n) * bd_max

cdef double get_lctol(double[::1, :] a, double[::1] b):
    cdef int m = a.shape[0]
    cdef int n = a.shape[1]
    if b.shape[0] != m:
        raise ValueError('Constraint shapes are inconsistent')
    cdef double b_max = 1.0
    cdef Py_ssize_t i
    for i in range(m):
        if isfinite(b[i]):
            b_max = max(b_max, fabs(b[i]))
    return get_tol(max(m, n)) * b_max