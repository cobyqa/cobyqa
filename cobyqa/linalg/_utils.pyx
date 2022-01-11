# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

from libc.math cimport fabs, fmax, isfinite

import numpy as np
cimport numpy as np
np.import_array()

# Avoid namespace lookup for NumPy types
from numpy import float64 as np_float64

cdef double get_tol(int n):
    """
    Return the base tolerance.
    """
    return 10.0 * np.finfo(np_float64).eps * np_float64(n)

cdef double get_bdtol(double[::1] xl, double[::1] xu):
    """
    Return the tolerance on bounds.
    """
    cdef int n = xl.shape[0]
    if xu.shape[0] != n:
        raise ValueError('Bound shapes are inconsistent')
    cdef double bd_max = 1.0
    cdef Py_ssize_t i
    for i in range(n):
        if isfinite(xl[i]):
            bd_max = fmax(bd_max, fabs(xl[i]))
        if isfinite(xu[i]):
            bd_max = fmax(bd_max, fabs(xu[i]))
    return get_tol(n) * bd_max

cdef double get_lctol(double[::1, :] a, double[::1] b):
    """
    Return the tolerance on linear constraints.
    """
    cdef int m = a.shape[0]
    cdef int n = a.shape[1]
    if b.shape[0] != m:
        raise ValueError('Constraint shapes are inconsistent')
    cdef double b_max = 1.0
    cdef Py_ssize_t i
    for i in range(m):
        if isfinite(b[i]):
            b_max = fmax(b_max, fabs(b[i]))
    return get_tol(max(m, n)) * b_max
