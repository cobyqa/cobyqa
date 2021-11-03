# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np
np.import_array()

from . cimport cblas

ctypedef np.float64_t float64
ctypedef np.int64_t int64

# no namespace lookup for array creation
from numpy import arange as np_arange
from numpy import eye as np_eye

cpdef double drot(np.ndarray[float64] x, np.ndarray[float64] y, double a, double b):
    cdef int inc_x = x.strides[0] / x.itemsize
    cdef int inc_y = y.strides[0] / y.itemsize
    cdef double *a_ptr = &a
    cdef double *b_ptr = &b
    cdef double *c_ptr = <double *> malloc(sizeof(double))
    cdef double *s_ptr = <double *> malloc(sizeof(double))
    cblas.cblas_drotg(a_ptr, b_ptr, c_ptr, s_ptr)
    cblas.cblas_drot(x.size, <float64 *> x.data, inc_x, <float64 *> y.data, inc_y, c_ptr[0], s_ptr[0])
    free(c_ptr)
    free(s_ptr)
    return a_ptr[0]

cpdef void dgeqrf(np.ndarray[float64, ndim=2] a, np.ndarray[float64, ndim=2] q):
    cdef int inc = a.strides[0] / a.itemsize
    cdef double tiny = np.finfo(float).tiny
    cdef double cval, sval
    cdef Py_ssize_t i, j
    q[...] = np_eye(q.shape[0], q.shape[1])
    for j in range(a.shape[1]):
        for i in range(a.shape[0] - 1, j, -1):
            cval = a[j, j]
            sval = a[i, j]
            if abs(sval) > tiny * abs(cval):
                drot(q[j, :], q[i, :], cval, sval)
                drot(a[j, :], a[i, :], cval, sval)
            a[i, j] = 0.0
    q[...] = q.T

cpdef void dgeqp3(np.ndarray[float64, ndim=2] a, np.ndarray[float64, ndim=2] q, np.ndarray[int64] p):
    cdef int m = a.shape[0]
    cdef int n = a.shape[1]
    cdef int inc = a.strides[0] / a.itemsize
    cdef double tiny = np.finfo(float).tiny
    cdef double cval, sval
    cdef Py_ssize_t i, j, k
    q[...] = np_eye(m)
    p[...] = np_arange(n)
    for j in range(n):
        k = j + np.argmax(np.linalg.norm(a[j:, j:], axis=0))
        p[[j, k]] = p[[k, j]]
        a[:, [j, k]] = a[:, [k, j]]
        for i in range(m - 1, j, -1):
            cval = a[j, j]
            sval = a[i, j]
            if abs(sval) > tiny * abs(cval):
                drot(q[j, :], q[i, :], cval, sval)
                drot(a[j, :], a[i, :], cval, sval)
            a[i, j] = 0.0
    q[...] = q.T
