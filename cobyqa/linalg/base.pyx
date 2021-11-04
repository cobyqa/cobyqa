# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np
np.import_array()

from . cimport cblas, clapack

ctypedef clapack.lapack_int lapack_int
ctypedef np.float64_t float64

cdef struct lapack_layout:
    int layout
    lapack_int lda

cdef get_lapack_layout(np.ndarray[float64, ndim=2] a):
    cdef lapack_layout matrix_layout
    if a.flags['C_CONTIGUOUS']:
        if a.flags['F_CONTIGUOUS'] and a.shape[0] > a.shape[1]:
            matrix_layout.layout = clapack.LAPACK_COL_MAJOR
            matrix_layout.lda = a.shape[0]
        else:
            matrix_layout.layout = clapack.LAPACK_ROW_MAJOR
            matrix_layout.lda = a.shape[1]
    else:
        matrix_layout.layout = clapack.LAPACK_COL_MAJOR
        matrix_layout.lda = a.shape[0]
    return matrix_layout

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

cpdef lapack_int dgeqp3(np.ndarray[float64, ndim=2] a, np.ndarray[lapack_int] jpvt, np.ndarray[float64] tau):
    cdef lapack_layout matrix_layout = get_lapack_layout(a)
    return clapack.LAPACKE_dgeqp3(matrix_layout.layout, a.shape[0], a.shape[1], <float64 *> a.data, matrix_layout.lda, <lapack_int *> jpvt.data, <float64 *> tau.data)

cpdef lapack_int dgeqrf(np.ndarray[float64, ndim=2] a, np.ndarray[float64] tau):
    cdef lapack_layout matrix_layout = get_lapack_layout(a)
    return clapack.LAPACKE_dgeqrf(matrix_layout.layout, a.shape[0], a.shape[1], <float64 *> a.data, matrix_layout.lda, <float64 *> tau.data)

cpdef lapack_int dorgqr(np.ndarray[float64, ndim=2] a, np.ndarray[float64] tau):
    cdef lapack_layout matrix_layout = get_lapack_layout(a)
    return clapack.LAPACKE_dorgqr(matrix_layout.layout, a.shape[0], a.shape[1], tau.size, <float64 *> a.data, matrix_layout.lda, <float64 *> tau.data)
