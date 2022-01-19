# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

from libc.math cimport fabs, fmax, fmin, isfinite

from scipy.linalg.cython_blas cimport ddot, dgemv  # noqa


cdef void dot(double[::1, :] a, double[::1] b, double[::1] out, char* trans, double alpha, double beta):
    """
    Evaluate the product ``out = alpha * A @ b + beta * out`` if `trans` is 'n'
    and ``out = alpha * A.T @ b + beta * out`` if `trans` is 't' or 'c'.
    """
    cdef int m = a.shape[0]
    cdef int n = a.shape[1]
    cdef int inc_b = b.strides[0] / b.itemsize
    cdef int inc_out = out.strides[0] / out.itemsize
    dgemv(trans, &m, &n, &alpha, &a[0, 0], &m, &b[0], &inc_b, &beta, &out[0], &inc_out)


cdef double inner(double[::1] a, double[::1] b):
    """
    Return the inner product of `a` and `b`. 
    """
    cdef int n = a.shape[0]
    cdef int inc_a = a.strides[0] / a.itemsize
    cdef int inc_b = b.strides[0] / b.itemsize
    return ddot(&n, &a[0], &inc_a, &b[0], &inc_b)


cdef double max_array(double[::1] x):
    """
    Return the maximum.
    """
    cdef int n = x.shape[0]
    if n < 1:
        raise ValueError('Maximum of empty array undefined.')
    cdef double x_max = x[0]
    cdef Py_ssize_t i
    for i in range(1, n):
        x_max = fmax(x_max, x[i])
    return x_max


cdef double max_abs_array(double[::1] x, double initial):
    """
    Return the the maximum in absolute value lower-bounded by initial.
    """
    cdef int n = x.shape[0]
    cdef double x_max = initial
    cdef Py_ssize_t i
    for i in range(n):
        if isfinite(x[i]):
            x_max = fmax(x_max, fabs(x[i]))
    return x_max


cdef double min_array(double[::1] x):
    """
    Return the minimum.
    """
    cdef int n = x.shape[0]
    if n < 1:
        raise ValueError('Minimum of empty array undefined.')
    cdef double x_min = x[0]
    cdef Py_ssize_t i
    for i in range(1, n):
        x_min = fmin(x_min, x[i])
    return x_min
