# cython: language_level=3

cdef void dot(double[::1, :] a, double[::1] b, double[::1] out, char* trans, double alpha, double beta)
cdef double inner(double[::1] a, double[::1] b)
cdef double max_array(double[::1] x)
cdef double max_abs_array(double[::1] x, double initial)
cdef double min_array(double[::1] x)
