# cython: language_level=3

ctypedef double (*evalc_t)(int, double[:], tuple)

cdef void dot(double[::1, :], double[:], double[:], char*, double, double)
cdef double inner(double[:], double[:])
cdef double[::1, :] transpose(double[::1, :])
cdef double max_array(double[:])
cdef double min_array(double[:])
cdef double absmax_array(double[:], double)
cdef void qr(double[::1, :], double[::1, :], int[:])
cdef bint isact(int, int[:], int*)
cdef void getact(double[:], evalc_t, double[:], int[:], int, int*, double[::1, :], double[::1, :], double, double, tuple, double[:])
