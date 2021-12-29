# cython: language_level=3

cdef double get_tol(int n)
cdef double get_bdtol(double[::1] xl, double[::1] xu)
cdef double get_lctol(double[::1, :] a, double[::1] b)
