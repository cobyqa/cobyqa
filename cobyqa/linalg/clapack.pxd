# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

ctypedef int lapack_int

cdef extern from "lapacke.h" nogil:
    int LAPACK_ROW_MAJOR
    int LAPACK_COL_MAJOR
    int LAPACK_WORK_MEMORY_ERROR
    int LAPACK_TRANSPOSE_MEMORY_ERROR

    lapack_int LAPACKE_dgeqp3(int matrix_layout, lapack_int m, lapack_int n, double *a, lapack_int lda, lapack_int *jpvt, double *tau)
    lapack_int LAPACKE_dgeqrf(int matrix_layout, lapack_int m, lapack_int n, double *a, lapack_int lda, double *tau)
    lapack_int LAPACKE_dorgqr(int matrix_layout, lapack_int m, lapack_int n, lapack_int k, double *a, lapack_int lda, double *tau)