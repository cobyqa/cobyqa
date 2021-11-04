# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

cdef extern from "cblas.h" nogil:
    # =========================================================================
    # Enumerated and derived types
    # =========================================================================
    ctypedef enum CBLAS_ORDER:
        CblasRowMajor
        CblasColMajor
    ctypedef enum CBLAS_TRANSPOSE:
        CblasNoTrans
        CblasTrans
        CblasConjTrans
    ctypedef enum CBLAS_UPLO:
        CblasUpper
        CblasLower
    ctypedef enum CBLAS_DIAG:
        CblasNonUnit
        CblasUnit
    ctypedef enum CBLAS_SIDE:
        CblasLeft
        CblasRight

    # =========================================================================
    # Prototypes for level 1 BLAS functions
    # =========================================================================
    float cblas_sdsdot(int N, float alpha, float *X, int incX, float *Y, int incY)
    double cblas_dsdot(int N, float *X, int incX, float *Y, int incY)
    float cblas_sdot(int N, float *X, int incX, float *Y, int incY)
    double cblas_ddot(int N, double *X, int incX, double *Y, int incY)

    # Routines with standard 4 prefixes (s, d, c, z)
    size_t cblas_isamax(int N, float *X, int incX);
    size_t cblas_idamax(int N, double *X, int incX);
    size_t cblas_icamax(int N, void *X, int incX);
    size_t cblas_izamax(int N, void *X, int incX);

    # Functions with c and z prefixes only
    void cblas_cdotu_sub(int N, void *X, int incX, void *Y, int incY, void *dotu)
    void cblas_cdotc_sub(int N, void *X, int incX, void *Y, int incY, void *dotc)
    void cblas_zdotu_sub(int N, void *X, int incX, void *Y, int incY, void *dotu)
    void cblas_zdotc_sub(int N, void *X, int incX, void *Y, int incY, void *dotc)

    # Functions with s, d, sc, and dz prefixes
    float cblas_snrm2(int N, float *X, int incX)
    float cblas_sasum(int N, float *X, int incX)
    double cblas_dnrm2(int N, double *X, int incX)
    double cblas_dasum(int N, double *X, int incX)
    float cblas_scnrm2(int N, void *X, int incX)
    float cblas_scasum(int N, void *X, int incX)
    double cblas_dznrm2(int N, void *X, int incX)
    double cblas_dzasum(int N, void *X, int incX)

    # =========================================================================
    # Prototypes for level 1 BLAS routines
    # =========================================================================

    # Routines with standard 4 prefixes (s, d, c, z)
    void cblas_sswap(int N, float *X, int incX, float *Y, int incY)
    void cblas_scopy(int N, float *X, int incX, float *Y, int incY)
    void cblas_saxpy(int N, float alpha, float *X, int incX, float *Y, int incY)
    void cblas_dswap(int N, double *X, int incX, double *Y, int incY)
    void cblas_dcopy(int N, double *X, int incX, double *Y, int incY)
    void cblas_daxpy(int N, double alpha, double *X, int incX, double *Y, int incY)
    void cblas_cswap(int N, void *X, int incX, void *Y, int incY)
    void cblas_ccopy(int N, void *X, int incX, void *Y, int incY)
    void cblas_caxpy(int N, void *alpha, void *X, int incX, void *Y, int incY)
    void cblas_zswap(int N, void *X, int incX, void *Y, int incY)
    void cblas_zcopy(int N, void *X, int incX, void *Y, int incY)
    void cblas_zaxpy(int N, void *alpha, void *X, int incX, void *Y, int incY)

    # Routines with s and d prefixes only
    void cblas_srotg(float *a, float *b, float *c, float *s)
    void cblas_srotmg(float *d1, float *d2, float *b1, float b2, float *P)
    void cblas_srot(int N, float *X, int incX, float *Y, int incY, float c, float s)
    void cblas_srotm(int N, float *X, int incX, float *Y, int incY, float *P)
    void cblas_drotg(double *a, double *b, double *c, double *s)
    void cblas_drotmg(double *d1, double *d2, double *b1, double b2, double *P)
    void cblas_drot(int N, double *X, int incX, double *Y, int incY, double c, double s)
    void cblas_drotm(int N, double *X, int incX, double *Y, int incY, double *P)

    # Routines with s, d, c, z, cs, and zd prefixes
    void cblas_sscal(int N, float alpha, float *X, int incX)
    void cblas_dscal(int N, double alpha, double *X, int incX)
    void cblas_cscal(int N, void *alpha, void *X, int incX)
    void cblas_zscal(int N, void *alpha, void *X, int incX)
    void cblas_csscal(int N, float alpha, void *X, int incX)
    void cblas_zdscal(int N, double alpha, void *X, int incX)

    # =========================================================================
    # Prototypes for level 2 BLAS routines
    # =========================================================================

    # Routines with standard 4 prefixes (s, d, c, z)
    void cblas_sgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, int M, int N, float alpha, float *A, int lda, float *X, int incX, float beta, float *Y, int incY);
    void cblas_sgbmv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, int M, int N, int KL, int KU, float alpha, float *A, int lda, float *X, int incX, float beta, float *Y, int incY);
    void cblas_strmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, float *A, int lda, float *X, int incX);
    void cblas_stbmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, int K, float *A, int lda, float *X, int incX);
    void cblas_stpmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, float *Ap, float *X, int incX);
    void cblas_strsv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, float *A, int lda, float *X, int incX);
    void cblas_stbsv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, int K, float *A, int lda, float *X, int incX);
    void cblas_stpsv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, float *Ap, float *X, int incX);
    void cblas_dgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, int M, int N, double alpha, double *A, int lda, double *X, int incX, double beta, double *Y, int incY);
    void cblas_dgbmv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, int M, int N, int KL, int KU, double alpha, double *A, int lda, double *X, int incX, double beta, double *Y, int incY);
    void cblas_dtrmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, double *A, int lda, double *X, int incX);
    void cblas_dtbmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, int K, double *A, int lda, double *X, int incX);
    void cblas_dtpmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, double *Ap, double *X, int incX);
    void cblas_dtrsv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, double *A, int lda, double *X, int incX);
    void cblas_dtbsv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, int K, double *A, int lda, double *X, int incX);
    void cblas_dtpsv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, double *Ap, double *X, int incX);
    void cblas_cgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, int M, int N, void *alpha, void *A, int lda, void *X, int incX, void *beta, void *Y, int incY);
    void cblas_cgbmv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, int M, int N, int KL, int KU, void *alpha, void *A, int lda, void *X, int incX, void *beta, void *Y, int incY);
    void cblas_ctrmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, void *A, int lda, void *X, int incX);
    void cblas_ctbmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, int K, void *A, int lda, void *X, int incX);
    void cblas_ctpmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, void *Ap, void *X, int incX);
    void cblas_ctrsv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, void *A, int lda, void *X, int incX);
    void cblas_ctbsv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, int K, void *A, int lda, void *X, int incX);
    void cblas_ctpsv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, void *Ap, void *X, int incX);
    void cblas_zgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, int M, int N, void *alpha, void *A, int lda, void *X, int incX, void *beta, void *Y, int incY);
    void cblas_zgbmv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, int M, int N, int KL, int KU, void *alpha, void *A, int lda, void *X, int incX, void *beta, void *Y, int incY);
    void cblas_ztrmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, void *A, int lda, void *X, int incX);
    void cblas_ztbmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, int K, void *A, int lda, void *X, int incX);
    void cblas_ztpmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, void *Ap, void *X, int incX);
    void cblas_ztrsv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, void *A, int lda, void *X, int incX);
    void cblas_ztbsv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, int K, void *A, int lda, void *X, int incX);
    void cblas_ztpsv(CBLAS_ORDER order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int N, void *Ap, void *X, int incX);

    # Routines with s and d prefixes only
    void cblas_ssymv(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, float alpha, float *A, int lda, float *X, int incX, float beta, float *Y, int incY)
    void cblas_ssbmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, int K, float alpha, float *A, int lda, float *X, int incX, float beta, float *Y, int incY)
    void cblas_sspmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, float alpha, float *Ap, float *X, int incX, float beta, float *Y, int incY)
    void cblas_sger(CBLAS_ORDER order, int M, int N, float alpha, float *X, int incX, float *Y, int incY, float *A, int lda)
    void cblas_ssyr(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, float alpha, float *X, int incX, float *A, int lda)
    void cblas_sspr(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, float alpha, float *X, int incX, float *Ap)
    void cblas_ssyr2(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, float alpha, float *X, int incX, float *Y, int incY, float *A, int lda)
    void cblas_sspr2(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, float alpha, float *X, int incX, float *Y, int incY, float *A)
    void cblas_dsymv(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, double alpha, double *A, int lda, double *X, int incX, double beta, double *Y, int incY);
    void cblas_dsbmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, int K, double alpha, double *A, int lda, double *X, int incX, double beta, double *Y, int incY);
    void cblas_dspmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, double alpha, double *Ap, double *X, int incX, double beta, double *Y, int incY);
    void cblas_dger(CBLAS_ORDER order, int M, int N, double alpha, double *X, int incX, double *Y, int incY, double *A, int lda);
    void cblas_dsyr(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, double alpha, double *X, int incX, double *A, int lda);
    void cblas_dspr(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, double alpha, double *X, int incX, double *Ap);
    void cblas_dsyr2(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, double alpha, double *X, int incX, double *Y, int incY, double *A, int lda);
    void cblas_dspr2(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, double alpha, double *X, int incX, double *Y, int incY, double *A);

    # Routines with s and z prefixes only
    void cblas_chemv(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, void *alpha, void *A, int lda, void *X, int incX, void *beta, void *Y, int incY)
    void cblas_chbmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, int K, void *alpha, void *A, int lda, void *X, int incX, void *beta, void *Y, int incY)
    void cblas_chpmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, void *alpha, void *Ap, void *X, int incX, void *beta, void *Y, int incY)
    void cblas_cgeru(CBLAS_ORDER order, int M, int N, void *alpha, void *X, int incX, void *Y, int incY, void *A, int lda)
    void cblas_cgerc(CBLAS_ORDER order, int M, int N, void *alpha, void *X, int incX, void *Y, int incY, void *A, int lda)
    void cblas_cher(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, float alpha, void *X, int incX, void *A, int lda)
    void cblas_chpr(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, float alpha, void *X, int incX, void *A)
    void cblas_cher2(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, void *alpha, void *X, int incX, void *Y, int incY, void *A, int lda)
    void cblas_chpr2(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, void *alpha, void *X, int incX, void *Y, int incY, void *Ap)
    void cblas_zhemv(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, void *alpha, void *A, int lda, void *X, int incX, void *beta, void *Y, int incY)
    void cblas_zhbmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, int K, void *alpha, void *A, int lda, void *X, int incX, void *beta, void *Y, int incY)
    void cblas_zhpmv(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, void *alpha, void *Ap, void *X, int incX, void *beta, void *Y, int incY)
    void cblas_zgeru(CBLAS_ORDER order, int M, int N, void *alpha, void *X, int incX, void *Y, int incY, void *A, int lda)
    void cblas_zgerc(CBLAS_ORDER order, int M, int N, void *alpha, void *X, int incX, void *Y, int incY, void *A, int lda)
    void cblas_zher(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, double alpha, void *X, int incX, void *A, int lda)
    void cblas_zhpr(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, double alpha, void *X, int incX, void *A)
    void cblas_zher2(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, void *alpha, void *X, int incX, void *Y, int incY, void *A, int lda)
    void cblas_zhpr2(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N, void *alpha, void *X, int incX, void *Y, int incY, void *Ap)

    # =========================================================================
    # Prototypes for level 3 BLAS routines
    # =========================================================================

    # Routines with standard 4 prefixes (s, d, c, z)
    void cblas_sgemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, int M, int N, int K, float alpha, float *A, int lda, float *B, int ldb, float beta, float *C, int ldc)
    void cblas_ssymm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, int M, int N, float alpha, float *A, int lda, float *B, int ldb, float beta, float *C, int ldc)
    void cblas_ssyrk(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans, int N, int K, float alpha, float *A, int lda, float beta, float *C, int ldc)
    void cblas_ssyr2k(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans, int N, int K, float alpha, float *A, int lda, float *B, int ldb, float beta, float *C, int ldc)
    void cblas_strmm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int M, int N, float alpha, float *A, int lda, float *B, int ldb)
    void cblas_strsm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int M, int N, float alpha, float *A, int lda, float *B, int ldb)
    void cblas_dgemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, int M, int N, int K, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
    void cblas_dsymm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, int M, int N, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
    void cblas_dsyrk(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans, int N, int K, double alpha, double *A, int lda, double beta, double *C, int ldc)
    void cblas_dsyr2k(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans, int N, int K, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
    void cblas_dtrmm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int M, int N, double alpha, double *A, int lda, double *B, int ldb)
    void cblas_dtrsm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int M, int N, double alpha, double *A, int lda, double *B, int ldb)
    void cblas_cgemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, int M, int N, int K, void *alpha, void *A, int lda, void *B, int ldb, void *beta, void *C, int ldc)
    void cblas_csymm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, int M, int N, void *alpha, void *A, int lda, void *B, int ldb, void *beta, void *C, int ldc)
    void cblas_csyrk(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans, int N, int K, void *alpha, void *A, int lda, void *beta, void *C, int ldc)
    void cblas_csyr2k(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans, int N, int K, void *alpha, void *A, int lda, void *B, int ldb, void *beta, void *C, int ldc)
    void cblas_ctrmm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int M, int N, void *alpha, void *A, int lda, void *B, int ldb)
    void cblas_ctrsm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int M, int N, void *alpha, void *A, int lda, void *B, int ldb)
    void cblas_zgemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, int M, int N, int K, void *alpha, void *A, int lda, void *B, int ldb, void *beta, void *C, int ldc)
    void cblas_zsymm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, int M, int N, void *alpha, void *A, int lda, void *B, int ldb, void *beta, void *C, int ldc)
    void cblas_zsyrk(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans, int N, int K, void *alpha, void *A, int lda, void *beta, void *C, int ldc)
    void cblas_zsyr2k(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans, int N, int K, void *alpha, void *A, int lda, void *B, int ldb, void *beta, void *C, int ldc)
    void cblas_ztrmm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int M, int N, void *alpha, void *A, int lda, void *B, int ldb)
    void cblas_ztrsm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag, int M, int N, void *alpha, void *A, int lda, void *B, int ldb)

    # Routines with c and z prefixes only
    void cblas_chemm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, int M, int N, void *alpha, void *A, int lda, void *B, int ldb, void *beta, void *C, int ldc)
    void cblas_cherk(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans, int N, int K, float alpha, void *A, int lda, float beta, void *C, int ldc)
    void cblas_cher2k(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans, int N, int K, void *alpha, void *A, int lda, void *B, int ldb, float beta, void *C, int ldc)
    void cblas_zhemm(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo, int M, int N, void *alpha, void *A, int lda, void *B, int ldb, void *beta, void *C, int ldc)
    void cblas_zherk(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans, int N, int K, double alpha, void *A, int lda, double beta, void *C, int ldc)
    void cblas_zher2k(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans, int N, int K, void *alpha, void *A, int lda, void *B, int ldb, double beta, void *C, int ldc)
