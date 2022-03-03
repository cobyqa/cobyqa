# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

from libc.math cimport fabs, fmax, fmin, hypot, isfinite, sqrt

import numpy as np
cimport numpy as np
np.import_array()

# Avoid namespace lookup for NumPy types and array creation methods
from numpy import empty as np_empty
from numpy import zeros as np_zeros
from numpy import float64 as np_float64

from scipy.linalg.cython_blas cimport ddot, dgemv  # noqa
from scipy.linalg.cython_lapack cimport dgeqp3, dorgqr  # noqa


cdef void dot(double[::1, :] a, double[:] b, double[:] out, char* trans, double alpha, double beta):
    """
    Evaluate the product ``out = alpha * A @ b + beta * out`` if `trans` is 'n'
    and ``out = alpha * A.T @ b + beta * out`` if `trans` is 't' or 'c'.
    """
    cdef int m = a.shape[0]
    cdef int n = a.shape[1]
    cdef int inc_b = b.strides[0] / b.itemsize
    cdef int inc_out = out.strides[0] / out.itemsize
    cdef int m_out = out.shape[0]
    cdef Py_ssize_t i
    if min(m, n) > 0:
        dgemv(trans, &m, &n, &alpha, &a[0, 0], &m, &b[0], &inc_b, &beta, &out[0], &inc_out)
    else:
        for i in range(m_out):
            out[i] *= beta


cdef double inner(double[:] a, double[:] b):
    """
    Evaluate the inner product. 
    """
    cdef int n = a.shape[0]
    cdef int inc_a = a.strides[0] / a.itemsize
    cdef int inc_b = b.strides[0] / b.itemsize
    if n > 0:
        return ddot(&n, &a[0], &inc_a, &b[0], &inc_b)
    else:
        return 0.0


cdef double[::1, :] transpose(double[::1, :] a):
    """
    Transpose the matrix.
    """
    cdef int m = a.shape[0]
    cdef int n = a.shape[1]

    cdef double[::1, :] atr = np_empty((n, m), dtype=np_float64, order='F')
    cdef Py_ssize_t i, j
    for i in range(n):
        for j in range(m):
            atr[i, j] = a[j, i]
    return atr


cdef double max_array(double[:] x):
    """
    Evaluate the maximum.
    """
    cdef int n = x.shape[0]
    if n < 1:
        raise ValueError('Maximum of empty array undefined.')
    cdef double x_max = x[0]
    cdef Py_ssize_t i
    for i in range(1, n):
        x_max = fmax(x_max, x[i])
    return x_max


cdef double min_array(double[:] x):
    """
    Evaluate the minimum.
    """
    cdef int n = x.shape[0]
    if n < 1:
        raise ValueError('Minimum of empty array undefined.')
    cdef double x_min = x[0]
    cdef Py_ssize_t i
    for i in range(1, n):
        x_min = fmin(x_min, x[i])
    return x_min


cdef double absmax_array(double[:] x, double initial):
    """
    Evaluate the maximum in absolute value.
    """
    cdef int n = x.shape[0]
    cdef double x_max = initial
    cdef Py_ssize_t i
    for i in range(n):
        if isfinite(x[i]):
            x_max = fmax(x_max, fabs(x[i]))
    return x_max


cdef void qr(double[::1, :] a, double[::1, :] q, int[:] p):
    """
    Compute the QR factorization with column pivoting. On exit, the matrix `a`
    is overridden by the upper triangular matrix of the factorization and `q`
    holds the complete orthogonal matrix of the factorization.
    """
    cdef int m = a.shape[0]
    cdef int n = a.shape[1]
    cdef int k = min(m, n)

    cdef int lwork = max(3 * n + 1, m)
    cdef double[:] tau = np_empty(k, dtype=np_float64)
    cdef double[:] work = np_empty(lwork, dtype=np_float64)
    cdef int info
    cdef Py_ssize_t i, j
    if k > 0:
        # Compute the QR factorization with column pivoting using DGEQP3.
        p[:] = 0
        dgeqp3(&m, &n, &a[0, 0], &m, &p[0], &tau[0], &work[0], &lwork, &info)
        if info != 0:
            raise ValueError(f'{-info}-th argument of DGEQP3 received an illegal value')

        # Adapt the indices in p to match Python indexing and triangularize a.
        for i in range(n):
            p[i] -= 1
            for j in range(m):
                if i < m:
                    q[j, i] = a[j, i]
                if j > i:
                    a[j, i] = 0.0

        # Build the orthogonal matrix q.
        dorgqr(&m, &m, &k, &q[0, 0], &m, &tau[0], &work[0], &lwork, &info)
        if info != 0:
            raise ValueError(f'{-info}-th argument of DORGQR received an illegal value')
        for i in range(k):
            if a[i, i] < 0.0:
                for j in range(i, n):
                    a[i, j] = -a[i, j]
                for j in range(m):
                    q[j, i] = -q[j, i]
    else:
        q[:, :] = 0.0
        for i in range(m):
            q[i, i] = 1.0


cdef bint isact(int i, int[:] iact, int* nact):
    """
    Determine whether the `i`-th constraint is active.
    """
    cdef bint is_act = False
    cdef Py_ssize_t j
    for j in range(nact[0]):
        if i == iact[j]:
            is_act = True
            break
    return is_act


cdef void getact(double[:] gq, evalc_t evalc, double[:] resid, int[:] iact, int mleq, int* nact, double[::1, :] qfac, double[::1, :] rfac, double delta, double mu, tuple args, double[:] step):
    """
    Pick the current active set. The method seeks for the closest vector to
    `-gq` in Euclidean norm, subject to the linear constraints whose normalized
    residuals are upper bounded by ``mu * delta``. The method selects among
    these constraints a basis for the span their engender.
    """
    cdef int n = gq.shape[0]
    cdef int m = resid.shape[0]
    cdef double tdel = mu * delta

    # Remove from the current active set the constraints that are not considered
    # active anymore, that is those whose residuals exceed tdel.
    cdef Py_ssize_t k
    cdef int nactc = nact[0]
    for k in range(nactc - 1, -1, -1):
        if resid[iact[k]] > tdel:
            _rmact(k, mleq, nact, qfac, rfac)
            iact[k:nact[0]] = iact[k + 1:nact[0] + 1]

    # The vector vlam stores the Lagrange multipliers of the calculations (only
    # the first nact components are meaningful). Compute these Lagrange
    # multipliers, and remove from the current active set the constraints whose
    # Lagrange multipliers are nonnegative.
    cdef double[:] vlam = np_zeros(n, dtype=np_float64)
    cdef double temp
    k = nact[0] - 1
    while k >= 0:
        temp = inner(qfac[:, mleq + k], gq)
        temp -= inner(rfac[mleq + k, mleq + k + 1:mleq + nact[0]], vlam[k + 1:nact[0]])
        if temp >= 0:
            _rmact(k, mleq, nact, qfac, rfac)
            iact[k:nact[0]] = iact[k + 1:nact[0] + 1]
            vlam[k:nact[0]] = vlam[k + 1:nact[0] + 1]
            k = nact[0] - 1
        else:
            vlam[k] = temp / rfac[mleq + k, mleq + k]
            k -= 1

    # Start the iterative procedure. The calculations must be stopped if
    # nact + mleq equals n, as n linearly independent constraints would have
    # been found, which would make of the origin the only feasible point.
    cdef double[:] work = np_empty(n, dtype=np_float64)
    cdef double stepsq = inner(gq, gq)
    cdef double eps = np.finfo(np_float64).eps
    cdef double tiny = np.finfo(np_float64).tiny
    cdef double tol = 10.0 * eps * float(n)
    cdef double gqtol = tol * absmax_array(gq, 1.0)
    cdef double cosv, ctol, cval, lhs, sinv, ssq, sval, test, violmx, vmult
    cdef int ic, inext
    cdef Py_ssize_t i
    while nact[0] < n - mleq:
        # Set the new search direction. It is the vector that minimizes the
        # Euclidean norm of gq + step, subject to the active constraints. The
        # calculations are stopped if this vector is zero, of if its norm
        # exceeds the norm of the previous direction. In the original Fortran
        # code of GETACT, Powell stopped the computations whenever
        # ssq >= stepsq, returning the zero vector. However, such a comparison
        # is subjected to computer rounding errors (observed on the CUTEst
        # problem FEEDLOC). This property should be observed from a theoretical
        # standpoint, as ssq could be evaluated as
        #
        # >>> ssq = 0.0
        # >>> for j in range(mleq + nact, n):
        # ...     ssq += np.inner(qfac[:, j], gq) ** 2.0
        #
        # by using the orthogonality property of qfac. However, this
        # orthogonality is only approximate in practice.
        dot(qfac[:, mleq + nact[0]:], gq, work, 't', 1.0, 0.0)  # noqa
        dot(qfac[:, mleq + nact[0]:], work, step, 'n', -1.0, 0.0)  # noqa
        ssq = inner(step, step)
        if ssq - stepsq >= gqtol or sqrt(ssq) <= gqtol:
            step[:] = 0.0
            return
        stepsq = ssq

        # Select the index of the most violated constraint, if any. The step
        # that is considered in these calculations is the one of length delta
        # along the direction in the vector step. The selected constraint is
        # linearly independent with the current constraints in the working set
        # (both inequality and equality constraints) as the current step lies in
        # the subspace spanned by the boundaries of these constraints.
        test = sqrt(ssq) / delta
        inext = -1
        violmx = 0.0
        for i in range(m):
            if not isact(i, iact, nact) and resid[i] <= tdel:
                lhs = evalc(i, step, args)
                if lhs > fmax(test * resid[i], violmx):
                    inext = i
                    violmx = lhs

        # Return if no constraint is violated, or if the constraint violation
        # previously calculated is too small, as a positive value of violmx
        # might then be caused by computer rounding errors.
        ctol = 0.0
        if 0.0 < violmx < 1e-2 * delta:
            for k in range(nact[0]):
                ctol = fmax(ctol, fabs(evalc(iact[k], step, args)))
        ctol *= 10.0
        if inext == -1 or violmx <= ctol:
            return

        # Add the inext-th constraint to the active set, by applying Givens
        # rotations to the matrix qfac, and add the appropriate column to rfac.
        sval = 0.0
        for k in range(n - 1, -1, -1):
            cval = evalc(inext, qfac[:, k], args)
            if k < mleq + nact[0]:
                rfac[k, mleq + nact[0]] = cval
            elif fabs(sval) <= tol * fabs(cval):
                sval = cval
            else:
                sval = _drotg(cval, sval, &cosv, &sinv)
                for i in range(n):
                    temp = cosv * qfac[i, k] + sinv * qfac[i, k + 1]
                    qfac[i, k + 1] = cosv * qfac[i, k + 1] - sinv * qfac[i, k]
                    qfac[i, k] = temp
        if sval < 0.0:
            for i in range(n):
                qfac[i, mleq + nact[0]] = -qfac[i, mleq + nact[0]]
        rfac[mleq + nact[0], mleq + nact[0]] = fabs(sval)
        iact[nact[0]] = inext
        vlam[nact[0]] = 0.0
        nact[0] += 1

        while violmx > ctol:
            # Update the vector of the Lagrange multipliers of the calculations
            # to include the new constraint. When a constraint is added or
            # removed, all the Lagrange multipliers must be updated.
            work[nact[0] - 1] = (1.0 / rfac[mleq + nact[0] - 1, mleq + nact[0] - 1]) ** 2.0
            for k in range(nact[0] - 2, -1, -1):
                work[k] = -inner(rfac[mleq + k, mleq + k + 1:mleq + nact[0]], work[k + 1:nact[0]])
                work[k] /= rfac[mleq + k, mleq + k]
            vmult = violmx
            ic = -1
            for k in range(nact[0] - 1):
                if vlam[k] >= vmult * work[k]:
                    if fabs(work[k]) > tiny * fabs(vlam[k]):
                        ic = k
                        vmult = vlam[k] / work[k]
            for k in range(nact[0]):
                vlam[k] -= vmult * work[k]
            if ic >= 0:
                vlam[ic] = 0.0
                violmx = fmax(violmx - vmult, 0.0)
            else:
                violmx = 0.0

            # Remove from the active set the constraints whose Lagrange
            # multipliers are nonnegative. This mechanism ensures the active
            # constraints are linearly independent.
            nactc = nact[0]
            for k in range(nactc - 1, -1, -1):
                if vlam[k] >= 0.0:
                    _rmact(k, mleq, nact, qfac, rfac)
                    iact[k:nact[0]] = iact[k + 1:nact[0] + 1]
                    vlam[k:nact[0]] = vlam[k + 1:nact[0] + 1]

    step[:] = 0.0
    return


cdef double _drotg(double a, double b, double* c, double* s):
    """
    Prepare a Givens rotation.
    """
    cdef double r = hypot(a, b)
    if r == 0.0:
        c[0] = 1.0
        s[0] = 0.0
    else:
        c[0] = a / r
        s[0] = b / r
    return r


cdef void _rmact(int k, int mleq, int* nact, double[::1, :] qfac, double[::1, :] rfac):
    """
    Remove the `k`-th constraint from the active set. A constraint is removed
    from the active set by applying a sequence of Givens rotations to the
    factorization matrices given in `gfac` and `rfac`.
    """
    cdef int n = qfac.shape[0]
    cdef double cosv, cval, hval, sinv, sval, temp
    cdef Py_ssize_t i, j
    for j in range(mleq + k, mleq + nact[0] - 1):
        cval = rfac[j + 1, j + 1]
        sval = rfac[j, j + 1]
        hval = _drotg(cval, sval, &cosv, &sinv)
        rfac[j, j + 1] = sinv * rfac[j, j]
        rfac[j + 1, j + 1] = cosv * rfac[j, j]
        rfac[j, j] = hval
        for i in range(j + 2, mleq + nact[0]):
            temp = cosv * rfac[j + 1, i] + sinv * rfac[j, i]
            rfac[j + 1, i] = cosv * rfac[j, i] - sinv * rfac[j + 1, i]
            rfac[j, i] = temp
        for i in range(n):
            if i < j:
                temp = rfac[i, j + 1]
                rfac[i, j + 1] = rfac[i, j]
                rfac[i, j] = temp
            temp = cosv * qfac[i, j + 1] + sinv * qfac[i, j]
            qfac[i, j + 1] = cosv * qfac[i, j] - sinv * qfac[i, j + 1]
            qfac[i, j] = temp
    nact[0] -= 1
