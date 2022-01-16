# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

from libc.math cimport fabs

import numpy as np
cimport numpy as np
np.import_array()

# Avoid namespace lookup for NumPy types and array creation methods
from numpy import empty as np_empty
from numpy import zeros as np_zeros
from numpy import ones as np_ones
from numpy import int32 as np_int32
from numpy import float64 as np_float64

from scipy.linalg.cython_blas cimport ddot, dgemv  # noqa
from scipy.linalg.cython_lapack cimport dgelsy  # noqa
from scipy.linalg.lapack import dgelsy_lwork  # noqa

from ._utils cimport get_lctol

def nnls(double[::1, :] a, double[::1] b, int k, int maxiter):
    """
    Compute the least-squares solution to the equation ``a @ x = b`` subject to
    the nonnegativity constraints ``x[:k] >= 0``.

    Parameters
    ----------
    a : memoryview of numpy.ndarray, shape (m, n)
        Matrix `a` as shown above.
    b : memoryview of numpy.ndarray, shape (m,)
        Vector `b` as shown above.
    k : int
        Number of nonnegativity constraints. The first `k` components of the
        solution vector are nonnegative.
    maxiter : int
        Maximum number of inner iterations.

    Returns
    -------
    memoryview of numpy.ndarray, shape (n,)
        Solution vector ``x`` as shown above.
    """
    cdef int m = a.shape[0]
    cdef int n = a.shape[1]
    cdef int nact = k

    # Initialize the working arrays at the origin of the calculations.
    cdef bint[::1] iact = np_ones(k, dtype=np_int32)
    cdef double[::1] x = np_zeros(n, dtype=np_float64)
    cdef int incx = x.strides[0] // x.itemsize
    cdef double[::1] xact = np_empty(n, dtype=np_float64)
    cdef double[::1] resid = np_empty(m, dtype=np_float64)
    resid[:] = b

    # Evaluate the objective function of the linear least-squares problem at the
    # origin of the calculations.
    cdef int incr = resid.strides[0] // resid.itemsize
    cdef double lsx = 0.5 * ddot(&m, &resid[0], &incr, &resid[0], &incr)

    # Evaluate the gradient of the objective function of the linear
    # least-squares problem at the origin of the calculations.
    cdef double alpha = -1.0
    cdef double beta = 0.0
    cdef double[::1] grad = np_empty(n, dtype=np_float64)
    cdef int incg = grad.strides[0] // grad.itemsize
    dgemv('t', &m, &n, &alpha, &a[0, 0], &m, &resid[0], &incr, &beta, &grad[0], &incg)

    # Allocate the working arrays required by the GELSY driver.
    cdef double tiny = np.finfo(np_float64).tiny
    cdef double rcond = np.finfo(np_float64).eps * max(m, n)

    # Start the iterative procedure, and iterate until the approximate KKT
    # conditions hold for the current vector of variables.
    cdef int iterc = 0
    cdef double lctol = get_lctol(a, b)
    cdef double gamma, temp
    cdef Py_ssize_t i, inew
    while not check_kkt(grad, iact, lctol):
        # Remove from the working set the least-gradient coordinate. The case
        # nact > 0 is equivalent to k > 0 but is more comprehensible.
        if nact > 0:
            inew = -1
            for i in range(k):
                if iact[i] and (inew == -1 or grad[i] < grad[inew]):
                    inew = i
            iact[inew] = False
            nact -= 1

        # Solve the least-squares problem on the inactive columns.
        lstsq(a, b, iact, nact, rcond, xact)

        # Increase the working set if necessary.
        while not check_act(xact, iact):
            # Stop the computation if the maximum number of iterations has been
            # reached. The break statement of the outer loop is then reached
            # since the else statement below fails, which ends the computations.
            if iterc >= maxiter:
                x[:] = xact
                break
            iterc += 1

            # Update the current vector of variable to the closest vector from
            # xact along the line joining x to xact that is feasible.
            gamma = 1.0
            for i in range(k):
                temp = x[i] - xact[i]
                if not iact[i] and xact[i] <= 0.0 and x[i] < gamma * temp:
                    if fabs(temp) > tiny * fabs(x[i]):
                        gamma = x[i] / temp
            gamma = max(0.0, gamma)
            for i in range(n):
                x[i] += gamma * (xact[i] - x[i])

            # Update the working set at the new vector of variables.
            for i in range(k):
                if not iact[i] and fabs(x[i]) < lctol:
                    iact[i] = True
                    nact += 1

            # Solve the least-squares problem on the inactive columns.
            lstsq(a, b, iact, nact, rcond, xact)
        else:
            # Continue the computations if any progress has been made, as the
            # inner loop terminated correctly. If the relative change of the
            # objective function between the current iteration and the previous
            # one is insubstantial, stop the computations to prevent infinite
            # cycling, which may occur due to computer rounding errors.
            x[:] = xact
            resid[:] = b
            beta = 1.0
            dgemv('n', &m, &n, &alpha, &a[0, 0], &m, &x[0], &incx, &beta, &resid[0], &incr)
            temp = 0.5 * ddot(&m, &resid[0], &incr, &resid[0], &incr)
            if temp > (1.0 - lctol) * lsx:
                lsx = temp
                break
            lsx = temp
            beta = 0.0
            dgemv('t', &m, &n, &alpha, &a[0, 0], &m, &resid[0], &incr, &beta, &grad[0], &incg)
            continue
        break
    return x

cdef bint check_kkt(double[::1] grad, bint[::1] iact, double tol):
    """
    Check whether the approximate KKT conditions hold. The primal feasibility
    condition is assumed and is not checked by the function.

    Parameters
    ----------
    grad : memoryview of numpy.ndarray, shape (n,)
        Gradient of the linear least-squares objective function.
    iact : memoryview of numpy.ndarray, shape (k,)
        Working set. The ``i``-th variable in included in the working set if
        ``i < iact.shape[0]`` and ``iact[i] = True``.
    tol : double
        Tolerance on the approximate KKT conditions.

    Returns
    -------
    bint:
        Whether the approximate KKT conditions hold.
    """
    cdef int n = grad.shape[0]
    cdef int k = iact.shape[0]
    cdef bint kktc = True
    cdef Py_ssize_t i
    for i in range(n):
        if i < k and iact[i]:
            kktc = kktc and grad[i] >= -tol
        elif i >= k:
            kktc = kktc and fabs(grad[i]) <= tol
        if not kktc:
            break
    return kktc

cdef bint check_act(double[::1] x, bint[::1] iact):
    """
    Check whether any variable must be included in the working set.

    Parameters
    ----------
    x : memoryview of numpy.ndarray, shape (n,)
        Vector of variables.
    iact : memoryview of numpy.ndarray, shape (k,)
        Working set. The ``i``-th variable in included in the working set if
        ``i < iact.shape[0]`` and ``iact[i] = True``.

    Returns
    -------
    bint
        Whether any variable must be included in the working set.
    """
    cdef int k = iact.shape[0]
    cdef bint actc = True
    cdef Py_ssize_t i
    for i in range(k):
        if not iact[i] and x[i] <= 0.0:
            actc = False
            break
    return actc

cdef void lstsq(double[::1, :] a, double[::1] b, bint[::1] iact, int nact, double rcond, double[::1] x):
    """
    Compute the least-squares solution to the equation ``a @ x = b`` subject to
    ``x[i] = 0`` for any ``i <= iact.shape[0]`` such that ``iact[i] = True``.

    Parameters
    ----------
    a : memoryview of numpy.ndarray, shape (m, n)
        Matrix `a` as shown above.
    b : memoryview of numpy.ndarray, shape (m,)
        Vector `b` as shown above.
    iact : memoryview of numpy.ndarray, shape (k,)
        Indices of the constraints as mentioned above.
    nact : int
        Number of constraints. It must equals ``np.count_nonzero(iact)``, but
        must be provided for numerical efficiency.
    rcond : double
        Tolerance to determine the effective rank of `a`.
    x : memoryview of numpy.ndarray, shape (n,)
        The constrained least-squares solution array.
    """
    cdef int m = a.shape[0]
    cdef int n = a.shape[1]
    cdef int k = iact.shape[0]
    cdef int ldb = max((1, m, n - nact))
    cdef int nrhs = 1

    # Build the unconstrained linear least-squares problem corresponding to the
    # variables that are not included in the working set. The right-hand side
    # vector of the linear least-squares problem stored in xfree will be
    # overwritten by the least-squares solution of the problem.
    cdef int nfree = 0
    cdef double[::1, :] afree = np_empty((m, n - nact), dtype=np_float64, order='F')
    cdef double[::1, :] xfree = np_empty((ldb, nrhs), dtype=np_float64, order='F')
    cdef Py_ssize_t i
    for i in range(n):
        if i >= k or not iact[i]:
            afree[:, nfree] = a[:, i]
            nfree += 1
    xfree[:m, 0] = b[:]

    # Solve the unconstrained linear least-squares problem using the GELSY
    # driver, which builds a complete orthogonal factorization with column
    # pivoting of the matrix afree.
    cdef double temp
    cdef int info
    temp, info = dgelsy_lwork(m, nfree, 1, rcond)
    if info != 0:
        raise ValueError(f'Internal work array size computation failed: {info}')
    cdef int lwork = int(temp)
    cdef int[::1] jpvt = np_zeros(n, dtype=np_int32)
    cdef double[::1] work = np_empty(max(1, lwork), dtype=np_float64)
    cdef int rank
    dgelsy(&m, &nfree, &nrhs, &afree[0, 0], &m, &xfree[0, 0], &ldb, &jpvt[0], &rcond, &rank, &work[0], &lwork, &info)
    if info != 0:
        raise ValueError(f'{-info}-th argument of DGELSY received an illegal value')

    # Build the solution of the constrained linear least-squares problem, by
    # setting the variables included in the working set to zero.
    cdef Py_ssize_t ifree = 0
    for i in range(n):
        if i >= k or not iact[i]:
            x[i] = xfree[ifree, 0]
            ifree += 1
        else:
            x[i] = 0.0
