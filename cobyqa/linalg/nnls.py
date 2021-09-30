import numpy as np


def nnls(A, b, k=None, maxiter=None, **kwargs):
    """
    Compute the least-squares solution of ``A @ x = b`` subject to the
    nonnegativity constraints ``x[:k] >= 0``.

    Parameters
    ----------
    A : array_like, shape (m, n)
        Matrix `A` as shown above.
    b : array_like, shape (m,)
        Right-hand side vector `b` as shown above.
    k : int, optional
        Number of nonnegativity constraints. The first `k` components of the
        solution vector are nonnegative (the default is ``A.shape[1]``).
    maxiter : int, optional
        Maximum number of inner iterations (the default is ``3 * A.shape[1]``).

    Returns
    -------
    x : numpy.ndarray, shape (n,)
        Solution vector ``x`` as shown above.
    rnorm : float
        Residual at the solution.

    Other Parameters
    ----------------
    lstol : float, optional
        Tolerance on the approximate KKT conditions for the calculations of the
        least-squares Lagrange multipliers (the default is
        ``10 * eps * max(n, m) * max(1, max(abs(b)))``).

    Notes
    -----
    The method is adapted from the NNLS algorithm [1]_.

    References
    ----------
    .. [1] C. L. Lawson and R. J. Hanson. Solving Least Squares Problems.
       Classics Appl. Math. Philadelphia, PA, US: SIAM, 1974.
    """
    A = np.asarray(A)
    if A.dtype.kind in np.typecodes['AllInteger']:
        A = np.asarray(A, dtype=float)
    b = np.asarray(b)
    if b.dtype.kind in np.typecodes['AllInteger']:
        b = np.asarray(b, dtype=float)
    n = A.shape[1]
    if k is None:
        k = n
    if maxiter is None:
        maxiter = 3 * n

    # Define the tolerance to approximate the KKT conditions.
    eps = np.finfo(float).eps
    tiny = np.finfo(float).eps
    tol = 1e1 * eps * np.max(A.shape) * np.max(np.abs(b), initial=1.)
    tol = kwargs.get('lstol', tol)

    # Start the initialization procedure. The method sets
    # 1. X          solution vector;
    # 2. ACT        column activity flags for the first K components;
    # 3. INACT      column inactivity flags for all the components;
    # 4. IACT       indices of the active columns;
    # 5. RESID      residual of the current iterate;
    # 6. LSX        value of the least-squares objective function;
    # 7. W          gradient of the objective function of the least-squares
    #               problem at the initial guess, so that the first K
    #               components of W store the dual variables of the problem.
    x = np.zeros(n)
    act = np.ones(k, dtype=bool)
    inact = np.r_[np.zeros(k, dtype=bool), np.ones(n - k, dtype=bool)]
    iact = np.arange(k)
    resid = np.copy(b)
    lsx = np.inner(resid, resid)
    w = np.dot(A.T, resid)

    # Start the iterative procedure. The stopping criteria are the approximate
    # KKT conditions of the least-squares problem, up to a given tolerance.
    iterc = 0
    while np.any(act) and np.any(w[iact] > tol) or np.any(np.abs(w[k:]) > tol):
        # Update the active set. The copy of the first K elements of W is
        # crucial, are a slicing returns only a view of the subarray.
        if k > 0:
            wact = np.copy(w[:k])
            wact[inact[:k]] = -np.inf
            act[np.argmax(wact)] = False
            inact[:k] = np.logical_not(act)
            iact = np.flatnonzero(act)
        ipos = np.flatnonzero(inact[:k])

        # Solve the least-squares problem on the inactive columns.
        Ainact = np.delete(A, iact, 1)
        xlstsq, _, _, _ = np.linalg.lstsq(Ainact, b, None)
        xact = np.zeros_like(x)
        xact[inact] = xlstsq

        # Remove the indices from the inactive set which no longer belong.
        while np.any(xact[ipos] <= 0.):
            # Stop the computation if the maximum number of iterations has been
            # reached. The break statement of the outer loop is then reached
            # since the else statement below fails, which ends the computations.
            if iterc >= maxiter:
                x = xact
                break
            iterc += 1

            # Update X to keep the first K components nonnegative.
            idiv = np.greater(np.abs(x[:k] - xact[:k]), tiny * np.abs(x[:k]))
            upd = inact[:k] & (xact[:k] <= 0.) & idiv
            iupd = np.flatnonzero(upd)
            rxupd = x[iupd] / (x[iupd] - xact[iupd])
            x += np.min(rxupd) * (xact - x)

            # Update the active set according to the intermediate values of X.
            act[inact[:k] & (np.abs(x[:k]) < tol)] = True
            inact[:k] = np.logical_not(act)
            iact = np.flatnonzero(act)
            ipos = np.flatnonzero(inact[:k])

            # Solve the least-squares problem on the updated inactive columns.
            Ainact = np.delete(A, iact, 1)
            xlstsq, _, _, _ = np.linalg.lstsq(Ainact, b, None)
            xact = np.zeros_like(x)
            xact[inact] = xlstsq
        else:
            # Continue the computations if any progress has been made, as the
            # inner loop terminated correctly. If the relative change of the
            # objective function between the current iteration and the previous
            # one is insubstantial, stop the computations to prevent infinite
            # cycling, which may occur due to computer rounding errors.
            x = xact
            resid = b - np.dot(A, x)
            lsxnew = np.inner(resid, resid)
            if lsxnew > (1. - tol) * lsx:
                lsx = lsxnew
                break
            lsx = lsxnew
            w = np.dot(A.T, resid)
            continue
        break

    # Calculate the least-squares residual at the solution X.
    rnorm = np.sqrt(lsx)

    return x, rnorm
