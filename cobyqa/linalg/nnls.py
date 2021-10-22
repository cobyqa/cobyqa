import numpy as np

from .utils import get_lctol


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

    See Also
    --------
    bvtcg : Bounded variable truncated conjugate gradient
    cpqp : Convex piecewise quadratic programming
    lctcg : Linear constrained truncated conjugate gradient

    Notes
    -----
    The method is adapted from the NNLS algorithm [LaHa74]_.

    References
    ----------
    .. [LaHa74] C. L. Lawson and R. J. Hanson. Solving Least Squares Problems.
       Classics Appl. Math. Philadelphia, PA, US: SIAM, 1974.
    """
    A = np.atleast_2d(A)
    if A.dtype.kind in np.typecodes['AllInteger']:
        A = np.asarray(A, dtype=float)
    b = np.atleast_1d(b)
    if b.dtype.kind in np.typecodes['AllInteger']:
        b = np.asarray(b, dtype=float)
    n = A.shape[1]
    if k is None:
        k = n
    if maxiter is None:
        maxiter = 3 * n

    # Define the tolerance of the approximate KKT conditions.
    tiny = np.finfo(float).tiny
    tol = get_lctol(A, b, **kwargs)

    # Initialize the working arrays at the origin of the calculations.
    x = np.zeros(n)
    act = np.ones(k, dtype=bool)
    iact = np.arange(k)
    inact = np.r_[np.zeros(k, dtype=bool), np.ones(n - k, dtype=bool)]
    resid = np.copy(b)
    lsx = np.inner(resid, resid)
    w = np.dot(A.T, resid)

    # Start the iterative procedure. The stopping criteria are the approximate
    # KKT conditions of the least-squares problem, up to a given tolerance.
    iterc = 0
    while np.any(act) and np.any(w[iact] > tol) or np.any(np.abs(w[k:]) > tol):
        # Remove from the active set the least gradient coordinate.
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

        # Increase the active set if necessary.
        while np.any(xact[ipos] <= 0.):
            # Stop the computation if the maximum number of iterations has been
            # reached. The break statement of the outer loop is then reached
            # since the else statement below fails, which ends the computations.
            if iterc >= maxiter:
                x = xact
                break
            iterc += 1

            # Update the trial point, keeping the first components nonnegative.
            idiv = np.abs(x[:k] - xact[:k]) > tiny * np.abs(x[:k])
            upd = inact[:k] & (xact[:k] <= 0.) & idiv
            iupd = np.flatnonzero(upd)
            rxupd = x[iupd] / (x[iupd] - xact[iupd])
            x += np.min(rxupd) * (xact - x)

            # Update the active set according to the intermediate values.
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
