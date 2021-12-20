import numpy as np

from ._nnls import nnls as _nnls


def nnls(A, b, k=None, maxiter=None):
    """
    Compute the least-squares solution to the equation ``A @ x = b`` subject to
    the nonnegativity constraints ``x[:k] >= 0``.

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

    See Also
    --------
    bvtcg : Bounded variable truncated conjugate gradient
    cpqp : Convex piecewise quadratic programming
    lctcg : Linear constrained truncated conjugate gradient

    Notes
    -----
    The method is adapted from the NNLS algorithm [1]_.

    References
    ----------
    .. [1] C. L. Lawson and R. J. Hanson. Solving Least Squares Problems.
       Classics Appl. Math. Philadelphia, PA, US: SIAM, 1974.
    """
    A = np.atleast_2d(A)
    if A.dtype.kind in np.typecodes['AllInteger']:
        A = np.asarray(A, dtype=float)
    A = np.asfortranarray(A)
    b = np.atleast_1d(b)
    if b.dtype.kind in np.typecodes['AllInteger']:
        b = np.asarray(b, dtype=float)
    n = A.shape[1]
    if k is None:
        k = n
    if k < 0 or k > n:
        raise ValueError('Number of nonnegative constraints is invalid')
    if maxiter is None:
        maxiter = 3 * n

    x = _nnls(A, b, k, maxiter)  # noqa
    return np.array(x, dtype=float)
