import numpy as np
from numpy.testing import assert_


class NullProjectedDirectionException(Exception):
    """
    The projected direction computed by truncated conjugate gradient method is
    null, which must stop the computations.
    """
    pass


def givens(M, cval, sval, i, j, axis, slicing=None):
    r"""
    Perform a Givens rotation on the matrix ``M``.

    Parameters
    ----------
    M : array_like
        Matrix ``M`` as shown above. The Givens rotation is computed in-place.
    cval : float
        Multiple of the cosine value of the angle of rotation.
    sval : float
        Multiple of the sine value of the angle of rotation.
    i : int
        First index of the Givens rotation procedure.
    j : int
        Second index of the Givens rotation procedure.
    axis : int
        Axis over which to select values. If ``M`` is a matrix with two
        dimensions, the calculations will be applied to the rows by setting
        ``axis = 0`` and to the columns by setting ``axis = 1``.
    slicing : slice, optional
        Part of the data at which the Givens rotation should be applied.
        Default applies it to to all the components.

    Returns
    -------
    hval : float
        Length of the two-dimensional vector of components ``cval`` and
        ``sval``, given by :math:`\sqrt{ \mathtt{cval}^2 + \mathtt{sval}^2 }`.
    """
    if slicing is None:
        slicing = slice(None)
    hval = np.hypot(cval, sval)
    cosv = cval / hval
    sinv = sval / hval

    # Whenever M is of type numpy.ndarray, the function numpy.swapaxes returns
    # only a view of M, so that all calculations are carried out in place.
    Mc = np.swapaxes(M, 0, axis)
    Gr = np.array([[cosv, -sinv], [sinv, cosv]], dtype=float)
    Mc[[i, j], slicing] = np.dot(Gr, Mc[[i, j], slicing])

    return hval


def qr(a, overwrite_a=False, pivoting=False, check_finite=True):
    """
    Compute the QR factorization ``a = Q @ R`` where ``Q`` is an orthogonal
    matrix and ``R`` is an upper triangular matrix.

    Parameters
    ----------
    a : array_like, shape (m, n)
        Matrix to be factorized.
    overwrite_a : bool, optional
        Whether to overwrite the data in `a` with the matrix ``R`` (may improve
        the performance by limiting the memory cost).
    pivoting : bool, optional
        Whether the factorization should include column pivoting, in which case
        a permutation vector ``P`` is returned such that ``A[:, P] = Q @ R``.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.

    Returns
    -------
    Q : numpy.ndarray, shape (m, m)
        Above-mentioned orthogonal matrix ``Q``.
    R : numpy.ndarray, shape (m, n)
        Above-mentioned upper triangular matrix ``R``.
    P : numpy.ndarray, shape (n,)
        Indices of the permutation. Not returned if ``pivoting=False``.

    Raises
    ------
    AssertionError
        The matrix `a` is not two-dimensional.
    """
    a = np.asarray_chkfinite(a) if check_finite else np.asarray(a)
    assert_(a.ndim == 2)

    m, n = a.shape
    Q = np.eye(m, dtype=float)
    R = a if overwrite_a else np.copy(a)
    P = np.arange(n, dtype=int)
    for j in range(n):
        if pivoting:
            k = j + np.argmax(np.linalg.norm(R[j:, j:], axis=0))
            P[[j, k]] = P[[k, j]]
            R[:, [j, k]] = R[:, [k, j]]
        for i in range(j + 1, m):
            cval, sval = R[j, j], R[i, j]
            givens(Q, cval, sval, i, j, 1)
            givens(R, cval, sval, i, j, 0)
            R[i, j] = 0.
    if pivoting:
        return Q, R, P
    else:
        return Q, R
