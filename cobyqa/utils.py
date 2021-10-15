import numpy as np


class RestartRequiredException(Exception):
    """
    Indicate that a trust-region iteration should be restarted.
    """
    pass


def omega_product(zmat, idz, x):
    """
    Compute the products ``zmat @ np.c_[-zmat[:, :idz], zmat[:, idz:]].T @ x``,
    corresponding to the product of the leading submatrix of the inverse KKT
    matrix of interpolation with a vector `x`.

    Parameters
    ----------
    zmat : numpy.ndarray, shape (npt, npt - n - 1)
        Above-mentioned matrix `zmat`.
    idz : int
        Above-mentioned index `idz`.
    x : int or numpy.ndarray, shape (npt,)
        Above-mentioned vector `x`. An integer value represents the
        ``npt``-dimensional vector whose components are all zero, except the
        `x`-th one whose value is one.

    Returns
    -------
    numpy.ndarray, shape (npt,)
        Product ``zmat @ np.c_[-zmat[:, :idz], zmat[:, idz:]].T @ x``.
    """
    if isinstance(x, (int, np.integer)):
        temp = np.r_[-zmat[x, :idz], zmat[x, idz:]]
    else:
        temp = np.dot(np.c_[-zmat[:, :idz], zmat[:, idz:]].T, x)
    return np.dot(zmat, temp)


def normalize_constraint(A, b):
    """
    Normalize linear constraints.

    Each linear constraint is normalized, so that the Euclidean norm of its
    gradient is one (if not zero).

    Parameters
    ----------
    A : array_like, shape (m, n)
        Jacobian matrix of the linear constraints. Each row of `A` stores the
        gradient of a linear constraint.
    b : array_like, shape (m,)
        Right-hand side vector of the linear constraints ``A @ x = b`` or
        ``A @ x <= b``, where ``x`` is ``n``-dimensional.

    Returns
    -------
    A : numpy.ndarray, shape (m, n)
        Normalized Jacobian matrix of the linear constraints. If the input `A`
        is of type ``numpy.ndarray`` with correct size and float data type, then
        the computations is made in-place.
    b : numpy.ndarray, shape (m,)
        Normalized right-hand side vector of the linear constraints. If the
        input `b` is of type ``numpy.ndarray`` with correct size and float data
        type, then the computations is made in-place.
    """
    A = np.atleast_2d(A)
    if A.dtype.kind in np.typecodes['AllInteger']:
        A = np.asarray(A, dtype=float)
    b = np.atleast_1d(b)
    if b.dtype.kind in np.typecodes['AllInteger']:
        b = np.asarray(b, dtype=float)

    tiny = np.finfo(float).tiny
    norm = np.linalg.norm(A, axis=1)
    isafe = norm > tiny * np.max(np.abs(A), axis=1)
    isafe = isafe & (norm > tiny * np.abs(b))
    if np.any(isafe):
        A[isafe, :] = A[isafe, :] / norm[isafe, np.newaxis]
        b[isafe] = b[isafe] / norm[isafe]
    return A, b
