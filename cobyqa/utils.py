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
