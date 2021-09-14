import numpy as np


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
