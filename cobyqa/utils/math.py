import numpy as np


def get_arrays_tol(*arrays):
    """
    Get a relative tolerance for a set of arrays.

    Parameters
    ----------
    *arrays: tuple
        Set of `arrays` to get the tolerance for.

    Returns
    -------
    float
        Relative tolerance for the set of arrays.
    """
    if len(arrays) == 0:
        raise ValueError('At least one array must be provided.')
    size = max(array.size for array in arrays)
    weight = max(np.max(np.abs(array[np.isfinite(array)]), initial=1.0) for array in arrays)
    return 10.0 * np.finfo(float).eps * max(size, 1.0) * weight
