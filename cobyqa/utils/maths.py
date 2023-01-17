import numpy as np


def huge(dtype):
    """
    Get a large value.
    """
    return 2.0 ** min(100.0, 0.5 * np.finfo(dtype).maxexp)


def max_abs_arrays(*arrays, initial=1.0):
    """
    Get the largest absolute value among several arrays.
    """
    return max(map(lambda array: np.max(np.abs(array[np.isfinite(array)]), initial=initial), arrays))
