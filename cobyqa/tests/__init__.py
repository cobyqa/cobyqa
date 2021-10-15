import operator

import numpy as np
from numpy.testing import assert_, assert_array_compare


def assert_array_less_equal(x, y, err_msg='', verbose=True):
    """
    Raise an AssertionError if two objects are not less-or-equal-ordered.

    Parameters
    ----------
    x : array_like
        Smaller object to check.
    y : array_like
        Larger object to compare.
    err_msg : str, optional
        Error message to be printed in case of failure.
    verbose : bool, optional
        Whether the conflicting values are appended to the error message
        (default is True).

    Raises
    ------
    AssertionError
        The two arrays are not less-or-equal-ordered.
    """
    assert_array_compare(operator.__le__, x, y, err_msg, verbose,
                         'Arrays are not less-or-equal-ordered')


def assert_dtype_equal(actual, desired):
    """
    Compare the data type of two arrays.

    Parameters
    ----------
    actual : array_like or type
        Array obtained.
    desired : array_like or type
        Array desired.

    Raises
    ------
    AssertionError
        The two arrays do not share the same data type.
    """
    if isinstance(actual, np.ndarray):
        act = actual.dtype
    else:
        act = np.dtype(actual)
    if isinstance(desired, np.ndarray):
        des = desired.dtype
    else:
        des = np.dtype(desired)
    assert_(act == des, f'dtype mismatch: "{act}" (should be "{des}")')
