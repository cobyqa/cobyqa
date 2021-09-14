import numpy as np


class RestartRequiredException(Exception):
    pass


def omega_product(zmat, idz, x):
    if isinstance(x, (int, np.integer)):
        temp = np.r_[-zmat[x, :idz], zmat[x, idz:]]
    else:
        temp = np.dot(np.c_[-zmat[:, :idz], zmat[:, idz:]].T, x)
    return np.dot(zmat, temp)
