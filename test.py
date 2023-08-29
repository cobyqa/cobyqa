import numpy as np

from cobyqa import minimize


def fun(x, name):
    if name == 'ackley':
        return -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2.0) / x.size)) - np.exp(np.sum(np.cos(2.0 * np.pi * x)) / x.size) + 20.0 + np.exp(1.0)
    elif name == 'bukin6':
        assert x.size == 2
        return 100.0 * np.sqrt(np.abs(x[1] - 0.01 * x[0] ** 2.0)) + 0.01 * np.abs(x[0] + 10.0)
    else:
        raise NotImplementedError


def x0(n, name):
    if name == 'ackley':
        return 0.01 * np.ones(n)
    elif name == 'bukin6':
        assert n == 2
        return np.array([-9.0, 0.0])
    else:
        raise NotImplementedError


def xl(n, name):
    if name == 'ackley':
        return -32.768 * np.ones(n)
    elif name == 'bukin6':
        assert n == 2
        return np.array([-15.0, -3.0])
    else:
        raise NotImplementedError


def xu(n, name):
    if name == 'ackley':
        return 32.768 * np.ones(n)
    elif name == 'bukin6':
        assert n == 2
        return np.array([-5.0, 3.0])
    else:
        raise NotImplementedError


if __name__ == '__main__':
    n = 2
    name = 'bukin6'

    res = minimize(fun, x0(n, name), name)
    print(res)
