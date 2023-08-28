#!/usr/bin/env python3
"""
Solve the feasibility problem sum(sin(x)) = sum(cos(x)) = 0 in 5 variables.
"""
import numpy as np

from cobyqa import minimize


def sum_trig(x):
    return [np.sum(np.sin(x)), np.sum(np.cos(x))]


if __name__ == '__main__':
    options = {'verbose': True}
    minimize(None, np.zeros(5), ceq=sum_trig, options=options)