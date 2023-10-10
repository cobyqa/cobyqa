import sys
from enum import Enum

import numpy as np


# Exit status.
class ExitStatus(Enum):
    """
    Exit statuses.
    """
    RADIUS_SUCCESS = 0
    TARGET_SUCCESS = 1
    FIXED_SUCCESS = 2
    MAX_EVAL_WARNING = 3
    MAX_ITER_WARNING = 4
    INFEASIBLE_ERROR = -1


class Options(str, Enum):
    """
    Option names.
    """
    DEBUG = 'debug'
    FEASIBILITY_TOL = 'feasibility_tol'
    FILTER_SIZE = 'filter_size'
    HISTORY_SIZE = 'history_size'
    MAX_EVAL = 'max_eval'
    MAX_ITER = 'max_iter'
    NPT = 'npt'
    RADIUS_INIT = 'radius_init'
    RADIUS_FINAL = 'radius_final'
    STORE_HISTORY = 'store_history'
    TARGET = 'target'
    VERBOSE = 'verbose'


# Default options.
DEFAULT_OPTIONS = {
    Options.DEBUG.value: False,
    Options.FEASIBILITY_TOL.value: np.sqrt(np.finfo(float).eps),
    Options.FILTER_SIZE.value: sys.maxsize,
    Options.HISTORY_SIZE.value: sys.maxsize,
    Options.MAX_EVAL.value: lambda n: 500 * n,
    Options.MAX_ITER.value: lambda n: 1000 * n,
    Options.NPT.value: lambda n: 2 * n + 1,
    Options.RADIUS_INIT.value: 1.0,
    Options.RADIUS_FINAL.value: 1e-6,
    Options.STORE_HISTORY.value: False,
    Options.TARGET.value: -np.inf,
    Options.VERBOSE.value: False,
}
