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
    CALLBACK_SUCCESS = 3
    FEASIBLE_SUCCESS = 4
    MAX_EVAL_WARNING = 5
    MAX_ITER_WARNING = 6
    INFEASIBLE_ERROR = -1
    LINALG_ERROR = -2


class Options(str, Enum):
    """
    Option names.
    """
    DEBUG = 'debug'
    FEASIBILITY_TOL = 'feasibility_tol'
    FILTER_SIZE = 'filter_size'
    HISTORY_SIZE = 'history_size'
    MAX_EVAL = 'maxfev'
    MAX_ITER = 'maxiter'
    NPT = 'nb_points'
    RHOBEG = 'radius_init'
    RHOEND = 'radius_final'
    SCALE = 'scale'
    STORE_HISTORY = 'store_history'
    TARGET = 'target'
    VERBOSE = 'disp'


# Default options.
DEFAULT_OPTIONS = {
    Options.DEBUG.value: False,
    Options.FEASIBILITY_TOL.value: np.sqrt(np.finfo(float).eps),
    Options.FILTER_SIZE.value: sys.maxsize,
    Options.HISTORY_SIZE.value: sys.maxsize,
    Options.MAX_EVAL.value: lambda n: 500 * n,
    Options.MAX_ITER.value: lambda n: 1000 * n,
    Options.NPT.value: lambda n: 2 * n + 1,
    Options.RHOBEG.value: 1.0,
    Options.RHOEND.value: 1e-6,
    Options.SCALE.value: False,
    Options.STORE_HISTORY.value: False,
    Options.TARGET.value: -np.inf,
    Options.VERBOSE.value: False,
}


# Printing options.
PRINT_OPTIONS = {
    'threshold': 6,
    'edgeitems': 2,
    'linewidth': sys.maxsize,
    'formatter': {'float_kind': lambda x: np.format_float_scientific(x, precision=3, unique=False, pad_left=2)}
}


# Constants.
BARRIER = 2.0 ** min(100, np.finfo(float).maxexp // 2, -np.finfo(float).minexp // 2)
