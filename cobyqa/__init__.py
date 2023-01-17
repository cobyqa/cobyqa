import functools
import logging
from datetime import datetime

from .optimize import minimize
from .utils import OptimizeResult, show_versions

_log = logging.getLogger(__name__)


@functools.lru_cache()
def _ensure_handler():
    """
    Attach a file handler to the root logger.

    The handler is created and attached to the root logger only the first time
    this function is called (the first call is memoized).
    """
    handler = logging.FileHandler(f"{__name__}_{datetime.now().isoformat()}.log")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s"))
    _log.addHandler(handler)
    return handler


def set_loglevel(level):
    """
    Set the root logger and the root logger basic handler levels to `level`,
    creating the handler if it does not exist yet.

    The possible values for `level` are given below.

    ==========  ===============
     Level       Numeric value
    ==========  ===============
     CRITICAL    50
     ERROR       40
     WARNING     30
     INFO        20
     DEBUG       10
     NOTSET      0
    ==========  ===============

    Parameters
    ----------
    level : {int, str}
        Level of the root logger and the root logger basic handler. For example,
        to set the INFO level, use ``logging.INFO``, ``"INFO"``, or ``20``.
    """
    _log.setLevel(level)
    _ensure_handler().setLevel(level)


# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Final release markers:
#   X.Y.0   # For first release after an increment in Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#
# Dev branch marker is: "X.Y.dev" or "X.Y.devN" where N is an integer.
# "X.Y.dev0" is the canonical version of "X.Y.dev"
#
__version__ = "1.0.dev1"

__all__ = ["OptimizeResult", "minimize", "set_loglevel", "show_versions"]
