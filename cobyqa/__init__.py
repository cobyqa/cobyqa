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
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = '1.0.dev0'

try:
    # Enable subpackage importing when binaries are not yet built.
    __COBYQA_SETUP__  # noqa
except NameError:
    __COBYQA_SETUP__ = False

if not __COBYQA_SETUP__:
    from .optimize import OptimizeResult
    from .main import minimize
    from .utils import show_versions

    __all__ = ['OptimizeResult', 'minimize', 'show_versions']
