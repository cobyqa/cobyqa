import os
import platform
import re
import sys
from importlib.metadata import PackageNotFoundError, version

import toml


def _get_sys_info():
    """
    Get system-related information.
    """
    return {
        "python": sys.version.replace(os.linesep, " "),
        "executable": sys.executable,
        "machine": platform.platform(),
    }


def _get_deps_info():
    """
    Get information on the package and its dependencies.
    """
    # TODO: Use `tomllib` (only for Python 3.11 onwards).
    #  End-of-life (EOF) of Python version 3.10: October 2026.
    deps = ["setuptools", "pip", "cobyqa"]
    with open(os.path.abspath("pyproject.toml")) as f:
        for extra_deps in toml.loads(f.read())["project"]["dependencies"]:
            prog = re.compile(r"\s*(?P<dep>\w+).*", flags=re.ASCII)
            match = prog.match(extra_deps)
            if match:
                deps.append(match.group("dep") if match else extra_deps)
    deps_info = {}
    for module in deps:
        try:
            deps_info[module] = version(module)
        except PackageNotFoundError:
            deps_info[module] = None
    return deps_info


def show_versions():
    """
    Print debugging information.
    """
    print("System settings")
    print("---------------")
    sys_info = _get_sys_info()
    sys_width = max(map(len, sys_info.keys())) + 1
    for k, stat in sys_info.items():
        print(f"{k:>{sys_width}}: {stat}")

    print()
    print("Python dependencies")
    print("-------------------")
    deps_info = _get_deps_info()
    deps_width = max(map(len, deps_info.keys())) + 1
    for k, stat in sorted(deps_info.items()):
        print(f"{k:>{deps_width}}: {stat}")
