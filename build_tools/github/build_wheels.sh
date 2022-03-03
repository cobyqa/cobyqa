#!/usr/bin/env bash

set -e
set -x

if [[ "$RUNNER_OS" == macOS ]]; then
    if [[ "$CIBW_BUILD" == *-macosx_arm64 ]]; then
        # ARM64 builds must cross compile because CI runs on x64
        export PYTHON_CROSSENV=1

        # SciPy requires 12.0 on ARM architectures to prevent kernel panics
        # https://github.com/scipy/scipy/issues/14688
        export MACOSX_DEPLOYMENT_TARGET=12.0
    else
        export MACOSX_DEPLOYMENT_TARGET=10.13
    fi
fi

# Build the wheel distribution
# Option for development: --platform {auto,linux,macos,windows}
python -m pip install --progress-bar=off cibuildwheel
python -m cibuildwheel --output-dir wheelhouse
