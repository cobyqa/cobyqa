#!/usr/bin/env bash

set -e
set -x

# Generate the coverage report
if [[ $(python -c "import sys; print(sys.version_info[0])") -lt 3 ]]; then
    alias python="python3"
fi
python -m pip install --progress-bar=off numpy scipy cython
python -m pip install --progress-bar=off pytest pytest-cov
python setup.py build_ext --inplace
python -m pytest --cov=. --cov-report=xml
