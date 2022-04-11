#!/usr/bin/env bash

set -e
set -x

# Generate the coverage report
python3 -m pip install --progress-bar=off numpy scipy cython
python3 -m pip install --progress-bar=off pytest pytest-cov
python3 setup.py build_ext --inplace
python3 -m pytest --cov=. --cov-report=xml
