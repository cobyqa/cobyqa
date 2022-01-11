#!/usr/bin/env bash

set -e
set -x

# Generate the coverage report
python -m pip install --progress-bar=off numpy scipy cython
python -m pip install --progress-bar=off pytest pytest-cov
python setup.py build_ext --inplace
python -m pytest --cov=. --cov-report=xml