#!/usr/bin/env bash

set -x
set -e

# Install the dependencies and generate coverage report
python -m pip install --progress-bar=off numpy cython pytest pytest-cov
python setup.py build_ext --inplace
python -m pytest --cov=. --cov-report=xml