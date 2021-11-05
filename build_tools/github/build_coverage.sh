#!/usr/bin/env bash

set -e
set -x

python -m pip install --progress-bar=off numpy scipy # cython
python -m pip install --progress-bar=off pytest pytest-cov
python -m pytest --cov=. --cov-report=xml