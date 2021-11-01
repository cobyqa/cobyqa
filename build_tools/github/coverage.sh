#!/usr/bin/env bash

set -x
set -e

# Install the dependencies and generate coverage report
python -m pip install --progress-bar=off numpy pytest pytest-cov
python -m pytest --cov=. --cov-report=xml