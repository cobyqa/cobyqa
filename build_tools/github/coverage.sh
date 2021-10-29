#!/usr/bin/env bash

set -x
set -e

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install the dependencies and generate coverage report
python -m pip install --progress-bar=off numpy pytest pytest-cov
python -m pytest --cov=./ --cov-report=xml