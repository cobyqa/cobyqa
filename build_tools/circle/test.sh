#!/usr/bin/env bash

set -e
set -x

# Install the dependencies and build the solver
python -m pip install --progress-bar=off numpy cython
python setup.py install

# Install the dependencies and test the solver
python -m pip install --progress-bar=off pytest
python -m pytest