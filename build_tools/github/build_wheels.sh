#!/usr/bin/env bash

set -e
set -x

# Build the wheel distribution
python -m pip install --progress-bar=off cibuildwheel
python -m cibuildwheel --output-dir wheelhouse