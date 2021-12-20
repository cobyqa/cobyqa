#!/usr/bin/env bash

set -e
set -x

# Create a virtual environment outside of the source folder
cd ../../
python -m venv build_env
source build_env/bin/activate

# Build the wheel distribution
cd cobyqa/cobyqa
python -m pip install --progress-bar=off cibuildwheel
python -m cibuildwheel --output-dir wheelhouse