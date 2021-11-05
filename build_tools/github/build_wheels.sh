#!/usr/bin/env bash

set -e
set -x

python -m pip install cibuildwheel
python -m cibuildwheel --output-dir wheelhouse