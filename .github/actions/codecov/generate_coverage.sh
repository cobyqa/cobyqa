#!/usr/bin/env bash

set -e
set -x

python -m pip install --progress-bar=off .[tests]
python -m pytest --cov=. --cov-report=xml
