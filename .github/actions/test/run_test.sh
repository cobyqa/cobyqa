#!/usr/bin/env bash

set -e
set -x

python -m pip install --progress-bar=off .
python .github/actions/test/test.py
