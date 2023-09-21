#!/usr/bin/env bash

set -e
set -x

python -m pip install --progress-bar=off cobyqa
cd benchmarks
python unconstrained.py
