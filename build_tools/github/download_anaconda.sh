#!/usr/bin/env bash

set -e
set -x

# Fetch the distributions uploaded to Anaconda
COBYQA_URL="https://pypi.anaconda.org/ragonneau/simple/cobyqa/"
python3 -m pip install --progress-bar=off wheelhouse_uploader
python3 -m wheelhouse_uploader fetch --version "$COBYQA_VERSION" --local-folder dist/ cobyqa "$COBYQA_URL"
