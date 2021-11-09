#!/usr/bin/env bash

set -e
set -x

# Fetch the distributions uploaded to Anaconda
ANACONDA_ORG="ragonneau"
COBYQA_URL="https://pypi.anaconda.org/$ANACONDA_ORG/simple/cobyqa/"
python -m pip install --progress-bar=off wheelhouse_uploader
python -m wheelhouse_uploader fetch --version "$COBYQA_VERSION" --local-folder dist/ cobyqa "$COBYQA_URL"