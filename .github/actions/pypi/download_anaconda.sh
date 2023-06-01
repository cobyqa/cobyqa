#!/usr/bin/env bash

set -e
set -x

python -m pip install --progress-bar=off wheelhouse_uploader
python -m wheelhouse_uploader fetch --version "$VERSION" --local-folder dist/ cobyqa "$URL"
