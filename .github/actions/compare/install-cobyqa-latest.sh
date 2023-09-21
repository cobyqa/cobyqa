#!/usr/bin/env bash

set -e
set -x

git clone --depth 1 "https://github.com/$GITHUB_REPOSITORY.git" "$GITHUB_WORKSPACE/cobyqa_latest"
cd "$GITHUB_WORKSPACE/cobyqa_latest"
find . -type f -exec sed -i "s/cobyqa/cobyqa_latest/g" {} +
mv cobyqa "cobyqa_latest"
python -m pip install --progress-bar=off .[benchmarks]
