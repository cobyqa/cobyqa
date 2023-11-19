#!/usr/bin/env bash

set -e
set -x

# Install cobyqa_latest
find . -type f -maxdepth 1 -exec sed -i "s/cobyqa/cobyqa_latest/g" {} +
find cobyqa -type f -exec sed -i "s/cobyqa/cobyqa_latest/g" {} +
mv cobyqa cobyqa_latest
python -m pip install --progress-bar=off .[benchmarks]

# Install cobyqa
python -m pip install --progress-bar=off cobyqa

# Download CUTEst and its dependencies
mkdir "$GITHUB_WORKSPACE/cutest"
git clone --depth 1 --branch v2.1.28 https://github.com/ralna/ARCHDefs.git "$GITHUB_WORKSPACE/cutest/archdefs"
git clone --depth 1 --branch v2.0.8 https://github.com/ralna/SIFDecode.git "$GITHUB_WORKSPACE/cutest/sifdecode"
git clone --depth 1 --branch v2.0.24 https://github.com/ralna/CUTEst.git "$GITHUB_WORKSPACE/cutest/cutest"
git clone --depth 1 --branch v0.5 https://bitbucket.org/optrove/sif.git "$GITHUB_WORKSPACE/cutest/mastsif"

# Set the environment variables
export ARCHDEFS="$GITHUB_WORKSPACE/cutest/archdefs"
export SIFDECODE="$GITHUB_WORKSPACE/cutest/sifdecode"
export CUTEST="$GITHUB_WORKSPACE/cutest/cutest"
export MASTSIF="$GITHUB_WORKSPACE/cutest/mastsif"
export MYARCH=pc64.lnx.gfo
{
  echo "ARCHDEFS=$ARCHDEFS"
  echo "SIFDECODE=$SIFDECODE"
  echo "CUTEST=$CUTEST"
  echo "MASTSIF=$MASTSIF"
  echo "MYARCH=$MYARCH"
} >> "$GITHUB_ENV"

# Build and install CUTEst
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/jfowkes/pycutest/master/.install_cutest.sh)"

# Add the benchmarks to the PYTHONPATH
echo "PYTHONPATH=$PWD/benchmarks:$PYTHONPATH" >> "$GITHUB_ENV"
