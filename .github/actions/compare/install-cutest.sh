#!/usr/bin/env bash

set -e
set -x

# Download CUTEst and its dependencies
mkdir "$GITHUB_WORKSPACE/cutest"
cd "$GITHUB_WORKSPACE/cutest"
git clone --depth 1 --branch v2.1.24 https://github.com/ralna/ARCHDefs.git ./archdefs
git clone --depth 1 --branch v2.0.6 https://github.com/ralna/SIFDecode.git ./sifdecode
git clone --depth 1 --branch v2.0.17 https://github.com/ralna/CUTEst.git ./cutest
git clone --depth 1 --branch v0.5 https://bitbucket.org/optrove/sif.git ./mastsif

# Set the environment variables
{
  echo "ARCHDEFS=$GITHUB_WORKSPACE/cutest/archdefs"
  echo "SIFDECODE=$GITHUB_WORKSPACE/cutest/sifdecode"
  echo "CUTEST=$GITHUB_WORKSPACE/cutest/cutest"
  echo "MASTSIF=$GITHUB_WORKSPACE/cutest/mastsif"
  echo "MYARCH=pc64.lnx.gfo"
} >> "$GITHUB_ENV"

# Build and install CUTEst
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/jfowkes/pycutest/master/.install_cutest.sh)"
