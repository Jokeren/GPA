#!/bin/bash

SOURCE_DIR=$(pwd)
DIR=""

if [ $# -ne 1 ]; then
  if [ $# -eq 0 ]; then
    DIR=$(pwd)/gpa
  fi
else
  DIR=$1
fi

if [ -z "$DIR" ]; then
  echo "Wrong prefix"
  exit
fi

mkdir $DIR
cd $DIR

# Install spack
git clone https://github.com/spack/spack.git
export SPACK_ROOT=$(pwd)/spack
export PATH=${SPACK_ROOT}/bin:${PATH}
source ${SPACK_ROOT}/share/spack/setup-env.sh

# Install hpctoolkit dependencies
spack install --only dependencies hpctoolkit ^dyninst@master

CUDA_PATH ?= /usr/local/cuda/

# Find spack dir
B=$(spack find --path boost | tail -n 1 | cut -d ' ' -f 3)
S=${B%/*}

# install hpctoolkit
cd $SOURCE_DIR
cd hpctoolkit
mkdir build
cd build
../configure --prefix=$DIR/hpctoolkit --with-cuda=$CUDA_PATH \
  --with-cupti=$CUDA_PATH --with-spack=$S
make install -j8

echo "Install in "$DIR"/hpctoolkit"

export PATH=$DIR/bin:${PATH}
