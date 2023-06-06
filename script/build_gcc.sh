#!/bin/sh

set -eux

GCC_COMMAND=${C_COMPILER:-"gcc"}
GXX_COMMAND=${CXX_COMPILER:-"g++"}

USE_TEST=${USE_TEST:-"Yes"}
USE_GPU=${USE_GPU:-"No"}

CMAKE_OPS="-D CMAKE_C_COMPILER=$GCC_COMMAND
  -D CMAKE_CXX_COMPILER=$GXX_COMMAND \
  -D CMAKE_BUILD_TYPE=Release \
  -D USE_GPU=${USE_GPU} \
  -D USE_TEST=${USE_TEST}"

mkdir -p ./build
cmake -B build -G Ninja ${CMAKE_OPS}
ninja -C build -j $(nproc)
