#!/bin/sh

set -eux

script/configure

if [ "$(uname)" = 'Darwin' ]; then
  NPROC=$(sysctl -n hw.logicalcpu)
else
  NPROC=$(nproc)
fi

ninja -C build -j${NPROC}
