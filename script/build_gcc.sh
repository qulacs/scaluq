#!/bin/sh

set -eux

script/configure_dev

if [ "$(uname)" = 'Darwin' ]; then
  NPROC=$(sysctl -n hw.phisicalcpu)
else
  NPROC=$(nproc)
fi

ninja -C build -j${NPROC}
