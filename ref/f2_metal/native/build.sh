#!/bin/sh
# Plain clang++ build; MSL is compiled at runtime, no offline metal toolchain needed.
set -e
cd "$(dirname "$0")"
clang++ -std=c++17 -O2 -I"$HOME/tools/metal-f2/metal-cpp" native_bench.cpp -o native_bench \
  -framework Metal -framework Foundation -framework QuartzCore
