#!/bin/sh

set -eux

KOKKOS_TOOLS_DIR="./kokkos-tools"

if [ ! -d "$KOKKOS_TOOLS_DIR" ]; then
    echo "Cloning Kokkos Tools for profiling and debugging..."
    git clone --depth 1 https://github.com/kokkos/kokkos-tools.git "$KOKKOS_TOOLS_DIR"
else
    echo "Kokkos Tools already cloned."
fi

mkdir -p "$KOKKOS_TOOLS_DIR/build"
cd "$KOKKOS_TOOLS_DIR/build"
cmake ..
make
