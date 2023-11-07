# See here for image contents: https://github.com/microsoft/vscode-dev-containers/blob/main/containers/python-3/.devcontainer/base.Dockerfile

# Pick a Python version: 3, 3.10, 3.9, 3.8, 3.7, 3.6
# Append -bullseye or -buster to pin to an OS version.
# Use -bullseye variants on local on arm64/Apple Silicon.
FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.9-bullseye

ENV DEBIAN_FRONTEND=noninteractive

ENV PROJECT_DIR="/workspaces/qulacs"
# Add build artifact to PYTHONPATH and python can find qulacs.
# Egg file name might vary depending on qulacs and python version.
ENV PYTHONPATH="${PROJECT_DIR}/dist:${PYTHONPATH}"
ENV PYTHONPATH="${PROJECT_DIR}/build:${PYTHONPATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    clang-format \
    g++-10 \
    git \
    gdb \
    cmake \
    libboost-dev \
    ninja-build \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip \
    && pip install black flake8 openfermion mypy pybind11-stubgen

RUN git clone --recursive https://github.com/kokkos/kokkos.git /tmp/kokkos --depth 1 && \
    mkdir /tmp/kokkos/build && \
    cd /tmp/kokkos/build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/tmp/kokkos-build && \
    make install && \
    cp -r /tmp/kokkos-build/include /usr/local/include/kokkos && \
    rm -rf /tmp/kokkos /tmp/kokkos-build

USER vscode
