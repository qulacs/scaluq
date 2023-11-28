FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

ENV PROJECT_DIR="/workspaces/qulacs"
# Add build artifact to PYTHONPATH and python can find qulacs.
# Egg file name might vary depending on qulacs and python version.
ENV PYTHONPATH="${PROJECT_DIR}/dist:${PYTHONPATH}"
ENV PYTHONPATH="${PROJECT_DIR}/build:${PYTHONPATH}"

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y gcc-10 g++-10 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100 \
    # Remove imagemagick due to https://security-tracker.debian.org/tracker/CVE-2019-10131, which is done in Microsoft's devcontainer image.
    && apt-get purge -y imagemagick imagemagick-6-common \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    clang-format \
    cmake \
    curl \
    git \
    gdb \
    libboost-dev \
    libpython3-dev \
    manpages \
    man-db \
    ninja-build \
    python3 \
    python3-distutils \
    python3-pip \
    wget \
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
