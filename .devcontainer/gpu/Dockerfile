FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ENV PROJECT_DIR="/workspaces/scaluq"
# Add build artifact to PYTHONPATH and python can find scaluq.
# Egg file name might vary depending on scaluq and python version.
ENV PYTHONPATH="${PROJECT_DIR}/dist:${PYTHONPATH}"
ENV PYTHONPATH="${PROJECT_DIR}/build:${PYTHONPATH}"

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    apt-get install -y gcc-11 g++-11 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 \
    # Remove imagemagick due to https://security-tracker.debian.org/tracker/CVE-2019-10131, which is done in Microsoft's devcontainer image.
    && apt-get purge -y imagemagick imagemagick-6-common \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    clang-format \
    curl \
    git \
    gdb \
    libboost-dev \
    libpython3-dev \
    ninja-build \
    python3 \
    python3-distutils \
    python3-pip \
    wget \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/Kitware/CMake/releases/download/v3.28.0/cmake-3.28.0-Linux-x86_64.sh -q -O /tmp/cmake-install.sh && \
    chmod u+x /tmp/cmake-install.sh && \
    mkdir /opt/cmake-3.28.0 && \
    /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.28.0 && \
    rm /tmp/cmake-install.sh && \
    ln -s /opt/cmake-3.28.0/bin/* /usr/local/bin

RUN git clone --recursive https://github.com/kokkos/kokkos.git /tmp/kokkos --depth 1 && \
    mkdir /tmp/kokkos/build && \
    cd /tmp/kokkos/build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/tmp/kokkos-build && \
    make install && \
    cp -r /tmp/kokkos-build/include /usr/local/include/kokkos && \
    rm -rf /tmp/kokkos /tmp/kokkos-build

USER vscode
