FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PROJECT_DIR="/workspaces/scaluq"
ENV PYTHONPATH="${PROJECT_DIR}/dist:${PROJECT_DIR}/build"

RUN apt-get update && \
    apt-get install -y gcc-13 g++-13 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100 && \
    apt-get install -y --no-install-recommends \
    build-essential \
    autoconf \
    automake \
    ca-certificates \
    ccache \
    clang-format \
    clang-tidy \
    curl \
    gdb \
    git \
    libboost-dev \
    libomp-dev \
    libpython3-dev \
    libtool \
    libzstd-dev \
    locales \
    ninja-build \
    pkg-config \
    python3 \
    python3-pip \
    sudo \
    wget \
    clang-18 \
    clang++-18 \
    llvm-18-dev \
    lld-18 && \
    apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* && \
    update-alternatives --set gcc /usr/bin/gcc-13 && \
    update-alternatives --set g++ /usr/bin/g++-13 && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-18 100

RUN wget https://github.com/Kitware/CMake/releases/download/v3.28.0/cmake-3.28.0-Linux-x86_64.sh -q -O /tmp/cmake-install.sh && \
    chmod u+x /tmp/cmake-install.sh && \
    mkdir /opt/cmake-3.28.0 && \
    /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.28.0 && \
    rm /tmp/cmake-install.sh && \
    ln -s /opt/cmake-3.28.0/bin/* /usr/local/bin

ARG DPCPP_COMMIT=6fc5c4b110365fb9d79117829e4adf9c4f29835d
RUN git init /tmp/llvm && \
    git -C /tmp/llvm remote add origin https://github.com/intel/llvm.git && \
    git -C /tmp/llvm fetch --depth 1 origin ${DPCPP_COMMIT} && \
    git -C /tmp/llvm checkout --detach FETCH_HEAD && \
    CC=gcc-13 CXX=g++-13 python3 /tmp/llvm/buildbot/configure.py \
      --cuda \
      --no-assertions \
      --disable-jit \
      --level_zero_adapter_version V1 \
      --obj-dir /tmp/llvm-build \
      --cmake-opt=-DSYCL_INCLUDE_TESTS=OFF \
      --cmake-opt=-DCUDA_Toolkit_ROOT=/usr/local/cuda && \
    python3 /tmp/llvm/buildbot/compile.py \
      --obj-dir /tmp/llvm-build \
      --build-target deploy-sycl-toolchain && \
    mv /tmp/llvm-build/install /opt/dpcpp && \
    rm -rf /tmp/llvm /tmp/llvm-build

ENV PATH="/opt/dpcpp/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/dpcpp/lib"
ENV CC=clang
ENV CXX=clang++

RUN git clone --depth 1 https://github.com/uxlfoundation/oneDPL.git /tmp/oneDPL && \
    CXXFLAGS=-fsycl cmake -S /tmp/oneDPL -B /tmp/oneDPL/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DONEDPL_BACKEND=dpcpp_only \
    -DONEDPL_ENABLE_SIMD=OFF && \
    cmake --install /tmp/oneDPL/build && \
    rm -rf /tmp/oneDPL

RUN git clone --recursive https://github.com/kokkos/kokkos.git /tmp/kokkos --depth 1 && \
    mkdir /tmp/kokkos/build && \
    cd /tmp/kokkos/build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/tmp/kokkos-build && \
    make install && \
    cp -r /tmp/kokkos-build/include /usr/local/include/kokkos && \
    rm -rf /tmp/kokkos /tmp/kokkos-build

RUN pip install nanobind==2.0.0 --break-system-packages
RUN locale-gen en_US.UTF-8

ARG USERNAME=ubuntu
RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

RUN printf '%s\n' \
    'export PATH=/opt/dpcpp/bin:$PATH' \
    'export LD_LIBRARY_PATH=/opt/dpcpp/lib:${LD_LIBRARY_PATH:-}' \
    'export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-cuda:*}' \
    > /etc/profile.d/10-sycl-nvidia.sh

USER ubuntu
ENV PATH="/home/ubuntu/.local/bin:${PATH}"
