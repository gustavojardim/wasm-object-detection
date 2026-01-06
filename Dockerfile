# syntax=docker/dockerfile:1
FROM ubuntu:24.04

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    xz-utils \
    python3 \
    python3-pip \
    libssl-dev \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

## Install wasmtime 40.0.0 (latest available)
RUN wget https://github.com/bytecodealliance/wasmtime/releases/download/v40.0.0/wasmtime-v40.0.0-x86_64-linux.tar.xz \
    && tar -xf wasmtime-v40.0.0-x86_64-linux.tar.xz \
    && mv wasmtime-v40.0.0-x86_64-linux/wasmtime /usr/local/bin/wasmtime \
    && rm -rf wasmtime-v40.0.0-x86_64-linux*

# Install libtorch 2.4.1 CPU cxx11abi
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcpu.zip \
    && apt-get update && apt-get install -y unzip \
    && unzip libtorch-cxx11-abi-shared-with-deps-2.4.1+cpu.zip -d /opt \
    && rm libtorch-cxx11-abi-shared-with-deps-2.4.1+cpu.zip \
    && apt-get remove -y unzip && apt-get autoremove -y

ENV LD_LIBRARY_PATH=/opt/libtorch/lib

# Copy server and model files
COPY bundle /opt/wasm/

# Expose the server port
EXPOSE 9001

# Set entrypoint (adjust if your server binary is elsewhere or needs args)
ENTRYPOINT ["/opt/wasm/apps/server"]
