# syntax=docker/dockerfile:1

# ==============================================================================

# STAGE 1: Wasmtime Builder (Using Ubuntu + Rust)
# ===============================================================================
FROM ubuntu:24.04 AS wasmtime-builder

# 1. Install build dependencies and Rust
RUN apt-get update && apt-get install -y \
    git cmake clang libclang-dev curl unzip build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (nightly)
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y --default-toolchain nightly
ENV PATH="/root/.cargo/bin:$PATH"

# 2. Download LibTorch (x86_64 / CUDA)
WORKDIR /deps
RUN curl -L https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcu124.zip -o libtorch.zip \
    && unzip libtorch.zip \
    && rm libtorch.zip

ENV LIBTORCH=/deps/libtorch
ENV LD_LIBRARY_PATH=/deps/libtorch/lib

# 3. Clone and Checkout Specific Commit
WORKDIR /usr/src
RUN git clone https://github.com/bytecodealliance/wasmtime.git

WORKDIR /usr/src/wasmtime
RUN git checkout 69ef9afc1

# 4. Patch Root Cargo.toml
#    Inject 'features = ["pytorch"]' into the 'wasmtime-wasi-nn' dependency.
RUN sed -i 's|path = "crates/wasi-nn"|path = "crates/wasi-nn", features = ["pytorch"]|' Cargo.toml

# 5. Build the CLI
#    FIX: We define this environment variable to allow using LibTorch 2.7.1
#    even though the Rust crate expects 2.4.0.
ENV LIBTORCH_BYPASS_VERSION_CHECK=1
ENV CXXFLAGS "-D_GLIBCXX_USE_CXX11_ABI=1"

RUN cargo build --release -p wasmtime-cli --features "wasi-nn"

# ==============================================================================
# STAGE 2: Final Runtime Image
# ==============================================================================
FROM debian:stable-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=wasmtime-builder /deps/libtorch /opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib

COPY --from=wasmtime-builder /usr/src/wasmtime/target/release/wasmtime /usr/local/bin/wasmtime

# Copy your WASM and models to /app
COPY target/wasm32-wasip2/release/inference.wasm /app/inference.wasm
COPY models /app/models

WORKDIR /app

EXPOSE 8081/udp

ENTRYPOINT ["wasmtime", "run", "-S", "cli=y", "-S", "nn=y", "-S", "inherit-network=y", "-S", "tcp=y", "--dir", "/app/models::/models", "/app/inference.wasm"]
CMD ["--device", "cpu", "--udp"]