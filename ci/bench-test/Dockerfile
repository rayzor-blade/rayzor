# Dockerfile for testing intermittent benchmark crashes on amd64 Linux
#
# Mirrors the CI environment (Rust stable + LLVM 18 + all-backends).
#
# Build:
#   docker build --platform linux/amd64 -t rayzor-bench-test -f ci/bench-test/Dockerfile .
#
# Run stress test (default: 20 iterations):
#   docker run --platform linux/amd64 rayzor-bench-test
#
# Run with custom iterations:
#   docker run --platform linux/amd64 rayzor-bench-test ./ci/bench-test/stress-test.sh 50
#
# Interactive shell for debugging:
#   docker run --platform linux/amd64 -it rayzor-bench-test bash

FROM rust:bookworm

# Install LLVM 18 on Debian Bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    lsb-release \
    git \
    ca-certificates \
    && wget -qO /usr/share/keyrings/llvm-snapshot.gpg.key https://apt.llvm.org/llvm-snapshot.gpg.key \
    && gpg --dearmor < /usr/share/keyrings/llvm-snapshot.gpg.key > /usr/share/keyrings/llvm-snapshot.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/llvm-snapshot.gpg] http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-18 main" \
       > /etc/apt/sources.list.d/llvm-18.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends llvm-18 llvm-18-dev libpolly-18-dev \
    && rm -rf /var/lib/apt/lists/*

ENV LLVM_SYS_180_PREFIX=/usr/lib/llvm-18
ENV CARGO_TERM_COLOR=always
# Use system git for fetches (libgit2 OOMs on large wasmtime submodules under emulation)
ENV CARGO_NET_GIT_FETCH_WITH_CLI=true
# Limit parallelism to avoid OOM under emulation
ENV CARGO_BUILD_JOBS=2

WORKDIR /rayzor

# Copy full source (manifests + code together to simplify build)
COPY . .

# Build the benchmark runner in release mode with all backends
RUN cargo build --release --package compiler --features all-backends \
    --example benchmark_runner

# Build the SIGSEGV handler shared library for crash diagnosis
# (seghandler.c is already copied via COPY . . above)
RUN gcc -shared -fPIC -o /usr/lib/libseghandler.so ci/bench-test/seghandler.c -ldl -rdynamic

# Make stress test script executable
RUN chmod +x ci/bench-test/stress-test.sh

CMD ["./ci/bench-test/stress-test.sh"]
