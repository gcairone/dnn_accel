# fastmlp dev environment.
#
# - build-essential/cmake/git: the C++ core, tests and benchmarks.
# - libblis-serial-dev: BLIS, a correctness oracle and performance baseline
#   for the hand-written AVX2 kernels only, never linked by fastmlp_core.
# - python3-dev + nanobind: building the Python extension.
# - pytest: the Python binding/API tests.
# - torch (CPU-only wheel, so no unused CUDA stack): the benchmark baseline,
#   never a runtime dependency of fastmlp.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        util-linux \
        libblis-serial-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir numpy nanobind pytest
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /workspace

# Kept alive as a long-running dev container; the repo is bind-mounted into
# /workspace at `docker run`/`docker create` time, not copied in here.
CMD ["sleep", "infinity"]
