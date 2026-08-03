# fastmlp dev environment.
#
# Grows incrementally per implementation milestone (see the plan in
# the project notes / benchmarks/hardware_analysis/hw.md) rather than front-loading
# every dependency. Hardware analysis (phase 1-3) is done: target is an AMD
# Ryzen 7 4800H (Zen 2, AVX2+FMA3+F16C only), and BLIS-serial was confirmed
# to build/run correctly and to be the stronger of the two benchmarked
# reference libraries (see hw.md) — it's kept here purely as a correctness
# oracle and performance comparison point for the C++ core's own hand-written
# AVX2 kernel (idea.md: never a shipped runtime dependency of fastmlp_core
# itself). OpenBLAS and PyTorch are NOT installed here — OpenBLAS's baseline
# run is already recorded in hw.md and isn't needed again; PyTorch is only
# needed for the final M12 acceptance benchmark, added then.
#
# M10 adds python3-dev (Python.h, needed to build any extension module) and
# nanobind (idea.md's chosen binding tool over pybind11).
#
# M11 adds pytest (dev-only, runs the Python binding/API test suite).
#
# M12 adds PyTorch CPU (benchmarking-only dependency, idea.md -- the
# acceptance criterion is matching/beating its single-thread training
# throughput, never a runtime dependency of fastmlp itself). The CPU-only
# wheel index keeps this from pulling in an unused CUDA stack.

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
