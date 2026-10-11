cmake_minimum_required(VERSION 3.22)
project(fastmlp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# fastmlp_core is a static lib linked into both plain executables (tests/
# benchmarks) and the nanobind Python extension (a shared object) -- the
# latter requires every object it links to be position-independent.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(FASTMLP_BUILD_TESTS "Build the C++ test suite" ON)
option(FASTMLP_BUILD_BENCHMARKS "Build the C++ benchmark suite" OFF)
option(FASTMLP_BUILD_PYTHON "Build the nanobind Python extension" ON)
option(FASTMLP_BUILD_EXAMPLES "Build the C++-API example training programs" OFF)

# SCALAR is the dev/testing ground-truth backend; AVX2 is what the shipped
# library actually uses. This is a compile-time selector for building/testing
# two independent implementations side by side, not a runtime dispatch
# mechanism (see core/include/fastmlp/simd/gemm_kernel.hpp).
set(FASTMLP_SIMD_BACKEND "SCALAR" CACHE STRING "Active fastmlp_core GEMM backend: SCALAR or AVX2")
set_property(CACHE FASTMLP_SIMD_BACKEND PROPERTY STRINGS SCALAR AVX2)

include(cmake/DetectISA.cmake)

# BLIS: correctness-oracle / performance-comparison only (idea.md -- never a
# shipped runtime dependency of fastmlp_core). Discovered via explicit paths
# already confirmed working in the dev container: Debian's blis-serial
# package doesn't ship a blis.pc pkg-config file, so we don't assume one.
set(FASTMLP_BLIS_INCLUDE_DIR "/usr/include/x86_64-linux-gnu/blis-serial" CACHE PATH "BLIS header directory")
set(FASTMLP_BLIS_LIB_DIR "/usr/lib/x86_64-linux-gnu/blis-serial" CACHE PATH "BLIS library directory")
find_path(FASTMLP_BLIS_HEADER blis.h PATHS ${FASTMLP_BLIS_INCLUDE_DIR} NO_DEFAULT_PATH)
find_library(FASTMLP_BLIS_LIBRARY NAMES blis PATHS ${FASTMLP_BLIS_LIB_DIR} NO_DEFAULT_PATH)
if(FASTMLP_BLIS_HEADER AND FASTMLP_BLIS_LIBRARY)
  set(FASTMLP_BLIS_FOUND TRUE)
  message(STATUS "fastmlp: BLIS found (oracle/benchmark use only) - ${FASTMLP_BLIS_LIBRARY}")
else()
  set(FASTMLP_BLIS_FOUND FALSE)
  message(STATUS "fastmlp: BLIS not found - BLIS-dependent tests/benchmarks will be skipped")
endif()

add_subdirectory(core)

if(FASTMLP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()

if(FASTMLP_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks/cpp)
  add_subdirectory(benchmarks/suite)
endif()

if(FASTMLP_BUILD_PYTHON)
  add_subdirectory(bindings/python)
endif()

if(FASTMLP_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
