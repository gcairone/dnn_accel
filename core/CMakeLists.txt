# fastmlp_core: the C++ core. NEVER links BLIS or any BLAS library --
# that's the structural enforcement of idea.md's "no shipped BLAS
# dependency" mandate. BLIS only ever appears in tests/cpp and
# benchmarks/cpp, gated behind FASTMLP_BLIS_FOUND.
add_library(fastmlp_core STATIC
  src/simd/gemm_kernel_scalar.cpp
  src/simd/gemm_kernel_avx2.cpp
  src/simd/activation_scalar.cpp
  src/simd/activation_avx2.cpp
  src/activation.cpp
  src/layer.cpp
  src/module.cpp
  src/optimizer.cpp
  src/mlpf.cpp
)

target_include_directories(fastmlp_core PUBLIC include)

fastmlp_apply_arch_flags(fastmlp_core)

if(FASTMLP_SIMD_BACKEND STREQUAL "AVX2")
  target_compile_definitions(fastmlp_core PUBLIC FASTMLP_BACKEND_AVX2)
endif()
