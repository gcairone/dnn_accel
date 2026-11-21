# fastmlp_core never links BLIS or any other BLAS: the library ships with no
# BLAS dependency. BLIS only appears in tests/cpp and benchmarks/cpp, as an
# oracle, gated behind FASTMLP_BLIS_FOUND.
add_library(fastmlp_core STATIC
  src/simd/gemm_kernel_scalar.cpp
  src/simd/gemm_kernel_avx2.cpp
  src/simd/gemm_kernel_avx512.cpp
  src/simd/activation_scalar.cpp
  src/simd/activation_avx2.cpp
  src/simd/activation_avx512.cpp
  src/activation.cpp
  src/layer.cpp
  src/module.cpp
  src/optimizer.cpp
  src/mlpf.cpp
)

target_include_directories(fastmlp_core PUBLIC include)

fastmlp_apply_arch_flags(fastmlp_core)
set_source_files_properties(src/simd/gemm_kernel_avx512.cpp src/simd/activation_avx512.cpp
  PROPERTIES COMPILE_OPTIONS "${FASTMLP_AVX512_SOURCE_FLAGS}")

if(FASTMLP_SIMD_BACKEND STREQUAL "AVX2")
  target_compile_definitions(fastmlp_core PUBLIC FASTMLP_BACKEND_AVX2)
elseif(FASTMLP_SIMD_BACKEND STREQUAL "AVX512")
  target_compile_definitions(fastmlp_core PUBLIC FASTMLP_BACKEND_AVX512)
endif()
