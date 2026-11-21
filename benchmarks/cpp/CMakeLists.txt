add_executable(bench_gemm bench_gemm.cpp)
target_link_libraries(bench_gemm PRIVATE fastmlp_core)
fastmlp_apply_arch_flags(bench_gemm)
if(FASTMLP_BLIS_FOUND)
  target_compile_definitions(bench_gemm PRIVATE FASTMLP_HAVE_BLIS)
  target_include_directories(bench_gemm PRIVATE ${FASTMLP_BLIS_INCLUDE_DIR})
  target_link_libraries(bench_gemm PRIVATE ${FASTMLP_BLIS_LIBRARY})
  set_target_properties(bench_gemm PROPERTIES
    BUILD_RPATH ${FASTMLP_BLIS_LIB_DIR}
    INSTALL_RPATH ${FASTMLP_BLIS_LIB_DIR})
endif()

add_executable(bench_layer bench_layer.cpp)
target_link_libraries(bench_layer PRIVATE fastmlp_core)
fastmlp_apply_arch_flags(bench_layer)

add_executable(bench_train_step bench_train_step.cpp)
target_link_libraries(bench_train_step PRIVATE fastmlp_core)
fastmlp_apply_arch_flags(bench_train_step)

# Phase-by-phase breakdown of a real training step -- the analysis tool that
# locates the bottleneck, as opposed to bench_gemm which tunes a kernel in
# isolation. See its header comment for why both are needed.
add_executable(bench_breakdown bench_breakdown.cpp)
target_link_libraries(bench_breakdown PRIVATE fastmlp_core)
fastmlp_apply_arch_flags(bench_breakdown)

# Per-activation scalar vs avx2 cost, forward and backward.
add_executable(bench_activation bench_activation.cpp)
target_link_libraries(bench_activation PRIVATE fastmlp_core)
fastmlp_apply_arch_flags(bench_activation)

# Per-kernel avx2 vs avx512 vs MKL on the suite's real shapes, timed against
# a zmm FMA reference (see its header comment). Always built with the
# Emerald Rapids ISA: the reference loop and the avx512 kernels need it.
add_executable(bench_kernels bench_kernels.cpp)
target_link_libraries(bench_kernels PRIVATE fastmlp_core)
fastmlp_apply_arch_flags(bench_kernels)
target_compile_options(bench_kernels PRIVATE ${FASTMLP_AVX512_SOURCE_FLAGS})
if(FASTMLP_MKL_FOUND)
  target_compile_definitions(bench_kernels PRIVATE FASTMLP_HAVE_MKL)
  target_include_directories(bench_kernels PRIVATE ${FASTMLP_MKL_HEADER})
  target_link_libraries(bench_kernels PRIVATE ${FASTMLP_MKL_LIBRARY})
  get_filename_component(FASTMLP_MKL_LIBDIR ${FASTMLP_MKL_LIBRARY} DIRECTORY)
  set_target_properties(bench_kernels PROPERTIES BUILD_RPATH ${FASTMLP_MKL_LIBDIR})
endif()
