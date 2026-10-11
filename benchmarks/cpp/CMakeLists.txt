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
