add_executable(test_gemm_scalar test_gemm_scalar.cpp)
target_link_libraries(test_gemm_scalar PRIVATE fastmlp_core)
fastmlp_apply_arch_flags(test_gemm_scalar)
add_test(NAME test_gemm_scalar COMMAND test_gemm_scalar)

add_executable(test_activation test_activation.cpp)
target_link_libraries(test_activation PRIVATE fastmlp_core)
fastmlp_apply_arch_flags(test_activation)
add_test(NAME test_activation COMMAND test_activation)

add_executable(test_layer test_layer.cpp)
target_link_libraries(test_layer PRIVATE fastmlp_core)
fastmlp_apply_arch_flags(test_layer)
add_test(NAME test_layer COMMAND test_layer)

add_executable(test_module_tree test_module_tree.cpp)
target_link_libraries(test_module_tree PRIVATE fastmlp_core)
fastmlp_apply_arch_flags(test_module_tree)
add_test(NAME test_module_tree COMMAND test_module_tree)

add_executable(test_optimizer test_optimizer.cpp)
target_link_libraries(test_optimizer PRIVATE fastmlp_core)
fastmlp_apply_arch_flags(test_optimizer)
add_test(NAME test_optimizer COMMAND test_optimizer)

add_executable(test_mlpf_roundtrip test_mlpf_roundtrip.cpp)
target_link_libraries(test_mlpf_roundtrip PRIVATE fastmlp_core)
fastmlp_apply_arch_flags(test_mlpf_roundtrip)
add_test(NAME test_mlpf_roundtrip COMMAND test_mlpf_roundtrip)

add_executable(test_simd_kernel test_simd_kernel.cpp)
target_link_libraries(test_simd_kernel PRIVATE fastmlp_core)
fastmlp_apply_arch_flags(test_simd_kernel)
if(FASTMLP_BLIS_FOUND)
  target_compile_definitions(test_simd_kernel PRIVATE FASTMLP_HAVE_BLIS)
  target_include_directories(test_simd_kernel PRIVATE ${FASTMLP_BLIS_INCLUDE_DIR})
  target_link_libraries(test_simd_kernel PRIVATE ${FASTMLP_BLIS_LIBRARY})
  set_target_properties(test_simd_kernel PROPERTIES
    BUILD_RPATH ${FASTMLP_BLIS_LIB_DIR}
    INSTALL_RPATH ${FASTMLP_BLIS_LIB_DIR})
endif()
add_test(NAME test_simd_kernel COMMAND test_simd_kernel)

if(FASTMLP_BLIS_FOUND)
  add_executable(test_blis_smoke test_blis_smoke.cpp)
  target_include_directories(test_blis_smoke PRIVATE ${FASTMLP_BLIS_INCLUDE_DIR})
  target_link_libraries(test_blis_smoke PRIVATE ${FASTMLP_BLIS_LIBRARY})
  set_target_properties(test_blis_smoke PROPERTIES
    BUILD_RPATH ${FASTMLP_BLIS_LIB_DIR}
    INSTALL_RPATH ${FASTMLP_BLIS_LIB_DIR})
  fastmlp_apply_arch_flags(test_blis_smoke)
  add_test(NAME test_blis_smoke COMMAND test_blis_smoke)
else()
  message(WARNING "fastmlp: BLIS not found, skipping test_blis_smoke")
endif()
