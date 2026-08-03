# Example training programs -- the C++-core-API leg of the three-way
# (C++ core / Python fastmlp / PyTorch) comparison in examples/two_moons
# and examples/mnist. Gated behind FASTMLP_BUILD_EXAMPLES (default OFF,
# same convention as FASTMLP_BUILD_BENCHMARKS): these are demos, not part
# of the correctness/regression gate.
add_executable(two_moons_cpp two_moons/train_cpp.cpp)
target_link_libraries(two_moons_cpp PRIVATE fastmlp_core)
fastmlp_apply_arch_flags(two_moons_cpp)

add_executable(mnist_cpp mnist/train_cpp.cpp)
target_link_libraries(mnist_cpp PRIVATE fastmlp_core)
fastmlp_apply_arch_flags(mnist_cpp)
