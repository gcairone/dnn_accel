# The C++ leg of benchmarks/suite (driven by run.py). Built with the
# benchmarks because it is one: same AVX2 flags, never shipped.
add_executable(suite_cpp suite_cpp.cpp)
target_link_libraries(suite_cpp PRIVATE fastmlp_core)
fastmlp_apply_arch_flags(suite_cpp)
