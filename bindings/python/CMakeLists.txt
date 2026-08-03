# The only place nanobind and fastmlp_core meet (the project notes' "one rule").
# Discovery follows the standard nanobind CMake pattern -- same "explicit
# paths, don't assume a *.pc/*Config.cmake exists" caution already used for
# BLIS at the top level, except nanobind DOES ship its own cmake_dir helper.
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE FASTMLP_NANOBIND_CMAKE_DIR)
list(APPEND CMAKE_PREFIX_PATH "${FASTMLP_NANOBIND_CMAKE_DIR}")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_fastmlp
  NB_STATIC
  bindings.cpp
  optim_bindings.cpp
)
target_link_libraries(_fastmlp PRIVATE fastmlp_core)
fastmlp_apply_arch_flags(_fastmlp)

# Build straight into the installable package dir so `import fastmlp` (from
# python/) picks up a freshly built extension with no separate install step.
set_target_properties(_fastmlp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/fastmlp"
)
