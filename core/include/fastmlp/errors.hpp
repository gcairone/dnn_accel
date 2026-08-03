#pragma once
#include <stdexcept>

namespace fastmlp {

// batch > max_batch, mismatched leaf-input count, malformed .mlpf shapes, etc.
// The C++ core has no Python safety net of its own callers -- tests/cpp and
// benchmarks/cpp call directly, so these checks live here, not only at the
// Python binding layer.
struct ShapeError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Misuse of API state, e.g. calling backward() on an inference_only Linear.
struct StateError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

}  // namespace fastmlp
