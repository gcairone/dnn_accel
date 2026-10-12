#pragma once
#include <stdexcept>

namespace fastmlp {

// batch > max_batch, mismatched leaf-input count, malformed .mlpf shapes, etc.
// Checked in the core itself, not only in the Python bindings, since C++
// callers have no other safety net.
struct ShapeError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Misuse of API state, e.g. calling backward() on an inference_only Linear.
struct StateError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

}  // namespace fastmlp
