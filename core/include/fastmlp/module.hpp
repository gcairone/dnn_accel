#pragma once
#include "fastmlp/layer.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace fastmlp {

struct ModuleConfig {
  std::vector<int> layer_sizes;  // [0] = this module's own-stack input width
                                  // (raw external width if leaf, sum of
                                  // children's out_features if composite);
                                  // rest = each own-layer's output width.
  // Applied to every own layer unless `activations` is set.
  ActivationKind activation = ActivationKind::ReLU;
  // Per-layer override: empty, or exactly layer_sizes.size()-1 entries (one
  // per own Linear, input side first) -- anything else is a ShapeError.
  std::vector<ActivationKind> activations;
  int max_batch;
  bool inference_only = false;
  // A composite module always needs its own-stack input gradient -- that is
  // what it splits among its children. A LEAF's is a dead end: nothing
  // upstream consumes it, so by default the first layer skips that GEMM
  // entirely (and skips allocating its buffers). Set this to keep it anyway,
  // which is the only way own_input_grad() stays readable on a leaf; the
  // gradient tests in tests/cpp use it as their finite-difference oracle.
  bool retain_input_grad = false;
  std::optional<uint64_t> seed = std::nullopt;
};

// One concrete type mirrors the tree directly (idea.md's Module Design): a
// Module is either a leaf (inputs empty) or has children whose outputs get
// concatenated feature-wise before feeding this module's own Linear stack.
//
// Children are NON-OWNING (raw pointers): usages.md's freeze/load_weights
// example uses a child module independently after passing it into a
// parent's inputs=, which requires it to stay independently alive -- an
// owning tree would make that impossible. The core's contract is simply
// "don't outlive your children"; the Python binding layer is what actually
// keeps children alive via refcounted references.
class Module {
 public:
  explicit Module(const ModuleConfig& cfg, std::vector<Module*> inputs = {});

  // leaf_inputs: one raw pointer per LEAF module in this subtree, flattened
  // in depth-first (left-to-right) order over leaves -- matches usages.md's
  // "flattened, depth-first order over leaves" note. The core has no
  // concept of leaf names/dicts; that's resolved entirely at the Python
  // binding layer before calling in with this fixed positional order.
  // Throws ShapeError if leaf_inputs.size() != num_leaves() or batch > max_batch.
  const float* forward(const std::vector<const float*>& leaf_inputs, int batch);

  // dOut: (batch, out_features()) gradient w.r.t. this module's output.
  // Recurses into children, accumulating weight grads through the whole
  // subtree. Returns nothing (matches usages.md: model.backward() returns
  // nothing). Throws StateError if inference_only.
  void backward(const float* dOut, int batch);

  // Leaf: gradient w.r.t. the raw external input passed to forward(),
  // valid after backward(). Composite: gradient w.r.t. the concatenated
  // child-output buffer (an internal quantity, exposed for debugging/
  // testing). Only valid if !inference_only, and on a leaf only if the
  // config asked for it (ModuleConfig::retain_input_grad) -- otherwise
  // it was never computed and this returns nullptr rather than stale zeros.
  const float* own_input_grad() const { return dx_own_.empty() ? nullptr : dx_own_.data(); }

  void freeze();      // this module's own layers only
  void unfreeze();
  void freeze_tree();  // this + every descendant, recursively
  void unfreeze_tree();
  void zero_grad_tree();

  // Depth-first pre-order (this module first, then children left-to-right).
  // Only a mutable overload: nothing in this codebase needs a const-only
  // traversal, and providing both triggers a real (GCC-flagged) overload
  // ambiguity since std::function's converting constructor makes the two
  // candidates equally viable for a const-callback argument.
  void for_each_in_tree(const std::function<void(Module&)>& fn);

  const std::vector<Module*>& inputs() const { return inputs_; }
  bool is_leaf() const { return inputs_.empty(); }
  int num_leaves() const { return num_leaves_; }

  int in_features() const { return cfg_.layer_sizes.front(); }
  int out_features() const { return cfg_.layer_sizes.back(); }
  int max_batch() const { return cfg_.max_batch; }
  bool inference_only() const { return cfg_.inference_only; }
  // Per layer: layers()[i].activation().

  std::vector<Linear>& layers() { return layers_; }
  const std::vector<Linear>& layers() const { return layers_; }

 private:
  const float* forward_impl(const std::vector<const float*>& leaf_inputs,
                             size_t& next_leaf, int batch);

  ModuleConfig cfg_;
  std::vector<Module*> inputs_;  // non-owning
  std::vector<Linear> layers_;
  int num_leaves_;

  std::vector<float> concat_buf_;  // composite only: own-stack input scratch
  std::vector<float> dx_own_;      // grad w.r.t. own-stack input (!inference_only)
  std::vector<std::vector<float>> child_dgrad_bufs_;  // composite && !inference_only
};

}  // namespace fastmlp
