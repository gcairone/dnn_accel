#include "fastmlp/module.hpp"

#include <cstring>

#include "fastmlp/errors.hpp"

namespace fastmlp {

namespace {
int sum_child_out_features(const std::vector<Module*>& inputs) {
  int total = 0;
  for (auto* m : inputs) total += m->out_features();
  return total;
}
}  // namespace

Module::Module(const ModuleConfig& cfg, std::vector<Module*> inputs)
    : cfg_(cfg), inputs_(std::move(inputs)) {
  if (cfg_.layer_sizes.size() < 2) {
    throw ShapeError("Module: layer_sizes must have at least 2 entries");
  }
  if (!inputs_.empty() &&
      sum_child_out_features(inputs_) != cfg_.layer_sizes.front()) {
    throw ShapeError(
        "Module: layer_sizes[0] must equal the sum of children's out_features");
  }

  if (inputs_.empty()) {
    num_leaves_ = 1;
  } else {
    num_leaves_ = 0;
    for (auto* m : inputs_) num_leaves_ += m->num_leaves();
  }

  layers_.reserve(cfg_.layer_sizes.size() - 1);
  for (size_t i = 0; i + 1 < cfg_.layer_sizes.size(); i++) {
    LinearConfig lc;
    lc.in_features = cfg_.layer_sizes[i];
    lc.out_features = cfg_.layer_sizes[i + 1];
    lc.max_batch = cfg_.max_batch;
    lc.activation = cfg_.activation;
    lc.inference_only = cfg_.inference_only;
    lc.seed = cfg_.seed.has_value() ? std::optional<uint64_t>(*cfg_.seed + i)
                                     : std::nullopt;
    layers_.emplace_back(lc);
  }

  const size_t mb = (size_t)cfg_.max_batch;
  const size_t own_in = (size_t)cfg_.layer_sizes.front();

  if (!inputs_.empty()) {
    concat_buf_.resize(mb * own_in);
  }
  if (!cfg_.inference_only) {
    dx_own_.resize(mb * own_in);
    if (!inputs_.empty()) {
      child_dgrad_bufs_.resize(inputs_.size());
      for (size_t c = 0; c < inputs_.size(); c++) {
        child_dgrad_bufs_[c].resize(mb * (size_t)inputs_[c]->out_features());
      }
    }
  }
}

const float* Module::forward_impl(const std::vector<const float*>& leaf_inputs,
                                   size_t& next_leaf, int batch) {
  const float* stack_input;
  if (is_leaf()) {
    stack_input = leaf_inputs[next_leaf++];
  } else {
    const int total_w = cfg_.layer_sizes.front();
    int col = 0;
    for (auto* child : inputs_) {
      const float* child_out = child->forward_impl(leaf_inputs, next_leaf, batch);
      const int cw = child->out_features();
      for (int b = 0; b < batch; b++) {
        std::memcpy(concat_buf_.data() + (size_t)b * total_w + col,
                    child_out + (size_t)b * cw, sizeof(float) * cw);
      }
      col += cw;
    }
    stack_input = concat_buf_.data();
  }

  const float* cur = stack_input;
  for (auto& layer : layers_) cur = layer.forward(cur, batch);
  return cur;
}

const float* Module::forward(const std::vector<const float*>& leaf_inputs, int batch) {
  if ((int)leaf_inputs.size() != num_leaves_) {
    throw ShapeError("Module::forward: leaf_inputs count does not match num_leaves()");
  }
  if (batch <= 0 || batch > cfg_.max_batch) {
    throw ShapeError("Module::forward: batch out of range");
  }
  size_t next_leaf = 0;
  return forward_impl(leaf_inputs, next_leaf, batch);
}

void Module::backward(const float* dOut, int batch) {
  if (cfg_.inference_only) {
    throw StateError("Module::backward: called on an inference_only module");
  }
  if (batch <= 0 || batch > cfg_.max_batch) {
    throw ShapeError("Module::backward: batch out of range");
  }

  const float* cur = dOut;
  for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
    cur = it->backward(cur, batch);
  }
  // `cur` aliases the first layer's internal dx buffer, which the next
  // forward()/backward() call on that layer would clobber -- copy it out.
  std::memcpy(dx_own_.data(), cur, sizeof(float) * dx_own_.size());

  if (!is_leaf()) {
    const int total_w = cfg_.layer_sizes.front();
    int col = 0;
    for (size_t c = 0; c < inputs_.size(); c++) {
      Module* child = inputs_[c];
      const int cw = child->out_features();
      float* seg = child_dgrad_bufs_[c].data();
      for (int b = 0; b < batch; b++) {
        std::memcpy(seg + (size_t)b * cw,
                    dx_own_.data() + (size_t)b * total_w + col,
                    sizeof(float) * cw);
      }
      col += cw;
      child->backward(seg, batch);
    }
  }
}

void Module::freeze() {
  for (auto& l : layers_) l.freeze();
}
void Module::unfreeze() {
  for (auto& l : layers_) l.unfreeze();
}
void Module::freeze_tree() {
  for_each_in_tree([](Module& m) { m.freeze(); });
}
void Module::unfreeze_tree() {
  for_each_in_tree([](Module& m) { m.unfreeze(); });
}
void Module::zero_grad_tree() {
  for_each_in_tree([](Module& m) {
    for (auto& l : m.layers_) l.zero_grad();
  });
}

void Module::for_each_in_tree(const std::function<void(Module&)>& fn) {
  fn(*this);
  for (auto* child : inputs_) child->for_each_in_tree(fn);
}

}  // namespace fastmlp
