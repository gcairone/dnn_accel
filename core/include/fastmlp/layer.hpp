#pragma once
#include "fastmlp/activation.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace fastmlp {

struct LinearConfig {
  int in_features;
  int out_features;
  int max_batch;
  ActivationKind activation;
  bool inference_only = false;
  // False for the FIRST layer of a stack whose input gradient nobody reads --
  // the dX GEMM is then dead work exactly as expensive as the dW one. Module
  // sets it (see ModuleConfig::retain_input_grad); constructing a Linear
  // directly keeps the full behaviour.
  bool needs_input_grad = true;
  // nullopt = seeded from std::random_device (nondeterministic); set = a
  // reproducible init, used by tests and anywhere deterministic behavior matters.
  std::optional<uint64_t> seed = std::nullopt;
};

// y = act(x @ W^T + b). Weight layout: row-major (out_features, in_features),
// matching nn.Linear -- contiguous per-output-neuron rows.
//
// Every buffer is allocated exactly once, in the constructor; forward()/
// backward() never allocate (idea.md: zero allocation in the hot path).
class Linear {
 public:
  explicit Linear(const LinearConfig& cfg);

  // x: (batch, in_features) row-major, batch <= max_batch (else ShapeError).
  // Returns a pointer into this layer's own persistent output buffer
  // (batch, out_features) -- valid until the next forward() call on this
  // layer overwrites it (idea.md's preallocate-and-reuse-via-slicing contract).
  const float* forward(const float* x, int batch);

  // dY: (batch, out_features), gradient w.r.t. this layer's POST-activation
  // output. Accumulates into weight_grad()/bias_grad() unless frozen (a
  // frozen layer still computes and returns dX, propagating gradient
  // upstream, but never touches its own weight gradients). Throws
  // StateError if inference_only (no gradient/cache buffers exist).
  // Returns nullptr if the config asked for no input gradient.
  const float* backward(const float* dY, int batch);

  void zero_grad();
  void freeze() { frozen_ = true; }
  void unfreeze() { frozen_ = false; }
  bool is_frozen() const { return frozen_; }

  // Handing out a mutable pointer to W_ invalidates the packed copy
  // forward() uses (see w_dirty_ below), so the non-const accessor marks it
  // stale. The const overload does not -- reading weights is free.
  float* weights() {
    w_dirty_ = true;
    return W_.data();
  }
  float* bias() { return b_.data(); }
  const float* weights() const { return W_.data(); }
  const float* bias() const { return b_.data(); }
  const float* weight_grad() const { return dW_.data(); }
  const float* bias_grad() const { return db_.data(); }

  int in_features() const { return cfg_.in_features; }
  int out_features() const { return cfg_.out_features; }
  int max_batch() const { return cfg_.max_batch; }
  bool inference_only() const { return cfg_.inference_only; }
  ActivationKind activation() const { return cfg_.activation; }

 private:
  LinearConfig cfg_;
  bool frozen_ = false;

  std::vector<float> W_, b_;  // out_features*in_features, out_features

  // W_ transposed into (in_features, ldWp_) for the forward microkernel --
  // see the "PACKED FORWARD" note in simd/gemm_kernel.hpp for why the
  // forward pass wants the other orientation. Rebuilt lazily on the first
  // forward() after any mutable access to W_, so a training step pays the
  // O(K*N) repack once per optimizer step (against O(B*K*N) of GEMM), and
  // an inference_only model pays it exactly once, at construction.
  std::vector<float> Wp_;
  int ldWp_ = 0;
  bool w_dirty_ = true;

  // Always allocated: GEMM writes into z_buf_, activation reads it into a_buf_
  // (the buffer forward() returns).
  std::vector<float> z_buf_, a_buf_;

  // Only allocated when !inference_only:
  std::vector<float> x_cache_;  // input cache, needed for dW in backward()
  std::vector<float> dyt_buf_;  // dz_buf_ transposed, for gemm_dW_accum_t_f32
  std::vector<float> dW_, db_;
  std::vector<float> dz_buf_;  // scratch for activation_backward output
  std::vector<float> dx_buf_;  // the buffer backward() returns
  // L1 scratch the dX/dW kernels copy their 16-wide K-slice into, sized for
  // whichever of the two needs more rows -- see pack_scratch_floats().
  std::vector<float> pack_buf_;
};

}  // namespace fastmlp
