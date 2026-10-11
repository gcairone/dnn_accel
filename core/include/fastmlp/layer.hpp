#pragma once
#include "fastmlp/activation.hpp"
#include "fastmlp/simd/gemm_kernel.hpp"

#include <algorithm>
#include <cstddef>
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

  // O(1): marks the gradients as zero without touching them. The next
  // backward() overwrites instead of accumulating, so the zeroing pass and
  // the read of the old dW both disappear (on a model whose gradients live
  // in DRAM those were ~5% and ~10% of a step). Reading the gradients before
  // that backward still sees zeros -- weight_grad()/bias_grad() materialize
  // them on demand.
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
  const float* weight_grad() const {
    materialize_zero_grad();
    return dW_.data();
  }
  const float* bias_grad() const {
    materialize_zero_grad();
    return db_.data();
  }

  // The optimizer's way to change W: calls update(w, g, offset, count) on
  // consecutive row blocks of W and its gradient (both already advanced by
  // `offset` elements, which the optimizer applies to its own state arrays),
  // and transposes each block into the forward's packed copy right after,
  // while it is still in L1. A separate repack on the next forward() would
  // read W back from memory a second time -- from DRAM, once the model's
  // state outgrows the L3.
  template <class F>
  void update_weights(F&& update) {
    materialize_zero_grad();
    const int K = cfg_.in_features, N = cfg_.out_features;
    // Blocks of 16 rows, so each transposed block writes one whole 64-byte
    // line per Wp row (8 rows = half lines, touched twice) -- measured best
    // of 8/16/32/64/whole-matrix -- and of >= ~2k elements, so a narrow
    // layer is not cut into many tiny optimizer calls.
    const int rows = std::max(16, ((2048 / std::max(K, 1)) + 15) / 16 * 16);
    for (int r0 = 0; r0 < N; r0 += rows) {
      const int r1 = std::min(N, r0 + rows);
      const size_t off = (size_t)r0 * K;
      update(W_.data() + off, dW_.data() + off, off, (size_t)(r1 - r0) * K);
      simd::transpose_rows_f32(W_.data(), Wp_.data(), r0, r1, K, ldWp_);
    }
    // Wp_'s padding columns [N, ldWp_) are zero from construction and never
    // written, so the packed copy is now complete.
    w_dirty_ = false;
  }

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
  // mutable: weight_grad()/bias_grad() are logically const but may have to
  // write the zeros a pending zero_grad() deferred.
  mutable std::vector<float> dW_, db_;
  mutable bool grad_zero_pending_ = false;
  void materialize_zero_grad() const;
  std::vector<float> dz_buf_;  // scratch for activation_backward output
  std::vector<float> dx_buf_;  // the buffer backward() returns
  // L1 scratch the dX/dW kernels copy their 16-wide K-slice into, sized for
  // whichever of the two needs more rows -- see pack_scratch_floats().
  std::vector<float> pack_buf_;
};

}  // namespace fastmlp
