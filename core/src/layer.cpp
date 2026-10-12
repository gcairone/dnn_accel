#include "fastmlp/layer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

#include "fastmlp/errors.hpp"
#include "fastmlp/simd/gemm_kernel.hpp"

namespace fastmlp {

Linear::Linear(const LinearConfig& cfg) : cfg_(cfg) {
  const size_t in = (size_t)cfg_.in_features;
  const size_t out = (size_t)cfg_.out_features;
  const size_t mb = (size_t)cfg_.max_batch;

  W_.resize(out * in);
  b_.resize(out, 0.0f);
  z_buf_.resize(mb * out);
  a_buf_.resize(mb * out);

  ldWp_ = simd::packed_ldw(cfg_.out_features);
  Wp_.resize(in * (size_t)ldWp_);

  if (!cfg_.inference_only) {
    x_cache_.resize(mb * in);
    dyt_buf_.resize(out * mb);
    dW_.resize(out * in, 0.0f);
    db_.resize(out, 0.0f);
    dz_buf_.resize(mb * out);
    if (cfg_.needs_input_grad) dx_buf_.resize(mb * in);
    // dX packs N rows, dW packs `batch` rows; one buffer serves both since
    // they never run at the same time.
    pack_buf_.resize(simd::pack_scratch_floats((int)std::max(out, mb)));
  }

  // Kaiming-uniform: bound = sqrt(6 / fan_in).
  std::mt19937_64 rng(cfg_.seed.has_value() ? *cfg_.seed : std::random_device{}());
  float bound = std::sqrt(6.0f / (float)cfg_.in_features);
  std::uniform_real_distribution<float> dist(-bound, bound);
  for (auto& w : W_) w = dist(rng);
}

const float* Linear::forward(const float* x, int batch) {
  if (batch <= 0 || batch > cfg_.max_batch) {
    throw ShapeError("Linear::forward: batch out of range");
  }
  if (batch == 1) {
    // The packed tile needs batch rows to fill its accumulators and to amortise
    // the repack; at batch 1 it has neither, so take the row-streaming path
    // over the unpacked W instead (and skip the repack entirely -- w_dirty_
    // stays set, so a later batched call still repacks before it needs Wp_).
    simd::gemv_xwt_f32(x, W_.data(), b_.data(), z_buf_.data(), cfg_.in_features,
                        cfg_.out_features);
  } else {
    if (w_dirty_) {
      simd::pack_transpose_f32(W_.data(), Wp_.data(), cfg_.out_features, cfg_.in_features, ldWp_);
      w_dirty_ = false;
    }
    simd::gemm_xwtp_f32(x, Wp_.data(), b_.data(), z_buf_.data(), batch, cfg_.in_features,
                         cfg_.out_features, ldWp_);
  }
  activation_forward(cfg_.activation, z_buf_.data(), a_buf_.data(),
                      batch * cfg_.out_features);

  if (!cfg_.inference_only) {
    std::memcpy(x_cache_.data(), x, sizeof(float) * (size_t)batch * cfg_.in_features);
  }
  return a_buf_.data();
}

const float* Linear::backward(const float* dY, int batch) {
  if (cfg_.inference_only) {
    throw StateError("Linear::backward: called on an inference_only layer");
  }
  if (batch <= 0 || batch > cfg_.max_batch) {
    throw ShapeError("Linear::backward: batch out of range");
  }

  activation_backward(cfg_.activation, z_buf_.data(), a_buf_.data(), dY,
                       dz_buf_.data(), batch * cfg_.out_features);

  // The dX GEMM costs the same as the dW one; on the first layer of a stack
  // whose caller never reads the input gradient it is pure waste (17% of a
  // whole training step, measured on [512,256,128,1] at B=256).
  if (cfg_.needs_input_grad) {
    simd::gemm_dX_f32(dz_buf_.data(), W_.data(), dx_buf_.data(), batch,
                       cfg_.in_features, cfg_.out_features, pack_buf_.data());
  }

  if (!frozen_) {
    // dW's reduction runs down dz_buf_'s columns; transposing it first turns
    // that into sequential streams (see gemm_dW_accum_t_f32). The transpose
    // is O(batch*out) against the accumulation's O(batch*in*out), so it pays
    // for itself as soon as in_features is more than a handful.
    simd::pack_transpose_f32(dz_buf_.data(), dyt_buf_.data(), batch, cfg_.out_features, batch);
    // After zero_grad() the gradients are stale-but-logically-zero: write
    // them instead of accumulating (see zero_grad()).
    const bool accumulate = !grad_zero_pending_;
    simd::gemm_dW_accum_t_f32(x_cache_.data(), dyt_buf_.data(), dW_.data(), batch,
                               cfg_.in_features, cfg_.out_features, pack_buf_.data(), accumulate);
    simd::bias_grad_accum_f32(dz_buf_.data(), db_.data(), batch, cfg_.out_features, accumulate);
    grad_zero_pending_ = false;
  }

  return cfg_.needs_input_grad ? dx_buf_.data() : nullptr;
}

void Linear::zero_grad() {
  if (cfg_.inference_only) return;
  grad_zero_pending_ = true;
}

void Linear::materialize_zero_grad() const {
  if (!grad_zero_pending_) return;
  std::fill(dW_.begin(), dW_.end(), 0.0f);
  std::fill(db_.begin(), db_.end(), 0.0f);
  grad_zero_pending_ = false;
}

}  // namespace fastmlp
