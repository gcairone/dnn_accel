#include "fastmlp/optimizer.hpp"

#include <cmath>

namespace fastmlp {

Optimizer::Optimizer(Module& root) {
  root.for_each_in_tree([&](Module& m) {
    for (auto& l : m.layers()) params_.push_back(&l);
  });
}

// --- SGD ------------------------------------------------------------

SGD::SGD(Module& root, float lr, float momentum, float weight_decay)
    : Optimizer(root), lr_(lr), momentum_(momentum), weight_decay_(weight_decay) {
  vel_W_.resize(params_.size());
  vel_b_.resize(params_.size());
  for (size_t i = 0; i < params_.size(); i++) {
    vel_W_[i].assign((size_t)params_[i]->out_features() * params_[i]->in_features(), 0.0f);
    vel_b_[i].assign((size_t)params_[i]->out_features(), 0.0f);
  }
}

// Same __restrict rationale as adam_update below: reached through members,
// the compiler cannot prove p/g/vel do not alias and leaves the loop scalar.
// The momentum branch is hoisted out of the loop body too -- it is loop-
// invariant, but leaving it inside blocks vectorization of the common
// momentum==0 path.
static void sgd_update(float* __restrict p, const float* __restrict g_in,
                        float* __restrict vel, int n, float lr, float momentum,
                        float weight_decay) {
  if (momentum != 0.0f) {
    for (int j = 0; j < n; j++) {
      float g = g_in[j] + weight_decay * p[j];
      float vj = momentum * vel[j] + g;
      vel[j] = vj;
      p[j] -= lr * vj;
    }
  } else {
    for (int j = 0; j < n; j++) {
      float g = g_in[j] + weight_decay * p[j];
      p[j] -= lr * g;
    }
  }
}

void SGD::step() {
  for (size_t i = 0; i < params_.size(); i++) {
    Linear* l = params_[i];
    if (l->is_frozen()) continue;

    float* vel = vel_W_[i].data();
    l->update_weights([&](float* w, const float* g, size_t off, size_t n) {
      sgd_update(w, g, vel + off, (int)n, lr_, momentum_, weight_decay_);
    });
    sgd_update(l->bias(), l->bias_grad(), vel_b_[i].data(), l->out_features(), lr_, momentum_,
                /*weight_decay=*/0.0f);
  }
}

void SGD::zero_grad() {
  for (auto* l : params_) l->zero_grad();
}

// --- Adam -------------------------------------------------------------

Adam::Adam(Module& root, float lr, float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(root), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps),
      weight_decay_(weight_decay) {
  m_W_.resize(params_.size());
  v_W_.resize(params_.size());
  m_b_.resize(params_.size());
  v_b_.resize(params_.size());
  for (size_t i = 0; i < params_.size(); i++) {
    size_t nW = (size_t)params_[i]->out_features() * params_[i]->in_features();
    size_t nb = (size_t)params_[i]->out_features();
    m_W_[i].assign(nW, 0.0f);
    v_W_[i].assign(nW, 0.0f);
    m_b_[i].assign(nb, 0.0f);
    v_b_[i].assign(nb, 0.0f);
  }
}

// One Adam update over a contiguous parameter array.
//
// The __restrict qualifiers are load-bearing, not decoration: with the
// arrays reached through members (m_W_[i][j] etc.) the compiler cannot prove
// W does not alias m/v, so it kept the whole loop scalar -- and a scalar
// Adam step measured 69 cycles/parameter (bench_breakdown: 16.9% of an
// entire MNIST training step, at 1.7 GB/s). Hoisting to restrict-qualified
// locals lets it vectorize the divide and sqrt 8-wide.
//
// The two bias corrections are passed pre-reciprocated so the per-element
// work is a multiply rather than a divide; only the final
// lr*mhat/(sqrt(vhat)+eps) stays a real division, since folding that into a
// reciprocal-multiply changes results by more than test_optimizer's 1e-4
// tolerance against the closed-form reference.
static void adam_update(float* __restrict p, const float* __restrict g_in,
                         float* __restrict m, float* __restrict v, int n, float lr,
                         float beta1, float beta2, float eps, float weight_decay,
                         float inv_bc1, float inv_bc2) {
  const float one_minus_b1 = 1.0f - beta1;
  const float one_minus_b2 = 1.0f - beta2;
  for (int j = 0; j < n; j++) {
    float g = g_in[j] + weight_decay * p[j];
    float mj = beta1 * m[j] + one_minus_b1 * g;
    float vj = beta2 * v[j] + one_minus_b2 * g * g;
    m[j] = mj;
    v[j] = vj;
    float mhat = mj * inv_bc1;
    float vhat = vj * inv_bc2;
    p[j] -= lr * mhat / (std::sqrt(vhat) + eps);
  }
}

void Adam::step() {
  t_++;
  float inv_bc1 = 1.0f / (1.0f - std::pow(beta1_, (float)t_));
  float inv_bc2 = 1.0f / (1.0f - std::pow(beta2_, (float)t_));

  for (size_t i = 0; i < params_.size(); i++) {
    Linear* l = params_[i];
    if (l->is_frozen()) continue;

    float* m = m_W_[i].data();
    float* v = v_W_[i].data();
    l->update_weights([&](float* w, const float* g, size_t off, size_t n) {
      adam_update(w, g, m + off, v + off, (int)n, lr_, beta1_, beta2_, eps_, weight_decay_,
                   inv_bc1, inv_bc2);
    });
    adam_update(l->bias(), l->bias_grad(), m_b_[i].data(), v_b_[i].data(), l->out_features(),
                 lr_, beta1_, beta2_, eps_, /*weight_decay=*/0.0f, inv_bc1, inv_bc2);
  }
}

void Adam::zero_grad() {
  for (auto* l : params_) l->zero_grad();
}

}  // namespace fastmlp
