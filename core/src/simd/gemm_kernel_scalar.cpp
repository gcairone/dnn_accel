#include "fastmlp/simd/gemm_kernel.hpp"

#include <cstddef>

namespace fastmlp::simd::scalar {

void gemm_xwt_f32(const float* X, const float* W, const float* bias,
                   float* Y, int B, int K, int N) {
  for (int b = 0; b < B; b++) {
    const float* x_row = X + (size_t)b * K;
    for (int n = 0; n < N; n++) {
      const float* w_row = W + (size_t)n * K;
      float acc = bias ? bias[n] : 0.0f;
      for (int k = 0; k < K; k++) acc += x_row[k] * w_row[k];
      Y[(size_t)b * N + n] = acc;
    }
  }
}

void gemv_xwt_f32(const float* x, const float* W, const float* bias, float* y, int K, int N) {
  gemm_xwt_f32(x, W, bias, y, 1, K, N);
}

// `pack` is the AVX2 backend's L1 scratch (see the header). The scalar
// reference has no cache-blocking to do, so it ignores it -- the parameter
// exists only to keep both backends' signatures identical, which is what
// lets test_simd_kernel call them interchangeably.
void gemm_dX_f32(const float* dY, const float* W, float* dX, int B, int K,
                  int N, float* /*pack*/) {
  for (int b = 0; b < B; b++) {
    const float* dy_row = dY + (size_t)b * N;
    float* dx_row = dX + (size_t)b * K;
    for (int k = 0; k < K; k++) dx_row[k] = 0.0f;
    for (int n = 0; n < N; n++) {
      float dy = dy_row[n];
      const float* w_row = W + (size_t)n * K;
      for (int k = 0; k < K; k++) dx_row[k] += dy * w_row[k];
    }
  }
}

void gemm_dW_accum_f32(const float* X, const float* dY, float* dW, int B,
                        int K, int N) {
  for (int n = 0; n < N; n++) {
    float* dw_row = dW + (size_t)n * K;
    for (int b = 0; b < B; b++) {
      float dy = dY[(size_t)b * N + n];
      const float* x_row = X + (size_t)b * K;
      for (int k = 0; k < K; k++) dw_row[k] += dy * x_row[k];
    }
  }
}

void bias_grad_accum_f32(const float* dY, float* db, int B, int N, bool accumulate) {
  if (!accumulate)
    for (int n = 0; n < N; n++) db[n] = 0.0f;
  for (int b = 0; b < B; b++) {
    const float* dy_row = dY + (size_t)b * N;
    for (int n = 0; n < N; n++) db[n] += dy_row[n];
  }
}

void transpose_rows_f32(const float* src, float* dst, int r0, int r1, int src_cols, int ld_dst) {
  for (int c = 0; c < src_cols; c++)
    for (int r = r0; r < r1; r++) dst[(size_t)c * ld_dst + r] = src[(size_t)r * src_cols + c];
}

void pack_transpose_f32(const float* src, float* dst, int src_rows, int src_cols, int ld_dst) {
  for (int c = 0; c < src_cols; c++) {
    float* d = dst + (size_t)c * ld_dst;
    for (int r = 0; r < src_rows; r++) d[r] = src[(size_t)r * src_cols + c];
    for (int r = src_rows; r < ld_dst; r++) d[r] = 0.0f;  // pad, so the kernel never masks
  }
}

void gemm_dW_accum_t_f32(const float* X, const float* dYt, float* dW, int B, int K, int N,
                          float* /*pack*/, bool accumulate) {
  for (int n = 0; n < N; n++) {
    float* dw_row = dW + (size_t)n * K;
    if (!accumulate)
      for (int k = 0; k < K; k++) dw_row[k] = 0.0f;
    const float* dyt_row = dYt + (size_t)n * B;
    for (int b = 0; b < B; b++) {
      float dy = dyt_row[b];
      const float* x_row = X + (size_t)b * K;
      for (int k = 0; k < K; k++) dw_row[k] += dy * x_row[k];
    }
  }
}

void gemm_xwtp_f32(const float* X, const float* Wp, const float* bias, float* Y, int B, int K,
                    int N, int ldWp) {
  for (int b = 0; b < B; b++) {
    float* y_row = Y + (size_t)b * N;
    const float* x_row = X + (size_t)b * K;
    for (int n = 0; n < N; n++) y_row[n] = bias ? bias[n] : 0.0f;
    for (int k = 0; k < K; k++) {
      float xk = x_row[k];
      const float* wp_row = Wp + (size_t)k * ldWp;
      for (int n = 0; n < N; n++) y_row[n] += xk * wp_row[n];
    }
  }
}

}  // namespace fastmlp::simd::scalar
