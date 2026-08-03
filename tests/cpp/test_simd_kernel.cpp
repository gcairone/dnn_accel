// M8: the hand-written AVX2+FMA microkernel checked against TWO independent
// oracles -- the project's own scalar reference (always) and BLIS's
// cblas_sgemm (when available), across hw.md's real shapes, tail/edge
// shapes (K/N/B not multiples of the tile), and seeded random fuzz.
#include "fastmlp/simd/gemm_kernel.hpp"

#include <cmath>
#include <random>
#include <vector>

#include "test_harness.hpp"

#ifdef FASTMLP_HAVE_BLIS
#include <cblas.h>
#endif

using namespace fastmlp::simd;

namespace {

void fill_pattern(std::vector<float>& v, int period, int offset) {
  for (size_t i = 0; i < v.size(); i++) {
    v[i] = 0.1f * (float)((int)(i % (size_t)period) - offset);
  }
}

// Relative+absolute tolerance: FMA/accumulation-order differences between
// the scalar reference, the AVX2 kernel, and BLIS are expected (three
// genuinely different implementations), so exact equality isn't the bar --
// but a real correctness bug (wrong index, wrong sign, uninitialized
// memory) shows up as a difference far bigger than float32 rounding noise.
void assert_close(float a, float b) {
  double tol = 1e-4 + 1e-3 * std::fabs((double)a);
  if (std::fabs((double)a - (double)b) > tol) {
    throw fastmlp_test::TestFailure{
        __FILE__, __LINE__,
        "values differ beyond tolerance: " + std::to_string(a) + " vs " + std::to_string(b)};
  }
}

void check_gemm_xwt(int B, int K, int N, bool with_bias) {
  std::vector<float> X((size_t)B * K), W((size_t)N * K), bias((size_t)N);
  fill_pattern(X, 13, 6);
  fill_pattern(W, 7, 3);
  fill_pattern(bias, 5, 2);
  const float* bias_ptr = with_bias ? bias.data() : nullptr;

  std::vector<float> Y_scalar((size_t)B * N), Y_avx2((size_t)B * N);
  scalar::gemm_xwt_f32(X.data(), W.data(), bias_ptr, Y_scalar.data(), B, K, N);
  avx2::gemm_xwt_f32(X.data(), W.data(), bias_ptr, Y_avx2.data(), B, K, N);
  for (size_t i = 0; i < Y_scalar.size(); i++) assert_close(Y_scalar[i], Y_avx2[i]);

#ifdef FASTMLP_HAVE_BLIS
  std::vector<float> Y_blis((size_t)B * N, 0.0f);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, B, N, K, 1.0f, X.data(), K,
              W.data(), K, 0.0f, Y_blis.data(), N);
  if (with_bias) {
    for (int b = 0; b < B; b++)
      for (int n = 0; n < N; n++) Y_blis[(size_t)b * N + n] += bias[n];
  }
  for (size_t i = 0; i < Y_scalar.size(); i++) assert_close(Y_scalar[i], Y_blis[i]);
#endif
}

void check_gemm_dX(int B, int K, int N) {
  std::vector<float> dY((size_t)B * N), W((size_t)N * K);
  fill_pattern(dY, 9, 4);
  fill_pattern(W, 11, 5);
  std::vector<float> dX_scalar((size_t)B * K), dX_avx2((size_t)B * K);
  scalar::gemm_dX_f32(dY.data(), W.data(), dX_scalar.data(), B, K, N);
  avx2::gemm_dX_f32(dY.data(), W.data(), dX_avx2.data(), B, K, N);
  for (size_t i = 0; i < dX_scalar.size(); i++) assert_close(dX_scalar[i], dX_avx2[i]);
}

void check_gemm_dW(int B, int K, int N) {
  std::vector<float> X((size_t)B * K), dY((size_t)B * N);
  fill_pattern(X, 13, 6);
  fill_pattern(dY, 7, 3);
  std::vector<float> dW_scalar((size_t)N * K, 0.0f), dW_avx2((size_t)N * K, 0.0f);
  scalar::gemm_dW_accum_f32(X.data(), dY.data(), dW_scalar.data(), B, K, N);
  avx2::gemm_dW_accum_f32(X.data(), dY.data(), dW_avx2.data(), B, K, N);
  for (size_t i = 0; i < dW_scalar.size(); i++) assert_close(dW_scalar[i], dW_avx2[i]);

  // accumulation semantics: a second call must add, not overwrite, identically.
  scalar::gemm_dW_accum_f32(X.data(), dY.data(), dW_scalar.data(), B, K, N);
  avx2::gemm_dW_accum_f32(X.data(), dY.data(), dW_avx2.data(), B, K, N);
  for (size_t i = 0; i < dW_scalar.size(); i++) assert_close(dW_scalar[i], dW_avx2[i]);
}

void check_bias_grad(int B, int N) {
  std::vector<float> dY((size_t)B * N);
  fill_pattern(dY, 9, 4);
  std::vector<float> db_scalar((size_t)N, 0.0f), db_avx2((size_t)N, 0.0f);
  scalar::bias_grad_accum_f32(dY.data(), db_scalar.data(), B, N);
  avx2::bias_grad_accum_f32(dY.data(), db_avx2.data(), B, N);
  for (int i = 0; i < N; i++) assert_close(db_scalar[i], db_avx2[i]);
}

void check_pack_transpose(int rows, int cols) {
  const int ld = packed_ldw(rows);
  std::vector<float> src((size_t)rows * cols);
  fill_pattern(src, 11, 5);
  std::vector<float> dst_scalar((size_t)cols * ld, -1.0f), dst_avx2((size_t)cols * ld, -2.0f);

  scalar::pack_transpose_f32(src.data(), dst_scalar.data(), rows, cols, ld);
  avx2::pack_transpose_f32(src.data(), dst_avx2.data(), rows, cols, ld);

  for (int c = 0; c < cols; c++) {
    for (int r = 0; r < ld; r++) {
      // Independent oracle: the definition of a transpose, plus the
      // contract that [rows, ld) is zero-padded (the kernels read it).
      float expect = (r < rows) ? src[(size_t)r * cols + c] : 0.0f;
      assert_close(dst_scalar[(size_t)c * ld + r], expect);
      assert_close(dst_avx2[(size_t)c * ld + r], expect);
    }
  }
}

// The packed forward must agree with the unpacked gemm_xwt_f32 -- that is
// the real contract, since Linear::forward switched to the packed path and
// gemm_xwt_f32 stayed behind as the reference.
void check_gemm_xwtp(int B, int K, int N, bool with_bias) {
  const int ldWp = packed_ldw(N);
  std::vector<float> X((size_t)B * K), W((size_t)N * K), bias((size_t)N);
  fill_pattern(X, 13, 6);
  fill_pattern(W, 7, 3);
  fill_pattern(bias, 5, 2);
  const float* bias_ptr = with_bias ? bias.data() : nullptr;

  std::vector<float> Wp((size_t)K * ldWp, 0.0f);
  scalar::pack_transpose_f32(W.data(), Wp.data(), N, K, ldWp);

  std::vector<float> Y_ref((size_t)B * N, -1.0f), Y_scalar((size_t)B * N, -2.0f),
      Y_avx2((size_t)B * N, -3.0f);
  scalar::gemm_xwt_f32(X.data(), W.data(), bias_ptr, Y_ref.data(), B, K, N);
  scalar::gemm_xwtp_f32(X.data(), Wp.data(), bias_ptr, Y_scalar.data(), B, K, N, ldWp);
  avx2::gemm_xwtp_f32(X.data(), Wp.data(), bias_ptr, Y_avx2.data(), B, K, N, ldWp);

  for (size_t i = 0; i < Y_ref.size(); i++) {
    assert_close(Y_scalar[i], Y_ref[i]);
    assert_close(Y_avx2[i], Y_ref[i]);
  }
}

// Likewise the transposed-dY dW accumulation must agree with the untransposed
// gemm_dW_accum_f32, including the "+=" semantics (pre-seeded dW below).
void check_gemm_dW_t(int B, int K, int N) {
  std::vector<float> X((size_t)B * K), dY((size_t)B * N);
  fill_pattern(X, 13, 6);
  fill_pattern(dY, 9, 4);

  std::vector<float> dYt((size_t)N * B, 0.0f);
  scalar::pack_transpose_f32(dY.data(), dYt.data(), B, N, B);

  std::vector<float> dW_ref((size_t)N * K), dW_scalar, dW_avx2;
  fill_pattern(dW_ref, 5, 2);  // non-zero start: verifies accumulation, not assignment
  dW_scalar = dW_ref;
  dW_avx2 = dW_ref;

  scalar::gemm_dW_accum_f32(X.data(), dY.data(), dW_ref.data(), B, K, N);
  scalar::gemm_dW_accum_t_f32(X.data(), dYt.data(), dW_scalar.data(), B, K, N);
  avx2::gemm_dW_accum_t_f32(X.data(), dYt.data(), dW_avx2.data(), B, K, N);

  for (size_t i = 0; i < dW_ref.size(); i++) {
    assert_close(dW_scalar[i], dW_ref[i]);
    assert_close(dW_avx2[i], dW_ref[i]);
  }
}

void check_all(int B, int K, int N) {
  check_gemm_xwt(B, K, N, /*with_bias=*/true);
  check_gemm_xwt(B, K, N, /*with_bias=*/false);
  check_gemm_xwtp(B, K, N, /*with_bias=*/true);
  check_gemm_xwtp(B, K, N, /*with_bias=*/false);
  check_gemm_dX(B, K, N);
  check_gemm_dW(B, K, N);
  check_gemm_dW_t(B, K, N);
  check_bias_grad(B, N);
  check_pack_transpose(B, N);
  check_pack_transpose(N, K);
}

struct Shape {
  int B, K, N;
};

}  // namespace

TEST(hw_md_shape_table) {
  // The exact (B,K,N) triples benchmarked in hw.md, so avx2 correctness is
  // validated on precisely the shapes this project's own hardware analysis
  // cares about, not synthetic square matrices.
  Shape shapes[] = {
      {128, 20, 64}, {128, 64, 32}, {128, 32, 1}, {64, 512, 128},
      {64, 128, 64}, {256, 16, 32}, {256, 32, 16}, {1, 64, 128},
  };
  for (auto& s : shapes) check_all(s.B, s.K, s.N);
}

TEST(tail_and_edge_shapes) {
  // Deliberately includes values that are NOT multiples of the AVX2 width
  // (8) or the microkernel tile (MR=4, NR=2), since usages.md's own shapes
  // (e.g. in_features=20) are not multiples of 8 either.
  int Ks[] = {1, 2, 3, 7, 9, 15, 17};
  int Ns[] = {1, 2, 3, 5, 8, 9};
  int Bs[] = {1, 2, 3, 4, 5, 7, 9};
  for (int K : Ks) {
    for (int N : Ns) {
      for (int B : Bs) {
        check_all(B, K, N);
      }
    }
  }
}

TEST(randomized_fuzz) {
  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> dim(1, 40);
  for (int trial = 0; trial < 200; trial++) {
    check_all(dim(rng), dim(rng), dim(rng));
  }
}

FASTMLP_TEST_MAIN()
