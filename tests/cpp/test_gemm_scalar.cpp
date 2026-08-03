// M1: the naive scalar GEMM is the numerical ground truth every later
// milestone (Linear, Module, and eventually the AVX2 kernel in M8) is
// checked against. These hand-computed values are the literal ground truth
// referenced later.
#include "fastmlp/simd/gemm_kernel.hpp"
#include "test_harness.hpp"

using namespace fastmlp::simd::scalar;

TEST(gemm_xwt_hand_case) {
  const int B = 2, K = 3, N = 2;
  float X[B * K] = {1, 2, 3, 4, 5, 6};
  float W[N * K] = {1, 0, 1, 0, 1, 1};
  float bias[N] = {10, 20};
  float Y[B * N] = {0};

  gemm_xwt_f32(X, W, bias, Y, B, K, N);

  float expected[B * N] = {14, 25, 20, 31};
  for (int i = 0; i < B * N; i++) ASSERT_EQ(Y[i], expected[i]);
}

TEST(gemm_xwt_no_bias) {
  const int B = 2, K = 3, N = 2;
  float X[B * K] = {1, 2, 3, 4, 5, 6};
  float W[N * K] = {1, 0, 1, 0, 1, 1};
  float Y[B * N] = {0};

  gemm_xwt_f32(X, W, /*bias=*/nullptr, Y, B, K, N);

  float expected[B * N] = {4, 5, 10, 11};
  for (int i = 0; i < B * N; i++) ASSERT_EQ(Y[i], expected[i]);
}

TEST(gemm_xwt_edge_K1) {
  const int B = 2, K = 1, N = 2;
  float X[B * K] = {2, 3};
  float W[N * K] = {4, 5};
  float bias[N] = {0, 0};
  float Y[B * N] = {0};

  gemm_xwt_f32(X, W, bias, Y, B, K, N);

  float expected[B * N] = {8, 10, 12, 15};
  for (int i = 0; i < B * N; i++) ASSERT_EQ(Y[i], expected[i]);
}

TEST(gemm_xwt_edge_N1) {
  const int B = 2, K = 3, N = 1;
  float X[B * K] = {1, 2, 3, 4, 5, 6};
  float W[N * K] = {1, 1, 1};
  float bias[N] = {100};
  float Y[B * N] = {0};

  gemm_xwt_f32(X, W, bias, Y, B, K, N);

  float expected[B * N] = {106, 115};
  for (int i = 0; i < B * N; i++) ASSERT_EQ(Y[i], expected[i]);
}

TEST(gemm_xwt_edge_B1) {
  const int B = 1, K = 3, N = 2;
  float X[B * K] = {2, 3, 4};
  float W[N * K] = {1, 0, 0, 0, 1, 0};
  float bias[N] = {0, 0};
  float Y[B * N] = {0};

  gemm_xwt_f32(X, W, bias, Y, B, K, N);

  float expected[B * N] = {2, 3};
  for (int i = 0; i < B * N; i++) ASSERT_EQ(Y[i], expected[i]);
}

TEST(gemm_dX_hand_case) {
  const int B = 2, K = 3, N = 2;
  float dY[B * N] = {1, 2, 3, 4};
  float W[N * K] = {1, 0, 1, 0, 1, 1};
  float dX[B * K] = {0};

  gemm_dX_f32(dY, W, dX, B, K, N, nullptr);

  float expected[B * K] = {1, 2, 3, 3, 4, 7};
  for (int i = 0; i < B * K; i++) ASSERT_EQ(dX[i], expected[i]);
}

TEST(gemm_dW_accum_case) {
  const int B = 2, K = 3, N = 2;
  float X[B * K] = {1, 2, 3, 4, 5, 6};
  float dY[B * N] = {1, 2, 3, 4};
  float dW[N * K] = {0, 0, 0, 0, 0, 0};

  gemm_dW_accum_f32(X, dY, dW, B, K, N);
  float expected_once[N * K] = {13, 17, 21, 18, 24, 30};
  for (int i = 0; i < N * K; i++) ASSERT_EQ(dW[i], expected_once[i]);

  // accumulates, does not overwrite
  gemm_dW_accum_f32(X, dY, dW, B, K, N);
  float expected_twice[N * K] = {26, 34, 42, 36, 48, 60};
  for (int i = 0; i < N * K; i++) ASSERT_EQ(dW[i], expected_twice[i]);
}

TEST(bias_grad_accum_case) {
  const int B = 2, N = 2;
  float dY[B * N] = {1, 2, 3, 4};
  float db[N] = {0, 0};

  bias_grad_accum_f32(dY, db, B, N);
  ASSERT_EQ(db[0], 4.0f);
  ASSERT_EQ(db[1], 6.0f);

  // accumulates, does not overwrite
  bias_grad_accum_f32(dY, db, B, N);
  ASSERT_EQ(db[0], 8.0f);
  ASSERT_EQ(db[1], 12.0f);
}

FASTMLP_TEST_MAIN()
