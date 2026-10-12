/*
 * Minimal single-core SGEMM benchmark against a CBLAS-compatible backend
 * (OpenBLAS-serial or BLIS-serial, selected at link time — same source,
 * two binaries). Manual harness on purpose: clock_gettime + warm-up +
 * repeated-block timing + median, no external benchmark framework.
 *
 * Computes Y(B,N) = X(B,K) @ W(K,N), row-major, matching the actual
 * fastmlp forward-pass shape (batch x in_features) @ (in_features x
 * out_features) — not a generic square-matrix benchmark.
 */
#define _GNU_SOURCE
#include <cblas.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef BACKEND_NAME
#define BACKEND_NAME "unknown"
#endif

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
  double da = *(const double *)a, db = *(const double *)b;
  return (da > db) - (da < db);
}

static double median(double *v, int n) {
  qsort(v, n, sizeof(double), cmp_double);
  return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

typedef struct {
  int B, K, N;
  const char *label;
} shape_t;

static void run_shape(const shape_t *s, int inner_iters, int outer_reps, int warmup_blocks) {
  float *X = malloc((size_t)s->B * s->K * sizeof(float));
  float *W = malloc((size_t)s->K * s->N * sizeof(float));
  float *Y = malloc((size_t)s->B * s->N * sizeof(float));

  for (int i = 0; i < s->B * s->K; i++) X[i] = (float)((i % 13) - 6) * 0.1f;
  for (int i = 0; i < s->K * s->N; i++) W[i] = (float)((i % 7) - 3) * 0.1f;

  for (int b = 0; b < warmup_blocks; b++) {
    for (int i = 0; i < inner_iters; i++) {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, s->B, s->N, s->K,
                  1.0f, X, s->K, W, s->N, 0.0f, Y, s->N);
    }
  }

  double *samples = malloc((size_t)outer_reps * sizeof(double));
  for (int r = 0; r < outer_reps; r++) {
    double t0 = now_sec();
    for (int i = 0; i < inner_iters; i++) {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, s->B, s->N, s->K,
                  1.0f, X, s->K, W, s->N, 0.0f, Y, s->N);
    }
    double t1 = now_sec();
    samples[r] = (t1 - t0) / inner_iters;
  }

  double med_sec = median(samples, outer_reps);
  double flops = 2.0 * s->B * s->K * s->N; /* one multiply + one add per MAC */
  double gflops = flops / med_sec / 1e9;

  printf("%-8s %-24s B=%-4d K=%-4d N=%-4d  median=%10.3f us  %8.3f GFLOP/s\n",
         BACKEND_NAME, s->label, s->B, s->K, s->N, med_sec * 1e6, gflops);

  free(X); free(W); free(Y); free(samples);
}

int main(void) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(0, &set);
  if (sched_setaffinity(0, sizeof(set), &set) != 0) {
    fprintf(stderr, "warning: sched_setaffinity failed, not pinned to core 0\n");
  }

  /* (batch, in, out) per layer of small, realistic MLPs: the small-shape
   * regime that must not regress. */
  shape_t shapes[] = {
      {128, 20, 64,  "small L1 (20->64)"},
      {128, 64, 32,  "small L2 (64->32)"},
      {128, 32, 1,   "small L3 (32->1)"},
      {64,  512,128, "enc_img L1 (512->128)"},
      {64,  128,64,  "enc_img L2 (128->64)"},
      {256, 16, 32,  "enc_a L1 (16->32)"},
      {256, 32, 16,  "enc_a L2 (32->16)"},
      {1,   64, 128, "inference B=1 (64->128)"},
  };
  int n_shapes = (int)(sizeof(shapes) / sizeof(shapes[0]));

  printf("=== %s single-core SGEMM benchmark ===\n", BACKEND_NAME);
  for (int i = 0; i < n_shapes; i++) {
    run_shape(&shapes[i], /*inner_iters=*/500, /*outer_reps=*/15, /*warmup_blocks=*/3);
  }
  return 0;
}
