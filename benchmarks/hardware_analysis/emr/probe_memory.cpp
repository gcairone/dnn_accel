// Memory-side probe for the Emerald Rapids target:
//   1. load / store port throughput on L1-resident data (zmm, ymm, broadcast,
//      masked), and the cost of loads that split a cache line
//   2. load-to-use latency per level (pointer chase), 4 KB pages vs THP
//   3. bandwidth per level: read / write / copy / rmw / non-temporal write
//   4. set-conflict map: cost per load of R lines spaced S bytes apart
//      (-> L1 / L2 aliasing strides, the rule behind packed_ldw())
//   5. hardware prefetcher: strided and multi-stream reads from DRAM
//   6. DTLB reach: random page walk, 4 KB pages vs THP
//
//   g++ -O3 -march=emeraldrapids -o probe_memory probe_memory.cpp
#include <random>

#include "common.hpp"

using namespace probe;

alignas(4096) static float g_l1buf[16384];  // 64 KB, the first 2-16 KB of it used as "L1-resident"

// --- 1. ports -------------------------------------------------------------------
// L16 issues 16 independent loads (one per register, no consumer) over 16
// distinct lines; each function below issues 32 (48 for ld_mix) per rep.
// Pure load-port work.
#define L16(INS, REG, STEP)                                                                     \
  INS " 0*" STEP "(%0), %%" REG "0\n\t" INS " 1*" STEP "(%0), %%" REG "1\n\t" INS " 2*" STEP   \
      "(%0), %%" REG "2\n\t" INS " 3*" STEP "(%0), %%" REG "3\n\t" INS " 4*" STEP "(%0), %%" REG \
      "4\n\t" INS " 5*" STEP "(%0), %%" REG "5\n\t" INS " 6*" STEP "(%0), %%" REG "6\n\t" INS    \
      " 7*" STEP "(%0), %%" REG "7\n\t" INS " 8*" STEP "(%0), %%" REG "8\n\t" INS " 9*" STEP     \
      "(%0), %%" REG "9\n\t" INS " 10*" STEP "(%0), %%" REG "10\n\t" INS " 11*" STEP            \
      "(%0), %%" REG "11\n\t" INS " 12*" STEP "(%0), %%" REG "12\n\t" INS " 13*" STEP           \
      "(%0), %%" REG "13\n\t" INS " 14*" STEP "(%0), %%" REG "14\n\t" INS " 15*" STEP           \
      "(%0), %%" REG "15\n\t"
#define CLOB16(R)                                                                                  \
  #R "0", #R "1", #R "2", #R "3", #R "4", #R "5", #R "6", #R "7", #R "8", #R "9", #R "10", #R "11", \
      #R "12", #R "13", #R "14", #R "15"

constexpr int kPortReps = 4000;
__attribute__((noinline)) static void ld_zmm() {
  const float* p = g_l1buf;
  for (int r = 0; r < kPortReps; ++r) {
    asm volatile(L16("vmovaps", "zmm", "64") L16("vmovaps", "zmm", "64+1024") : : "r"(p) : CLOB16(xmm));
  }
}
__attribute__((noinline)) static void ld_ymm() {
  const float* p = g_l1buf;
  for (int r = 0; r < kPortReps; ++r) {
    asm volatile(L16("vmovaps", "ymm", "64") L16("vmovaps", "ymm", "64+1024") : : "r"(p) : CLOB16(xmm));
  }
}
__attribute__((noinline)) static void ld_bcast() {
  const float* p = g_l1buf;
  for (int r = 0; r < kPortReps; ++r) {
    asm volatile(L16("vbroadcastss", "zmm", "64") L16("vbroadcastss", "zmm", "64+1024") : : "r"(p) : CLOB16(xmm));
  }
}
__attribute__((noinline)) static void ld_scalar() {
  const float* p = g_l1buf;
  for (int r = 0; r < kPortReps; ++r) {
    asm volatile(L16("vmovss", "xmm", "64") L16("vmovss", "xmm", "64+1024") : : "r"(p) : CLOB16(xmm));
  }
}
// 2 zmm loads + 1 broadcast pattern (what a 2-wide tile issues per k) x 32
__attribute__((noinline)) static void ld_mix() {
  const float* p = g_l1buf;
  for (int r = 0; r < kPortReps; ++r) {
    asm volatile(L16("vmovaps", "zmm", "64") L16("vbroadcastss", "zmm", "64+1024") L16("vmovaps", "zmm", "64+2048")
                 : : "r"(p) : CLOB16(xmm));
  }
}
__attribute__((noinline)) static void ld_masked() {
  const float* p = g_l1buf;
  __mmask16 k = 0x0fff;
  asm volatile("kmovw %0, %%k1" : : "r"((int)k) : "k1");
  for (int r = 0; r < kPortReps; ++r) {
    asm volatile(
        "vmovups 0(%0), %%zmm0%{%%k1%}%{z%}\n\tvmovups 64(%0), %%zmm1%{%%k1%}%{z%}\n\t"
        "vmovups 128(%0), %%zmm2%{%%k1%}%{z%}\n\tvmovups 192(%0), %%zmm3%{%%k1%}%{z%}\n\t"
        "vmovups 256(%0), %%zmm4%{%%k1%}%{z%}\n\tvmovups 320(%0), %%zmm5%{%%k1%}%{z%}\n\t"
        "vmovups 384(%0), %%zmm6%{%%k1%}%{z%}\n\tvmovups 448(%0), %%zmm7%{%%k1%}%{z%}\n\t"
        "vmovups 512(%0), %%zmm8%{%%k1%}%{z%}\n\tvmovups 576(%0), %%zmm9%{%%k1%}%{z%}\n\t"
        "vmovups 640(%0), %%zmm10%{%%k1%}%{z%}\n\tvmovups 704(%0), %%zmm11%{%%k1%}%{z%}\n\t"
        "vmovups 768(%0), %%zmm12%{%%k1%}%{z%}\n\tvmovups 832(%0), %%zmm13%{%%k1%}%{z%}\n\t"
        "vmovups 896(%0), %%zmm14%{%%k1%}%{z%}\n\tvmovups 960(%0), %%zmm15%{%%k1%}%{z%}\n\t"
        : : "r"(p) : CLOB16(xmm));
  }
}
__attribute__((noinline)) static void st_zmm() {
  float* p = g_l1buf;
  for (int r = 0; r < kPortReps; ++r) {
    asm volatile(
        "vmovaps %%zmm0, 0(%0)\n\tvmovaps %%zmm1, 64(%0)\n\tvmovaps %%zmm2, 128(%0)\n\t"
        "vmovaps %%zmm3, 192(%0)\n\tvmovaps %%zmm4, 256(%0)\n\tvmovaps %%zmm5, 320(%0)\n\t"
        "vmovaps %%zmm6, 384(%0)\n\tvmovaps %%zmm7, 448(%0)\n\tvmovaps %%zmm8, 512(%0)\n\t"
        "vmovaps %%zmm9, 576(%0)\n\tvmovaps %%zmm10, 640(%0)\n\tvmovaps %%zmm11, 704(%0)\n\t"
        "vmovaps %%zmm12, 768(%0)\n\tvmovaps %%zmm13, 832(%0)\n\tvmovaps %%zmm14, 896(%0)\n\t"
        "vmovaps %%zmm15, 960(%0)\n\t"
        : : "r"(p) : "memory");
  }
}
__attribute__((noinline)) static void st_ymm() {
  float* p = g_l1buf;
  for (int r = 0; r < kPortReps; ++r) {
    asm volatile(
        "vmovaps %%ymm0, 0(%0)\n\tvmovaps %%ymm1, 32(%0)\n\tvmovaps %%ymm2, 64(%0)\n\t"
        "vmovaps %%ymm3, 96(%0)\n\tvmovaps %%ymm4, 128(%0)\n\tvmovaps %%ymm5, 160(%0)\n\t"
        "vmovaps %%ymm6, 192(%0)\n\tvmovaps %%ymm7, 224(%0)\n\tvmovaps %%ymm8, 256(%0)\n\t"
        "vmovaps %%ymm9, 288(%0)\n\tvmovaps %%ymm10, 320(%0)\n\tvmovaps %%ymm11, 352(%0)\n\t"
        "vmovaps %%ymm12, 384(%0)\n\tvmovaps %%ymm13, 416(%0)\n\tvmovaps %%ymm14, 448(%0)\n\t"
        "vmovaps %%ymm15, 480(%0)\n\t"
        : : "r"(p) : "memory");
  }
}

// Sequential zmm loads over a 16 KB L1-resident block at byte offset `off`
// from line alignment: off != 0 makes every load straddle two lines.
static int g_off = 0;
__attribute__((noinline)) static void ld_offset() {
  const char* base = (const char*)g_l1buf + g_off;
  __m512 s0 = _mm512_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
  for (int r = 0; r < 64; ++r) {
    for (int i = 0; i < 16384 - 64 * 4; i += 64 * 4) {
      s0 = _mm512_or_ps(s0, _mm512_loadu_ps((const float*)(base + i)));
      s1 = _mm512_or_ps(s1, _mm512_loadu_ps((const float*)(base + i + 64)));
      s2 = _mm512_or_ps(s2, _mm512_loadu_ps((const float*)(base + i + 128)));
      s3 = _mm512_or_ps(s3, _mm512_loadu_ps((const float*)(base + i + 192)));
    }
  }
  do_not_optimize(s0);
  do_not_optimize(s1);
  do_not_optimize(s2);
  do_not_optimize(s3);
}

static void section_ports() {
  std::printf("== 1. L1 load/store throughput per core cycle ==\n");
  std::printf("both cycle measures: via the zmm FMA reference / via the clock read right after\n");
  auto row = [](const char* name, double n, auto f) {
    std::printf("%-30s %5.2f  %5.2f /cyc\n", name, n / zmm_cycles(f), n / cycles_it(f).med);
  };
  row("zmm load  (vmovaps)", 32.0 * kPortReps, ld_zmm);
  row("ymm load  (vmovaps)", 32.0 * kPortReps, ld_ymm);
  row("zmm bcast (vbroadcastss)", 32.0 * kPortReps, ld_bcast);
  row("scalar    (vmovss)", 32.0 * kPortReps, ld_scalar);
  row("2 zmm load + 1 bcast mix", 48.0 * kPortReps, ld_mix);
  row("zmm masked load {k}{z}", 16.0 * kPortReps, ld_masked);
  row("zmm store (vmovaps)", 16.0 * kPortReps, st_zmm);
  row("ymm store (vmovaps)", 16.0 * kPortReps, st_ymm);
  std::printf("zmm loadu at offset from a 64 B line (sequential, 16 KB block):\n");
  const double nl = 64.0 * ((16384 - 256) / 256) * 4;
  for (int off : {0, 4, 16, 32, 48, 60}) {
    g_off = off;
    char name[32];
    std::snprintf(name, sizeof name, "  +%2d B", off);
    row(name, nl, ld_offset);
  }
  std::printf("\n");
}

// --- 2. latency ---------------------------------------------------------------
// Random cyclic permutation over the lines of a buffer; one pointer per line.
static double chase_ns(size_t bytes, bool huge, double* cycles) {
  const size_t lines = bytes / 64;
  char* buf = (char*)alloc_pages(bytes, huge);
  std::vector<size_t> order(lines);
  for (size_t i = 0; i < lines; ++i) order[i] = i;
  std::mt19937_64 rng(42);
  std::shuffle(order.begin() + 1, order.end(), rng);
  for (size_t i = 0; i < lines; ++i) {
    *(char**)(buf + order[i] * 64) = buf + order[(i + 1) % lines] * 64;
  }
  const long steps = 2000000;
  char* p = buf;
  auto t = time_it([&] {
    char* q = p;
    for (long i = 0; i < steps; ++i) q = *(char**)q;
    p = q;
  }, 0.15, 5);
  do_not_optimize(p);
  double hz = add_chain_hz(0.05);
  std::free(buf);
  double ns = t.min / steps * 1e9;
  *cycles = ns * hz / 1e9;
  return ns;
}

static void section_latency() {
  std::printf("== 2. load-to-use latency, random pointer chase over the whole buffer ==\n");
  std::printf("%10s %10s %8s %10s %8s\n", "size", "4K ns", "cyc", "THP ns", "cyc");
  for (size_t kb : {16, 32, 40, 48, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 8192, 16384, 32768, 65536,
                    131072, 262144, 524288, 1048576}) {
    double c4, c2;
    double n4 = chase_ns(kb * 1024, false, &c4);
    double n2 = chase_ns(kb * 1024, true, &c2);
    std::printf("%8zu K %10.2f %8.1f %10.2f %8.1f\n", kb, n4, c4, n2, c2);
  }
  std::printf("\n");
}

// --- 3. bandwidth ---------------------------------------------------------------
__attribute__((noinline)) static float bw_read(const float* p, size_t n) {
  __m512 s[8];
  for (auto& v : s) v = _mm512_setzero_ps();
  for (size_t i = 0; i < n; i += 128)
    for (int j = 0; j < 8; ++j) s[j] = _mm512_add_ps(s[j], _mm512_load_ps(p + i + 16 * j));
  for (int j = 1; j < 8; ++j) s[0] = _mm512_add_ps(s[0], s[j]);
  return _mm512_reduce_add_ps(s[0]);
}
__attribute__((noinline)) static void bw_write(float* p, size_t n) {
  const __m512 x = _mm512_set1_ps(1e-3f);
  for (size_t i = 0; i < n; i += 16) _mm512_store_ps(p + i, x);
}
__attribute__((noinline)) static void bw_write_nt(float* p, size_t n) {
  const __m512 x = _mm512_set1_ps(1e-3f);
  for (size_t i = 0; i < n; i += 16) _mm512_stream_ps(p + i, x);
  _mm_sfence();
}
__attribute__((noinline)) static void bw_copy(float* __restrict d, const float* __restrict s, size_t n) {
  for (size_t i = 0; i < n; i += 16) _mm512_store_ps(d + i, _mm512_load_ps(s + i));
}
// a += s*b: one read stream + one read-modify-write stream (axpy / SGD shape)
__attribute__((noinline)) static void bw_rmw(float* __restrict a, const float* __restrict b, size_t n) {
  const __m512 x = _mm512_set1_ps(1e-9f);
  for (size_t i = 0; i < n; i += 16)
    _mm512_store_ps(a + i, _mm512_fmadd_ps(x, _mm512_load_ps(b + i), _mm512_load_ps(a + i)));
}

static void section_bandwidth() {
  std::printf("== 3. bandwidth, GB/s and bytes/core-cycle (footprint = size; rmw/copy split it in two) ==\n");
  std::printf("%10s %13s %13s %13s %13s %13s\n", "size", "read", "write", "write-NT", "copy", "rmw(axpy)");
  for (size_t kb : {16, 32, 40, 64, 128, 256, 512, 1024, 1536, 1792, 2048, 3072, 4096, 8192, 32768, 131072, 524288}) {
    const size_t n = kb * 1024 / 4;
    float* a = (float*)alloc_pages(n * 4, true);
    float* b = (float*)alloc_pages(n * 4, true);
    for (size_t i = 0; i < n; ++i) a[i] = b[i] = 1e-3f;
    volatile float sink = 0;
    double hz = add_chain_hz(0.02);
    auto gbs = [&](auto f, double bytes) {
      auto t = time_it(f, 0.08, 7);
      return bytes / t.med / 1e9;
    };
    double r = gbs([&] { sink = sink + bw_read(a, n); }, n * 4.0);
    double w = gbs([&] { bw_write(a, n); }, n * 4.0);
    double wnt = gbs([&] { bw_write_nt(a, n); }, n * 4.0);
    double c = gbs([&] { bw_copy(a, b, n / 2); }, (n / 2) * 8.0);
    double m = gbs([&] { bw_rmw(a, b, n / 2); }, (n / 2) * 12.0);
    auto bpc = [&](double g) { return g * 1e9 / hz; };
    std::printf("%8zu K %6.1f (%4.1f) %6.1f (%4.1f) %6.1f (%4.1f) %6.1f (%4.1f) %6.1f (%4.1f)\n", kb, r, bpc(r), w,
                bpc(w), wnt, bpc(wnt), c, bpc(c), m, bpc(m));
    std::free(a);
    std::free(b);
  }
  std::printf("\n");
}

// --- 4. set conflicts ---------------------------------------------------------
// R lines at stride S, one zmm load each, swept `reps` times: the access
// pattern of a kernel walking R rows of a matrix with row stride S. Cycles per
// load vs (S, R) shows where an L1 / L2 set overflows its ways.
static char* g_conf = nullptr;
static size_t g_S = 0;
static int g_R = 0;
__attribute__((noinline)) static void conflict_walk() {
  __m512 s0 = _mm512_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
  const char* p = g_conf;
  for (int rep = 0; rep < 64; ++rep) {
    int r = 0;
    for (; r + 4 <= g_R; r += 4) {
      s0 = _mm512_or_ps(s0, _mm512_load_ps((const float*)(p + (size_t)r * g_S)));
      s1 = _mm512_or_ps(s1, _mm512_load_ps((const float*)(p + (size_t)(r + 1) * g_S)));
      s2 = _mm512_or_ps(s2, _mm512_load_ps((const float*)(p + (size_t)(r + 2) * g_S)));
      s3 = _mm512_or_ps(s3, _mm512_load_ps((const float*)(p + (size_t)(r + 3) * g_S)));
    }
  }
  do_not_optimize(s0);
  do_not_optimize(s1);
  do_not_optimize(s2);
  do_not_optimize(s3);
}

static void section_conflicts() {
  std::printf("== 4. set-conflict map: core cycles per zmm load, R lines at stride S (THP) ==\n");
  const int Rs[] = {8, 12, 16, 20, 24, 32, 64, 128, 256};
  const size_t Ss[] = {64, 128, 256, 512, 1024, 2048, 4096, 4096 + 64, 8192, 8192 + 64, 16384, 32768, 65536,
                       131072, 131072 + 64, 262144};
  const size_t max_bytes = 256 * 262144 + 4096;
  g_conf = (char*)alloc_pages(max_bytes, true);
  std::printf("%10s", "S \\ R");
  for (int R : Rs) std::printf(" %6d", R);
  std::printf("\n");
  for (size_t S : Ss) {
    std::printf("%10zu", S);
    for (int R : Rs) {
      g_S = S;
      g_R = R;
      double c = cycles_it(conflict_walk, 0.003, 7).med / (64.0 * R);
      std::printf(" %6.2f", c);
    }
    std::printf("\n");
  }
  std::free(g_conf);
  std::printf("\n");
}

// --- 5. prefetcher ----------------------------------------------------------------
// (a) one stream, touching one line every `stride` lines: ns per touched line
// (b) `k` sequential streams interleaved one line at a time: aggregate GB/s
static void section_prefetch() {
  std::printf("== 5. hardware prefetch, 1 GB buffer (THP) ==\n");
  const size_t bytes = 1ull << 30;
  char* buf = (char*)alloc_pages(bytes, true);
  std::printf("(a) one stream, one line every `stride` lines: ns / line, effective GB/s of touched lines\n");
  for (size_t stride : {1, 2, 3, 4, 8, 16, 32, 64, 65}) {
    const size_t step = stride * 64;
    const size_t n = std::min<size_t>(bytes / step, 1 << 20);
    auto t = time_it([&] {
      __m512 s = _mm512_setzero_ps();
      for (size_t i = 0; i < n; ++i) s = _mm512_or_ps(s, _mm512_load_ps((const float*)(buf + i * step)));
      do_not_optimize(s);
    }, 0.2, 5);
    double ns = t.med / n * 1e9;
    std::printf("  stride %3zu lines  %6.2f ns/line  %6.1f GB/s\n", stride, ns, 64.0 / ns);
  }
  std::printf("(b) k interleaved sequential streams: aggregate read GB/s\n");
  for (int k : {1, 2, 4, 8, 12, 16, 24, 32, 48, 64}) {
    const size_t per = bytes / k / 64 * 64;
    const size_t lines = std::min<size_t>(per / 64, (64u << 20) / 64 / k);
    auto t = time_it([&] {
      __m512 s[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
      for (size_t l = 0; l < lines; ++l)
        for (int j = 0; j < k; ++j)
          s[j & 3] = _mm512_or_ps(s[j & 3], _mm512_load_ps((const float*)(buf + j * per + l * 64)));
      for (auto& v : s) do_not_optimize(v);
    }, 0.3, 5);
    std::printf("  %2d streams  %6.1f GB/s\n", k, lines * k * 64.0 / t.med / 1e9);
  }
  std::free(buf);
  std::printf("\n");
}

// --- 6. TLB -------------------------------------------------------------------------
static void section_tlb() {
  std::printf("== 6. DTLB reach: random chase, one line per 4 KB page (random line offset) ==\n");
  std::printf("%8s %12s %12s %12s\n", "pages", "4K: ns", "span", "THP: ns");
  for (size_t pages : {16, 32, 64, 96, 128, 256, 512, 1024, 1536, 2048, 4096, 8192, 16384, 32768}) {
    double r[2];
    for (int h = 0; h < 2; ++h) {
      const size_t bytes = pages * 4096;
      char* buf = (char*)alloc_pages(bytes, h == 1);
      std::vector<size_t> order(pages);
      for (size_t i = 0; i < pages; ++i) order[i] = i;
      std::mt19937_64 rng(7);
      std::shuffle(order.begin() + 1, order.end(), rng);
      // Line offset within the page drawn at random: an offset that is a
      // function of the page number (e.g. page % 64) maps every line of a
      // THP-backed buffer onto 64 of the L2's 2048 sets and measures L2
      // conflicts instead of the TLB.
      std::vector<size_t> off(pages);
      for (auto& o : off) o = rng() % 64;
      auto addr = [&](size_t i) { return buf + order[i] * 4096 + off[order[i]] * 64; };
      for (size_t i = 0; i < pages; ++i) *(char**)addr(i) = addr((i + 1) % pages);
      const long steps = 1000000;
      char* p = addr(0);
      auto t = time_it([&] {
        char* q = p;
        for (long i = 0; i < steps; ++i) q = *(char**)q;
        p = q;
      }, 0.1, 5);
      do_not_optimize(p);
      r[h] = t.min / steps * 1e9;
      std::free(buf);
    }
    std::printf("%8zu %12.2f %10zu K %12.2f\n", pages, r[0], pages * 4, r[1]);
  }
  std::printf("\n");
}

int main(int argc, char** argv) {
  pin(core_from_env());
  for (auto& v : g_l1buf) v = 1e-3f;
  std::printf("# probe_memory, core %d\n\n", core_from_env());
  const char* only = argc > 1 ? argv[1] : "";
  auto want = [&](const char* s) { return !*only || std::strstr(only, s); };
  if (want("ports")) section_ports();
  if (want("latency")) section_latency();
  if (want("bandwidth")) section_bandwidth();
  if (want("conflict")) section_conflicts();
  if (want("prefetch")) section_prefetch();
  if (want("tlb")) section_tlb();
  return 0;
}
