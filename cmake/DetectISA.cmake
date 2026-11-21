# Two fixed, known targets -- chosen here at build time, never detected at
# runtime:
#
#   SCALAR / AVX2 backends: AMD Ryzen 7 4800H (Zen 2). No AVX-512 there, so
#     AVX2+FMA3+F16C is the only SIMD path.
#   AVX512 backend: Intel Xeon, Emerald Rapids (family 6, model 207; Raptor
#     Cove cores), the development VM. AVX-512 with 2 x 512-bit FMA units; see
#     benchmarks/hardware_analysis/results/emr/ for every figure the kernels
#     are tuned against.
#
# -fno-math-errno is a measured necessity, not boilerplate: by default GCC
# must assume std::sqrt sets errno, so it cannot contract it to a bare
# vsqrtps and guards every call with a branch -- which blocks vectorization
# of any loop containing one. That kept Adam::step scalar (bench_breakdown
# measured it at 16.9% of a whole MNIST training step). This flag only
# asserts "we never read errno after a math call", which is true here; it
# does NOT relax rounding or allow reassociation the way -ffast-math would,
# so numerical results stay bit-identical.
#
# -mprefer-vector-width=512 on the Emerald Rapids target: GCC's default for
# -march=emeraldrapids is 256-bit auto-vectorization (a Skylake-SP-era
# frequency-license precaution). Measured on this part the AVX-512 license
# costs ~10% of clock (results/emr/compute.txt, section 1) against 2x the
# work per instruction, so the compiler-vectorized loops (optimizer updates,
# the cheap activations) should use zmm too.
set(FASTMLP_ARCH_FLAGS_ZEN2 -march=znver2 -mavx2 -mfma -mf16c -fno-math-errno)
set(FASTMLP_ARCH_FLAGS_EMR -march=emeraldrapids -mprefer-vector-width=512 -fno-math-errno)

if(FASTMLP_SIMD_BACKEND STREQUAL "AVX512")
  set(FASTMLP_ARCH_FLAGS ${FASTMLP_ARCH_FLAGS_EMR})
else()
  set(FASTMLP_ARCH_FLAGS ${FASTMLP_ARCH_FLAGS_ZEN2})
endif()

# The AVX-512 kernel sources are compiled in every build (tests compare all
# backends against scalar, as with avx2), so they need the Emerald Rapids ISA
# even when the rest of the library targets Zen 2. A later -march overrides
# an earlier one. Those translation units deliberately use no STL or other
# shared inline code, so no AVX-512 copy of an inline function can win the
# ODR merge and leak into a Zen 2 build.
set(FASTMLP_AVX512_SOURCE_FLAGS -march=emeraldrapids -mprefer-vector-width=512)

function(fastmlp_apply_arch_flags target)
  target_compile_options(${target} PRIVATE ${FASTMLP_ARCH_FLAGS} -O3)
endfunction()
