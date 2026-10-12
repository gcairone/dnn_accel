# Fixed, known target: AMD Ryzen 7 4800H (Zen 2), which has no AVX-512, so
# AVX2+FMA3+F16C is the only SIMD path. Chosen here at build time, never
# detected at runtime: the target CPU is fixed.
#
# -fno-math-errno is a measured necessity, not boilerplate: by default GCC
# must assume std::sqrt sets errno, so it cannot contract it to a bare
# vsqrtps and guards every call with a branch -- which blocks vectorization
# of any loop containing one. That kept Adam::step scalar (bench_breakdown
# measured it at 16.9% of a whole MNIST training step). This flag only
# asserts "we never read errno after a math call", which is true here; it
# does NOT relax rounding or allow reassociation the way -ffast-math would,
# so numerical results stay bit-identical.
set(FASTMLP_ARCH_FLAGS -march=znver2 -mavx2 -mfma -mf16c -fno-math-errno)

function(fastmlp_apply_arch_flags target)
  target_compile_options(${target} PRIVATE ${FASTMLP_ARCH_FLAGS} -O3)
endfunction()
