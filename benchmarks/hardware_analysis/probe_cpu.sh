#!/usr/bin/env bash
# Hardware probe: lscpu, /proc/cpuinfo, ISA flags of interest, and per-core
# cache topology. No load testing (src/roofline.cpp measures clock, FMA peak
# and bandwidth).
#
# Run inside the fastmlp-dev container, the environment the kernels are
# built and benchmarked in.
set -euo pipefail

OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results"
mkdir -p "$OUT_DIR"

{
  echo "=== date (UTC) ==="
  date -u
  echo
  echo "=== lscpu ==="
  lscpu
  echo
  echo "=== lscpu -C (cache topology, if supported) ==="
  lscpu -C 2>/dev/null || echo "(lscpu -C not supported by this util-linux version)"
} > "$OUT_DIR/lscpu.txt"

cp /proc/cpuinfo "$OUT_DIR/cpuinfo.txt"

FLAGS="$(grep -m1 '^flags' /proc/cpuinfo | cut -d: -f2)"

check_flag() {
  if grep -qw -- "$1" <<< "$FLAGS"; then echo "yes"; else echo "no"; fi
}

{
  echo "=== ISA flags of interest ==="
  for f in sse4_2 avx avx2 fma f16c \
           avx512f avx512cd avx512bw avx512dq avx512vl \
           avx512_vnni avx512bf16 avx512_fp16 amx_tile amx_bf16; do
    printf '%-16s %s\n' "$f" "$(check_flag "$f")"
  done
  echo
  echo "=== register width implied by ISA (not directly reported by lscpu) ==="
  echo "AVX2 only    -> 16x 256-bit YMM registers"
  echo "AVX-512F+    -> 32x 512-bit ZMM registers (widens YMM0-15 usage to 32 regs)"
} > "$OUT_DIR/isa_flags.txt"

{
  echo "=== cache topology (cpu0, from /sys) ==="
  for idx in /sys/devices/system/cpu/cpu0/cache/index*; do
    [ -d "$idx" ] || continue
    printf 'index=%s level=%s type=%-12s size=%-8s ways=%-4s sets=%-8s line=%-4s shared_cpus=%s\n' \
      "$(basename "$idx")" \
      "$(cat "$idx/level" 2>/dev/null)" \
      "$(cat "$idx/type" 2>/dev/null)" \
      "$(cat "$idx/size" 2>/dev/null)" \
      "$(cat "$idx/ways_of_associativity" 2>/dev/null)" \
      "$(cat "$idx/number_of_sets" 2>/dev/null)" \
      "$(cat "$idx/coherency_line_size" 2>/dev/null)" \
      "$(cat "$idx/shared_cpu_list" 2>/dev/null)"
  done
} > "$OUT_DIR/cache_topology.txt"

echo "Wrote: $OUT_DIR/{lscpu,cpuinfo,isa_flags,cache_topology}.txt"
