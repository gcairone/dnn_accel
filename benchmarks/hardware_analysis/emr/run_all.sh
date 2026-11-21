#!/usr/bin/env bash
# Builds and runs every Emerald Rapids probe, pinned to one core, and writes
# the outputs to ../results/emr. About 15 minutes.
#   PROBE_CORE=<n> selects the core (default 0).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/../results/emr"
BIN="${BIN_DIR:-$(mktemp -d)}"
mkdir -p "$OUT"
for p in probe_compute probe_memory probe_noise; do
  g++ -O3 -march=emeraldrapids -o "$BIN/$p" "$HERE/$p.cpp"
done
OUT_DIR="$OUT" "$HERE/../probe_cpu.sh"
"$BIN/probe_compute" > "$OUT/compute.txt"
"$BIN/probe_memory"  > "$OUT/memory.txt"
"$BIN/probe_noise"   > "$OUT/noise.txt"
echo "Wrote: $OUT/{compute,memory,noise}.txt"
