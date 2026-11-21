#pragma once
// 64-byte-aligned storage for every hot buffer (weights, gradients,
// activations, optimizer state).
//
// glibc's malloc returns 16-byte-aligned blocks (large ones come from mmap at
// page + 16), so a std::vector<float> starts 16 bytes into a cache line and,
// on a 512-bit target, every full-vector load or store of it straddles two
// lines. Measured on the Emerald Rapids target (interleaved A/B, same kernel,
// 16- vs 64-byte-aligned buffers): the GEMMs 2-20% slower, an elementwise
// activation 38-90% slower (benchmarks/hardware_analysis/results/emr/
// memory.txt section 1: a line-split zmm load runs at ~0.73 of an aligned
// one). Huge pages were measured too (madvise THP on the big buffers): within
// +-5% either way, and not reliably granted, so not done.
#include <cstddef>
#include <cstdlib>
#include <new>
#include <vector>

namespace fastmlp::simd {

template <class T>
struct AlignedAllocator {
  using value_type = T;
  static constexpr std::size_t kAlign = 64;

  AlignedAllocator() noexcept = default;
  template <class U>
  AlignedAllocator(const AlignedAllocator<U>&) noexcept {}

  T* allocate(std::size_t n) {
    // std::aligned_alloc needs the size to be a multiple of the alignment.
    const std::size_t bytes = (n * sizeof(T) + kAlign - 1) / kAlign * kAlign;
    void* p = std::aligned_alloc(kAlign, bytes ? bytes : kAlign);
    if (!p) throw std::bad_alloc();
    return static_cast<T*>(p);
  }
  void deallocate(T* p, std::size_t) noexcept { std::free(p); }

  template <class U>
  bool operator==(const AlignedAllocator<U>&) const noexcept {
    return true;
  }
  template <class U>
  bool operator!=(const AlignedAllocator<U>&) const noexcept {
    return false;
  }
};

template <class T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

}  // namespace fastmlp::simd
