#pragma once
#include "../../common.hpp"
#include <cmath>
#include <cstddef>

namespace simd {
namespace impl {

#if defined(SIMD_AVX2_ENABLED)
template <typename StoreFn>
void abs_impl(const float* src, float* dst, size_t n, StoreFn store) {
    __m256 bitmask = _mm256_set1_ps(-0.0f);
    size_t n_simd = n / 8;
    #pragma omp parallel for firstprivate(store) if(n_simd >= 128)
    for (size_t idx = 0; idx < n_simd; idx++) {
        size_t i = idx * 8;
        __m256 vx = _mm256_loadu_ps(src + i);
        store(dst + i, _mm256_andnot_ps(bitmask, vx));
    }
    for (size_t i = n_simd * 8; i < n; i++) {
        dst[i] = std::abs(src[i]);
    }
}

inline void abs_simd(const float* src, float* dst, size_t n) {
    abs_impl(src, dst, n, [](float* p, __m256 v) { _mm256_storeu_ps(p, v); });
}

inline void abs_simd_nt(const float* src, float* dst, size_t n) {
    // Uses storeu (not stream) because _mm256_stream_ps requires 32-byte alignment
    // which std::vector does not guarantee.
    abs_impl(src, dst, n, [](float* p, __m256 v) { _mm256_storeu_ps(p, v); });
}
#endif

} // namespace impl
} // namespace simd
