#pragma once
#include "../../common.hpp"
#include <algorithm>
#include <cstddef>

namespace simd {
namespace impl {

#if defined(SIMD_AVX2_ENABLED)
template <typename StoreFn>
void clamp_impl(const float* src, float* dst, size_t n, float lo, float hi, StoreFn store) {
    __m256 vlo = _mm256_set1_ps(lo);
    __m256 vhi = _mm256_set1_ps(hi);
    size_t n_simd = n / 8;
    #pragma omp parallel for firstprivate(store) if(n_simd >= 128)
    for (size_t idx = 0; idx < n_simd; idx++) {
        size_t i = idx * 8;
        __m256 vx = _mm256_loadu_ps(src + i);
        vx = _mm256_max_ps(vx, vlo);
        vx = _mm256_min_ps(vx, vhi);
        store(dst + i, vx);
    }
    for (size_t i = n_simd * 8; i < n; i++) {
        dst[i] = std::min(std::max(src[i], lo), hi);
    }
}

inline void clamp_simd(const float* src, float* dst, size_t n, float lo, float hi) {
    clamp_impl(src, dst, n, lo, hi, [](float* p, __m256 v) { _mm256_storeu_ps(p, v); });
}

inline void clamp_simd_nt(const float* src, float* dst, size_t n, float lo, float hi) {
    clamp_impl(src, dst, n, lo, hi, [](float* p, __m256 v) { _mm256_storeu_ps(p, v); });
}
#endif

} // namespace impl
} // namespace simd
