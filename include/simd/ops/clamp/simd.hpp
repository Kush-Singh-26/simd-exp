#pragma once
#include "../../common.hpp"
#include <algorithm>
#include <cstddef>

namespace simd {
namespace impl {

#if defined(SIMD_AVX2_ENABLED)
inline void clamp_simd(const float* src, float* dst, size_t n, float lo, float hi) {
    __m256 vlo = _mm256_set1_ps(lo);
    __m256 vhi = _mm256_set1_ps(hi);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(src + i);
        vx = _mm256_max_ps(vx, vlo);
        vx = _mm256_min_ps(vx, vhi);
        _mm256_storeu_ps(dst + i, vx);
    }
    for (; i < n; i++) {
        dst[i] = std::min(std::max(src[i], lo), hi);
    }
}

inline void clamp_simd_nt(const float* src, float* dst, size_t n, float lo, float hi) {
    __m256 vlo = _mm256_set1_ps(lo);
    __m256 vhi = _mm256_set1_ps(hi);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(src + i);
        vx = _mm256_max_ps(vx, vlo);
        vx = _mm256_min_ps(vx, vhi);
        _mm256_stream_ps(dst + i, vx);
    }
    for (; i < n; i++) {
        dst[i] = std::min(std::max(src[i], lo), hi);
    }
}
#endif

} // namespace impl
} // namespace simd
