#pragma once
#include "../../common.hpp"
#include <cmath>
#include <cstddef>

namespace simd {
namespace impl {

#if defined(SIMD_AVX2_ENABLED)
inline void abs_simd(const float* src, float* dst, size_t n) {
    size_t i = 0;
    __m256 bitmask = _mm256_set1_ps(-0.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_andnot_ps(bitmask, vx));
    }
    for (; i < n; i++) {
        dst[i] = std::abs(src[i]);
    }
}

inline void abs_simd_nt(const float* src, float* dst, size_t n) {
    size_t i = 0;
    __m256 bitmask = _mm256_set1_ps(-0.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(src + i);
        _mm256_stream_ps(dst + i, _mm256_andnot_ps(bitmask, vx));
    }
    for (; i < n; i++) {
        dst[i] = std::abs(src[i]);
    }
}
#endif

} // namespace impl
} // namespace simd
