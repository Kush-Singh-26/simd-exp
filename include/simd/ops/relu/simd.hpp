#pragma once
#include "../../common.hpp"
#include <cstddef>

namespace simd {
namespace impl {

#if defined(SIMD_AVX2_ENABLED)
inline void relu_simd(const float* src, float* dst, size_t n) {
    size_t i = 0;
    __m256 zero = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(src + i);
        vx = _mm256_max_ps(vx, zero);
        _mm256_storeu_ps(dst + i, vx);
    }
    for (; i < n; i++) {
        dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
    }
}

inline void relu_simd_nt(const float* src, float* dst, size_t n) {
    size_t i = 0;
    __m256 zero = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(src + i);
        vx = _mm256_max_ps(vx, zero);
        _mm256_stream_ps(dst + i, vx);
    }
    for (; i < n; i++) {
        dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
    }
}
#endif

} // namespace impl
} // namespace simd
