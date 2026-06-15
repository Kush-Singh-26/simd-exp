#pragma once
#include "../../math_utils.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace simd {
namespace impl {

#if defined(SIMD_AVX2_ENABLED)

inline float exp_sub_max_simd(const float* src, float* dst, size_t n) {
    __m256 vmax = _mm256_set1_ps(-INFINITY);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(src + i));
    }
    float max_val = hmax_ps(vmax);

    // tail
    for (; i < n; i++) {
        max_val = std::max(max_val, src[i]);
    }

    vmax = _mm256_set1_ps(max_val);
    __m256 vsum = _mm256_setzero_ps();

    // exp(v - vmax)
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        v = avx2_exp_ps(_mm256_sub_ps(v, vmax));
        _mm256_storeu_ps(dst + i, v);
        vsum = _mm256_add_ps(vsum, v);
    }

    // horizontal sum
    float exp_sum = hsum_ps(vsum);

    // tail
    for (; i < n; i++) {
        exp_sum += (dst[i] = std::exp(src[i] - max_val));
    }
    return exp_sum;
}

inline void softmax_simd(const float* src, float* dst, size_t n) {
    if (n == 0) return;
    float exp_sum = exp_sub_max_simd(src, dst, n);

    __m256 vrcp = _mm256_set1_ps(1.0f / exp_sum);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(dst + i);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(v, vrcp));
    }

    float rcp = 1.0f / exp_sum;
    for (; i < n; ++i) {
        dst[i] *= rcp;
    }
}

inline float exp_sub_max_simd_nt(const float* src, float* dst, size_t n) {
    __m256 vmax = _mm256_set1_ps(-INFINITY);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(src + i));
    }
    float max_val = hmax_ps(vmax);
    for (; i < n; i++) {
        max_val = std::max(max_val, src[i]);
    }

    vmax = _mm256_set1_ps(max_val);
    __m256 vsum = _mm256_setzero_ps();

    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        v = avx2_exp_ps(_mm256_sub_ps(v, vmax));
        _mm256_stream_ps(dst + i, v);
        vsum = _mm256_add_ps(vsum, v);
    }

    float exp_sum = hsum_ps(vsum);

    for (; i < n; i++) {
        exp_sum += (dst[i] = std::exp(src[i] - max_val));
    }
    return exp_sum;
}

inline void softmax_simd_nt(const float* src, float* dst, size_t n) {
    if (n == 0) return;
    float exp_sum = exp_sub_max_simd_nt(src, dst, n);
    __m256 vrcp = _mm256_set1_ps(1.0f / exp_sum);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(dst + i);
        _mm256_stream_ps(dst + i, _mm256_mul_ps(v, vrcp));
    }
    float rcp = 1.0f / exp_sum;
    for (; i < n; ++i) {
        dst[i] *= rcp;
    }
}

#endif

} // namespace impl
} // namespace simd
