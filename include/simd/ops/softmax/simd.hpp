#pragma once
#include "../../math_utils.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace simd {
namespace impl {

#if defined(SIMD_AVX2_ENABLED)

inline float exp_sub_max_simd(const float* src, float* dst, size_t n) {
    __m256 vmax = _mm256_set1_ps(-__builtin_inff());
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(src + i));
    }
    // horizontal max
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 m = _mm_max_ps(hi, lo);
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ps(m, _mm_movehdup_ps(m));

    float max_val = _mm_cvtss_f32(m);

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
    lo = _mm256_castps256_ps128(vsum);
    hi = _mm256_extractf128_ps(vsum, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ps(s, _mm_movehdup_ps(s));
    float exp_sum = _mm_cvtss_f32(s);

    // tail
    for (; i < n; i++) {
        exp_sum += (dst[i] = std::exp(src[i] - max_val));
    }
    return exp_sum;
}

inline void softmax_simd(const float* src, float* dst, size_t n) {
    float exp_sum = exp_sub_max_simd(src, dst, n);

    // reciprocal
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
    __m256 vmax = _mm256_set1_ps(-__builtin_inff());
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(src + i));
    }
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 m = _mm_max_ps(hi, lo);
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ps(m, _mm_movehdup_ps(m));
    float max_val = _mm_cvtss_f32(m);
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

    lo = _mm256_castps256_ps128(vsum);
    hi = _mm256_extractf128_ps(vsum, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ps(s, _mm_movehdup_ps(s));
    float exp_sum = _mm_cvtss_f32(s);

    for (; i < n; i++) {
        exp_sum += (dst[i] = std::exp(src[i] - max_val));
    }
    return exp_sum;
}

inline void softmax_simd_nt(const float* src, float* dst, size_t n) {
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
