#pragma once
#include "../../math_utils.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace simd {
namespace impl {

#if defined(SIMD_AVX2_ENABLED)

template <typename StoreFn>
float exp_sub_max_impl(const float* src, float* dst, size_t n, StoreFn store) {
    size_t n_simd = n / 8;

    // Pass 1: parallel SIMD max reduction
    float max_val = -INFINITY;
    #pragma omp parallel reduction(max:max_val) if(n_simd >= 512)
    {
        __m256 vlocal = _mm256_set1_ps(-INFINITY);
        #pragma omp for nowait
        for (size_t idx = 0; idx < n_simd; idx++) {
            vlocal = _mm256_max_ps(vlocal, _mm256_loadu_ps(src + idx * 8));
        }
        max_val = hmax_ps(vlocal);
    }
    for (size_t i = n_simd * 8; i < n; i++) {
        max_val = std::max(max_val, src[i]);
    }

    // Pass 2: parallel SIMD exp(x - max) + sum
    __m256 vbcast_max = _mm256_set1_ps(max_val);
    float exp_sum = 0.0f;
    #pragma omp parallel reduction(+:exp_sum) if(n_simd >= 512)
    {
        StoreFn local_store = store;
        __m256 vlocal = _mm256_setzero_ps();
        #pragma omp for nowait
        for (size_t idx = 0; idx < n_simd; idx++) {
            __m256 v = _mm256_loadu_ps(src + idx * 8);
            v = avx2_exp_ps(_mm256_sub_ps(v, vbcast_max));
            local_store(dst + idx * 8, v);
            vlocal = _mm256_add_ps(vlocal, v);
        }
        exp_sum += hsum_ps(vlocal);
    }
    for (size_t i = n_simd * 8; i < n; i++) {
        exp_sum += (dst[i] = std::exp(src[i] - max_val));
    }
    return exp_sum;
}

template <typename StoreFn>
void softmax_impl(const float* src, float* dst, size_t n, StoreFn store) {
    if (n == 0) return;
    float exp_sum = exp_sub_max_impl(src, dst, n, store);

    size_t n_simd = n / 8;
    __m256 vrcp = _mm256_set1_ps(1.0f / exp_sum);
    #pragma omp parallel for firstprivate(store) if(n_simd >= 128)
    for (size_t idx = 0; idx < n_simd; idx++) {
        __m256 v = _mm256_loadu_ps(dst + idx * 8);
        store(dst + idx * 8, _mm256_mul_ps(v, vrcp));
    }
    float rcp = 1.0f / exp_sum;
    for (size_t i = n_simd * 8; i < n; ++i) {
        dst[i] *= rcp;
    }
}

inline float exp_sub_max_simd(const float* src, float* dst, size_t n) {
    return exp_sub_max_impl(src, dst, n, [](float* p, __m256 v) { _mm256_storeu_ps(p, v); });
}

inline void softmax_simd(const float* src, float* dst, size_t n) {
    softmax_impl(src, dst, n, [](float* p, __m256 v) { _mm256_storeu_ps(p, v); });
}

inline float exp_sub_max_simd_nt(const float* src, float* dst, size_t n) {
    return exp_sub_max_impl(src, dst, n, [](float* p, __m256 v) { _mm256_storeu_ps(p, v); });
}

inline void softmax_simd_nt(const float* src, float* dst, size_t n) {
    softmax_impl(src, dst, n, [](float* p, __m256 v) { _mm256_storeu_ps(p, v); });
}

#endif

} // namespace impl
} // namespace simd
