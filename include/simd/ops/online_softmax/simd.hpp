#pragma once
#include "../../math_utils.hpp"
#include "scalar.hpp"
#include <cmath>
#include <cstddef>

namespace simd {
namespace impl {

#if defined(SIMD_AVX2_ENABLED)

inline void online_exp_sum_max_simd(const float* src, size_t n, float* out_max, float* out_sum){
    if (n < 8) {
        online_exp_sum_max_scalar(src, n, out_max, out_sum);
        return;
    }

    __m256 v_m = _mm256_loadu_ps(src);
    __m256 v_d = _mm256_set1_ps(1.0f);

    size_t i = 8;
    for(; i + 8 <= n; i += 8){
        __m256 v_src = _mm256_loadu_ps(src + i);
        __m256 v_m_prev = v_m;

        v_m = _mm256_max_ps(v_m, v_src);

        __m256 v_mask = _mm256_cmp_ps(v_m, v_m_prev, _CMP_NEQ_OQ);
        __m256 v_correction_factor = _mm256_set1_ps(1.0f);
        __m256 v_correction_exp = avx2_exp_ps(_mm256_sub_ps(v_m_prev, v_m));
        v_correction_factor = _mm256_blendv_ps(v_correction_factor, v_correction_exp, v_mask);

        __m256 v_new_elements = avx2_exp_ps(_mm256_sub_ps(v_src, v_m));

        v_d = _mm256_fmadd_ps(v_d, v_correction_factor, v_new_elements);
    }

    // Horizontal Max of v_m
    __m128 m_lo = _mm256_castps256_ps128(v_m);
    __m128 m_hi = _mm256_extractf128_ps(v_m, 1);
    __m128 m_max = _mm_max_ps(m_lo, m_hi);
    m_max = _mm_max_ps(m_max, _mm_shuffle_ps(m_max, m_max, _MM_SHUFFLE(1, 0, 3, 2)));
    m_max = _mm_max_ps(m_max, _mm_shuffle_ps(m_max, m_max, _MM_SHUFFLE(2, 3, 0, 1)));
    float global_max = _mm_cvtss_f32(m_max);

    __m256 v_global_max = _mm256_set1_ps(global_max);
    
    __m256 v_d_correction = avx2_exp_ps(_mm256_sub_ps(v_m, v_global_max));
    v_d = _mm256_mul_ps(v_d, v_d_correction);

    __m128 d_lo = _mm256_castps256_ps128(v_d);
    __m128 d_hi = _mm256_extractf128_ps(v_d, 1);
    __m128 d_sum = _mm_add_ps(d_lo, d_hi);
    // Shuffle down and add to reduce 4 elements to 1
    d_sum = _mm_add_ps(d_sum, _mm_shuffle_ps(d_sum, d_sum, _MM_SHUFFLE(1, 0, 3, 2)));
    d_sum = _mm_add_ps(d_sum, _mm_shuffle_ps(d_sum, d_sum, _MM_SHUFFLE(2, 3, 0, 1)));
    float global_sum = _mm_cvtss_f32(d_sum);

    // Tail
    for (; i < n; ++i) {
        float m_prev = global_max;
        if (src[i] > global_max) {
            global_max = src[i];
            global_sum = global_sum * std::exp(m_prev - global_max) + 1.0f;
        } else {
            global_sum += std::exp(src[i] - global_max);
        }
    }

    *out_max = global_max;
    *out_sum = global_sum;
}

inline void online_softmax_simd(const float *src, float *dst, size_t n){
    if (n == 0)
        return;
    
    float global_max = 0.0f;
    float exp_sum = 0.0f;
    
    online_exp_sum_max_simd(src, n,  &global_max, &exp_sum);
    
    __m256 v_global_max = _mm256_set1_ps(global_max);
    float rcp = 1.0f / exp_sum;
    __m256 vrcp = _mm256_set1_ps(rcp);
    
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v_src = _mm256_loadu_ps(src + i);
        v_src = avx2_exp_ps(_mm256_sub_ps(v_src, v_global_max));
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(v_src, vrcp));
    }

    
    for (; i < n; ++i) {
        dst[i] = std::exp(src[i] - global_max) * rcp;
    }
}
#endif

} // namespace impl
} // namespace simd