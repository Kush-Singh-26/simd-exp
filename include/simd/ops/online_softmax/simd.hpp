#pragma once
#include "../../math_utils.hpp"
#include "scalar.hpp"
#include <cmath>
#include <cstddef>

namespace simd {
namespace impl {

#if defined(SIMD_AVX2_ENABLED)

// Internal: SIMD online exp_sum_max for a contiguous chunk (single-threaded)
inline void online_exp_sum_max_simd_chunk(const float* src, size_t n, float* out_max, float* out_sum){
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

    float global_max = hmax_ps(v_m);

    __m256 v_global_max = _mm256_set1_ps(global_max);
    
    __m256 v_d_correction = avx2_exp_ps(_mm256_sub_ps(v_m, v_global_max));
    v_d = _mm256_mul_ps(v_d, v_d_correction);

    float global_sum = hsum_ps(v_d);

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

// Threaded wrapper: segments + merge
inline void online_exp_sum_max_simd(const float* src, size_t n, float* out_max, float* out_sum){
    if (n < 8 * 8) {
        online_exp_sum_max_simd_chunk(src, n, out_max, out_sum);
        return;
    }

    int nt = 1;
    #pragma omp parallel
    { nt = omp_get_num_threads(); }

    std::vector<float> local_max(nt);
    std::vector<float> local_sum(nt);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        size_t seg_size = (n + nt - 1) / nt;
        size_t start = tid * seg_size;
        size_t end = std::min(start + seg_size, n);
        online_exp_sum_max_simd_chunk(src + start, end - start,
                                       &local_max[tid], &local_sum[tid]);
    }

    online_merge_segments(nt, local_max.data(), local_sum.data(), *out_max, *out_sum);
}

inline void online_softmax_simd(const float *src, float *dst, size_t n){
    if (n == 0)
        return;
    
    float global_max = 0.0f;
    float exp_sum = 0.0f;
    
    online_exp_sum_max_simd(src, n, &global_max, &exp_sum);
    
    __m256 v_global_max = _mm256_set1_ps(global_max);
    float rcp = 1.0f / exp_sum;
    __m256 vrcp = _mm256_set1_ps(rcp);
    
    size_t n_simd = n / 8;
    #pragma omp parallel for if(n_simd >= 128)
    for (size_t idx = 0; idx < n_simd; idx++) {
        __m256 v_src = _mm256_loadu_ps(src + idx * 8);
        v_src = avx2_exp_ps(_mm256_sub_ps(v_src, v_global_max));
        _mm256_storeu_ps(dst + idx * 8, _mm256_mul_ps(v_src, vrcp));
    }
    
    for (size_t i = n_simd * 8; i < n; ++i) {
        dst[i] = std::exp(src[i] - global_max) * rcp;
    }
}
#endif

} // namespace impl
} // namespace simd