#pragma once
#include <cmath>
#include <cstddef>
#include <omp.h>
#include <vector>

namespace simd {
namespace impl {

inline void online_merge_segments(int nt, const float* local_max,
                                   const float* local_sum,
                                   float& global_max, float& global_sum) {
    global_max = local_max[0];
    global_sum = local_sum[0];
    for (int t = 1; t < nt; t++) {
        if (local_max[t] > global_max) {
            global_sum = global_sum * std::exp(global_max - local_max[t]) + local_sum[t];
            global_max = local_max[t];
        } else {
            global_sum += local_sum[t] * std::exp(local_max[t] - global_max);
        }
    }
}

inline void online_exp_sum_max_scalar(const float* src, size_t n, float* out_max, float* out_sum) {
    float m = -INFINITY;
    float d = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float m_prev = m;
        if (src[i] > m) {
            m = src[i];
            d = d * std::exp(m_prev - m) + 1.0f;
        } else {
            d = d + std::exp(src[i] - m);
        }
    }

    *out_max = m;
    *out_sum = d;
}

inline void online_softmax_scalar(const float* src, float* dst, size_t n) {
    if (n == 0)
        return;
    float global_max = 0.0f;
    float exp_sum = 0.0f;

    online_exp_sum_max_scalar(src, n, &global_max, &exp_sum);

    for (size_t i = 0; i < n; i++)
        dst[i] = std::exp(src[i] - global_max) / exp_sum;
}

inline void online_exp_sum_max_scalar_parallel(const float* src, size_t n, float* out_max, float* out_sum) {
    if (n < 8 * 8) {
        online_exp_sum_max_scalar(src, n, out_max, out_sum);
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

        float m = -INFINITY;
        float d = 0.0f;
        for (size_t i = start; i < end; i++) {
            float m_prev = m;
            if (src[i] > m) {
                m = src[i];
                d = d * std::exp(m_prev - m) + 1.0f;
            } else {
                d += std::exp(src[i] - m);
            }
        }
        local_max[tid] = m;
        local_sum[tid] = d;
    }

    online_merge_segments(nt, local_max.data(), local_sum.data(), *out_max, *out_sum);
}

inline void online_softmax_scalar_parallel(const float* src, float* dst, size_t n) {
    if (n == 0) return;

    float global_max, exp_sum;
    online_exp_sum_max_scalar_parallel(src, n, &global_max, &exp_sum);

    float rcp = 1.0f / exp_sum;
    #pragma omp parallel for if(n >= 1024)
    for (size_t i = 0; i < n; i++)
        dst[i] = std::exp(src[i] - global_max) * rcp;
}

} // namespace impl
} // namespace simd
