#pragma once
#include <cmath>
#include <cstddef>

namespace simd {
namespace impl {

    inline void online_exp_sum_max_scalar(const float* src, size_t n, float* out_max, float* out_sum){

        float m = -INFINITY;
        float d = 0.0f;

        // pass 1
        for(size_t i = 0; i < n; i++){
            float m_prev = m;
            if(src[i] > m){
                m = src[i];
                d = d * std::exp(m_prev - m) + 1.0f; // because src[i]-m = 1
            } 
            else {
                d = d + std::exp(src[i] - m);
            }
        }

        *out_max = m;
        *out_sum = d;
    }
    inline void online_softmax_scalar(const float* src, float* dst, size_t n){
        if (n == 0)
            return;
        float global_max = 0.0f;
        float exp_sum = 0.0f;

        online_exp_sum_max_scalar(src, n, &global_max, &exp_sum);

        //pass 2
        for(size_t i = 0; i < n; i++)
            dst[i] = std::exp(src[i] - global_max) / exp_sum;
    }
} //namespace impl
} //namespace simd