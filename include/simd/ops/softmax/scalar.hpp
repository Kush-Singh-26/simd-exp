#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace simd {
namespace impl {

inline float exp_sub_max_scalar(const float* src, float* dst, size_t n) {
    float exp_sum = 0.0f;
    // pass 1
    float max_val = *std::max_element(src, src + n);
    
    // pass 2
    for (size_t i = 0; i < n; ++i) {
        exp_sum += (dst[i] = std::exp(src[i] - max_val));
    }
    return exp_sum;
}

inline void softmax_scalar(const float* src, float* dst, size_t n) {
    std::vector<float> vexp(n);
    float exp_sum = exp_sub_max_scalar(src, vexp.data(), n);
    // pass 3   
    for (size_t i = 0; i < n; ++i) {
        dst[i] = vexp[i] / exp_sum;
    }
}

} // namespace impl
} // namespace simd
