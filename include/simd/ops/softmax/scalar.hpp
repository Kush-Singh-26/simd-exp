#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace simd {
namespace impl {

inline float exp_sub_max_scalar(const float* src, float* dst, size_t n) {
    if (n == 0) return 0.0f;

    float max_val = src[0];
    for (size_t i = 1; i < n; ++i) {
        max_val = std::max(max_val, src[i]);
    }

    float exp_sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        exp_sum += (dst[i] = std::exp(src[i] - max_val));
    }
    return exp_sum;
}

inline void softmax_scalar(const float* src, float* dst, size_t n) {
    if (n == 0) return;

    float max_val = src[0];
    for (size_t i = 1; i < n; ++i) {
        max_val = std::max(max_val, src[i]);
    }

    float exp_sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        dst[i] = std::exp(src[i] - max_val);
        exp_sum += dst[i];
    }

    float rcp = 1.0f / exp_sum;
    for (size_t i = 0; i < n; ++i) {
        dst[i] *= rcp;
    }
}

} // namespace impl
} // namespace simd
