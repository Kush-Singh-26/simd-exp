#pragma once
#include <cstddef>

namespace simd {
namespace impl {

inline void relu_scalar(const float* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
    }
}

inline void relu_scalar_parallel(const float* src, float* dst, size_t n) {
    #pragma omp parallel for if(n >= 1024)
    for (size_t i = 0; i < n; i++) {
        dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
    }
}

} // namespace impl
} // namespace simd
