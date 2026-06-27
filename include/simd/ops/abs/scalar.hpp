#pragma once
#include <cmath>
#include <cstddef>

namespace simd {
namespace impl {

inline void abs_scalar(const float* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = std::abs(src[i]);
    }
}

inline void abs_scalar_parallel(const float* src, float* dst, size_t n) {
    #pragma omp parallel for if(n >= 1024)
    for (size_t i = 0; i < n; i++) {
        dst[i] = std::abs(src[i]);
    }
}

} // namespace impl
} // namespace simd
