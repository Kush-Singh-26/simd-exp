#pragma once
#include <cstddef>

namespace simd {
namespace impl {

inline float sum_scalar(const float* data, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) {
        s += data[i];
    }
    return s;
}

inline float sum_scalar_parallel(const float* data, size_t n) {
    float s = 0.0f;
    #pragma omp parallel for reduction(+:s) if(n >= 4096)
    for (size_t i = 0; i < n; i++) {
        s += data[i];
    }
    return s;
}

} // namespace impl
} // namespace simd
