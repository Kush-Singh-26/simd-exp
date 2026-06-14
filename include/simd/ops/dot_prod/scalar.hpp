#pragma once
#include <cstddef>

namespace simd {
namespace impl {

inline float dot_prod_scalar(const float* data1, const float* data2, size_t n) {
    float d = 0.0f;
    for (size_t i = 0; i < n; i++) {
        d += data1[i] * data2[i];
    }
    return d;
}

} // namespace impl
} // namespace simd
