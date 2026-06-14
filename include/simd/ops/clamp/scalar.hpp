#pragma once
#include <algorithm>
#include <cstddef>

namespace simd {
namespace impl {

inline void clamp_scalar(const float* src, float* dst, size_t n, float lo, float hi) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = std::min(std::max(src[i], lo), hi);
    }
}

} // namespace impl
} // namespace simd
