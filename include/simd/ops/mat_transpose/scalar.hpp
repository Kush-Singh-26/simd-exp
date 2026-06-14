#pragma once
#include <cstddef>

namespace simd {
namespace impl {

inline void transpose_scalar(const float* src, float* dst) {
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            dst[j * 4 + i] = src[i * 4 + j];
        }
    }
}

} // namespace impl
} // namespace simd
