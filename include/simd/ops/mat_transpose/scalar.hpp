#pragma once
#include <cstddef>
#include <mdspan>

namespace simd {
namespace impl {

// Generalized transpose: src is rows x cols, dst is cols x rows
inline void transpose_scalar(const float* src, float* dst, size_t rows, size_t cols) {
    std::mdspan<const float, std::dextents<size_t, 2>> src_md(src, rows, cols);
    std::mdspan<float, std::dextents<size_t, 2>> dst_md(dst, cols, rows);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            dst_md[j, i] = src_md[i, j];
        }
    }
}

// 4x4 convenience overload (keeps existing tests and SIMD dispatch working)
inline void transpose_scalar_4x4(const float* src, float* dst) {
    transpose_scalar(src, dst, 4, 4);
}

} // namespace impl
} // namespace simd
