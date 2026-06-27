#pragma once
#include "scalar.hpp"
#include "simd.hpp"
#include <span>

namespace simd {

// Generalized transpose: src is rows x cols, dst is cols x rows
// Uses SIMD 8x8 block for square 8x8, 4x4 block for square 4x4,
// scalar path for everything else.
inline void transpose(std::span<const float> src, std::span<float> dst, size_t rows, size_t cols) {
#if defined(SIMD_AVX2_ENABLED)
    if (rows == 8 && cols == 8) {
        impl::transpose8x8_simd(src.data(), dst.data());
        return;
    }
    if (rows == 4 && cols == 4) {
        impl::transpose_simd(src.data(), dst.data());
        return;
    }
#endif
    impl::transpose_scalar(src.data(), dst.data(), rows, cols);
}

// NT (non-temporal) variant: uses streaming stores for large matrices
inline void transpose_nt(std::span<const float> src, std::span<float> dst, size_t rows, size_t cols) {
#if defined(SIMD_AVX2_ENABLED)
    if (rows == 8 && cols == 8) {
        impl::transpose8x8_simd_nt(src.data(), dst.data());
        return;
    }
    if (rows == 4 && cols == 4) {
        impl::transpose_simd_nt(src.data(), dst.data());
        return;
    }
#endif
    impl::transpose_scalar(src.data(), dst.data(), rows, cols);
}

} // namespace simd
