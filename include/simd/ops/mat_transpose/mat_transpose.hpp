#pragma once
#include "scalar.hpp"
#include "simd.hpp"

namespace simd {

// Generalized transpose: src is rows x cols, dst is cols x rows
// Uses SIMD 4x4 block for square 4x4, scalar path for everything else.
inline void transpose(const float* src, float* dst, size_t rows, size_t cols) {
#if defined(SIMD_AVX2_ENABLED)
    if (rows == 4 && cols == 4) {
        impl::transpose_simd(src, dst);
        return;
    }
#endif
    impl::transpose_scalar(src, dst, rows, cols);
}

// NT (non-temporal) variant: uses streaming stores for large matrices
inline void transpose_nt(const float* src, float* dst, size_t rows, size_t cols) {
#if defined(SIMD_AVX2_ENABLED)
    if (rows == 4 && cols == 4) {
        impl::transpose_simd_nt(src, dst);
        return;
    }
#endif
    impl::transpose_scalar(src, dst, rows, cols);
}

} // namespace simd
