#pragma once
#include "scalar.hpp"
#include "simd.hpp"

namespace simd {

inline void transpose(const float* src, float* dst) {
#if defined(SIMD_AVX2_ENABLED)
    impl::transpose_simd(src, dst);
#else
    impl::transpose_scalar(src, dst);
#endif
}

inline void transpose_nt(const float* src, float* dst) {
#if defined(SIMD_AVX2_ENABLED)
    impl::transpose_simd_nt(src, dst);
#else
    impl::transpose_scalar(src, dst);
#endif
}

} // namespace simd
