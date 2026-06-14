#pragma once
#include "scalar.hpp"
#include "simd.hpp"
#include <cstddef>

namespace simd {

inline void abs(const float* src, float* dst, size_t n) {
#if defined(SIMD_AVX2_ENABLED)
    impl::abs_simd(src, dst, n);
#else
    impl::abs_scalar(src, dst, n);
#endif
}

inline void abs_nt(const float* src, float* dst, size_t n) {
#if defined(SIMD_AVX2_ENABLED)
    impl::abs_simd_nt(src, dst, n);
#else
    impl::abs_scalar(src, dst, n);
#endif
}

} // namespace simd
