#pragma once
#include "scalar.hpp"
#include "simd.hpp"
#include <cstddef>

namespace simd {

inline void clamp(const float* src, float* dst, size_t n, float lo, float hi) {
#if defined(SIMD_AVX2_ENABLED)
    impl::clamp_simd(src, dst, n, lo, hi);
#else
    impl::clamp_scalar(src, dst, n, lo, hi);
#endif
}

inline void clamp_nt(const float* src, float* dst, size_t n, float lo, float hi) {
#if defined(SIMD_AVX2_ENABLED)
    impl::clamp_simd_nt(src, dst, n, lo, hi);
#else
    impl::clamp_scalar(src, dst, n, lo, hi);
#endif
}

} // namespace simd
