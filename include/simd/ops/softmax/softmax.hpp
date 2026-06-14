#pragma once
#include "scalar.hpp"
#include "simd.hpp"
#include <cstddef>

namespace simd {

inline void softmax(const float* src, float* dst, size_t n) {
#if defined(SIMD_AVX2_ENABLED)
    impl::softmax_simd(src, dst, n);
#else
    impl::softmax_scalar(src, dst, n);
#endif
}

inline void softmax_nt(const float* src, float* dst, size_t n) {
#if defined(SIMD_AVX2_ENABLED)
    impl::softmax_simd_nt(src, dst, n);
#else
    impl::softmax_scalar(src, dst, n);
#endif
}

} // namespace simd
