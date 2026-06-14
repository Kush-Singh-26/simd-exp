#pragma once
#include "scalar.hpp"
#include "simd.hpp"
#include <cstddef>

namespace simd {

inline void relu(const float* src, float* dst, size_t n) {
#if defined(SIMD_AVX2_ENABLED)
    impl::relu_simd(src, dst, n);
#else
    impl::relu_scalar(src, dst, n);
#endif
}

inline void relu_nt(const float* src, float* dst, size_t n) {
#if defined(SIMD_AVX2_ENABLED)
    impl::relu_simd_nt(src, dst, n);
#else
    impl::relu_scalar(src, dst, n);
#endif
}

} // namespace simd
