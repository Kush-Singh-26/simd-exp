#pragma once
#include "scalar.hpp"
#include "simd.hpp"
#include <span>

namespace simd {

inline void softmax(std::span<const float> src, std::span<float> dst) {
#if defined(SIMD_AVX2_ENABLED)
    impl::softmax_simd(src.data(), dst.data(), src.size());
#else
    impl::softmax_scalar_parallel(src.data(), dst.data(), src.size());
#endif
}

inline void softmax_nt(std::span<const float> src, std::span<float> dst) {
#if defined(SIMD_AVX2_ENABLED)
    impl::softmax_simd_nt(src.data(), dst.data(), src.size());
#else
    impl::softmax_scalar_parallel(src.data(), dst.data(), src.size());
#endif
}

} // namespace simd
