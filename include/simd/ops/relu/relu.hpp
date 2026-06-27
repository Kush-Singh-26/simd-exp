#pragma once
#include "scalar.hpp"
#include "simd.hpp"
#include <span>

namespace simd {

inline void relu(std::span<const float> src, std::span<float> dst) {
#if defined(SIMD_AVX2_ENABLED)
    impl::relu_simd(src.data(), dst.data(), src.size());
#else
    impl::relu_scalar_parallel(src.data(), dst.data(), src.size());
#endif
}

inline void relu_nt(std::span<const float> src, std::span<float> dst) {
#if defined(SIMD_AVX2_ENABLED)
    impl::relu_simd_nt(src.data(), dst.data(), src.size());
#else
    impl::relu_scalar_parallel(src.data(), dst.data(), src.size());
#endif
}

} // namespace simd
