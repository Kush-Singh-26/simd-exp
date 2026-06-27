#pragma once
#include "scalar.hpp"
#include "simd.hpp"
#include <span>

namespace simd {

inline void abs(std::span<const float> src, std::span<float> dst) {
#if defined(SIMD_AVX2_ENABLED)
    impl::abs_simd(src.data(), dst.data(), src.size());
#else
    impl::abs_scalar_parallel(src.data(), dst.data(), src.size());
#endif
}

inline void abs_nt(std::span<const float> src, std::span<float> dst) {
#if defined(SIMD_AVX2_ENABLED)
    impl::abs_simd_nt(src.data(), dst.data(), src.size());
#else
    impl::abs_scalar_parallel(src.data(), dst.data(), src.size());
#endif
}

} // namespace simd
