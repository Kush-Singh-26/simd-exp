#pragma once
#include "scalar.hpp"
#include "simd.hpp"
#include <span>

namespace simd {

inline void clamp(std::span<const float> src, std::span<float> dst, float lo, float hi) {
#if defined(SIMD_AVX2_ENABLED)
    impl::clamp_simd(src.data(), dst.data(), src.size(), lo, hi);
#else
    impl::clamp_scalar_parallel(src.data(), dst.data(), src.size(), lo, hi);
#endif
}

inline void clamp_nt(std::span<const float> src, std::span<float> dst, float lo, float hi) {
#if defined(SIMD_AVX2_ENABLED)
    impl::clamp_simd_nt(src.data(), dst.data(), src.size(), lo, hi);
#else
    impl::clamp_scalar_parallel(src.data(), dst.data(), src.size(), lo, hi);
#endif
}

} // namespace simd
