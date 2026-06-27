#pragma once
#include "scalar.hpp"
#include "simd.hpp"
#include <span>

namespace simd {

inline void online_softmax(std::span<const float> src, std::span<float> dst) {
#if defined(SIMD_AVX2_ENABLED)
    impl::online_softmax_simd(src.data(), dst.data(), src.size());
#else
    impl::online_softmax_scalar_parallel(src.data(), dst.data(), src.size());
#endif
}

} // namespace simd
