#pragma once
#include "scalar.hpp"
#include "simd.hpp"
#include <span>

namespace simd {

inline float sum(std::span<const float> data) {
#if defined(SIMD_AVX2_ENABLED)
    return impl::sum_simd(data.data(), data.size());
#else
    return impl::sum_scalar_parallel(data.data(), data.size());
#endif
}

} // namespace simd
