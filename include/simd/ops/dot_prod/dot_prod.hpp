#pragma once
#include "scalar.hpp"
#include "simd.hpp"
#include <span>

namespace simd {

inline float dot_prod(std::span<const float> data1, std::span<const float> data2) {
#if defined(SIMD_AVX2_ENABLED) && defined(__FMA__)
    return impl::dot_prod_simd(data1.data(), data2.data(), data1.size());
#else
    return impl::dot_prod_scalar_parallel(data1.data(), data2.data(), data1.size());
#endif
}

} // namespace simd
