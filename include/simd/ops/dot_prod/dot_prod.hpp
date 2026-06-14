#pragma once
#include "scalar.hpp"
#include "simd.hpp"
#include <cstddef>

namespace simd {

inline float dot_prod(const float* data1, const float* data2, size_t n) {
#if defined(SIMD_AVX2_ENABLED) && defined(__FMA__)
    return impl::dot_prod_simd(data1, data2, n);
#else
    return impl::dot_prod_scalar(data1, data2, n);
#endif
}

} // namespace simd
