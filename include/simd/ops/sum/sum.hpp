#pragma once
#include "scalar.hpp"
#include "simd.hpp"
#include <cstddef>

namespace simd {

inline float sum(const float* data, size_t n) {
#if defined(SIMD_AVX2_ENABLED)
    return impl::sum_simd(data, n);
#else
    return impl::sum_scalar(data, n);
#endif
}

} // namespace simd
