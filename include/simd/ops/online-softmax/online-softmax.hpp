#pragma once
#include "scalar.hpp"
#include "simd.hpp"
#include <cstddef>

namespace simd {

    inline void online_softmax(const float* src, float* dst, size_t n) {
    #if defined(SIMD_AVX2_ENABLED)
        impl::online_softmax_simd(src, dst, n);
    #else
        impl::online_softmax_scalar(src, dst, n);
    #endif
    } 

}// namespace simd