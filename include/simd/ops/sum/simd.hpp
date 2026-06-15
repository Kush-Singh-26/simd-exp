#pragma once
#include "../../math_utils.hpp"
#include <cstddef>

namespace simd {
namespace impl {

#if defined(SIMD_AVX2_ENABLED)
inline float sum_simd(const float* data, size_t n) {
    __m256 vsum = _mm256_setzero_ps();
    size_t i = 0;
    
    // Vertical summing
    for (; i + 8 <= n; i += 8) {
        vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(data + i));
    }
    
    float s = hsum_ps(vsum);
    
    // Summing tail elements
    for (; i < n; i++) {
        s += data[i];
    }
    
    return s;
}
#endif

} // namespace impl
} // namespace simd
