#pragma once
#include "../../common.hpp"
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
    
    // Horizontal reductions
    __m256 swapped = _mm256_permute2f128_ps(vsum, vsum, 1);
    __m256 folded = _mm256_add_ps(vsum, swapped);
    __m256 h1 = _mm256_hadd_ps(folded, folded);
    __m256 h2 = _mm256_hadd_ps(h1, h1);
    
    float s = _mm_cvtss_f32(_mm256_castps256_ps128(h2));
    
    // Summing tail elements
    for (; i < n; i++) {
        s += data[i];
    }
    
    return s;
}
#endif

} // namespace impl
} // namespace simd
