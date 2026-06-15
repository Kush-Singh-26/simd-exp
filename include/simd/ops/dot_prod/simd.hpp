#pragma once
#include "../../math_utils.hpp"
#include <cstddef>

namespace simd {
namespace impl {

#if defined(SIMD_AVX2_ENABLED) && defined(__FMA__)
inline float dot_prod_simd(const float* data1, const float* data2, size_t n) {
    __m256 vsum0 = _mm256_setzero_ps();
    __m256 vsum1 = _mm256_setzero_ps();
    __m256 vsum2 = _mm256_setzero_ps();
    __m256 vsum3 = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        vsum0 = _mm256_fmadd_ps(_mm256_loadu_ps(data1 + i),      _mm256_loadu_ps(data2 + i),      vsum0);
        vsum1 = _mm256_fmadd_ps(_mm256_loadu_ps(data1 + i + 8),  _mm256_loadu_ps(data2 + i + 8),  vsum1);
        vsum2 = _mm256_fmadd_ps(_mm256_loadu_ps(data1 + i + 16), _mm256_loadu_ps(data2 + i + 16), vsum2);
        vsum3 = _mm256_fmadd_ps(_mm256_loadu_ps(data1 + i + 24), _mm256_loadu_ps(data2 + i + 24), vsum3);
    }

    vsum0 = _mm256_add_ps(vsum0, vsum1);
    vsum2 = _mm256_add_ps(vsum2, vsum3);
    vsum0 = _mm256_add_ps(vsum0, vsum2);

    float d = hsum_ps(vsum0);
    
    for (; i < n; i++) {
        d += data1[i] * data2[i];
    }

    return d;    
}
#endif

} // namespace impl
} // namespace simd
