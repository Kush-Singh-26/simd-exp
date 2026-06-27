#pragma once
#include "../../math_utils.hpp"
#include <cstddef>

namespace simd {
namespace impl {

#if defined(SIMD_AVX2_ENABLED) && defined(__FMA__)
inline float dot_prod_simd(const float* data1, const float* data2, size_t n) {
    float d = 0.0f;
    size_t n_simd = n / 32;
    #pragma omp parallel reduction(+:d) if(n_simd >= 128)
    {
        __m256 v0 = _mm256_setzero_ps();
        __m256 v1 = _mm256_setzero_ps();
        __m256 v2 = _mm256_setzero_ps();
        __m256 v3 = _mm256_setzero_ps();
        #pragma omp for nowait
        for (size_t idx = 0; idx < n_simd; idx++) {
            size_t i = idx * 32;
            v0 = _mm256_fmadd_ps(_mm256_loadu_ps(data1 + i),      _mm256_loadu_ps(data2 + i),      v0);
            v1 = _mm256_fmadd_ps(_mm256_loadu_ps(data1 + i + 8),  _mm256_loadu_ps(data2 + i + 8),  v1);
            v2 = _mm256_fmadd_ps(_mm256_loadu_ps(data1 + i + 16), _mm256_loadu_ps(data2 + i + 16), v2);
            v3 = _mm256_fmadd_ps(_mm256_loadu_ps(data1 + i + 24), _mm256_loadu_ps(data2 + i + 24), v3);
        }
        __m256 vsum = _mm256_add_ps(_mm256_add_ps(v0, v1), _mm256_add_ps(v2, v3));
        d += hsum_ps(vsum);
    }
    for (size_t i = n_simd * 32; i < n; i++) {
        d += data1[i] * data2[i];
    }
    return d;    
}
#endif

} // namespace impl
} // namespace simd
