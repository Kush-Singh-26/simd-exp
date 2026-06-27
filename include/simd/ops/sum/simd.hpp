#pragma once
#include "../../math_utils.hpp"
#include <cstddef>

namespace simd {
namespace impl {

#if defined(SIMD_AVX2_ENABLED)
inline float sum_simd(const float* data, size_t n) {
    size_t n_simd = n / 8;
    float s = 0.0f;
    #pragma omp parallel reduction(+:s) if(n_simd >= 512)
    {
        __m256 vlocal = _mm256_setzero_ps();
        #pragma omp for nowait
        for (size_t idx = 0; idx < n_simd; idx++) {
            vlocal = _mm256_add_ps(vlocal, _mm256_loadu_ps(data + idx * 8));
        }
        s += hsum_ps(vlocal);
    }
    for (size_t i = n_simd * 8; i < n; i++) {
        s += data[i];
    }
    return s;
}
#endif

} // namespace impl
} // namespace simd
