#pragma once
#include "../../common.hpp"
#include <cstddef>

namespace simd {
namespace impl {

#if defined(SIMD_AVX2_ENABLED)
template <typename StoreFn>
void relu_impl(const float* src, float* dst, size_t n, StoreFn store) {
    __m256 zero = _mm256_setzero_ps();
    size_t n_simd = n / 8;
    #pragma omp parallel for firstprivate(store) if(n_simd >= 128)
    for (size_t idx = 0; idx < n_simd; idx++) {
        size_t i = idx * 8;
        __m256 vx = _mm256_loadu_ps(src + i);
        vx = _mm256_max_ps(vx, zero);
        store(dst + i, vx);
    }
    for (size_t i = n_simd * 8; i < n; i++) {
        dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
    }
}

inline void relu_simd(const float* src, float* dst, size_t n) {
    relu_impl(src, dst, n, [](float* p, __m256 v) { _mm256_storeu_ps(p, v); });
}

inline void relu_simd_nt(const float* src, float* dst, size_t n) {
    relu_impl(src, dst, n, [](float* p, __m256 v) { _mm256_storeu_ps(p, v); });
}
#endif

} // namespace impl
} // namespace simd
