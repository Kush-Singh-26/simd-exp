#pragma once
#include "../../common.hpp"
#include <cstddef>

namespace simd {
namespace impl {

#if defined(SIMD_AVX2_ENABLED)
inline void transpose_simd(const float* src, float* dst) {
    __m128 r1 = _mm_loadu_ps(&src[0]);
    __m128 r2 = _mm_loadu_ps(&src[4]);
    __m128 r3 = _mm_loadu_ps(&src[8]);
    __m128 r4 = _mm_loadu_ps(&src[12]);

    __m128 t1 = _mm_unpacklo_ps(r1, r2);
    __m128 t2 = _mm_unpacklo_ps(r3, r4);
    __m128 t3 = _mm_unpackhi_ps(r1, r2);
    __m128 t4 = _mm_unpackhi_ps(r3, r4);

    __m128 f1 = _mm_shuffle_ps(t1, t2, _MM_SHUFFLE(1, 0, 1, 0));
    __m128 f2 = _mm_shuffle_ps(t1, t2, _MM_SHUFFLE(3, 2, 3, 2));
    __m128 f3 = _mm_shuffle_ps(t3, t4, _MM_SHUFFLE(1, 0, 1, 0));
    __m128 f4 = _mm_shuffle_ps(t3, t4, _MM_SHUFFLE(3, 2, 3, 2));

    _mm_storeu_ps(&dst[0], f1);
    _mm_storeu_ps(&dst[4], f2);
    _mm_storeu_ps(&dst[8], f3);
    _mm_storeu_ps(&dst[12], f4);
}

inline void transpose_simd_nt(const float* src, float* dst) {
    __m128 r1 = _mm_loadu_ps(&src[0]);
    __m128 r2 = _mm_loadu_ps(&src[4]);
    __m128 r3 = _mm_loadu_ps(&src[8]);
    __m128 r4 = _mm_loadu_ps(&src[12]);

    __m128 t1 = _mm_unpacklo_ps(r1, r2);
    __m128 t2 = _mm_unpacklo_ps(r3, r4);
    __m128 t3 = _mm_unpackhi_ps(r1, r2);
    __m128 t4 = _mm_unpackhi_ps(r3, r4);

    __m128 f1 = _mm_shuffle_ps(t1, t2, _MM_SHUFFLE(1, 0, 1, 0));
    __m128 f2 = _mm_shuffle_ps(t1, t2, _MM_SHUFFLE(3, 2, 3, 2));
    __m128 f3 = _mm_shuffle_ps(t3, t4, _MM_SHUFFLE(1, 0, 1, 0));
    __m128 f4 = _mm_shuffle_ps(t3, t4, _MM_SHUFFLE(3, 2, 3, 2));

    _mm_stream_ps(&dst[0], f1);
    _mm_stream_ps(&dst[4], f2);
    _mm_stream_ps(&dst[8], f3);
    _mm_stream_ps(&dst[12], f4);
}
#endif

} // namespace impl
} // namespace simd
