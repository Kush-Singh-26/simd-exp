#pragma once
#include "../../common.hpp"
#include <cstddef>

namespace simd {
namespace impl {

#if defined(SIMD_AVX2_ENABLED)

// ── 4x4 SSE transpose (128-bit) ────────────────────────────────────────────

template <typename StoreFn>
void transpose_impl(const float* src, float* dst, StoreFn store) {
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

    store(dst + 0,  f1);
    store(dst + 4,  f2);
    store(dst + 8,  f3);
    store(dst + 12, f4);
}

inline void transpose_simd(const float* src, float* dst) {
    transpose_impl(src, dst, [](float* p, __m128 v) { _mm_storeu_ps(p, v); });
}

inline void transpose_simd_nt(const float* src, float* dst) {
    transpose_impl(src, dst, [](float* p, __m128 v) { _mm_storeu_ps(p, v); });
}

// ── 8x8 AVX2 strided transpose (256-bit) ────────────────────────────────────
// Used for blocked transpose of larger NxM matrices. Supports arbitrary
// src_stride and dst_stride for non-contiguous 8x8 blocks.

template <typename StoreFn>
void transpose8x8_impl(const float* src, float* dst,
                       size_t src_stride, size_t dst_stride,
                       StoreFn store) {
    __m256 row0 = _mm256_loadu_ps(src + 0 * src_stride);
    __m256 row1 = _mm256_loadu_ps(src + 1 * src_stride);
    __m256 row2 = _mm256_loadu_ps(src + 2 * src_stride);
    __m256 row3 = _mm256_loadu_ps(src + 3 * src_stride);
    __m256 row4 = _mm256_loadu_ps(src + 4 * src_stride);
    __m256 row5 = _mm256_loadu_ps(src + 5 * src_stride);
    __m256 row6 = _mm256_loadu_ps(src + 6 * src_stride);
    __m256 row7 = _mm256_loadu_ps(src + 7 * src_stride);

    __m256 t0 = _mm256_unpacklo_ps(row0, row1);
    __m256 t1 = _mm256_unpackhi_ps(row0, row1);
    __m256 t2 = _mm256_unpacklo_ps(row2, row3);
    __m256 t3 = _mm256_unpackhi_ps(row2, row3);
    __m256 t4 = _mm256_unpacklo_ps(row4, row5);
    __m256 t5 = _mm256_unpackhi_ps(row4, row5);
    __m256 t6 = _mm256_unpacklo_ps(row6, row7);
    __m256 t7 = _mm256_unpackhi_ps(row6, row7);

    __m256 m0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 m1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 m2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 m3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 m4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 m5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 m6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 m7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

    store(dst + 0 * dst_stride, _mm256_permute2f128_ps(m0, m4, 0x20));
    store(dst + 1 * dst_stride, _mm256_permute2f128_ps(m1, m5, 0x20));
    store(dst + 2 * dst_stride, _mm256_permute2f128_ps(m2, m6, 0x20));
    store(dst + 3 * dst_stride, _mm256_permute2f128_ps(m3, m7, 0x20));
    store(dst + 4 * dst_stride, _mm256_permute2f128_ps(m0, m4, 0x31));
    store(dst + 5 * dst_stride, _mm256_permute2f128_ps(m1, m5, 0x31));
    store(dst + 6 * dst_stride, _mm256_permute2f128_ps(m2, m6, 0x31));
    store(dst + 7 * dst_stride, _mm256_permute2f128_ps(m3, m7, 0x31));
}

// Strided 8x8 (for blocked transpose of larger NxM)
inline void transpose8x8_strided_simd(const float* src, float* dst,
                                       size_t src_stride, size_t dst_stride) {
    transpose8x8_impl(src, dst, src_stride, dst_stride,
        [](float* p, __m256 v) { _mm256_storeu_ps(p, v); });
}

inline void transpose8x8_strided_simd_nt(const float* src, float* dst,
                                          size_t src_stride, size_t dst_stride) {
    transpose8x8_impl(src, dst, src_stride, dst_stride,
        [](float* p, __m256 v) { _mm256_stream_ps(p, v); });
}

// Flat 8x8 (contiguous, stride == 8)
inline void transpose8x8_simd(const float* src, float* dst) {
    transpose8x8_strided_simd(src, dst, 8, 8);
}

inline void transpose8x8_simd_nt(const float* src, float* dst) {
    transpose8x8_strided_simd_nt(src, dst, 8, 8);
}
#endif

} // namespace impl
} // namespace simd
