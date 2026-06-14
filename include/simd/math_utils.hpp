#pragma once
#include "common.hpp"

#if defined(SIMD_AVX2_ENABLED)
#include <immintrin.h>

namespace simd {
namespace impl {

// Minimax polynomial constants for fast exp approximation
static const float EXP_HI = 88.3762626647950f;
static const float EXP_LO = -88.3762626647950f;
static const float LOG2EF = 1.44269504088896341f;
static const float LN2_HI = 6.93145751953125e-1f;
static const float LN2_LO = 1.42860682030941723212e-6f;
static const float P0 = 1.0000000754895593f;
static const float P1 = 6.931472284335791e-1f;
static const float P2 = 2.402264895851545e-1f;
static const float P3 = 5.550332399887598e-2f;
static const float P4 = 9.618038735174234e-3f;
static const float P5 = 1.339045359498462e-3f;
static const float P6 = 1.540357332908606e-4f;

inline __m256 avx2_exp_ps(__m256 x) {
    x = _mm256_min_ps(x, _mm256_set1_ps(EXP_HI));
    x = _mm256_max_ps(x, _mm256_set1_ps(EXP_LO));
    __m256 fx = _mm256_fmadd_ps(x, _mm256_set1_ps(LOG2EF), _mm256_set1_ps(0.5f));
    fx = _mm256_floor_ps(fx);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(LN2_HI), x);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(LN2_LO), x);
    __m256 y = _mm256_set1_ps(P6);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(P5));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(P4));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(P3));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(P2));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(P1));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(P0));
    __m256i imm0 = _mm256_cvttps_epi32(fx);
    imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
    imm0 = _mm256_slli_epi32(imm0, 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(imm0));
}

} // namespace impl
} // namespace simd

#endif
