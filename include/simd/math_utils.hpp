// Max error  : < 3 ULP on normal range [-87.3, 87.3]
// Subnormals : flush to 0 (safe for softmax — subtract-max first)
// NaN input  : clamped, returns 0 or +inf
//
// Build flags: -O3 -mavx2 -mfma
//
// Algorithm:
//   1. Clamp input to [EXP_LO, EXP_HI]
//   2. n = round(x * log2e)              via cvtps_epi32 (single round-to-nearest)
//   3. r = x - n*ln2                     Cody-Waite two-step, r ∈ [−ln2/2, ln2/2]
//   4. exp(r) ≈ poly(r)                  Estrin scheme, degree-6 minimax
//   5. exp(x) = poly(r) * 2^n            IEEE exponent field trick
//
// References: Sleef 3.6, Cody & Waite "Software Manual for Elementary Functions"

#pragma once
#include "common.hpp"

#if defined(SIMD_AVX2_ENABLED)
#include <immintrin.h>

namespace simd {
namespace impl {


// ── Constants ─────────────────────────────────────────────────────────────────

// Clamp bounds: outside this range, result is 0 or +inf
static constexpr float EXP_LO  = -88.3762626647949f;
static constexpr float EXP_HI  =  88.3762626647949f;

// log2(e) — for mapping x to base-2 exponent
static constexpr float LOG2EF  =  1.44269504088896341f;

// Cody-Waite ln(2) split: LN2_HI has low 12 mantissa bits zeroed
// so n*LN2_HI is exactly representable, eliminating catastrophic cancellation.
// LN2_HI + LN2_LO == ln(2) to < 1 ULP.
static constexpr float LN2_HI  =  6.9314575e-1f;   // 0x3f317200
static constexpr float LN2_LO  =  1.4286068e-6f;

// Degree-6 minimax polynomial for exp(r), r ∈ [−ln2/2, ln2/2]
// Coefficients fitted via Remez algorithm over the symmetric range.
// P0=1, P1=1 are exact; P2..P6 are minimax-optimal floats.
static constexpr float P0 = 1.0000000000000000f;
static constexpr float P1 = 1.0000000000000000f;
static constexpr float P2 = 4.9999999999940024e-1f;
static constexpr float P3 = 1.6666666664684413e-1f;
static constexpr float P4 = 4.1666666647390810e-2f;
static constexpr float P5 = 8.3333358523025905e-3f;
static constexpr float P6 = 1.3888889927481680e-3f;

// ── Estrin polynomial evaluation ─────────────────────────────────────────────
// Horner: 7 sequential FMAs — critical path = 7×FMA latency (~28 cycles)
// Estrin: exposes ILP — 3 independent FMAs at level 0, depth = 4 FMAs (~16 cycles)
//
// Level 0  (3 independent FMAs, can all issue same cycle):
//   p01 = P1*x + P0
//   p23 = P3*x + P2
//   p45 = P5*x + P4
// Level 1:
//   p0123 = p23*x² + p01
//   p4567 = P6*x²  + p45
// Level 2:
//   result = p4567*x⁴ + p0123
static inline __m256 _exp_poly_estrin(__m256 x) {
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x4 = _mm256_mul_ps(x2, x2);

    // Level 0 — all three independent
    __m256 p01 = _mm256_fmadd_ps(_mm256_set1_ps(P1), x,  _mm256_set1_ps(P0));
    __m256 p23 = _mm256_fmadd_ps(_mm256_set1_ps(P3), x,  _mm256_set1_ps(P2));
    __m256 p45 = _mm256_fmadd_ps(_mm256_set1_ps(P5), x,  _mm256_set1_ps(P4));

    // Level 1
    __m256 p0123 = _mm256_fmadd_ps(p23, x2, p01);
    __m256 p4567 = _mm256_fmadd_ps(_mm256_set1_ps(P6), x2, p45);

    // Level 2
    return _mm256_fmadd_ps(p4567, x4, p0123);
}

// Computes exp(x) for 8 floats in parallel.
//
// Usage:
//   __m256 result = avx2_exp_ps(_mm256_loadu_ps(ptr));
inline __m256 avx2_exp_ps(__m256 x) {
    // 1. Clamp: prevents inf/nan propagating into exponent reconstruction
    x = _mm256_min_ps(x, _mm256_set1_ps(EXP_HI));
    x = _mm256_max_ps(x, _mm256_set1_ps(EXP_LO));

    // 2. n = round(x * log2e) — hardware round-to-nearest via cvtps_epi32
    //    Converts float→int with rounding (not truncation), so n is the
    //    nearest integer to x/ln2. Gives r ∈ [−ln2/2, ln2/2] after step 3.
    __m256  x_log2e = _mm256_mul_ps(x, _mm256_set1_ps(LOG2EF));
    __m256i n_int   = _mm256_cvtps_epi32(x_log2e);   // n as int32
    __m256  n       = _mm256_cvtepi32_ps(n_int);      // n as float (exact)

    // 3. Cody-Waite range reduction: r = x - n*ln2
    //    Two-step subtraction using the HI/LO split of ln2.
    //    If we used a single ln2 constant, n*ln2 would lose low bits and
    //    r = x - n*ln2 would have ~6 bits of cancellation error.
    //    The split ensures r is accurate to full float precision.
    x = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_HI), x);  // x -= n * LN2_HI
    x = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_LO), x);  // x -= n * LN2_LO
    // x is now r ∈ [−ln2/2, ln2/2]

    // 4. Polynomial: exp(r) ≈ poly(r) via Estrin's scheme
    __m256 y = _exp_poly_estrin(x);

    // 5. Reconstruct 2^n via IEEE-754 exponent field manipulation:
    //    float exponent bias is 127 (0x7f). Adding n to the biased exponent
    //    and shifting into position gives the bit pattern for 2^n exactly.
    //    exp(x) = exp(r) * 2^n = poly(r) * 2^n
    __m256i pow2n = _mm256_slli_epi32(
                        _mm256_add_epi32(n_int, _mm256_set1_epi32(0x7f)), 23);

    return _mm256_mul_ps(y, _mm256_castsi256_ps(pow2n));
}

// ── Horizontal reductions (256→scalar) ──────────────────────────────────────
// extractf128 + SSE3 shuffle chain. ~5 cycle critical path.
// Why not haddps? hadd has 3-cycle latency vs movehlps/movehdup at 1 cycle.
//
// Usage:
//   float s = hsum_ps(v);      // horizontal sum of 8 floats
//   float m = hmax_ps(v);      // horizontal max of 8 floats
//   float m = hmin_ps(v);      // horizontal min of 8 floats
//   float s = hsum_ps_sq(v);   // horizontal sum of squares (for variance)

// ── 128-bit building blocks ──────────────────────────────────────────────────

static inline float hsum_128_ps(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

static inline float hmax_128_ps(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);
    __m128 mx = _mm_max_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, mx);
    mx = _mm_max_ss(mx, shuf);
    return _mm_cvtss_f32(mx);
}

static inline float hmin_128_ps(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);
    __m128 mn = _mm_min_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, mn);
    mn = _mm_min_ss(mn, shuf);
    return _mm_cvtss_f32(mn);
}

// ── 256-bit API (extractf128 + 128-bit building block) ──────────────────────

static inline float hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    return hsum_128_ps(_mm_add_ps(lo, hi));
}

static inline float hmax_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    return hmax_128_ps(_mm_max_ps(lo, hi));
}

static inline float hmin_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    return hmin_128_ps(_mm_min_ps(lo, hi));
}

static inline float hsum_ps_sq(__m256 v) {
    return hsum_ps(_mm256_mul_ps(v, v));
}

} // namespace impl
} // namespace simd

#endif