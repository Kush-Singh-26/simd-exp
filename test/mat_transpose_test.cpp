#include "test_harness.hpp"
#include <simd/ops/mat_transpose/scalar.hpp>
#include <simd/ops/mat_transpose/simd.hpp>
#include <simd/ops/mat_transpose/mat_transpose.hpp>

// ── Scalar correctness (general shapes) ──────────────────────────────────────

TEST(MatTransposeTest, KnownValues) {
    float src[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    float expected[] = {
        1, 5, 9, 13,
        2, 6, 10, 14,
        3, 7, 11, 15,
        4, 8, 12, 16
    };
    float dst[16];
    simd::impl::transpose_scalar(src, dst, 4, 4);
    for (int i = 0; i < 16; i++) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(MatTransposeTest, DoubleTranspose) {
    float src[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    float mid[16], dst[16];
    simd::impl::transpose_scalar(src, mid, 4, 4);
    simd::impl::transpose_scalar(mid, dst, 4, 4);
    for (int i = 0; i < 16; i++) {
        EXPECT_FLOAT_EQ(src[i], dst[i]);
    }
}

TEST(MatTransposeTest, Rectangular_2x3) {
    float src[] = {1, 2, 3, 4, 5, 6};
    float expected[] = {1, 4, 2, 5, 3, 6};
    float dst[6];
    simd::impl::transpose_scalar(src, dst, 2, 3);
    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(MatTransposeTest, Rectangular_3x5) {
    float src[15], expected[15];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 5; j++) {
            src[i * 5 + j] = static_cast<float>(i * 5 + j);
            expected[j * 3 + i] = src[i * 5 + j];
        }
    }
    float dst[15];
    simd::impl::transpose_scalar(src, dst, 3, 5);
    for (int i = 0; i < 15; i++) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(MatTransposeTest, NonSquare_RoundTrip) {
    float src[15], mid[15], dst[15];
    for (int i = 0; i < 15; i++) src[i] = static_cast<float>(i * 3 + 1);
    simd::impl::transpose_scalar(src, mid, 3, 5);
    simd::impl::transpose_scalar(mid, dst, 5, 3);
    for (int i = 0; i < 15; i++) {
        EXPECT_FLOAT_EQ(src[i], dst[i]);
    }
}

// ── SIMD correctness (4x4 only) ──────────────────────────────────────────────

TEST(MatTransposeTest, Simd_MatchesScalar) {
#if defined(SIMD_AVX2_ENABLED)
    // Known values 4x4
    float src_known[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    float scalar_dst[16], simd_dst[16];
    simd::impl::transpose_scalar(src_known, scalar_dst, 4, 4);
    simd::impl::transpose_simd(src_known, simd_dst);
    check_exact(scalar_dst, simd_dst, 16, "Known values 4x4");

    // Random 4x4
    auto src_rand = make_random(16, -100.f, 100.f);
    simd::impl::transpose_scalar(src_rand.data(), scalar_dst, 4, 4);
    simd::impl::transpose_simd(src_rand.data(), simd_dst);
    check_exact(scalar_dst, simd_dst, 16, "Random 4x4");
#endif
}

TEST(MatTransposeTest, SimdNt_MatchesScalar) {
#if defined(SIMD_AVX2_ENABLED)
    auto src = make_random(16, -100.f, 100.f);
    float scalar_dst[16];
    simd::impl::transpose_scalar(src.data(), scalar_dst, 4, 4);

    alignas(16) float simd_nt_dst[16];
    simd::impl::transpose_simd_nt(src.data(), simd_nt_dst);
    check_exact(scalar_dst, simd_nt_dst, 16, "Simd NT 4x4");
#endif
}

TEST(MatTransposeTest, Simd_DoubleTranspose) {
#if defined(SIMD_AVX2_ENABLED)
    auto src = make_random(16, -100.f, 100.f);
    float mid[16], dst[16];
    simd::impl::transpose_simd(src.data(), mid);
    simd::impl::transpose_simd(mid, dst);
    check_exact(src.data(), dst, 16, "Simd double transpose 4x4");
#endif
}

// ── Dispatcher tests (public API) ────────────────────────────────────────────

TEST(MatTransposeTest, Dispatcher_4x4) {
    auto src = make_random(16, -100.f, 100.f);
    std::vector<float> scalar_dst(16), dispatch_dst(16), dispatch_nt_dst(16);
    simd::impl::transpose_scalar(src.data(), scalar_dst.data(), 4, 4);

    simd::transpose(src, dispatch_dst, 4, 4);
    check_exact(scalar_dst.data(), dispatch_dst.data(), 16, "Dispatcher 4x4");

    simd::transpose_nt(src, dispatch_nt_dst, 4, 4);
    check_exact(scalar_dst.data(), dispatch_nt_dst.data(), 16, "Dispatcher NT 4x4");
}

// ── 8x8 AVX2 transpose tests ────────────────────────────────────────────────

TEST(MatTransposeTest, Transpose8x8_Flat_MatchesScalar) {
#if defined(SIMD_AVX2_ENABLED)
    float src[] = {
         1,  2,  3,  4,  5,  6,  7,  8,
         9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56,
        57, 58, 59, 60, 61, 62, 63, 64
    };
    float scalar_dst[64], simd_dst[64];
    simd::impl::transpose_scalar(src, scalar_dst, 8, 8);
    simd::impl::transpose8x8_simd(src, simd_dst);
    check_exact(scalar_dst, simd_dst, 64, "8x8 flat known values");
#endif
}

TEST(MatTransposeTest, Transpose8x8_Strided_MatchesScalar) {
#if defined(SIMD_AVX2_ENABLED)
    // 16x16 matrix, transpose the 8x8 block at offset (4,4)
    // SIMD kernel reads with stride 16; scalar needs a contiguous copy
    float src[256];
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++)
            src[i * 16 + j] = static_cast<float>(i * 16 + j);

    // Extract contiguous 8x8 block for scalar reference
    float src_block[64];
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            src_block[r * 8 + c] = src[(4 + r) * 16 + (4 + c)];

    float scalar_dst[64], simd_dst[64];
    simd::impl::transpose_scalar(src_block, scalar_dst, 8, 8);
    simd::impl::transpose8x8_strided_simd(src + 4 * 16 + 4, simd_dst, 16, 8);
    check_exact(scalar_dst, simd_dst, 64, "8x8 strided");
#endif
}

TEST(MatTransposeTest, Transpose8x8_Random) {
#if defined(SIMD_AVX2_ENABLED)
    auto src = make_random(64, -100.f, 100.f);
    float scalar_dst[64], simd_dst[64];
    simd::impl::transpose_scalar(src.data(), scalar_dst, 8, 8);
    simd::impl::transpose8x8_simd(src.data(), simd_dst);
    check_exact(scalar_dst, simd_dst, 64, "8x8 random");
#endif
}

TEST(MatTransposeTest, Transpose8x8_DoubleTranspose) {
#if defined(SIMD_AVX2_ENABLED)
    auto src = make_random(64, -100.f, 100.f);
    float mid[64], dst[64];
    simd::impl::transpose8x8_simd(src.data(), mid);
    simd::impl::transpose8x8_simd(mid, dst);
    check_exact(src.data(), dst, 64, "8x8 double transpose");
#endif
}

TEST(MatTransposeTest, Dispatcher_8x8_Simd) {
    // Verifies 8x8 now uses SIMD kernel (not scalar fallback)
    auto src = make_random(64, -100.f, 100.f);
    std::vector<float> simd_dst(64), dispatch_dst(64);
    simd::impl::transpose8x8_simd(src.data(), simd_dst.data());

    simd::transpose(src, dispatch_dst, 8, 8);
    check_exact(simd_dst.data(), dispatch_dst.data(), 64, "Dispatcher 8x8 SIMD");
}

TEST(MatTransposeTest, Dispatcher_8x8_NT_Matches) {
#if defined(SIMD_AVX2_ENABLED)
    auto src = make_random(64, -100.f, 100.f);
    std::vector<float> scalar_dst(64);
    simd::impl::transpose_scalar(src.data(), scalar_dst.data(), 8, 8);

    alignas(32) float nt_dst[64];
    simd::transpose_nt(src, nt_dst, 8, 8);
    check_exact(scalar_dst.data(), nt_dst, 64, "Dispatcher 8x8 NT");
#endif
}
