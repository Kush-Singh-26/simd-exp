#include <gtest/gtest.h>
#include <simd/ops/mat_transpose/scalar.hpp>
#include <vector>

// ── 4x4 tests (existing coverage, using generalized API) ─────────────────────

TEST(MatTransposeTest, IdentityMatrix) {
    float src[] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    float dst[16];
    simd::impl::transpose_scalar(src, dst, 4, 4);
    for (int i = 0; i < 16; i++) {
        EXPECT_FLOAT_EQ(src[i], dst[i]);
    }
}

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

TEST(MatTransposeTest, SymmetricMatrix) {
    float src[] = {
        1, 2, 3, 4,
        2, 5, 6, 7,
        3, 6, 8, 9,
        4, 7, 9, 10
    };
    float dst[16];
    simd::impl::transpose_scalar(src, dst, 4, 4);
    for (int i = 0; i < 16; i++) {
        EXPECT_FLOAT_EQ(src[i], dst[i]);
    }
}

TEST(MatTransposeTest, AllZeros) {
    float src[16] = {};
    float dst[16];
    simd::impl::transpose_scalar(src, dst, 4, 4);
    for (int i = 0; i < 16; i++) {
        EXPECT_FLOAT_EQ(dst[i], 0.0f);
    }
}

TEST(MatTransposeTest, NegativeValues) {
    float src[] = {
        -1, -2, -3, -4,
        -5, -6, -7, -8,
        -9, -10, -11, -12,
        -13, -14, -15, -16
    };
    float expected[] = {
        -1, -5, -9, -13,
        -2, -6, -10, -14,
        -3, -7, -11, -15,
        -4, -8, -12, -16
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

TEST(MatTransposeTest, AllOnes) {
    float src[16];
    for (int i = 0; i < 16; i++) src[i] = 1.0f;
    float dst[16];
    simd::impl::transpose_scalar(src, dst, 4, 4);
    for (int i = 0; i < 16; i++) {
        EXPECT_FLOAT_EQ(dst[i], 1.0f);
    }
}

TEST(MatTransposeTest, RandomData) {
    float src[16];
    for (int i = 0; i < 16; i++) src[i] = static_cast<float>(i * 7 + 3);
    float dst[16];
    simd::impl::transpose_scalar(src, dst, 4, 4);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            EXPECT_FLOAT_EQ(dst[j * 4 + i], src[i * 4 + j]);
        }
    }
}

TEST(MatTransposeTest, LargeScale) {
    float src[16], mid[16], dst[16];
    for (int i = 0; i < 16; i++) src[i] = static_cast<float>(i);
    for (int iter = 0; iter < 1000; iter++) {
        simd::impl::transpose_scalar(src, mid, 4, 4);
        simd::impl::transpose_scalar(mid, dst, 4, 4);
        for (int i = 0; i < 16; i++) {
            EXPECT_FLOAT_EQ(src[i], dst[i]);
        }
    }
}

// ── Non-4x4 tests (new C++23 mdspan coverage) ───────────────────────────────

TEST(MatTransposeTest, Rectangular_2x3) {
    // src: 2 rows x 3 cols
    // dst: 3 rows x 2 cols
    float src[] = {1, 2, 3, 4, 5, 6};
    float expected[] = {1, 4, 2, 5, 3, 6};
    float dst[6];
    simd::impl::transpose_scalar(src, dst, 2, 3);
    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(MatTransposeTest, Rectangular_3x2) {
    // src: 3 rows x 2 cols
    // dst: 2 rows x 3 cols
    float src[] = {1, 2, 3, 4, 5, 6};
    float expected[] = {1, 3, 5, 2, 4, 6};
    float dst[6];
    simd::impl::transpose_scalar(src, dst, 3, 2);
    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(MatTransposeTest, Rectangular_3x5) {
    // src: 3 rows x 5 cols = 15 elements
    // dst: 5 rows x 3 cols = 15 elements
    float src[15], expected[15];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 5; j++) {
            src[i * 5 + j] = static_cast<float>(i * 5 + j);
            expected[j * 3 + i] = src[i * 5 + j];
        }
    float dst[15];
    simd::impl::transpose_scalar(src, dst, 3, 5);
    for (int i = 0; i < 15; i++) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(MatTransposeTest, Square_8x8) {
    float src[64], expected[64];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) {
            src[i * 8 + j] = static_cast<float>(i * 8 + j);
            expected[j * 8 + i] = src[i * 8 + j];
        }
    float dst[64];
    simd::impl::transpose_scalar(src, dst, 8, 8);
    for (int i = 0; i < 64; i++) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(MatTransposeTest, Square_16x16) {
    std::vector<float> src(256), expected(256), dst(256);
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++) {
            src[i * 16 + j] = static_cast<float>(i * 16 + j);
            expected[j * 16 + i] = src[i * 16 + j];
        }
    simd::impl::transpose_scalar(src.data(), dst.data(), 16, 16);
    for (int i = 0; i < 256; i++) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(MatTransposeTest, Rectangular_1x8) {
    // Row vector -> column vector
    float src[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float expected[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float dst[8];
    simd::impl::transpose_scalar(src, dst, 1, 8);
    for (int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(MatTransposeTest, Rectangular_8x1) {
    // Column vector -> row vector
    float src[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float expected[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float dst[8];
    simd::impl::transpose_scalar(src, dst, 8, 1);
    for (int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(MatTransposeTest, NonSquare_RoundTrip) {
    // transpose then transpose back should give original
    float src[15], mid[15], dst[15];
    for (int i = 0; i < 15; i++) src[i] = static_cast<float>(i * 3 + 1);
    simd::impl::transpose_scalar(src, mid, 3, 5);
    simd::impl::transpose_scalar(mid, dst, 5, 3);
    for (int i = 0; i < 15; i++) {
        EXPECT_FLOAT_EQ(src[i], dst[i]);
    }
}

// ── SIMD 4x4 tests ───────────────────────────────────────────────────────────

#if defined(SIMD_AVX2_ENABLED)
#include <simd/ops/mat_transpose/simd.hpp>

TEST(MatTransposeTest, SIMD_KnownValues) {
    float src[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    float scalar_dst[16], simd_dst[16];
    simd::impl::transpose_scalar(src, scalar_dst, 4, 4);
    simd::impl::transpose_simd(src, simd_dst);
    for (int i = 0; i < 16; i++) {
        EXPECT_FLOAT_EQ(scalar_dst[i], simd_dst[i]);
    }
}

TEST(MatTransposeTest, SIMD_RandomData) {
    float src[16];
    for (int i = 0; i < 16; i++) src[i] = static_cast<float>(i * 7 + 3);
    float scalar_dst[16], simd_dst[16];
    simd::impl::transpose_scalar(src, scalar_dst, 4, 4);
    simd::impl::transpose_simd(src, simd_dst);
    for (int i = 0; i < 16; i++) {
        EXPECT_FLOAT_EQ(scalar_dst[i], simd_dst[i]);
    }
}

TEST(MatTransposeTest, SIMD_DoubleTranspose) {
    float src[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    float mid[16], dst[16];
    simd::impl::transpose_simd(src, mid);
    simd::impl::transpose_simd(mid, dst);
    for (int i = 0; i < 16; i++) {
        EXPECT_FLOAT_EQ(src[i], dst[i]);
    }
}

TEST(MatTransposeTest, SIMD_NTMatchesScalar) {
    float src[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    float scalar_dst[16], simd_nt_dst[16];
    simd::impl::transpose_scalar(src, scalar_dst, 4, 4);
    simd::impl::transpose_simd_nt(src, simd_nt_dst);
    for (int i = 0; i < 16; i++) {
        EXPECT_FLOAT_EQ(scalar_dst[i], simd_nt_dst[i]);
    }
}
#endif

// ── Dispatcher tests (public API) ────────────────────────────────────────────

#include <simd/ops/mat_transpose/mat_transpose.hpp>

TEST(MatTransposeTest, Dispatcher_4x4) {
    float src[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float expected[] = {1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16};
    float dst[16];
    simd::transpose(src, dst, 4, 4);
    for (int i = 0; i < 16; i++) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(MatTransposeTest, Dispatcher_8x8) {
    float src[64], expected[64];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) {
            src[i * 8 + j] = static_cast<float>(i * 8 + j);
            expected[j * 8 + i] = src[i * 8 + j];
        }
    float dst[64];
    simd::transpose(src, dst, 8, 8);
    for (int i = 0; i < 64; i++) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}
