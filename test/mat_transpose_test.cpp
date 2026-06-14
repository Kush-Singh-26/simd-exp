#include <gtest/gtest.h>
#include <simd/ops/mat_transpose/scalar.hpp>

TEST(MatTransposeTest, IdentityMatrix) {
    float src[] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    float dst[16];
    simd::impl::transpose_scalar(src, dst);
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
    simd::impl::transpose_scalar(src, dst);
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
    simd::impl::transpose_scalar(src, dst);
    for (int i = 0; i < 16; i++) {
        EXPECT_FLOAT_EQ(src[i], dst[i]);
    }
}

TEST(MatTransposeTest, AllZeros) {
    float src[16] = {};
    float dst[16];
    simd::impl::transpose_scalar(src, dst);
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
    simd::impl::transpose_scalar(src, dst);
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
    simd::impl::transpose_scalar(src, mid);
    simd::impl::transpose_scalar(mid, dst);
    for (int i = 0; i < 16; i++) {
        EXPECT_FLOAT_EQ(src[i], dst[i]);
    }
}

TEST(MatTransposeTest, AllOnes) {
    float src[16];
    for (int i = 0; i < 16; i++) src[i] = 1.0f;
    float dst[16];
    simd::impl::transpose_scalar(src, dst);
    for (int i = 0; i < 16; i++) {
        EXPECT_FLOAT_EQ(dst[i], 1.0f);
    }
}

TEST(MatTransposeTest, RandomData) {
    float src[16];
    for (int i = 0; i < 16; i++) src[i] = static_cast<float>(i * 7 + 3);
    float dst[16];
    simd::impl::transpose_scalar(src, dst);
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
        simd::impl::transpose_scalar(src, mid);
        simd::impl::transpose_scalar(mid, dst);
        for (int i = 0; i < 16; i++) {
            EXPECT_FLOAT_EQ(src[i], dst[i]);
        }
    }
}

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
    simd::impl::transpose_scalar(src, scalar_dst);
    simd::impl::transpose_simd(src, simd_dst);
    for (int i = 0; i < 16; i++) {
        EXPECT_FLOAT_EQ(scalar_dst[i], simd_dst[i]);
    }
}

TEST(MatTransposeTest, SIMD_RandomData) {
    float src[16];
    for (int i = 0; i < 16; i++) src[i] = static_cast<float>(i * 7 + 3);
    float scalar_dst[16], simd_dst[16];
    simd::impl::transpose_scalar(src, scalar_dst);
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
#endif
