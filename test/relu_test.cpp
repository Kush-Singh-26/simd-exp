#include <gtest/gtest.h>
#include <simd/common.hpp>
#include <simd/ops/relu/scalar.hpp>
#include <vector>
#include <random>

TEST(ReluTest, KnownValues) {
    float src[] = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f, -0.5f, 0.5f, 100.0f};
    float dst[8];
    simd::impl::relu_scalar(src, dst, 8);
    float expected[] = {0.0f, 0.0f, 0.0f, 1.0f, 3.0f, 0.0f, 0.5f, 100.0f};
    for (int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(ReluTest, AllPositive) {
    std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float dst[8];
    simd::impl::relu_scalar(src.data(), dst, 8);
    for (int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(dst[i], src[i]);
    }
}

TEST(ReluTest, AllNegative) {
    std::vector<float> src = {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f};
    float dst[8];
    simd::impl::relu_scalar(src.data(), dst, 8);
    for (int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(dst[i], 0.0f);
    }
}

TEST(ReluTest, TailElements) {
    float src[] = {-2.0f, 3.0f, -1.0f, 4.0f, 0.0f};
    float dst[5];
    simd::impl::relu_scalar(src, dst, 5);
    float expected[] = {0.0f, 3.0f, 0.0f, 4.0f, 0.0f};
    for (int i = 0; i < 5; i++) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(ReluTest, SingleElement) {
    float src[] = {-5.0f};
    float dst[1];
    simd::impl::relu_scalar(src, dst, 1);
    EXPECT_FLOAT_EQ(dst[0], 0.0f);
}

TEST(ReluTest, RandomData) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    std::vector<float> src(1024);
    for (auto& x : src) x = dist(rng);
    std::vector<float> dst(1024);
    simd::impl::relu_scalar(src.data(), dst.data(), 1024);
    for (size_t i = 0; i < 1024; i++) {
        EXPECT_GE(dst[i], 0.0f);
        EXPECT_FLOAT_EQ(dst[i], std::fmax(0.0f, src[i]));
    }
}

TEST(ReluTest, LargeInput) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);
    std::vector<float> src(1 << 20);
    for (auto& x : src) x = dist(rng);
    std::vector<float> dst(1 << 20);
    simd::impl::relu_scalar(src.data(), dst.data(), 1 << 20);
    for (size_t i = 0; i < (1 << 20); i++) {
        EXPECT_GE(dst[i], 0.0f);
        EXPECT_FLOAT_EQ(dst[i], std::fmax(0.0f, src[i]));
    }
}

#if defined(SIMD_AVX2_ENABLED)
#include <simd/ops/relu/simd.hpp>

TEST(ReluTest, SIMD_KnownValues) {
    float src[] = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f, -0.5f, 0.5f, 100.0f};
    float scalar_dst[8], simd_dst[8];
    simd::impl::relu_scalar(src, scalar_dst, 8);
    simd::impl::relu_simd(src, simd_dst, 8);
    for (int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(scalar_dst[i], simd_dst[i]);
    }
}

TEST(ReluTest, SIMD_RandomData) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    std::vector<float> src(1024);
    for (auto& x : src) x = dist(rng);
    std::vector<float> scalar_dst(1024), simd_dst(1024);
    simd::impl::relu_scalar(src.data(), scalar_dst.data(), 1024);
    simd::impl::relu_simd(src.data(), simd_dst.data(), 1024);
    for (size_t i = 0; i < 1024; i++) {
        EXPECT_FLOAT_EQ(scalar_dst[i], simd_dst[i]);
    }
}

TEST(ReluTest, SIMD_TailElements) {
    float src[] = {-2.0f, 3.0f, -1.0f, 4.0f, 0.0f};
    float scalar_dst[5], simd_dst[5];
    simd::impl::relu_scalar(src, scalar_dst, 5);
    simd::impl::relu_simd(src, simd_dst, 5);
    for (int i = 0; i < 5; i++) {
        EXPECT_FLOAT_EQ(scalar_dst[i], simd_dst[i]);
    }
}

TEST(ReluTest, SIMD_LargeInput) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);
    std::vector<float> src(1 << 20);
    for (auto& x : src) x = dist(rng);
    std::vector<float> scalar_dst(1 << 20), simd_dst(1 << 20);
    simd::impl::relu_scalar(src.data(), scalar_dst.data(), 1 << 20);
    simd::impl::relu_simd(src.data(), simd_dst.data(), 1 << 20);
    for (size_t i = 0; i < (1 << 20); i++) {
        EXPECT_FLOAT_EQ(scalar_dst[i], simd_dst[i]);
    }
}
#endif

#include <simd/ops/relu/relu.hpp>

TEST(ReluTest, Dispatcher_KnownValues) {
    float src[] = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f, -0.5f, 0.5f, 100.0f};
    float expected[] = {0.0f, 0.0f, 0.0f, 1.0f, 3.0f, 0.0f, 0.5f, 100.0f};
    float dst[8];
    simd::relu(src, dst, 8);
    for (int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(ReluTest, Dispatcher_RandomData) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    std::vector<float> src(1024);
    for (auto& x : src) x = dist(rng);
    std::vector<float> dst(1024);
    simd::relu(src.data(), dst.data(), 1024);
    for (size_t i = 0; i < 1024; i++) {
        EXPECT_GE(dst[i], 0.0f);
        EXPECT_FLOAT_EQ(dst[i], std::fmax(0.0f, src[i]));
    }
}

TEST(ReluTest, Dispatcher_NT_KnownValues) {
    float src[] = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f, -0.5f, 0.5f, 100.0f};
    float expected[] = {0.0f, 0.0f, 0.0f, 1.0f, 3.0f, 0.0f, 0.5f, 100.0f};
    float dst[8];
    simd::relu_nt(src, dst, 8);
    for (int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}
