#include <gtest/gtest.h>
#include <simd/common.hpp>
#include <simd/ops/abs/scalar.hpp>
#include <vector>
#include <cmath>
#include <random>

TEST(AbsTest, KnownValues) {
    float src[] = {-3.0f, -1.5f, 0.0f, 1.5f, 3.0f, -100.0f, 42.0f, -0.001f};
    float dst[8];
    simd::impl::abs_scalar(src, dst, 8);
    EXPECT_FLOAT_EQ(dst[0], 3.0f);
    EXPECT_FLOAT_EQ(dst[1], 1.5f);
    EXPECT_FLOAT_EQ(dst[2], 0.0f);
    EXPECT_FLOAT_EQ(dst[3], 1.5f);
    EXPECT_FLOAT_EQ(dst[4], 3.0f);
    EXPECT_FLOAT_EQ(dst[5], 100.0f);
    EXPECT_FLOAT_EQ(dst[6], 42.0f);
    EXPECT_FLOAT_EQ(dst[7], 0.001f);
}

TEST(AbsTest, PositiveData) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);
    std::vector<float> src(1024);
    for (auto& x : src) x = dist(rng);
    std::vector<float> dst(1024);
    simd::impl::abs_scalar(src.data(), dst.data(), 1024);
    for (size_t i = 0; i < 1024; i++) {
        EXPECT_FLOAT_EQ(dst[i], src[i]);
    }
}

TEST(AbsTest, NegativeData) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 0.0f);
    std::vector<float> src(1024);
    for (auto& x : src) x = dist(rng);
    std::vector<float> dst(1024);
    simd::impl::abs_scalar(src.data(), dst.data(), 1024);
    for (size_t i = 0; i < 1024; i++) {
        EXPECT_FLOAT_EQ(dst[i], -src[i]);
    }
}

TEST(AbsTest, MixedRandomData) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);
    std::vector<float> src(4096);
    for (auto& x : src) x = dist(rng);
    std::vector<float> dst(4096);
    simd::impl::abs_scalar(src.data(), dst.data(), 4096);
    for (size_t i = 0; i < 4096; i++) {
        EXPECT_GE(dst[i], 0.0f);
        EXPECT_FLOAT_EQ(dst[i], std::fabs(src[i]));
    }
}

TEST(AbsTest, TailElements) {
    float src[] = {-5.0f, 4.0f, -3.0f, 2.0f, -1.0f};
    float dst[5];
    simd::impl::abs_scalar(src, dst, 5);
    float expected[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    for (int i = 0; i < 5; i++) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(AbsTest, SingleElement) {
    float src[] = {-42.0f};
    float dst[1];
    simd::impl::abs_scalar(src, dst, 1);
    EXPECT_FLOAT_EQ(dst[0], 42.0f);
}

TEST(AbsTest, LargeInput) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    std::vector<float> src(1 << 20);
    for (auto& x : src) x = dist(rng);
    std::vector<float> dst(1 << 20);
    simd::impl::abs_scalar(src.data(), dst.data(), 1 << 20);
    for (size_t i = 0; i < (1 << 20); i++) {
        EXPECT_GE(dst[i], 0.0f);
        EXPECT_FLOAT_EQ(dst[i], std::fabs(src[i]));
    }
}

#if defined(SIMD_AVX2_ENABLED)
#include <simd/ops/abs/simd.hpp>

TEST(AbsTest, SIMD_KnownValues) {
    float src[] = {-3.0f, -1.5f, 0.0f, 1.5f, 3.0f, -100.0f, 42.0f, -0.001f};
    float scalar_dst[8], simd_dst[8];
    simd::impl::abs_scalar(src, scalar_dst, 8);
    simd::impl::abs_simd(src, simd_dst, 8);
    for (int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(scalar_dst[i], simd_dst[i]);
    }
}

TEST(AbsTest, SIMD_RandomData) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);
    std::vector<float> src(4096);
    for (auto& x : src) x = dist(rng);
    std::vector<float> scalar_dst(4096), simd_dst(4096);
    simd::impl::abs_scalar(src.data(), scalar_dst.data(), 4096);
    simd::impl::abs_simd(src.data(), simd_dst.data(), 4096);
    for (size_t i = 0; i < 4096; i++) {
        EXPECT_FLOAT_EQ(scalar_dst[i], simd_dst[i]);
    }
}

TEST(AbsTest, SIMD_TailElements) {
    float src[] = {-5.0f, 4.0f, -3.0f, 2.0f, -1.0f};
    float scalar_dst[5], simd_dst[5];
    simd::impl::abs_scalar(src, scalar_dst, 5);
    simd::impl::abs_simd(src, simd_dst, 5);
    for (int i = 0; i < 5; i++) {
        EXPECT_FLOAT_EQ(scalar_dst[i], simd_dst[i]);
    }
}

TEST(AbsTest, SIMD_LargeInput) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    std::vector<float> src(1 << 20);
    for (auto& x : src) x = dist(rng);
    std::vector<float> scalar_dst(1 << 20), simd_dst(1 << 20);
    simd::impl::abs_scalar(src.data(), scalar_dst.data(), 1 << 20);
    simd::impl::abs_simd(src.data(), simd_dst.data(), 1 << 20);
    for (size_t i = 0; i < (1 << 20); i++) {
        EXPECT_FLOAT_EQ(scalar_dst[i], simd_dst[i]);
    }
}
#endif

#include <simd/ops/abs/abs.hpp>

TEST(AbsTest, Dispatcher_KnownValues) {
    float src[] = {-3.0f, -1.5f, 0.0f, 1.5f, 3.0f, -100.0f, 42.0f, -0.001f};
    float expected[] = {3.0f, 1.5f, 0.0f, 1.5f, 3.0f, 100.0f, 42.0f, 0.001f};
    float dst[8];
    simd::abs(src, dst, 8);
    for (int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(AbsTest, Dispatcher_RandomData) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);
    std::vector<float> src(4096);
    for (auto& x : src) x = dist(rng);
    std::vector<float> dst(4096);
    simd::abs(src.data(), dst.data(), 4096);
    for (size_t i = 0; i < 4096; i++) {
        EXPECT_GE(dst[i], 0.0f);
        EXPECT_FLOAT_EQ(dst[i], std::fabs(src[i]));
    }
}

TEST(AbsTest, Dispatcher_NT_KnownValues) {
    float src[] = {-3.0f, -1.5f, 0.0f, 1.5f, 3.0f, -100.0f, 42.0f, -0.001f};
    float expected[] = {3.0f, 1.5f, 0.0f, 1.5f, 3.0f, 100.0f, 42.0f, 0.001f};
    float dst[8];
    simd::abs_nt(src, dst, 8);
    for (int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}
