#include <gtest/gtest.h>
#include <simd/common.hpp>
#include <simd/ops/clamp/scalar.hpp>
#include <vector>
#include <random>

TEST(ClampTest, KnownValues) {
    float src[] = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f, 15.0f, -1.0f, 3.0f};
    float dst[8];
    simd::impl::clamp_scalar(src, dst, 8, -5.0f, 5.0f);
    float expected[] = {-5.0f, -5.0f, 0.0f, 5.0f, 5.0f, 5.0f, -1.0f, 3.0f};
    for (int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(ClampTest, AllInRange) {
    std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float dst[8];
    simd::impl::clamp_scalar(src.data(), dst, 8, 0.0f, 10.0f);
    for (int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(dst[i], src[i]);
    }
}

TEST(ClampTest, AllBelow) {
    std::vector<float> src = {-10.0f, -20.0f, -30.0f, -40.0f, -50.0f, -60.0f, -70.0f, -80.0f};
    float dst[8];
    simd::impl::clamp_scalar(src.data(), dst, 8, 0.0f, 1.0f);
    for (int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(dst[i], 0.0f);
    }
}

TEST(ClampTest, AllAbove) {
    std::vector<float> src = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    float dst[8];
    simd::impl::clamp_scalar(src.data(), dst, 8, 0.0f, 1.0f);
    for (int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(dst[i], 1.0f);
    }
}

TEST(ClampTest, TailElements) {
    float src[] = {-5.0f, 3.0f, 10.0f, -1.0f, 7.0f};
    float dst[5];
    simd::impl::clamp_scalar(src, dst, 5, 0.0f, 5.0f);
    float expected[] = {0.0f, 3.0f, 5.0f, 0.0f, 5.0f};
    for (int i = 0; i < 5; i++) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(ClampTest, SingleElement) {
    float src[] = {42.0f};
    float dst[1];
    simd::impl::clamp_scalar(src, dst, 1, 0.0f, 10.0f);
    EXPECT_FLOAT_EQ(dst[0], 10.0f);
}

TEST(ClampTest, RandomData) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    std::vector<float> src(1024);
    for (auto& x : src) x = dist(rng);
    std::vector<float> dst(1024);
    simd::impl::clamp_scalar(src.data(), dst.data(), 1024, -10.0f, 10.0f);
    for (size_t i = 0; i < 1024; i++) {
        EXPECT_GE(dst[i], -10.0f);
        EXPECT_LE(dst[i], 10.0f);
        if (src[i] >= -10.0f && src[i] <= 10.0f) {
            EXPECT_FLOAT_EQ(dst[i], src[i]);
        }
    }
}

TEST(ClampTest, LargeInput) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    std::vector<float> src(1 << 20);
    for (auto& x : src) x = dist(rng);
    std::vector<float> dst(1 << 20);
    simd::impl::clamp_scalar(src.data(), dst.data(), 1 << 20, -50.0f, 50.0f);
    for (size_t i = 0; i < (1 << 20); i++) {
        EXPECT_GE(dst[i], -50.0f);
        EXPECT_LE(dst[i], 50.0f);
    }
}

#if defined(SIMD_AVX2_ENABLED)
#include <simd/ops/clamp/simd.hpp>

TEST(ClampTest, SIMD_KnownValues) {
    float src[] = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f, 15.0f, -1.0f, 3.0f};
    float scalar_dst[8], simd_dst[8];
    simd::impl::clamp_scalar(src, scalar_dst, 8, -5.0f, 5.0f);
    simd::impl::clamp_simd(src, simd_dst, 8, -5.0f, 5.0f);
    for (int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(scalar_dst[i], simd_dst[i]);
    }
}

TEST(ClampTest, SIMD_RandomData) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    std::vector<float> src(1024);
    for (auto& x : src) x = dist(rng);
    std::vector<float> scalar_dst(1024), simd_dst(1024);
    simd::impl::clamp_scalar(src.data(), scalar_dst.data(), 1024, -10.0f, 10.0f);
    simd::impl::clamp_simd(src.data(), simd_dst.data(), 1024, -10.0f, 10.0f);
    for (size_t i = 0; i < 1024; i++) {
        EXPECT_FLOAT_EQ(scalar_dst[i], simd_dst[i]);
    }
}

TEST(ClampTest, SIMD_TailElements) {
    float src[] = {-5.0f, 3.0f, 10.0f, -1.0f, 7.0f};
    float scalar_dst[5], simd_dst[5];
    simd::impl::clamp_scalar(src, scalar_dst, 5, 0.0f, 5.0f);
    simd::impl::clamp_simd(src, simd_dst, 5, 0.0f, 5.0f);
    for (int i = 0; i < 5; i++) {
        EXPECT_FLOAT_EQ(scalar_dst[i], simd_dst[i]);
    }
}

TEST(ClampTest, SIMD_LargeInput) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    std::vector<float> src(1 << 20);
    for (auto& x : src) x = dist(rng);
    std::vector<float> scalar_dst(1 << 20), simd_dst(1 << 20);
    simd::impl::clamp_scalar(src.data(), scalar_dst.data(), 1 << 20, -50.0f, 50.0f);
    simd::impl::clamp_simd(src.data(), simd_dst.data(), 1 << 20, -50.0f, 50.0f);
    for (size_t i = 0; i < (1 << 20); i++) {
        EXPECT_FLOAT_EQ(scalar_dst[i], simd_dst[i]);
    }
}
#endif

#include <simd/ops/clamp/clamp.hpp>

TEST(ClampTest, Dispatcher_KnownValues) {
    float src[] = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f, 15.0f, -1.0f, 3.0f};
    float expected[] = {-5.0f, -5.0f, 0.0f, 5.0f, 5.0f, 5.0f, -1.0f, 3.0f};
    float dst[8];
    simd::clamp(src, dst, 8, -5.0f, 5.0f);
    for (int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(ClampTest, Dispatcher_RandomData) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    std::vector<float> src(1024);
    for (auto& x : src) x = dist(rng);
    std::vector<float> dst(1024);
    simd::clamp(src.data(), dst.data(), 1024, -10.0f, 10.0f);
    for (size_t i = 0; i < 1024; i++) {
        EXPECT_GE(dst[i], -10.0f);
        EXPECT_LE(dst[i], 10.0f);
    }
}

TEST(ClampTest, Dispatcher_NT_KnownValues) {
    float src[] = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f, 15.0f, -1.0f, 3.0f};
    float expected[] = {-5.0f, -5.0f, 0.0f, 5.0f, 5.0f, 5.0f, -1.0f, 3.0f};
    float dst[8];
    simd::clamp_nt(src, dst, 8, -5.0f, 5.0f);
    for (int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}
