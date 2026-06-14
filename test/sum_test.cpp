#include <gtest/gtest.h>
#include <simd/ops/sum/scalar.hpp>
#include <vector>
#include <random>

TEST(SumTest, KnownValues) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float result = simd::impl::sum_scalar(data, 8);
    EXPECT_FLOAT_EQ(result, 36.0f);
}

TEST(SumTest, AllZeros) {
    float data[8] = {};
    float result = simd::impl::sum_scalar(data, 8);
    EXPECT_FLOAT_EQ(result, 0.0f);
}

TEST(SumTest, AllNegative) {
    float data[] = {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f};
    float result = simd::impl::sum_scalar(data, 8);
    EXPECT_FLOAT_EQ(result, -36.0f);
}

TEST(SumTest, MixedValues) {
    float data[] = {10.0f, -5.0f, 3.0f, -2.0f, 7.0f, -1.0f, 4.0f, -8.0f};
    float result = simd::impl::sum_scalar(data, 8);
    EXPECT_FLOAT_EQ(result, 8.0f);
}

TEST(SumTest, TailElements) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float result = simd::impl::sum_scalar(data, 5);
    EXPECT_FLOAT_EQ(result, 15.0f);
}

TEST(SumTest, SingleElement) {
    float data[] = {42.0f};
    float result = simd::impl::sum_scalar(data, 1);
    EXPECT_FLOAT_EQ(result, 42.0f);
}

TEST(SumTest, RandomData) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> data(1024);
    for (auto& x : data) x = dist(rng);
    float result = simd::impl::sum_scalar(data.data(), 1024);
    float expected = 0.0f;
    for (auto x : data) expected += x;
    EXPECT_NEAR(result, expected, 1e-2f);
}

TEST(SumTest, LargeInput) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(1 << 20);
    for (auto& x : data) x = dist(rng);
    float result = simd::impl::sum_scalar(data.data(), 1 << 20);
    float expected = 0.0f;
    for (auto x : data) expected += x;
    EXPECT_NEAR(result, expected, 1.0f);
}

#if defined(SIMD_AVX2_ENABLED)
#include <simd/ops/sum/simd.hpp>

TEST(SumTest, SIMD_KnownValues) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float scalar_result = simd::impl::sum_scalar(data, 8);
    float simd_result = simd::impl::sum_simd(data, 8);
    EXPECT_NEAR(scalar_result, simd_result, 1e-5f);
}

TEST(SumTest, SIMD_RandomData) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> data(1024);
    for (auto& x : data) x = dist(rng);
    float scalar_result = simd::impl::sum_scalar(data.data(), 1024);
    float simd_result = simd::impl::sum_simd(data.data(), 1024);
    EXPECT_NEAR(scalar_result, simd_result, 1e-2f);
}

TEST(SumTest, SIMD_LargeInput) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(1 << 20);
    for (auto& x : data) x = dist(rng);
    float scalar_result = simd::impl::sum_scalar(data.data(), 1 << 20);
    float simd_result = simd::impl::sum_simd(data.data(), 1 << 20);
    EXPECT_NEAR(scalar_result, simd_result, 1.0f);
}
#endif
