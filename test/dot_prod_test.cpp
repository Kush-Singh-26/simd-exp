#include <gtest/gtest.h>
#include <simd/common.hpp>
#include <simd/ops/dot_prod/scalar.hpp>
#include <vector>
#include <random>

TEST(DotProdTest, KnownValues) {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b[] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float result = simd::impl::dot_prod_scalar(a, b, 8);
    EXPECT_FLOAT_EQ(result, 120.0f);
}

TEST(DotProdTest, OrthogonalVectors) {
    float a[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float b[] = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float result = simd::impl::dot_prod_scalar(a, b, 8);
    EXPECT_FLOAT_EQ(result, 0.0f);
}

TEST(DotProdTest, SameVectors) {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float result = simd::impl::dot_prod_scalar(a, a, 8);
    EXPECT_FLOAT_EQ(result, 204.0f);
}

TEST(DotProdTest, TailElements) {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    float result = simd::impl::dot_prod_scalar(a, b, 3);
    EXPECT_FLOAT_EQ(result, 32.0f);
}

TEST(DotProdTest, SingleElement) {
    float a[] = {3.0f};
    float b[] = {7.0f};
    float result = simd::impl::dot_prod_scalar(a, b, 1);
    EXPECT_FLOAT_EQ(result, 21.0f);
}

TEST(DotProdTest, RandomData) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> a(1024), b(1024);
    for (size_t i = 0; i < 1024; i++) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }
    float result = simd::impl::dot_prod_scalar(a.data(), b.data(), 1024);
    float expected = 0.0f;
    for (size_t i = 0; i < 1024; i++) expected += a[i] * b[i];
    EXPECT_NEAR(result, expected, 1e-2f);
}

TEST(DotProdTest, LargeInput) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> a(1 << 20), b(1 << 20);
    for (size_t i = 0; i < (1 << 20); i++) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }
    float result = simd::impl::dot_prod_scalar(a.data(), b.data(), 1 << 20);
    float expected = 0.0f;
    for (size_t i = 0; i < (1 << 20); i++) expected += a[i] * b[i];
    EXPECT_NEAR(result, expected, 1.0f);
}

#if defined(SIMD_AVX2_ENABLED) && defined(__FMA__)
#include <simd/ops/dot_prod/simd.hpp>

TEST(DotProdTest, SIMD_KnownValues) {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b[] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float scalar_result = simd::impl::dot_prod_scalar(a, b, 8);
    float simd_result = simd::impl::dot_prod_simd(a, b, 8);
    EXPECT_NEAR(scalar_result, simd_result, 1e-5f);
}

TEST(DotProdTest, SIMD_RandomData) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> a(1024), b(1024);
    for (size_t i = 0; i < 1024; i++) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }
    float scalar_result = simd::impl::dot_prod_scalar(a.data(), b.data(), 1024);
    float simd_result = simd::impl::dot_prod_simd(a.data(), b.data(), 1024);
    EXPECT_NEAR(scalar_result, simd_result, 1e-2f);
}

TEST(DotProdTest, SIMD_LargeInput) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> a(1 << 20), b(1 << 20);
    for (size_t i = 0; i < (1 << 20); i++) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }
    float scalar_result = simd::impl::dot_prod_scalar(a.data(), b.data(), 1 << 20);
    float simd_result = simd::impl::dot_prod_simd(a.data(), b.data(), 1 << 20);
    EXPECT_NEAR(scalar_result, simd_result, 1.0f);
}
#endif

#include <simd/ops/dot_prod/dot_prod.hpp>

TEST(DotProdTest, Dispatcher_KnownValues) {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b[] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float result = simd::dot_prod(a, b, 8);
    EXPECT_FLOAT_EQ(result, 120.0f);
}

TEST(DotProdTest, Dispatcher_RandomData) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> a(1024), b(1024);
    for (size_t i = 0; i < 1024; i++) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }
    float result = simd::dot_prod(a.data(), b.data(), 1024);
    float expected = 0.0f;
    for (size_t i = 0; i < 1024; i++) expected += a[i] * b[i];
    EXPECT_NEAR(result, expected, 1e-2f);
}
