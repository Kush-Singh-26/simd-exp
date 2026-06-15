#include <gtest/gtest.h>
#include <simd/common.hpp>
#include <simd/ops/online_softmax/scalar.hpp>
#include <simd/ops/softmax/scalar.hpp>
#include <vector>
#include <random>
#include <cmath>

static void verify_online_softmax_scalar(const float* src, const float* dst, size_t n, float tol) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        EXPECT_GE(dst[i], 0.0f) << "Negative output at index " << i;
        sum += dst[i];
    }
    EXPECT_NEAR(sum, 1.0f, tol) << "Output does not sum to 1.0";
}

TEST(OnlineSoftmaxTest, KnownValues) {
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float dst[8];
    simd::impl::online_softmax_scalar(src, dst, 8);
    verify_online_softmax_scalar(src, dst, 8, 1e-5f);
}

TEST(OnlineSoftmaxTest, OutputSumsToOne) {
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float dst[8];
    simd::impl::online_softmax_scalar(src, dst, 8);
    float sum = 0.0f;
    for (int i = 0; i < 8; i++) sum += dst[i];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(OnlineSoftmaxTest, AllSameValues) {
    float src[] = {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f};
    float dst[8];
    simd::impl::online_softmax_scalar(src, dst, 8);
    for (int i = 0; i < 8; i++) {
        EXPECT_NEAR(dst[i], 0.125f, 1e-5f);
    }
}

TEST(OnlineSoftmaxTest, NegativeValues) {
    float src[] = {-10.0f, -5.0f, -3.0f, -1.0f, -8.0f, -6.0f, -2.0f, -4.0f};
    float dst[8];
    simd::impl::online_softmax_scalar(src, dst, 8);
    verify_online_softmax_scalar(src, dst, 8, 1e-5f);
}

TEST(OnlineSoftmaxTest, LargeValues) {
    float src[] = {100.0f, 200.0f, 300.0f, 400.0f, 100.0f, 200.0f, 300.0f, 400.0f};
    float dst[8];
    simd::impl::online_softmax_scalar(src, dst, 8);
    verify_online_softmax_scalar(src, dst, 8, 1e-5f);
}

TEST(OnlineSoftmaxTest, TailElements) {
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float dst[5];
    simd::impl::online_softmax_scalar(src, dst, 5);
    verify_online_softmax_scalar(src, dst, 5, 1e-5f);
}

TEST(OnlineSoftmaxTest, SingleElement) {
    float src[] = {42.0f};
    float dst[1];
    simd::impl::online_softmax_scalar(src, dst, 1);
    EXPECT_NEAR(dst[0], 1.0f, 1e-5f);
}

TEST(OnlineSoftmaxTest, RandomData) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::vector<float> src(1024);
    for (auto& x : src) x = dist(rng);
    std::vector<float> dst(1024);
    simd::impl::online_softmax_scalar(src.data(), dst.data(), 1024);
    verify_online_softmax_scalar(src.data(), dst.data(), 1024, 1e-5f);
}

TEST(OnlineSoftmaxTest, LargeInput) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> src(1 << 20);
    for (auto& x : src) x = dist(rng);
    std::vector<float> dst(1 << 20);
    simd::impl::online_softmax_scalar(src.data(), dst.data(), 1 << 20);
    verify_online_softmax_scalar(src.data(), dst.data(), 1 << 20, 1e-3f);
}

TEST(OnlineSoftmaxTest, MatchesPlainSoftmax) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::vector<float> src(1024);
    for (auto& x : src) x = dist(rng);
    std::vector<float> plain_dst(1024), online_dst(1024);
    simd::impl::softmax_scalar(src.data(), plain_dst.data(), 1024);
    simd::impl::online_softmax_scalar(src.data(), online_dst.data(), 1024);
    for (size_t i = 0; i < 1024; i++) {
        EXPECT_NEAR(plain_dst[i], online_dst[i], 1e-5f)
            << "Plain vs online mismatch at index " << i;
    }
}

#if defined(SIMD_AVX2_ENABLED)
#include <simd/ops/online_softmax/simd.hpp>

TEST(OnlineSoftmaxTest, SIMD_KnownValues) {
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float scalar_dst[8], simd_dst[8];
    simd::impl::online_softmax_scalar(src, scalar_dst, 8);
    simd::impl::online_softmax_simd(src, simd_dst, 8);
    for (int i = 0; i < 8; i++) {
        EXPECT_NEAR(scalar_dst[i], simd_dst[i], 1e-5f)
            << "Mismatch at index " << i;
    }
}

TEST(OnlineSoftmaxTest, SIMD_OutputSumsToOne) {
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float scalar_dst[8], simd_dst[8];
    simd::impl::online_softmax_scalar(src, scalar_dst, 8);
    simd::impl::online_softmax_simd(src, simd_dst, 8);
    float scalar_sum = 0.0f, simd_sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        scalar_sum += scalar_dst[i];
        simd_sum += simd_dst[i];
    }
    EXPECT_NEAR(scalar_sum, 1.0f, 1e-5f);
    EXPECT_NEAR(simd_sum, 1.0f, 1e-5f);
}

TEST(OnlineSoftmaxTest, SIMD_RandomData) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::vector<float> src(1024);
    for (auto& x : src) x = dist(rng);
    std::vector<float> scalar_dst(1024), simd_dst(1024);
    simd::impl::online_softmax_scalar(src.data(), scalar_dst.data(), 1024);
    simd::impl::online_softmax_simd(src.data(), simd_dst.data(), 1024);
    for (size_t i = 0; i < 1024; i++) {
        EXPECT_NEAR(scalar_dst[i], simd_dst[i], 1e-5f)
            << "Mismatch at index " << i;
    }
}

TEST(OnlineSoftmaxTest, SIMD_LargeInput) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> src(1 << 20);
    for (auto& x : src) x = dist(rng);
    std::vector<float> scalar_dst(1 << 20), simd_dst(1 << 20);
    simd::impl::online_softmax_scalar(src.data(), scalar_dst.data(), 1 << 20);
    simd::impl::online_softmax_simd(src.data(), simd_dst.data(), 1 << 20);
    for (size_t i = 0; i < (1 << 20); i++) {
        EXPECT_NEAR(scalar_dst[i], simd_dst[i], 1e-5f)
            << "Mismatch at index " << i;
    }
}
#endif

#include <simd/ops/online_softmax/online_softmax.hpp>

TEST(OnlineSoftmaxTest, Dispatcher_KnownValues) {
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float dst[8];
    simd::online_softmax(src, dst, 8);
    float sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        EXPECT_GE(dst[i], 0.0f);
        sum += dst[i];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(OnlineSoftmaxTest, Dispatcher_RandomData) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::vector<float> src(1024);
    for (auto& x : src) x = dist(rng);
    std::vector<float> dst(1024);
    simd::online_softmax(src.data(), dst.data(), 1024);
    float sum = 0.0f;
    for (size_t i = 0; i < 1024; i++) {
        EXPECT_GE(dst[i], 0.0f);
        sum += dst[i];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}
