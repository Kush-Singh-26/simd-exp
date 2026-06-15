#include <gtest/gtest.h>
#include <simd/common.hpp>
#include <simd/ops/softmax/scalar.hpp>
#include <vector>
#include <random>
#include <cmath>

static void verify_scalar_softmax(const float* src, const float* dst, size_t n, float tol) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        EXPECT_GE(dst[i], 0.0f) << "Negative output at index " << i;
        sum += dst[i];
    }
    EXPECT_NEAR(sum, 1.0f, tol) << "Output does not sum to 1.0";
}

TEST(SoftmaxTest, KnownValues) {
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float dst[8];
    simd::impl::softmax_scalar(src, dst, 8);
    verify_scalar_softmax(src, dst, 8, 1e-5f);
    for (int i = 0; i < 7; i++) {
        EXPECT_LE(dst[i], dst[i + 1]) << "Output not monotonically increasing";
    }
}

TEST(SoftmaxTest, OutputSumsToOne) {
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float dst[8];
    simd::impl::softmax_scalar(src, dst, 8);
    float sum = 0.0f;
    for (int i = 0; i < 8; i++) sum += dst[i];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(SoftmaxTest, AllSameValues) {
    float src[] = {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f};
    float dst[8];
    simd::impl::softmax_scalar(src, dst, 8);
    for (int i = 0; i < 8; i++) {
        EXPECT_NEAR(dst[i], 0.125f, 1e-5f);
    }
}

TEST(SoftmaxTest, LargeValues) {
    float src[] = {100.0f, 200.0f, 300.0f, 400.0f, 100.0f, 200.0f, 300.0f, 400.0f};
    float dst[8];
    simd::impl::softmax_scalar(src, dst, 8);
    verify_scalar_softmax(src, dst, 8, 1e-5f);
}

TEST(SoftmaxTest, NegativeValues) {
    float src[] = {-10.0f, -5.0f, -3.0f, -1.0f, -8.0f, -6.0f, -2.0f, -4.0f};
    float dst[8];
    simd::impl::softmax_scalar(src, dst, 8);
    verify_scalar_softmax(src, dst, 8, 1e-5f);
}

TEST(SoftmaxTest, TailElements) {
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float dst[5];
    simd::impl::softmax_scalar(src, dst, 5);
    verify_scalar_softmax(src, dst, 5, 1e-5f);
}

TEST(SoftmaxTest, SingleElement) {
    float src[] = {42.0f};
    float dst[1];
    simd::impl::softmax_scalar(src, dst, 1);
    EXPECT_NEAR(dst[0], 1.0f, 1e-5f);
}

TEST(SoftmaxTest, RandomData) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::vector<float> src(1024);
    for (auto& x : src) x = dist(rng);
    std::vector<float> dst(1024);
    simd::impl::softmax_scalar(src.data(), dst.data(), 1024);
    verify_scalar_softmax(src.data(), dst.data(), 1024, 1e-5f);
}

TEST(SoftmaxTest, LargeInput) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> src(1 << 20);
    for (auto& x : src) x = dist(rng);
    std::vector<float> dst(1 << 20);
    simd::impl::softmax_scalar(src.data(), dst.data(), 1 << 20);
    verify_scalar_softmax(src.data(), dst.data(), 1 << 20, 1e-3f);
}

#if defined(SIMD_AVX2_ENABLED)
#include <simd/ops/softmax/simd.hpp>

TEST(SoftmaxTest, SIMD_KnownValues) {
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float scalar_dst[8], simd_dst[8];
    simd::impl::softmax_scalar(src, scalar_dst, 8);
    simd::impl::softmax_simd(src, simd_dst, 8);
    for (int i = 0; i < 8; i++) {
        EXPECT_NEAR(scalar_dst[i], simd_dst[i], 1e-5f)
            << "Mismatch at index " << i;
    }
}

TEST(SoftmaxTest, SIMD_OutputSumsToOne) {
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float scalar_dst[8], simd_dst[8];
    simd::impl::softmax_scalar(src, scalar_dst, 8);
    simd::impl::softmax_simd(src, simd_dst, 8);
    float scalar_sum = 0.0f, simd_sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        scalar_sum += scalar_dst[i];
        simd_sum += simd_dst[i];
    }
    EXPECT_NEAR(scalar_sum, 1.0f, 1e-5f);
    EXPECT_NEAR(simd_sum, 1.0f, 1e-5f);
}

TEST(SoftmaxTest, SIMD_RandomData) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::vector<float> src(1024);
    for (auto& x : src) x = dist(rng);
    std::vector<float> scalar_dst(1024), simd_dst(1024);
    simd::impl::softmax_scalar(src.data(), scalar_dst.data(), 1024);
    simd::impl::softmax_simd(src.data(), simd_dst.data(), 1024);
    for (size_t i = 0; i < 1024; i++) {
        EXPECT_NEAR(scalar_dst[i], simd_dst[i], 1e-5f)
            << "Mismatch at index " << i;
    }
}

TEST(SoftmaxTest, SIMD_LargeInput) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> src(1 << 20);
    for (auto& x : src) x = dist(rng);
    std::vector<float> scalar_dst(1 << 20), simd_dst(1 << 20);
    simd::impl::softmax_scalar(src.data(), scalar_dst.data(), 1 << 20);
    simd::impl::softmax_simd(src.data(), simd_dst.data(), 1 << 20);
    for (size_t i = 0; i < (1 << 20); i++) {
        EXPECT_NEAR(scalar_dst[i], simd_dst[i], 1e-5f)
            << "Mismatch at index " << i;
    }
}
#endif

#include <simd/ops/softmax/softmax.hpp>

TEST(SoftmaxTest, Dispatcher_KnownValues) {
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float dst[8];
    simd::softmax(src, dst, 8);
    float sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        EXPECT_GE(dst[i], 0.0f);
        sum += dst[i];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(SoftmaxTest, Dispatcher_RandomData) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::vector<float> src(1024);
    for (auto& x : src) x = dist(rng);
    std::vector<float> dst(1024);
    simd::softmax(src.data(), dst.data(), 1024);
    float sum = 0.0f;
    for (size_t i = 0; i < 1024; i++) {
        EXPECT_GE(dst[i], 0.0f);
        sum += dst[i];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(SoftmaxTest, Dispatcher_NT_KnownValues) {
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float dst[8];
    simd::softmax_nt(src, dst, 8);
    float sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        EXPECT_GE(dst[i], 0.0f);
        sum += dst[i];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}
