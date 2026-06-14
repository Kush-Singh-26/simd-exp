#include <gtest/gtest.h>
#include <simd/math_utils.hpp>
#include <cmath>
#include <vector>
#include <random>

#if defined(SIMD_AVX2_ENABLED)

static float ulp_error(float computed, float reference) {
    if (computed == reference) return 0.0f;
    if (std::isinf(computed) || std::isinf(reference)) return 1e30f;
    if (std::isnan(computed) || std::isnan(reference)) return 1e30f;
    float diff = std::fabs(computed - reference);
    float ref_abs = std::fabs(reference);
    if (ref_abs < 1e-30f) return diff * 1e30f;
    return diff / ref_abs * (1 << 23);
}

TEST(MathUtilsTest, KnownValues) {
    float inputs[] = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f, std::log(2.0f)};
    constexpr size_t N = 8;

    float lane[8] = {};
    for (size_t i = 0; i < N; ++i) {
        lane[i] = inputs[i];
    }
    __m256 x = _mm256_loadu_ps(lane);
    __m256 result = simd::impl::avx2_exp_ps(x);
    float out[8];
    _mm256_storeu_ps(out, result);

    for (size_t i = 0; i < N; ++i) {
        float expected = std::exp(inputs[i]);
        EXPECT_LE(ulp_error(out[i], expected), 3.0f)
            << "exp(" << inputs[i] << ") = " << out[i]
            << ", expected " << expected;
    }
}

TEST(MathUtilsTest, PositiveRange) {
    float lane[8], out[8];
    for (int i = 0; i < 8; ++i) lane[i] = 0.0f;

    for (float x = 0.0f; x <= 87.0f; x += 0.7f) {
        for (int i = 0; i < 8; ++i) lane[i] = x + i * 0.01f;
        __m256 v = _mm256_loadu_ps(lane);
        __m256 result = simd::impl::avx2_exp_ps(v);
        _mm256_storeu_ps(out, result);
        for (int i = 0; i < 8; ++i) {
            float expected = std::exp(lane[i]);
            EXPECT_LE(ulp_error(out[i], expected), 3.0f)
                << "exp(" << lane[i] << ") = " << out[i]
                << ", expected " << expected;
        }
    }
}

TEST(MathUtilsTest, NegativeRange) {
    float lane[8], out[8];
    for (int i = 0; i < 8; ++i) lane[i] = 0.0f;

    for (float x = 0.0f; x >= -87.0f; x -= 0.7f) {
        for (int i = 0; i < 8; ++i) lane[i] = x - i * 0.01f;
        __m256 v = _mm256_loadu_ps(lane);
        __m256 result = simd::impl::avx2_exp_ps(v);
        _mm256_storeu_ps(out, result);
        for (int i = 0; i < 8; ++i) {
            float expected = std::exp(lane[i]);
            EXPECT_LE(ulp_error(out[i], expected), 3.0f)
                << "exp(" << lane[i] << ") = " << out[i]
                << ", expected " << expected;
        }
    }
}

TEST(MathUtilsTest, LargeValues) {
    float inputs[] = {88.0f, -88.0f, 88.3f, -88.3f};
    constexpr size_t N = sizeof(inputs) / sizeof(inputs[0]);
    float lane[8] = {}, out[8];
    for (size_t i = 0; i < N; ++i) lane[i] = inputs[i];

    __m256 v = _mm256_loadu_ps(lane);
    __m256 result = simd::impl::avx2_exp_ps(v);
    _mm256_storeu_ps(out, result);

    for (size_t i = 0; i < N; ++i) {
        float expected = std::exp(inputs[i]);
        if (std::isinf(expected)) {
            EXPECT_TRUE(std::isinf(out[i]) || out[i] > 1e30f)
                << "exp(" << inputs[i] << ") should be huge, got " << out[i];
        } else {
            EXPECT_LE(ulp_error(out[i], expected), 3.0f)
                << "exp(" << inputs[i] << ") = " << out[i]
                << ", expected " << expected;
        }
    }
}

TEST(MathUtilsTest, SubnormalFlushToZero) {
    float lane[8] = {-100.0f, -95.0f, -90.0f, -89.0f,
                     -100.0f, -95.0f, -90.0f, -89.0f};
    float out[8];
    __m256 v = _mm256_loadu_ps(lane);
    __m256 result = simd::impl::avx2_exp_ps(v);
    _mm256_storeu_ps(out, result);

    for (int i = 0; i < 8; ++i) {
        float expected = std::exp(lane[i]);
        if (expected < 1e-38f) {
            EXPECT_EQ(out[i], 0.0f)
                << "exp(" << lane[i] << ") should flush to 0, got " << out[i];
        }
    }
}

TEST(MathUtilsTest, RandomData) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    float max_err = 0.0f;
    float lane[8], out[8];
    for (int batch = 0; batch < 128; ++batch) {
        for (int i = 0; i < 8; ++i) lane[i] = dist(rng);
        __m256 v = _mm256_loadu_ps(lane);
        __m256 result = simd::impl::avx2_exp_ps(v);
        _mm256_storeu_ps(out, result);
        for (int i = 0; i < 8; ++i) {
            float expected = std::exp(lane[i]);
            float err = ulp_error(out[i], expected);
            if (err > max_err) max_err = err;
            EXPECT_LE(err, 3.0f)
                << "exp(" << lane[i] << ") = " << out[i]
                << ", expected " << expected;
        }
    }
    EXPECT_LE(max_err, 3.0f) << "Max ULP error across 1024 random values";
}

TEST(MathUtilsTest, BatchComparison) {
    float lane[8], out[8], ref[8];
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    for (int i = 0; i < 8; ++i) {
        lane[i] = dist(rng);
        ref[i] = std::exp(lane[i]);
    }

    __m256 v = _mm256_loadu_ps(lane);
    __m256 result = simd::impl::avx2_exp_ps(v);
    _mm256_storeu_ps(out, result);

    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(out[i], ref[i], std::fabs(ref[i]) * 1e-6f)
            << "Lane " << i << ": exp(" << lane[i] << ") = " << out[i]
            << ", expected " << ref[i];
    }
}

#endif
