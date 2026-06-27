#include "test_harness.hpp"
#include <simd/ops/online_softmax/scalar.hpp>
#include <simd/ops/online_softmax/simd.hpp>
#include <simd/ops/online_softmax/online_softmax.hpp>
#include <simd/ops/softmax/scalar.hpp>

TEST(OnlineSoftmaxTest, Scalar_KnownValues) {
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float dst[8];
    simd::impl::online_softmax_scalar(src, dst, 8);
    float sum = 0.0f;
    for (int i = 0; i < 8; i++) sum += dst[i];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(OnlineSoftmaxTest, Scalar_UniformInput) {
    float src[] = {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f};
    float dst[8];
    simd::impl::online_softmax_scalar(src, dst, 8);
    for (int i = 0; i < 8; i++) {
        EXPECT_NEAR(dst[i], 0.125f, 1e-5f);
    }
}

TEST(OnlineSoftmaxTest, Scalar_SumsToOne) {
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float dst[8];
    simd::impl::online_softmax_scalar(src, dst, 8);
    float sum = 0.0f;
    for (int i = 0; i < 8; i++) sum += dst[i];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(OnlineSoftmaxTest, MatchesPlainSoftmax) {
    for (size_t n : kStdSizes) {
        auto src = make_random(n, -5.f, 5.f);
        std::vector<float> plain_dst(n), online_dst(n);
        simd::impl::softmax_scalar(src.data(), plain_dst.data(), n);
        simd::impl::online_softmax_scalar(src.data(), online_dst.data(), n);
        check_near(plain_dst.data(), online_dst.data(), n, 1e-5f, "n=" + std::to_string(n));
    }
}

TEST(OnlineSoftmaxTest, Simd_MatchesScalar) {
#if defined(SIMD_AVX2_ENABLED)
    for (size_t n : kStdSizes) {
        auto src = make_random(n, -5.f, 5.f);
        std::vector<float> scalar_dst(n), simd_dst(n);
        simd::impl::online_softmax_scalar(src.data(), scalar_dst.data(), n);
        simd::impl::online_softmax_simd(src.data(), simd_dst.data(), n);
        check_near(scalar_dst.data(), simd_dst.data(), n, 1e-5f, "n=" + std::to_string(n));
    }
#endif
}

TEST(OnlineSoftmaxTest, Dispatcher_MatchesScalar) {
    size_t n = 1024;
    auto src = make_random(n, -5.f, 5.f);
    std::vector<float> scalar_dst(n), dispatch_dst(n);
    simd::impl::online_softmax_scalar(src.data(), scalar_dst.data(), n);
    
    simd::online_softmax(src, dispatch_dst);
    check_near(scalar_dst.data(), dispatch_dst.data(), n, 1e-5f, "online_softmax dispatcher");
}
