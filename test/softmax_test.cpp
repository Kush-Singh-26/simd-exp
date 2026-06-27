#include "test_harness.hpp"
#include <simd/ops/softmax/scalar.hpp>
#include <simd/ops/softmax/simd.hpp>
#include <simd/ops/softmax/softmax.hpp>

TEST(SoftmaxTest, Scalar_KnownValues) {
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float dst[8];
    simd::impl::softmax_scalar(src, dst, 8);
    for (int i = 0; i < 7; i++) {
        EXPECT_LE(dst[i], dst[i + 1]) << "Output not monotonically increasing";
    }
}

TEST(SoftmaxTest, Scalar_UniformInput) {
    float src[] = {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f};
    float dst[8];
    simd::impl::softmax_scalar(src, dst, 8);
    for (int i = 0; i < 8; i++) {
        EXPECT_NEAR(dst[i], 0.125f, 1e-5f);
    }
}

TEST(SoftmaxTest, Scalar_SumsToOne) {
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float dst[8];
    simd::impl::softmax_scalar(src, dst, 8);
    float sum = 0.0f;
    for (int i = 0; i < 8; i++) sum += dst[i];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(SoftmaxTest, Simd_MatchesScalar) {
#if defined(SIMD_AVX2_ENABLED)
    for (size_t n : kStdSizes) {
        auto src = make_random(n, -5.f, 5.f);
        std::vector<float> scalar_dst(n), simd_dst(n);
        simd::impl::softmax_scalar(src.data(), scalar_dst.data(), n);
        simd::impl::softmax_simd(src.data(), simd_dst.data(), n);
        check_near(scalar_dst.data(), simd_dst.data(), n, 1e-5f, "n=" + std::to_string(n));
    }
#endif
}

TEST(SoftmaxTest, SimdNt_MatchesScalar) {
#if defined(SIMD_AVX2_ENABLED)
    for (size_t n : {static_cast<size_t>(8), static_cast<size_t>(1024)}) {
        auto src = make_random(n, -5.f, 5.f);
        std::vector<float> scalar_dst(n);
        simd::impl::softmax_scalar(src.data(), scalar_dst.data(), n);
        
        auto alloc_result = simd::aligned_alloc(32, n * sizeof(float));
        ASSERT_TRUE(alloc_result.has_value());
        float* dst = static_cast<float*>(alloc_result.value());
        simd::impl::softmax_simd_nt(src.data(), dst, n);
        check_near(scalar_dst.data(), dst, n, 1e-5f, "n=" + std::to_string(n));
        simd::aligned_free(dst);
    }
#endif
}

TEST(SoftmaxTest, Dispatcher_MatchesScalar) {
    size_t n = 1024;
    auto src = make_random(n, -5.f, 5.f);
    std::vector<float> scalar_dst(n), dispatch_dst(n), dispatch_nt_dst(n);
    simd::impl::softmax_scalar(src.data(), scalar_dst.data(), n);
    
    simd::softmax(src, dispatch_dst);
    check_near(scalar_dst.data(), dispatch_dst.data(), n, 1e-5f, "softmax dispatcher");

    simd::softmax_nt(src, dispatch_nt_dst);
    check_near(scalar_dst.data(), dispatch_nt_dst.data(), n, 1e-5f, "softmax_nt dispatcher");
}
