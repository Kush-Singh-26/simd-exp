#include "test_harness.hpp"
#include <simd/ops/relu/scalar.hpp>
#include <simd/ops/relu/simd.hpp>
#include <simd/ops/relu/relu.hpp>

TEST(ReluTest, Scalar_KnownValues) {
    float src[] = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f, -0.5f, 0.5f, 100.0f};
    float dst[8];
    simd::impl::relu_scalar(src, dst, 8);
    float expected[] = {0.0f, 0.0f, 0.0f, 1.0f, 3.0f, 0.0f, 0.5f, 100.0f};
    for (int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(ReluTest, Simd_MatchesScalar) {
#if defined(SIMD_AVX2_ENABLED)
    for (size_t n : kStdSizes) {
        auto src = make_random(n, -50.f, 50.f);
        std::vector<float> scalar_dst(n), simd_dst(n);
        simd::impl::relu_scalar(src.data(), scalar_dst.data(), n);
        simd::impl::relu_simd(src.data(), simd_dst.data(), n);
        check_exact(scalar_dst.data(), simd_dst.data(), n, "n=" + std::to_string(n));
    }
#endif
}

TEST(ReluTest, SimdNt_MatchesScalar) {
#if defined(SIMD_AVX2_ENABLED)
    for (size_t n : {static_cast<size_t>(8), static_cast<size_t>(1024)}) {
        auto src = make_random(n, -50.f, 50.f);
        std::vector<float> scalar_dst(n);
        simd::impl::relu_scalar(src.data(), scalar_dst.data(), n);
        
        auto alloc_result = simd::aligned_alloc(32, n * sizeof(float));
        ASSERT_TRUE(alloc_result.has_value());
        float* dst = static_cast<float*>(alloc_result.value());
        simd::impl::relu_simd_nt(src.data(), dst, n);
        check_exact(scalar_dst.data(), dst, n, "n=" + std::to_string(n));
        simd::aligned_free(dst);
    }
#endif
}

TEST(ReluTest, Dispatcher_MatchesScalar) {
    size_t n = 1024;
    auto src = make_random(n, -50.f, 50.f);
    std::vector<float> scalar_dst(n), dispatch_dst(n), dispatch_nt_dst(n);
    simd::impl::relu_scalar(src.data(), scalar_dst.data(), n);
    
    simd::relu(src, dispatch_dst);
    check_exact(scalar_dst.data(), dispatch_dst.data(), n, "relu dispatcher");

    simd::relu_nt(src, dispatch_nt_dst);
    check_exact(scalar_dst.data(), dispatch_nt_dst.data(), n, "relu_nt dispatcher");
}
