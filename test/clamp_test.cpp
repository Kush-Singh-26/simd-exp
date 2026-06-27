#include "test_harness.hpp"
#include <simd/ops/clamp/scalar.hpp>
#include <simd/ops/clamp/simd.hpp>
#include <simd/ops/clamp/clamp.hpp>

TEST(ClampTest, Scalar_KnownValues) {
    float src[] = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f, 15.0f, -1.0f, 3.0f};
    float dst[8];
    simd::impl::clamp_scalar(src, dst, 8, -5.0f, 5.0f);
    float expected[] = {-5.0f, -5.0f, 0.0f, 5.0f, 5.0f, 5.0f, -1.0f, 3.0f};
    for (int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(ClampTest, Simd_MatchesScalar) {
#if defined(SIMD_AVX2_ENABLED)
    for (size_t n : kStdSizes) {
        auto src = make_boundary_stress(n, -50.f, 50.f);
        std::vector<float> scalar_dst(n), simd_dst(n);
        simd::impl::clamp_scalar(src.data(), scalar_dst.data(), n, -10.f, 10.f);
        simd::impl::clamp_simd(src.data(), simd_dst.data(), n, -10.f, 10.f);
        check_exact(scalar_dst.data(), simd_dst.data(), n, "n=" + std::to_string(n));
    }
#endif
}

TEST(ClampTest, SimdNt_MatchesScalar) {
#if defined(SIMD_AVX2_ENABLED)
    for (size_t n : {static_cast<size_t>(8), static_cast<size_t>(1024)}) {
        auto src = make_boundary_stress(n, -50.f, 50.f);
        std::vector<float> scalar_dst(n);
        simd::impl::clamp_scalar(src.data(), scalar_dst.data(), n, -10.f, 10.f);
        
        auto alloc_result = simd::aligned_alloc(32, n * sizeof(float));
        ASSERT_TRUE(alloc_result.has_value());
        float* dst = static_cast<float*>(alloc_result.value());
        simd::impl::clamp_simd_nt(src.data(), dst, n, -10.f, 10.f);
        check_exact(scalar_dst.data(), dst, n, "n=" + std::to_string(n));
        simd::aligned_free(dst);
    }
#endif
}

TEST(ClampTest, Dispatcher_MatchesScalar) {
    size_t n = 1024;
    auto src = make_boundary_stress(n, -50.f, 50.f);
    std::vector<float> scalar_dst(n), dispatch_dst(n), dispatch_nt_dst(n);
    simd::impl::clamp_scalar(src.data(), scalar_dst.data(), n, -10.f, 10.f);
    
    simd::clamp(src, dispatch_dst, -10.f, 10.f);
    check_exact(scalar_dst.data(), dispatch_dst.data(), n, "clamp dispatcher");

    simd::clamp_nt(src, dispatch_nt_dst, -10.f, 10.f);
    check_exact(scalar_dst.data(), dispatch_nt_dst.data(), n, "clamp_nt dispatcher");
}
