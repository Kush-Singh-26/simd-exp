#include "test_harness.hpp"
#include <simd/ops/abs/scalar.hpp>
#include <simd/ops/abs/simd.hpp>
#include <simd/ops/abs/abs.hpp>

TEST(AbsTest, Scalar_KnownValues) {
    float src[] = {-3.0f, -1.5f, 0.0f, 1.5f, 3.0f, -100.0f, 42.0f, -0.001f};
    float dst[8];
    simd::impl::abs_scalar(src, dst, 8);
    EXPECT_FLOAT_EQ(dst[0], 3.0f);
    EXPECT_FLOAT_EQ(dst[1], 1.5f);
    EXPECT_FLOAT_EQ(dst[2], 0.0f);
    EXPECT_FLOAT_EQ(dst[3], 1.5f);
    EXPECT_FLOAT_EQ(dst[4], 3.0f);
    EXPECT_FLOAT_EQ(dst[5], 100.0f);
    EXPECT_FLOAT_EQ(dst[6], 42.0f);
    EXPECT_FLOAT_EQ(dst[7], 0.001f);
}

TEST(AbsTest, Simd_MatchesScalar) {
#if defined(SIMD_AVX2_ENABLED)
    for (size_t n : kStdSizes) {
        auto src = make_random(n, -50.f, 50.f);
        std::vector<float> scalar_dst(n), simd_dst(n);
        simd::impl::abs_scalar(src.data(), scalar_dst.data(), n);
        simd::impl::abs_simd(src.data(), simd_dst.data(), n);
        check_exact(scalar_dst.data(), simd_dst.data(), n, "n=" + std::to_string(n));
    }
#endif
}

TEST(AbsTest, SimdNt_MatchesScalar) {
#if defined(SIMD_AVX2_ENABLED)
    for (size_t n : {static_cast<size_t>(8), static_cast<size_t>(1024)}) {
        auto src = make_random(n, -50.f, 50.f);
        std::vector<float> scalar_dst(n);
        simd::impl::abs_scalar(src.data(), scalar_dst.data(), n);
        
        auto alloc_result = simd::aligned_alloc(32, n * sizeof(float));
        ASSERT_TRUE(alloc_result.has_value());
        float* dst = static_cast<float*>(alloc_result.value());
        simd::impl::abs_simd_nt(src.data(), dst, n);
        check_exact(scalar_dst.data(), dst, n, "n=" + std::to_string(n));
        simd::aligned_free(dst);
    }
#endif
}

TEST(AbsTest, Dispatcher_MatchesScalar) {
    size_t n = 1024;
    auto src = make_random(n, -50.f, 50.f);
    std::vector<float> scalar_dst(n), dispatch_dst(n), dispatch_nt_dst(n);
    simd::impl::abs_scalar(src.data(), scalar_dst.data(), n);
    
    simd::abs(src, dispatch_dst);
    check_exact(scalar_dst.data(), dispatch_dst.data(), n, "abs dispatcher");

    simd::abs_nt(src, dispatch_nt_dst);
    check_exact(scalar_dst.data(), dispatch_nt_dst.data(), n, "abs_nt dispatcher");
}
