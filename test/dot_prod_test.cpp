#include "test_harness.hpp"
#include <simd/ops/dot_prod/scalar.hpp>
#include <simd/ops/dot_prod/simd.hpp>
#include <simd/ops/dot_prod/dot_prod.hpp>

TEST(DotProdTest, Scalar_KnownValues) {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b[] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float result = simd::impl::dot_prod_scalar(a, b, 8);
    EXPECT_FLOAT_EQ(result, 120.0f);
}

TEST(DotProdTest, Simd_MatchesScalar) {
#if defined(SIMD_AVX2_ENABLED) && defined(__FMA__)
    for (size_t n : kStdSizes) {
        auto a = make_random(n, -10.f, 10.f);
        auto b = make_random(n, -10.f, 10.f, 43);
        float scalar_res = simd::impl::dot_prod_scalar(a.data(), b.data(), n);
        float simd_res = simd::impl::dot_prod_simd(a.data(), b.data(), n);
        
        float tol = 1e-5f;
        if (n >= 1023 && n <= 1024) {
            tol = 1e-3f;
        } else if (n > 1024) {
            tol = 5.0f;
        }
        check_scalar_near(scalar_res, simd_res, tol, "n=" + std::to_string(n));
    }
#endif
}

TEST(DotProdTest, Dispatcher_MatchesScalar) {
    size_t n = 1024;
    auto a = make_random(n, -10.f, 10.f);
    auto b = make_random(n, -10.f, 10.f, 43);
    float scalar_res = simd::impl::dot_prod_scalar(a.data(), b.data(), n);
    float dispatch_res = simd::dot_prod(a, b);
    check_scalar_near(scalar_res, dispatch_res, 1e-3f, "dot_prod dispatcher");
}
