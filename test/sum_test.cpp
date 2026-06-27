#include "test_harness.hpp"
#include <simd/ops/sum/scalar.hpp>
#include <simd/ops/sum/simd.hpp>
#include <simd/ops/sum/sum.hpp>

TEST(SumTest, Scalar_KnownValues) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float result = simd::impl::sum_scalar(data, 8);
    EXPECT_FLOAT_EQ(result, 36.0f);
}

TEST(SumTest, Simd_MatchesScalar) {
#if defined(SIMD_AVX2_ENABLED)
    for (size_t n : kStdSizes) {
        auto data = make_random(n, -10.f, 10.f);
        float scalar_res = simd::impl::sum_scalar(data.data(), n);
        float simd_res = simd::impl::sum_simd(data.data(), n);
        
        float tol = 1e-5f;
        if (n >= 1023 && n <= 1024) {
            tol = 1e-3f;
        } else if (n > 1024) {
            tol = 1e-1f;
        }
        check_scalar_near(scalar_res, simd_res, tol, "n=" + std::to_string(n));
    }
#endif
}

TEST(SumTest, Dispatcher_MatchesScalar) {
    size_t n = 1024;
    auto data = make_random(n, -10.f, 10.f);
    float scalar_res = simd::impl::sum_scalar(data.data(), n);
    float dispatch_res = simd::sum(data);
    check_scalar_near(scalar_res, dispatch_res, 1e-3f, "sum dispatcher");
}
