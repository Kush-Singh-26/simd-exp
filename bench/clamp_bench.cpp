#include "bench_harness.hpp"
#include <simd/ops/clamp/scalar.hpp>
#include <simd/ops/clamp/simd.hpp>

SIMD_BENCH_UNARY_SCALAR_PARALLEL(Clamp, simd::impl::clamp_scalar_parallel, gen_data_const, 0.0f, 1.0f)
SIMD_BENCH_UNARY(Clamp,
    simd::impl::clamp_scalar,
    simd::impl::clamp_simd,
    simd::impl::clamp_simd_nt,
    gen_data_const,
    0.0f, 1.0f)

BENCHMARK_MAIN();
