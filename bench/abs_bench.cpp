#include "bench_harness.hpp"
#include <simd/ops/abs/scalar.hpp>
#include <simd/ops/abs/simd.hpp>

SIMD_BENCH_UNARY_SCALAR_PARALLEL(Abs, simd::impl::abs_scalar_parallel, gen_data_const)
SIMD_BENCH_UNARY(Abs,
    simd::impl::abs_scalar,
    simd::impl::abs_simd,
    simd::impl::abs_simd_nt,
    gen_data_const)

BENCHMARK_MAIN();
