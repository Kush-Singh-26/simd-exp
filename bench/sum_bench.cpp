#include "bench_harness.hpp"
#include <simd/ops/sum/scalar.hpp>
#include <simd/ops/sum/simd.hpp>

SIMD_BENCH_REDUCTION_SCALAR_PARALLEL_1(Sum, simd::impl::sum_scalar_parallel, gen_data_const)
SIMD_BENCH_REDUCTION(Sum,
    simd::impl::sum_scalar,
    simd::impl::sum_simd,
    gen_data_const, 1)

BENCHMARK_MAIN();
