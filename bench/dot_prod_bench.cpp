#include "bench_harness.hpp"
#include <simd/ops/dot_prod/scalar.hpp>
#include <simd/ops/dot_prod/simd.hpp>

SIMD_BENCH_REDUCTION_SCALAR_PARALLEL_2(DotProd, simd::impl::dot_prod_scalar_parallel, gen_data_random)
SIMD_BENCH_REDUCTION(DotProd,
    simd::impl::dot_prod_scalar,
    simd::impl::dot_prod_simd,
    gen_data_random, 2)

BENCHMARK_MAIN();
