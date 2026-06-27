#include "bench_harness.hpp"
#include <simd/ops/relu/scalar.hpp>
#include <simd/ops/relu/simd.hpp>

SIMD_BENCH_UNARY_SCALAR_PARALLEL(ReLU, simd::impl::relu_scalar_parallel, gen_data_const)
SIMD_BENCH_UNARY(ReLU,
    simd::impl::relu_scalar,
    simd::impl::relu_simd,
    simd::impl::relu_simd_nt,
    gen_data_const)

BENCHMARK_MAIN();
