#include "bench_harness.hpp"
#include <simd/ops/softmax/scalar.hpp>
#include <simd/ops/softmax/simd.hpp>

SIMD_BENCH_UNARY_SCALAR_PARALLEL(Softmax, simd::impl::softmax_scalar_parallel, gen_data_random)
SIMD_BENCH_UNARY(Softmax,
    simd::impl::softmax_scalar,
    simd::impl::softmax_simd,
    simd::impl::softmax_simd_nt,
    gen_data_random)

BENCHMARK_MAIN();
