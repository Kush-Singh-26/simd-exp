#include "bench_harness.hpp"
#include <simd/ops/online_softmax/scalar.hpp>
#include <simd/ops/online_softmax/simd.hpp>

SIMD_BENCH_UNARY_SCALAR_PARALLEL(OnlineSoftmax, simd::impl::online_softmax_scalar_parallel, gen_data_random)
SIMD_BENCH_UNARY_NO_NT(OnlineSoftmax,
    simd::impl::online_softmax_scalar,
    simd::impl::online_softmax_simd,
    gen_data_random)

BENCHMARK_MAIN();
