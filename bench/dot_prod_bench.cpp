#include <benchmark/benchmark.h>
#include <simd/simd.hpp>
#include <cstddef>
#include <vector>
#include "bench_utils.hpp"

static void BM_Dot_Prod_Scalar(benchmark::State& state, DataType dtype) {
  size_t n = state.range(0);
  std::vector<float> data1(n), data2(n);
  gen_data_const(data1, dtype);
  gen_data_const(data2, dtype);
  for (auto _ : state) {
    float r = simd::impl::dot_prod_scalar(data1.data(), data2.data(), n);
    benchmark::DoNotOptimize(r);
  }
}
BENCHMARK_CAPTURE(BM_Dot_Prod_Scalar, pos, DataType::POS)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
BENCHMARK_CAPTURE(BM_Dot_Prod_Scalar, neg, DataType::NEG)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
BENCHMARK_CAPTURE(BM_Dot_Prod_Scalar, rand, DataType::RAND)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);

#if defined(SIMD_AVX2_ENABLED) && defined(__FMA__)
static void BM_Dot_Prod_Simd(benchmark::State& state, DataType dtype) {
  size_t n = state.range(0);
  std::vector<float> data1(n), data2(n);
  gen_data_const(data1, dtype);
  gen_data_const(data2, dtype);
  for (auto _ : state) {
    float r = simd::impl::dot_prod_simd(data1.data(), data2.data(), n);
    benchmark::DoNotOptimize(r);
  }
}
BENCHMARK_CAPTURE(BM_Dot_Prod_Simd, pos, DataType::POS)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
BENCHMARK_CAPTURE(BM_Dot_Prod_Simd, neg, DataType::NEG)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
BENCHMARK_CAPTURE(BM_Dot_Prod_Simd, rand, DataType::RAND)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
#endif

BENCHMARK_MAIN();
