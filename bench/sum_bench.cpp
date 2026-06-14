#include <benchmark/benchmark.h>
#include <simd/simd.hpp>
#include <cstddef>
#include <vector>
#include "bench_utils.hpp"

static void BM_Sum_Scalar(benchmark::State& state, DataType dtype) {
  size_t n = state.range(0);
  std::vector<float> data(n);
  gen_data_const(data, dtype);
  for (auto _ : state) {
    float r = simd::impl::sum_scalar(data.data(), n);
    benchmark::DoNotOptimize(r);
  }
}
BENCHMARK_CAPTURE(BM_Sum_Scalar, pos, DataType::POS)->Arg(1<<20);
BENCHMARK_CAPTURE(BM_Sum_Scalar, neg, DataType::NEG)->Arg(1<<20);
BENCHMARK_CAPTURE(BM_Sum_Scalar, rand, DataType::RAND)->Arg(1<<20);

#if defined(SIMD_AVX2_ENABLED)
static void BM_Sum_Simd(benchmark::State& state, DataType dtype) {
  size_t n = state.range(0);
  std::vector<float> data(n);
  gen_data_const(data, dtype);
  for (auto _ : state) {
    float r = simd::impl::sum_simd(data.data(), n);
    benchmark::DoNotOptimize(r);
  }
}
BENCHMARK_CAPTURE(BM_Sum_Simd, pos, DataType::POS)->Arg(1<<20);
BENCHMARK_CAPTURE(BM_Sum_Simd, neg, DataType::NEG)->Arg(1<<20);
BENCHMARK_CAPTURE(BM_Sum_Simd, rand, DataType::RAND)->Arg(1<<20);
#endif

BENCHMARK_MAIN();
