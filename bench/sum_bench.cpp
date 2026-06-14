#include <benchmark/benchmark.h>
#include <simd/simd.hpp>
#include <cstddef>
#include <vector>

static void BM_Sum_Scalar(benchmark::State& state) {
  size_t n = state.range(0);
  std::vector<float> data(n, 1.0f);
  for (auto _ : state) {
    float r = simd::impl::sum_scalar(data.data(), n);
    benchmark::DoNotOptimize(r);
  }
}
BENCHMARK(BM_Sum_Scalar)->Arg(1<<20);

#if defined(SIMD_AVX2_ENABLED)
static void BM_Sum_Simd(benchmark::State& state) {
  size_t n = state.range(0);
  std::vector<float> data(n, 1.0f);
  for (auto _ : state) {
    float r = simd::impl::sum_simd(data.data(), n);
    benchmark::DoNotOptimize(r);
  }
}
BENCHMARK(BM_Sum_Simd)->Arg(1<<20);
#endif

BENCHMARK_MAIN();