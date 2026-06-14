#include <benchmark/benchmark.h>
#include <simd/simd.hpp>
#include <cstddef>
#include <vector>

static void BM_Dot_Prod_Scalar(benchmark::State& state) {
  size_t n = state.range(0);
  std::vector<float> data1(n, 1.0f);
  std::vector<float> data2(n, 1.0f);
  for (auto _ : state) {
    float r = simd::impl::dot_prod_scalar(data1.data(), data2.data(), n);
    benchmark::DoNotOptimize(r);
  }
}
BENCHMARK(BM_Dot_Prod_Scalar)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);

#if defined(SIMD_AVX2_ENABLED) && defined(__FMA__)
static void BM_Dot_Prod_Simd(benchmark::State& state) {
  size_t n = state.range(0);
  std::vector<float> data1(n, 1.0f);
  std::vector<float> data2(n, 1.0f);
  for (auto _ : state) {
    float r = simd::impl::dot_prod_simd(data1.data(), data2.data(), n);
    benchmark::DoNotOptimize(r);
  }
}
BENCHMARK(BM_Dot_Prod_Simd)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
#endif

BENCHMARK_MAIN();