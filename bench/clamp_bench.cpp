#include <benchmark/benchmark.h>
#include <simd/simd.hpp>
#include <cstddef>
#include <vector>

static void BM_Clamp_Scalar(benchmark::State& state) {
  size_t n = state.range(0);
  std::vector<float> src(n, 1.0f);
  std::vector<float> dst(n);
  for (auto _ : state) {
    simd::impl::clamp_scalar(src.data(), dst.data(), n, 0.0f, 1.0f);
    benchmark::DoNotOptimize(dst.data());
  }
}
BENCHMARK(BM_Clamp_Scalar)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);

#if defined(SIMD_AVX2_ENABLED)
static void BM_Clamp_Simd(benchmark::State& state) {
  size_t n = state.range(0);
  std::vector<float> src(n, 1.0f);
  std::vector<float> dst(n);
  for (auto _ : state) {
    simd::impl::clamp_simd(src.data(), dst.data(), n, 0.0f, 1.0f);
    benchmark::DoNotOptimize(dst.data());
  }
}
BENCHMARK(BM_Clamp_Simd)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);

static void BM_Clamp_Simd_NT(benchmark::State& state) {
  size_t n = state.range(0);
  std::vector<float> src(n, 1.0f);
  float* dst = static_cast<float*>(simd::aligned_alloc(32, n * sizeof(float)));
  for (auto _ : state) {
    simd::impl::clamp_simd_nt(src.data(), dst, n, 0.0f, 1.0f);
    benchmark::DoNotOptimize(dst);
  }
  simd::aligned_free(dst);
}
BENCHMARK(BM_Clamp_Simd_NT)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
#endif

BENCHMARK_MAIN();