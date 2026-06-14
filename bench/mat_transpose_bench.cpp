#include <benchmark/benchmark.h>
#include <simd/simd.hpp>
#include <cstddef>
#include <vector>

static void BM_Transpose_Scalar(benchmark::State& state) {
  size_t n = state.range(0);
  std::vector<float> src(n * 16);
  std::vector<float> dst(n * 16);
  for (size_t i = 0; i < n * 16; i++) src[i] = static_cast<float>(i);
  for (auto _ : state) {
    for (size_t i = 0; i < n; i++)
      simd::impl::transpose_scalar(src.data() + i * 16, dst.data() + i * 16);
    benchmark::DoNotOptimize(dst.data());
  }
}
BENCHMARK(BM_Transpose_Scalar)->Arg(1<<14)->Arg(1<<10)->Arg(1<<6);

#if defined(SIMD_AVX2_ENABLED)
static void BM_Transpose_Simd(benchmark::State& state) {
  size_t n = state.range(0);
  std::vector<float> src(n * 16);
  std::vector<float> dst(n * 16);
  for (size_t i = 0; i < n * 16; i++) src[i] = static_cast<float>(i);
  for (auto _ : state) {
    for (size_t i = 0; i < n; i++)
      simd::impl::transpose_simd(src.data() + i * 16, dst.data() + i * 16);
    benchmark::DoNotOptimize(dst.data());
  }
}
BENCHMARK(BM_Transpose_Simd)->Arg(1<<14)->Arg(1<<10)->Arg(1<<6);

static void BM_Transpose_Simd_NT(benchmark::State& state) {
  size_t n = state.range(0);
  std::vector<float> src(n * 16);
  float* dst = static_cast<float*>(simd::aligned_alloc(16, n * 16 * sizeof(float)));
  for (size_t i = 0; i < n * 16; i++) src[i] = static_cast<float>(i);
  for (auto _ : state) {
    for (size_t i = 0; i < n; i++)
      simd::impl::transpose_simd_nt(src.data() + i * 16, dst + i * 16);
    benchmark::DoNotOptimize(dst);
  }
  simd::aligned_free(dst);
}
BENCHMARK(BM_Transpose_Simd_NT)->Arg(1<<14)->Arg(1<<10)->Arg(1<<6);
#endif

BENCHMARK_MAIN();