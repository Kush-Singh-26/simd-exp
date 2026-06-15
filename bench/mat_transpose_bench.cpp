#include <benchmark/benchmark.h>
#include <simd/simd.hpp>
#include <simd/ops/mat_transpose/mat_transpose.hpp>
#include <cstddef>
#include <vector>

static void BM_Transpose_Scalar(benchmark::State& state) {
  size_t n = state.range(0);
  std::vector<float> src(n * 16);
  std::vector<float> dst(n * 16);
  for (size_t i = 0; i < n * 16; i++) src[i] = static_cast<float>(i);
  for (auto _ : state) {
    benchmark::DoNotOptimize(src.data());
    for (size_t i = 0; i < n; i++)
      simd::impl::transpose_scalar(src.data() + i * 16, dst.data() + i * 16, 4, 4);
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
    benchmark::DoNotOptimize(src.data());
    for (size_t i = 0; i < n; i++)
      simd::impl::transpose_simd(src.data() + i * 16, dst.data() + i * 16);
    benchmark::DoNotOptimize(dst.data());
  }
}
BENCHMARK(BM_Transpose_Simd)->Arg(1<<14)->Arg(1<<10)->Arg(1<<6);

static void BM_Transpose_Simd_NT(benchmark::State& state) {
  size_t n = state.range(0);
  std::vector<float> src(n * 16);
  auto alloc_result = simd::aligned_alloc(16, n * 16 * sizeof(float));
  if (!alloc_result.has_value()) return;
  float* dst = static_cast<float*>(alloc_result.value());
  for (size_t i = 0; i < n * 16; i++) src[i] = static_cast<float>(i);
  for (auto _ : state) {
    benchmark::DoNotOptimize(src.data());
    for (size_t i = 0; i < n; i++)
      simd::impl::transpose_simd_nt(src.data() + i * 16, dst + i * 16);
    benchmark::DoNotOptimize(dst);
  }
  simd::aligned_free(dst);
}
BENCHMARK(BM_Transpose_Simd_NT)->Arg(1<<14)->Arg(1<<10)->Arg(1<<6);
#endif

// ── Non-4x4 benchmarks (C++23 mdspan path) ───────────────────────────────────

static void BM_Transpose_Scalar_8x8(benchmark::State& state) {
  size_t n = state.range(0);
  std::vector<float> src(n * 64);
  std::vector<float> dst(n * 64);
  for (size_t i = 0; i < n * 64; i++) src[i] = static_cast<float>(i);
  for (auto _ : state) {
    benchmark::DoNotOptimize(src.data());
    for (size_t i = 0; i < n; i++)
      simd::impl::transpose_scalar(src.data() + i * 64, dst.data() + i * 64, 8, 8);
    benchmark::DoNotOptimize(dst.data());
  }
}
BENCHMARK(BM_Transpose_Scalar_8x8)->Arg(1<<14)->Arg(1<<10)->Arg(1<<6);

static void BM_Transpose_Scalar_16x16(benchmark::State& state) {
  size_t n = state.range(0);
  std::vector<float> src(n * 256);
  std::vector<float> dst(n * 256);
  for (size_t i = 0; i < n * 256; i++) src[i] = static_cast<float>(i);
  for (auto _ : state) {
    benchmark::DoNotOptimize(src.data());
    for (size_t i = 0; i < n; i++)
      simd::impl::transpose_scalar(src.data() + i * 256, dst.data() + i * 256, 16, 16);
    benchmark::DoNotOptimize(dst.data());
  }
}
BENCHMARK(BM_Transpose_Scalar_16x16)->Arg(1<<12)->Arg(1<<8)->Arg(1<<4);

static void BM_Transpose_Scalar_3x5(benchmark::State& state) {
  size_t n = state.range(0);
  std::vector<float> src(n * 15);
  std::vector<float> dst(n * 15);
  for (size_t i = 0; i < n * 15; i++) src[i] = static_cast<float>(i);
  for (auto _ : state) {
    benchmark::DoNotOptimize(src.data());
    for (size_t i = 0; i < n; i++)
      simd::impl::transpose_scalar(src.data() + i * 15, dst.data() + i * 15, 3, 5);
    benchmark::DoNotOptimize(dst.data());
  }
}
BENCHMARK(BM_Transpose_Scalar_3x5)->Arg(1<<14)->Arg(1<<10)->Arg(1<<6);

BENCHMARK_MAIN();
