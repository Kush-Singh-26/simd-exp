#include "bench_harness.hpp"
#include <simd/simd.hpp>
#include <simd/ops/mat_transpose/mat_transpose.hpp>
#include <cstddef>
#include <vector>

// 4x4 matrix transpose benchmarks (Scalar, SIMD, SIMD NT)
SIMD_BENCH_FIXED(Transpose,
                 simd::impl::transpose_scalar,
                 simd::impl::transpose_simd,
                 simd::impl::transpose_simd_nt,
                 4)

// ── 8x8 AVX2 transpose benchmarks ────────────────────────────────────────────
// These are in addition to the 4x4 fixed-size macro benchmarks above.

#if defined(SIMD_AVX2_ENABLED)
#define SIMD_BENCH_FIXED_8x8_SIMD(Name, SimdFn) \
static void BM_##Name##_Simd(benchmark::State& state) { \
  size_t n = state.range(0); \
  size_t size = 64; \
  std::vector<float> src(n * size); \
  std::vector<float> dst(n * size); \
  for (size_t i = 0; i < n * size; i++) src[i] = static_cast<float>(i); \
  for (auto _ : state) { \
    benchmark::DoNotOptimize(src.data()); \
    for (size_t i = 0; i < n; i++) \
      SimdFn(src.data() + i * size, dst.data() + i * size); \
    benchmark::DoNotOptimize(dst.data()); \
  } \
} \
BENCHMARK(BM_##Name##_Simd)->Arg(1<<14)->Arg(1<<10)->Arg(1<<6);

SIMD_BENCH_FIXED_8x8_SIMD(Transpose_8x8, simd::impl::transpose8x8_simd)
SIMD_BENCH_FIXED_8x8_SIMD(Transpose_8x8_Dispatch, simd::impl::transpose8x8_simd)

// Strided variant: transpose 8x8 blocks within a larger 256x256 matrix
static void BM_Transpose_8x8_Strided(benchmark::State& state) {
  size_t matrix_dim = 256;
  size_t blocks = (matrix_dim / 8) * (matrix_dim / 8);
  std::vector<float> src(matrix_dim * matrix_dim);
  std::vector<float> dst(matrix_dim * matrix_dim);
  for (size_t i = 0; i < src.size(); i++) src[i] = static_cast<float>(i);
  for (auto _ : state) {
    benchmark::DoNotOptimize(src.data());
    for (size_t bi = 0; bi < matrix_dim; bi += 8) {
      for (size_t bj = 0; bj < matrix_dim; bj += 8) {
        simd::impl::transpose8x8_strided_simd(
            src.data() + bi * matrix_dim + bj,
            dst.data() + bj * matrix_dim + bi,
            matrix_dim, matrix_dim);
      }
    }
    benchmark::DoNotOptimize(dst.data());
  }
}
BENCHMARK(BM_Transpose_8x8_Strided);
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
