#pragma once
#include <benchmark/benchmark.h>
#include <simd/simd.hpp>
#include <vector>
#include <cstddef>
#include "bench_utils.hpp"

// ── Unary helper macros ──────────────────────────────────────────────────────

#define SIMD_BENCH_UNARY_SCALAR_PARALLEL(Name, ScalarFn, GenFn, ...) \
static void BM_##Name##_Scalar_Parallel(benchmark::State& state, DataType dtype) { \
  size_t n = state.range(0); \
  std::vector<float> src(n); \
  GenFn(src, dtype); \
  std::vector<float> dst(n); \
  for (auto _ : state) { \
    ScalarFn(src.data(), dst.data(), n, ##__VA_ARGS__); \
    benchmark::DoNotOptimize(dst.data()); \
  } \
} \
BENCHMARK_CAPTURE(BM_##Name##_Scalar_Parallel, pos, DataType::POS)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10); \
BENCHMARK_CAPTURE(BM_##Name##_Scalar_Parallel, neg, DataType::NEG)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10); \
BENCHMARK_CAPTURE(BM_##Name##_Scalar_Parallel, rand, DataType::RAND)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);

#define SIMD_BENCH_UNARY_SCALAR(Name, ScalarFn, GenFn, ...) \
static void BM_##Name##_Scalar(benchmark::State& state, DataType dtype) { \
  size_t n = state.range(0); \
  std::vector<float> src(n); \
  GenFn(src, dtype); \
  std::vector<float> dst(n); \
  for (auto _ : state) { \
    ScalarFn(src.data(), dst.data(), n, ##__VA_ARGS__); \
    benchmark::DoNotOptimize(dst.data()); \
  } \
} \
BENCHMARK_CAPTURE(BM_##Name##_Scalar, pos, DataType::POS)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10); \
BENCHMARK_CAPTURE(BM_##Name##_Scalar, neg, DataType::NEG)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10); \
BENCHMARK_CAPTURE(BM_##Name##_Scalar, rand, DataType::RAND)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);

#if defined(SIMD_AVX2_ENABLED)
#define SIMD_BENCH_UNARY_SIMD(Name, SimdFn, GenFn, ...) \
static void BM_##Name##_Simd(benchmark::State& state, DataType dtype) { \
  size_t n = state.range(0); \
  std::vector<float> src(n); \
  GenFn(src, dtype); \
  std::vector<float> dst(n); \
  for (auto _ : state) { \
    SimdFn(src.data(), dst.data(), n, ##__VA_ARGS__); \
    benchmark::DoNotOptimize(dst.data()); \
  } \
} \
BENCHMARK_CAPTURE(BM_##Name##_Simd, pos, DataType::POS)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10); \
BENCHMARK_CAPTURE(BM_##Name##_Simd, neg, DataType::NEG)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10); \
BENCHMARK_CAPTURE(BM_##Name##_Simd, rand, DataType::RAND)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);

#define SIMD_BENCH_UNARY_SIMD_NT(Name, SimdNtFn, GenFn, ...) \
static void BM_##Name##_Simd_NT(benchmark::State& state, DataType dtype) { \
  size_t n = state.range(0); \
  std::vector<float> src(n); \
  GenFn(src, dtype); \
  auto alloc_result = simd::aligned_alloc(32, n * sizeof(float)); \
  if (!alloc_result.has_value()) return; \
  float* dst = static_cast<float*>(alloc_result.value()); \
  for (auto _ : state) { \
    SimdNtFn(src.data(), dst, n, ##__VA_ARGS__); \
    benchmark::DoNotOptimize(dst); \
  } \
  simd::aligned_free(dst); \
} \
BENCHMARK_CAPTURE(BM_##Name##_Simd_NT, pos, DataType::POS)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10); \
BENCHMARK_CAPTURE(BM_##Name##_Simd_NT, neg, DataType::NEG)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10); \
BENCHMARK_CAPTURE(BM_##Name##_Simd_NT, rand, DataType::RAND)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
#else
#define SIMD_BENCH_UNARY_SIMD(Name, SimdFn, GenFn, ...)
#define SIMD_BENCH_UNARY_SIMD_NT(Name, SimdNtFn, GenFn, ...)
#endif

#define SIMD_BENCH_UNARY(Name, ScalarFn, SimdFn, SimdNtFn, GenFn, ...) \
  SIMD_BENCH_UNARY_SCALAR(Name, ScalarFn, GenFn, ##__VA_ARGS__) \
  SIMD_BENCH_UNARY_SIMD(Name, SimdFn, GenFn, ##__VA_ARGS__) \
  SIMD_BENCH_UNARY_SIMD_NT(Name, SimdNtFn, GenFn, ##__VA_ARGS__)

#define SIMD_BENCH_UNARY_NO_NT(Name, ScalarFn, SimdFn, GenFn, ...) \
  SIMD_BENCH_UNARY_SCALAR(Name, ScalarFn, GenFn, ##__VA_ARGS__) \
  SIMD_BENCH_UNARY_SIMD(Name, SimdFn, GenFn, ##__VA_ARGS__)


// ── Reduction helper macros ──────────────────────────────────────────────────

#define SIMD_BENCH_REDUCTION_SCALAR_PARALLEL_1(Name, ScalarFn, GenFn) \
static void BM_##Name##_Scalar_Parallel(benchmark::State& state, DataType dtype) { \
  size_t n = state.range(0); \
  std::vector<float> data(n); \
  GenFn(data, dtype); \
  for (auto _ : state) { \
    benchmark::DoNotOptimize(data.data()); \
    float r = ScalarFn(data.data(), n); \
    benchmark::DoNotOptimize(r); \
  } \
} \
BENCHMARK_CAPTURE(BM_##Name##_Scalar_Parallel, pos, DataType::POS)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10); \
BENCHMARK_CAPTURE(BM_##Name##_Scalar_Parallel, neg, DataType::NEG)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10); \
BENCHMARK_CAPTURE(BM_##Name##_Scalar_Parallel, rand, DataType::RAND)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);

#define SIMD_BENCH_REDUCTION_SCALAR_1(Name, ScalarFn, GenFn) \
static void BM_##Name##_Scalar(benchmark::State& state, DataType dtype) { \
  size_t n = state.range(0); \
  std::vector<float> data(n); \
  GenFn(data, dtype); \
  for (auto _ : state) { \
    benchmark::DoNotOptimize(data.data()); \
    float r = ScalarFn(data.data(), n); \
    benchmark::DoNotOptimize(r); \
  } \
} \
BENCHMARK_CAPTURE(BM_##Name##_Scalar, pos, DataType::POS)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10); \
BENCHMARK_CAPTURE(BM_##Name##_Scalar, neg, DataType::NEG)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10); \
BENCHMARK_CAPTURE(BM_##Name##_Scalar, rand, DataType::RAND)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);

#define SIMD_BENCH_REDUCTION_SCALAR_PARALLEL_2(Name, ScalarFn, GenFn) \
static void BM_##Name##_Scalar_Parallel(benchmark::State& state, DataType dtype) { \
  size_t n = state.range(0); \
  std::vector<float> data1(n), data2(n); \
  GenFn(data1, dtype); \
  GenFn(data2, dtype); \
  for (auto _ : state) { \
    benchmark::DoNotOptimize(data1.data()); \
    benchmark::DoNotOptimize(data2.data()); \
    float r = ScalarFn(data1.data(), data2.data(), n); \
    benchmark::DoNotOptimize(r); \
  } \
} \
BENCHMARK_CAPTURE(BM_##Name##_Scalar_Parallel, pos, DataType::POS)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10); \
BENCHMARK_CAPTURE(BM_##Name##_Scalar_Parallel, neg, DataType::NEG)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10); \
BENCHMARK_CAPTURE(BM_##Name##_Scalar_Parallel, rand, DataType::RAND)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);

#define SIMD_BENCH_REDUCTION_SCALAR_2(Name, ScalarFn, GenFn) \
static void BM_##Name##_Scalar(benchmark::State& state, DataType dtype) { \
  size_t n = state.range(0); \
  std::vector<float> data1(n), data2(n); \
  GenFn(data1, dtype); \
  GenFn(data2, dtype); \
  for (auto _ : state) { \
    benchmark::DoNotOptimize(data1.data()); \
    benchmark::DoNotOptimize(data2.data()); \
    float r = ScalarFn(data1.data(), data2.data(), n); \
    benchmark::DoNotOptimize(r); \
  } \
} \
BENCHMARK_CAPTURE(BM_##Name##_Scalar, pos, DataType::POS)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10); \
BENCHMARK_CAPTURE(BM_##Name##_Scalar, neg, DataType::NEG)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10); \
BENCHMARK_CAPTURE(BM_##Name##_Scalar, rand, DataType::RAND)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);

#if defined(SIMD_AVX2_ENABLED)
#define SIMD_BENCH_REDUCTION_SUM_SIMD(Name, SimdFn, GenFn) \
static void BM_##Name##_Simd(benchmark::State& state, DataType dtype) { \
  size_t n = state.range(0); \
  std::vector<float> data(n); \
  GenFn(data, dtype); \
  for (auto _ : state) { \
    benchmark::DoNotOptimize(data.data()); \
    float r = SimdFn(data.data(), n); \
    benchmark::DoNotOptimize(r); \
  } \
} \
BENCHMARK_CAPTURE(BM_##Name##_Simd, pos, DataType::POS)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10); \
BENCHMARK_CAPTURE(BM_##Name##_Simd, neg, DataType::NEG)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10); \
BENCHMARK_CAPTURE(BM_##Name##_Simd, rand, DataType::RAND)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
#else
#define SIMD_BENCH_REDUCTION_SUM_SIMD(Name, SimdFn, GenFn)
#endif

#if defined(SIMD_AVX2_ENABLED) && defined(__FMA__)
#define SIMD_BENCH_REDUCTION_DOT_SIMD(Name, SimdFn, GenFn) \
static void BM_##Name##_Simd(benchmark::State& state, DataType dtype) { \
  size_t n = state.range(0); \
  std::vector<float> data1(n), data2(n); \
  GenFn(data1, dtype); \
  GenFn(data2, dtype); \
  for (auto _ : state) { \
    benchmark::DoNotOptimize(data1.data()); \
    benchmark::DoNotOptimize(data2.data()); \
    float r = SimdFn(data1.data(), data2.data(), n); \
    benchmark::DoNotOptimize(r); \
  } \
} \
BENCHMARK_CAPTURE(BM_##Name##_Simd, pos, DataType::POS)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10); \
BENCHMARK_CAPTURE(BM_##Name##_Simd, neg, DataType::NEG)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10); \
BENCHMARK_CAPTURE(BM_##Name##_Simd, rand, DataType::RAND)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
#else
#define SIMD_BENCH_REDUCTION_DOT_SIMD(Name, SimdFn, GenFn)
#endif

#define SIMD_BENCH_REDUCTION_SIMD_1(Name, SimdFn, GenFn) \
  SIMD_BENCH_REDUCTION_SUM_SIMD(Name, SimdFn, GenFn)

#define SIMD_BENCH_REDUCTION_SIMD_2(Name, SimdFn, GenFn) \
  SIMD_BENCH_REDUCTION_DOT_SIMD(Name, SimdFn, GenFn)

#define SIMD_BENCH_REDUCTION(Name, ScalarFn, SimdFn, GenFn, NInputs) \
  SIMD_BENCH_REDUCTION_SCALAR_##NInputs(Name, ScalarFn, GenFn) \
  SIMD_BENCH_REDUCTION_SIMD_##NInputs(Name, SimdFn, GenFn)


// ── Fixed-size helper macros ─────────────────────────────────────────────────

#define SIMD_BENCH_FIXED_SCALAR(Name, ScalarFn, TileSize) \
static void BM_##Name##_Scalar(benchmark::State& state) { \
  size_t n = state.range(0); \
  size_t size = TileSize * TileSize; \
  std::vector<float> src(n * size); \
  std::vector<float> dst(n * size); \
  for (size_t i = 0; i < n * size; i++) src[i] = static_cast<float>(i); \
  for (auto _ : state) { \
    benchmark::DoNotOptimize(src.data()); \
    for (size_t i = 0; i < n; i++) \
      ScalarFn(src.data() + i * size, dst.data() + i * size, TileSize, TileSize); \
    benchmark::DoNotOptimize(dst.data()); \
  } \
} \
BENCHMARK(BM_##Name##_Scalar)->Arg(1<<14)->Arg(1<<10)->Arg(1<<6);

#if defined(SIMD_AVX2_ENABLED)
#define SIMD_BENCH_FIXED_SIMD(Name, SimdFn, TileSize) \
static void BM_##Name##_Simd(benchmark::State& state) { \
  size_t n = state.range(0); \
  size_t size = TileSize * TileSize; \
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

#define SIMD_BENCH_FIXED_SIMD_NT(Name, SimdNtFn, TileSize) \
static void BM_##Name##_Simd_NT(benchmark::State& state) { \
  size_t n = state.range(0); \
  size_t size = TileSize * TileSize; \
  std::vector<float> src(n * size); \
  auto alloc_result = simd::aligned_alloc(16, n * size * sizeof(float)); \
  if (!alloc_result.has_value()) return; \
  float* dst = static_cast<float*>(alloc_result.value()); \
  for (size_t i = 0; i < n * size; i++) src[i] = static_cast<float>(i); \
  for (auto _ : state) { \
    benchmark::DoNotOptimize(src.data()); \
    for (size_t i = 0; i < n; i++) \
      SimdNtFn(src.data() + i * size, dst + i * size); \
    benchmark::DoNotOptimize(dst); \
  } \
  simd::aligned_free(dst); \
} \
BENCHMARK(BM_##Name##_Simd_NT)->Arg(1<<14)->Arg(1<<10)->Arg(1<<6);
#else
#define SIMD_BENCH_FIXED_SIMD(Name, SimdFn, TileSize)
#define SIMD_BENCH_FIXED_SIMD_NT(Name, SimdNtFn, TileSize)
#endif

#define SIMD_BENCH_FIXED(Name, ScalarFn, SimdFn, SimdNtFn, TileSize) \
  SIMD_BENCH_FIXED_SCALAR(Name, ScalarFn, TileSize) \
  SIMD_BENCH_FIXED_SIMD(Name, SimdFn, TileSize) \
  SIMD_BENCH_FIXED_SIMD_NT(Name, SimdNtFn, TileSize)
