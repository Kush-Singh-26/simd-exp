#include <benchmark/benchmark.h>
#include <simd/simd.hpp>
#include <vector>
#include <cstddef>
#include "bench_utils.hpp"

static void BM_ReLU_Scalar(benchmark::State& state, DataType dtype) {
  size_t n = state.range(0);
  std::vector<float> src(n);
  gen_data_const(src, dtype);
  std::vector<float> dst(n);
  for (auto _ : state) {
    simd::impl::relu_scalar(src.data(), dst.data(), n);
    benchmark::DoNotOptimize(dst.data());
  }
}
BENCHMARK_CAPTURE(BM_ReLU_Scalar, pos, DataType::POS)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
BENCHMARK_CAPTURE(BM_ReLU_Scalar, neg, DataType::NEG)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
BENCHMARK_CAPTURE(BM_ReLU_Scalar, rand, DataType::RAND)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);

#if defined(SIMD_AVX2_ENABLED)
static void BM_ReLU_Simd(benchmark::State& state, DataType dtype) {
  size_t n = state.range(0);
  std::vector<float> src(n);
  gen_data_const(src, dtype);
  std::vector<float> dst(n);
  for (auto _ : state) {
    simd::impl::relu_simd(src.data(), dst.data(), n);
    benchmark::DoNotOptimize(dst.data());
  }
}
BENCHMARK_CAPTURE(BM_ReLU_Simd, pos, DataType::POS)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
BENCHMARK_CAPTURE(BM_ReLU_Simd, neg, DataType::NEG)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
BENCHMARK_CAPTURE(BM_ReLU_Simd, rand, DataType::RAND)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);

static void BM_ReLU_Simd_NT(benchmark::State& state, DataType dtype) {
  size_t n = state.range(0);
  std::vector<float> src(n);
  gen_data_const(src, dtype);
  auto alloc_result = simd::aligned_alloc(32, n * sizeof(float));
  if (!alloc_result.has_value()) return;
  float* dst = static_cast<float*>(alloc_result.value());
  for (auto _ : state) {
    simd::impl::relu_simd_nt(src.data(), dst, n);
    benchmark::DoNotOptimize(dst);
  }
  simd::aligned_free(dst);
}
BENCHMARK_CAPTURE(BM_ReLU_Simd_NT, pos, DataType::POS)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
BENCHMARK_CAPTURE(BM_ReLU_Simd_NT, neg, DataType::NEG)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
BENCHMARK_CAPTURE(BM_ReLU_Simd_NT, rand, DataType::RAND)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
#endif

BENCHMARK_MAIN();
