#include <benchmark/benchmark.h>
#include <simd/simd.hpp>
#include <vector>
#include <cstddef>
#include "bench_utils.hpp"

static void BM_Online_Softmax_Scalar(benchmark::State& state, DataType dtype) {
  size_t n = state.range(0);
  std::vector<float> src(n);
  gen_data_random(src, dtype);
  std::vector<float> dst(n);
  for (auto _ : state) {
    benchmark::DoNotOptimize(src.data());
    simd::impl::online_softmax_scalar(src.data(), dst.data(), n);
    benchmark::DoNotOptimize(dst.data());
  }
}
BENCHMARK_CAPTURE(BM_Online_Softmax_Scalar, pos, DataType::POS)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
BENCHMARK_CAPTURE(BM_Online_Softmax_Scalar, neg, DataType::NEG)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
BENCHMARK_CAPTURE(BM_Online_Softmax_Scalar, rand, DataType::RAND)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);

#if defined(SIMD_AVX2_ENABLED)
static void BM_Online_Softmax_Simd(benchmark::State& state, DataType dtype) {
  size_t n = state.range(0);
  std::vector<float> src(n);
  gen_data_random(src, dtype);
  std::vector<float> dst(n);
  for (auto _ : state) {
    benchmark::DoNotOptimize(src.data());
    simd::impl::online_softmax_simd(src.data(), dst.data(), n);
    benchmark::DoNotOptimize(dst.data());
  }
}
BENCHMARK_CAPTURE(BM_Online_Softmax_Simd, pos, DataType::POS)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
BENCHMARK_CAPTURE(BM_Online_Softmax_Simd, neg, DataType::NEG)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
BENCHMARK_CAPTURE(BM_Online_Softmax_Simd, rand, DataType::RAND)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
#endif

BENCHMARK_MAIN();
