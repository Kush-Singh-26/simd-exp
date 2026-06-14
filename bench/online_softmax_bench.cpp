#include <benchmark/benchmark.h>
#include <simd/ops/online-softmax/scalar.hpp>
#include <simd/ops/online-softmax/simd.hpp>
#include <vector>
#include <cstddef>
#include <random>

enum DataType { POS, NEG, RAND };

static void gen_data(std::vector<float>& src, DataType dtype) {
    if (dtype == POS) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 10.0f);
        for (auto& x : src) x = dist(rng);
    } else if (dtype == NEG) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-10.0f, 0.0f);
        for (auto& x : src) x = dist(rng);
    } else {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
        for (auto& x : src) x = dist(rng);
    }
}

static void BM_Online_Softmax_Scalar(benchmark::State& state, DataType dtype) {
  size_t n = state.range(0);
  std::vector<float> src(n);
  gen_data(src, dtype);
  std::vector<float> dst(n);
  for (auto _ : state) {
    simd::impl::online_softmax_scalar(src.data(), dst.data(), n);
    benchmark::DoNotOptimize(dst.data());
  }
}
BENCHMARK_CAPTURE(BM_Online_Softmax_Scalar, pos, POS)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
BENCHMARK_CAPTURE(BM_Online_Softmax_Scalar, neg, NEG)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
BENCHMARK_CAPTURE(BM_Online_Softmax_Scalar, rand, RAND)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);

#if defined(SIMD_AVX2_ENABLED)
static void BM_Online_Softmax_Simd(benchmark::State& state, DataType dtype) {
  size_t n = state.range(0);
  std::vector<float> src(n);
  gen_data(src, dtype);
  std::vector<float> dst(n);
  for (auto _ : state) {
    simd::impl::online_softmax_simd(src.data(), dst.data(), n);
    benchmark::DoNotOptimize(dst.data());
  }
}
BENCHMARK_CAPTURE(BM_Online_Softmax_Simd, pos, POS)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
BENCHMARK_CAPTURE(BM_Online_Softmax_Simd, neg, NEG)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
BENCHMARK_CAPTURE(BM_Online_Softmax_Simd, rand, RAND)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
#endif

BENCHMARK_MAIN();
