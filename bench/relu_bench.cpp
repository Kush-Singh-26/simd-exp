#include <benchmark/benchmark.h>
#include <simd/simd.hpp>
#include <vector>
#include <cstddef>
#include <random>
#include <algorithm>

enum DataType { POS, NEG, RAND };

static void gen_data(std::vector<float>& src, DataType dtype) {
    if (dtype == POS) {
        std::fill(src.begin(), src.end(), 1.0f);
    } else if (dtype == NEG) {
        std::fill(src.begin(), src.end(), -1.0f);
    } else {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& x : src) x = dist(rng);
    }
}

static void BM_ReLU_Scalar(benchmark::State& state, DataType dtype) {
  size_t n = state.range(0);
  std::vector<float> src(n);
  gen_data(src, dtype);
  std::vector<float> dst(n);
  for (auto _ : state) {
    simd::impl::relu_scalar(src.data(), dst.data(), n);
    benchmark::DoNotOptimize(dst.data());
  }
}
BENCHMARK_CAPTURE(BM_ReLU_Scalar, pos, POS)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
BENCHMARK_CAPTURE(BM_ReLU_Scalar, neg, NEG)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
BENCHMARK_CAPTURE(BM_ReLU_Scalar, rand, RAND)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);

#if defined(SIMD_AVX2_ENABLED)
static void BM_ReLU_Simd(benchmark::State& state, DataType dtype) {
  size_t n = state.range(0);
  std::vector<float> src(n);
  gen_data(src, dtype);
  std::vector<float> dst(n);
  for (auto _ : state) {
    simd::impl::relu_simd(src.data(), dst.data(), n);
    benchmark::DoNotOptimize(dst.data());
  }
}
BENCHMARK_CAPTURE(BM_ReLU_Simd, pos, POS)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
BENCHMARK_CAPTURE(BM_ReLU_Simd, neg, NEG)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
BENCHMARK_CAPTURE(BM_ReLU_Simd, rand, RAND)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);

static void BM_ReLU_Simd_NT(benchmark::State& state, DataType dtype) {
  size_t n = state.range(0);
  std::vector<float> src(n);
  gen_data(src, dtype);
  float* dst = static_cast<float*>(simd::aligned_alloc(32, n * sizeof(float)));
  for (auto _ : state) {
    simd::impl::relu_simd_nt(src.data(), dst, n);
    benchmark::DoNotOptimize(dst);
  }
  simd::aligned_free(dst);
}
BENCHMARK_CAPTURE(BM_ReLU_Simd_NT, pos, POS)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
BENCHMARK_CAPTURE(BM_ReLU_Simd_NT, neg, NEG)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
BENCHMARK_CAPTURE(BM_ReLU_Simd_NT, rand, RAND)->Arg(1<<20)->Arg(1<<12)->Arg(1<<10);
#endif

BENCHMARK_MAIN();