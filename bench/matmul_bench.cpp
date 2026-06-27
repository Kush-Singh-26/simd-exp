#include "bench_harness.hpp"
#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <simd/ops/matmul/scalar.hpp>
#include <simd/ops/matmul/simd.hpp>
#include <simd/ops/matmul/f16.hpp>
#include <string>
#include <vector>

using GemmFn = void(*)(const float*, const float*, float*, size_t, size_t, size_t);

static std::vector<float> make_random_vec(size_t n, float lo, float hi, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> res(n);
    for (size_t i = 0; i < n; ++i) res[i] = dist(rng);
    return res;
}

static void BM_Gemm(benchmark::State& state, GemmFn fn, const char* label){
    size_t M = state.range(0);
    size_t N = state.range(1);
    size_t K = state.range(2);

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N, 0.0f);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for(auto& x: A) x = dist(rng);
    for(auto& x: B) x = dist(rng);
    for(auto _ : state){
        std::fill(C.begin(), C.end(), 0.0f);
        fn(A.data(), B.data(), C.data(), M, N, K);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    state.counters["GFLOPS"] = benchmark::Counter(
        2.0 * M * N * K * state.iterations() / 1e9, benchmark::Counter::kIsRate);
}

// Register all shape categories for a single kernel.
// Square: cache hierarchy sweep (64² → 2048²)
// Decode: M=1 autoregressive generation
// Non-square: tail handling (M%4, N%8, K%8 != 0)
static void register_gemm_bench(const char* name, GemmFn fn) {
    std::string base = std::string("BM_Gemm/") + name;

    benchmark::RegisterBenchmark(base.c_str(), BM_Gemm, fn, name)
        ->Args({64, 64, 64})
        ->Args({128, 128, 128})
        ->Args({256, 256, 256})
        ->Args({512, 512, 512})
        ->Args({1024, 1024, 1024})
        ->Args({2048, 2048, 2048})
        ->ArgNames({"M", "N", "K"});

    benchmark::RegisterBenchmark((base + "_decode").c_str(), BM_Gemm, fn, name)
        ->Args({1, 4096, 4096})
        ->Args({1, 51200, 4096})
        ->ArgNames({"M", "N", "K"});

    benchmark::RegisterBenchmark((base + "_nonsquare").c_str(), BM_Gemm, fn, name)
        ->Args({3, 5, 7})
        ->Args({128, 256, 64})
        ->Args({37, 1024, 512})
        ->ArgNames({"M", "N", "K"});
}

int main(int argc, char** argv) {
    register_gemm_bench("ijk", simd::impl::matmul_ijk);
    register_gemm_bench("ikj", simd::impl::matmul_ikj);
    register_gemm_bench("tiled_64", simd::impl::matmul_scalar_tiled<64, 64, 64>);
    register_gemm_bench("tiled_32", simd::impl::matmul_scalar_tiled<32, 32, 64>);
    register_gemm_bench("tiled_128", simd::impl::matmul_scalar_tiled<128, 128, 128>);
    register_gemm_bench("tiled_64x32", simd::impl::matmul_scalar_tiled<64, 32, 64>);
    register_gemm_bench("tiled_32x64", simd::impl::matmul_scalar_tiled<32, 64, 64>);
    register_gemm_bench("avx2_64", simd::impl::matmul_avx2<64, 64, 64>);
    register_gemm_bench("avx2_32", simd::impl::matmul_avx2<32, 32, 64>);
    register_gemm_bench("avx2_128", simd::impl::matmul_avx2<128, 128, 128>);
    register_gemm_bench("avx2_64x32", simd::impl::matmul_avx2<64, 32, 64>);
    register_gemm_bench("avx2_32x64", simd::impl::matmul_avx2<32, 64, 64>);
    register_gemm_bench("4x8_64", simd::impl::matmul_4x8<64, 64, 64>);
    register_gemm_bench("4x8_32", simd::impl::matmul_4x8<32, 32, 64>);
    register_gemm_bench("4x8_128", simd::impl::matmul_4x8<128, 128, 128>);
    register_gemm_bench("4x8_64x32", simd::impl::matmul_4x8<64, 32, 64>);
    register_gemm_bench("4x8_32x64", simd::impl::matmul_4x8<32, 64, 64>);
    register_gemm_bench("packed_64", simd::impl::matmul_packed<64, 64, 64>);
    register_gemm_bench("packed_32", simd::impl::matmul_packed<32, 32, 64>);
    register_gemm_bench("packed_128", simd::impl::matmul_packed<128, 128, 128>);
    register_gemm_bench("packed_64x32", simd::impl::matmul_packed<64, 32, 64>);
    register_gemm_bench("packed_32x64", simd::impl::matmul_packed<32, 64, 64>);
    register_gemm_bench("pf_64", simd::impl::matmul_packed_prefetch<64, 64, 64>);
    register_gemm_bench("pf_32", simd::impl::matmul_packed_prefetch<32, 32, 64>);
    register_gemm_bench("pf_128", simd::impl::matmul_packed_prefetch<128, 128, 128>);
    register_gemm_bench("pf_64x32", simd::impl::matmul_packed_prefetch<64, 32, 64>);
    register_gemm_bench("pf_32x64", simd::impl::matmul_packed_prefetch<32, 64, 64>);

    // M=1 specialized kernel — wrapper to adapt (N,K) signature to GemmFn (M,N,K)
    auto matmul_1x8_wrap = [](const float* A, const float* B, float* C,
                              size_t M, size_t N, size_t K) {
        simd::impl::matmul_1x8(A, B, C, N, K);
    };
    GemmFn fn_1x8 = matmul_1x8_wrap;
    register_gemm_bench("1x8", fn_1x8);

    // Batched GEMM benchmarks
    auto bm_batched = [](benchmark::State& state) {
        size_t M = state.range(0);
        size_t N = state.range(1);
        size_t K = state.range(2);
        size_t batch = state.range(3);
        int64_t stride_A = M * K, stride_B = K * N, stride_C = M * N;

        std::vector<float> A(batch * stride_A), B(batch * stride_B), C(batch * stride_C, 0.0f);
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& x : A) x = dist(rng);
        for (auto& x : B) x = dist(rng);

        for (auto _ : state) {
            std::fill(C.begin(), C.end(), 0.0f);
            simd::impl::matmul_strided_batched(A.data(), B.data(), C.data(),
                                                M, N, K, batch, stride_A, stride_B, stride_C);
            benchmark::DoNotOptimize(C.data());
            benchmark::ClobberMemory();
        }
        state.counters["GFLOPS"] = benchmark::Counter(
            2.0 * M * N * K * batch * state.iterations() / 1e9, benchmark::Counter::kIsRate);
    };

    auto bm_batched_shared = [](benchmark::State& state) {
        size_t M = state.range(0);
        size_t N = state.range(1);
        size_t K = state.range(2);
        size_t batch = state.range(3);
        int64_t stride_A = M * K, stride_B = K * N, stride_C = M * N;

        std::vector<float> A(batch * stride_A), B(batch * stride_B), C(batch * stride_C, 0.0f);
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& x : A) x = dist(rng);
        for (auto& x : B) x = dist(rng);

        for (auto _ : state) {
            std::fill(C.begin(), C.end(), 0.0f);
            simd::impl::matmul_strided_batched_shared_B(A.data(), B.data(), C.data(),
                                                         M, N, K, batch, stride_A, stride_B, stride_C);
            benchmark::DoNotOptimize(C.data());
            benchmark::ClobberMemory();
        }
        state.counters["GFLOPS"] = benchmark::Counter(
            2.0 * M * N * K * batch * state.iterations() / 1e9, benchmark::Counter::kIsRate);
    };

    // Multi-head attention: batch=32, M=1, N=64, K=64
    benchmark::RegisterBenchmark("BM_Batched/attn_32x1x64x64", bm_batched)
        ->Args({1, 64, 64, 32})->ArgNames({"M", "N", "K", "batch"});
    benchmark::RegisterBenchmark("BM_Batched/attn_32x1x128x128", bm_batched)
        ->Args({1, 128, 128, 32})->ArgNames({"M", "N", "K", "batch"});

    benchmark::RegisterBenchmark("BM_Batched_SharedB/attn_32x1x64x64", bm_batched_shared)
        ->Args({1, 64, 64, 32})->ArgNames({"M", "N", "K", "batch"});
    benchmark::RegisterBenchmark("BM_Batched_SharedB/attn_32x1x128x128", bm_batched_shared)
        ->Args({1, 128, 128, 32})->ArgNames({"M", "N", "K", "batch"});

    // Larger batch with M=1 (decode across batch)
    benchmark::RegisterBenchmark("BM_Batched/decode_8x1x4096x4096", bm_batched)
        ->Args({1, 4096, 4096, 8})->ArgNames({"M", "N", "K", "batch"});
    benchmark::RegisterBenchmark("BM_Batched_SharedB/decode_8x1x4096x4096", bm_batched_shared)
        ->Args({1, 4096, 4096, 8})->ArgNames({"M", "N", "K", "batch"});

    // ── Step 12: Quantized f32 × i8 benchmarks ───────────────────────────────
    auto bm_f32xi8 = [](benchmark::State& state) {
        size_t M = state.range(0);
        size_t N = state.range(1);
        size_t K = state.range(2);
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
        std::uniform_int_distribution<int> idist(-127, 127);

        std::vector<float> A(M * K), scales(N), C(M * N, 0.0f);
        std::vector<int8_t> B_i8(K * N);
        for (auto& x : A) x = fdist(rng);
        for (auto& x : B_i8) x = idist(rng);
        for (auto& x : scales) x = 0.5f + fdist(rng) * 0.5f;

        for (auto _ : state) {
            std::fill(C.begin(), C.end(), 0.0f);
            simd::impl::matmul_f32xi8_simd(A.data(), B_i8.data(), scales.data(),
                                             C.data(), M, N, K);
            benchmark::DoNotOptimize(C.data());
            benchmark::ClobberMemory();
        }
        state.counters["GFLOPS"] = benchmark::Counter(
            2.0 * M * N * K * state.iterations() / 1e9, benchmark::Counter::kIsRate);
    };

    benchmark::RegisterBenchmark("BM_F32xI8/packed_32", bm_f32xi8)->Args({32, 32, 64})->ArgNames({"M", "N", "K"});
    benchmark::RegisterBenchmark("BM_F32xI8/packed_64", bm_f32xi8)->Args({64, 64, 64})->ArgNames({"M", "N", "K"});
    benchmark::RegisterBenchmark("BM_F32xI8/packed_128", bm_f32xi8)->Args({128, 128, 128})->ArgNames({"M", "N", "K"});
    benchmark::RegisterBenchmark("BM_F32xI8/decode_1x4096x4096", bm_f32xi8)->Args({1, 4096, 4096})->ArgNames({"M", "N", "K"});

    // ── Step 13: Fused Epilogue benchmarks ────────────────────────────────────
    auto bm_bias_relu = [](benchmark::State& state) {
        size_t M = state.range(0), N = state.range(1), K = state.range(2);
        auto A = make_random_vec(M * K, -1, 1);
        auto B = make_random_vec(K * N, -1, 1);
        auto bias = make_random_vec(N, -1, 1);
        std::vector<float> C(M * N);

        for (auto _ : state) {
            std::fill(C.begin(), C.end(), 0.0f);
            simd::impl::matmul_bias_relu(A.data(), B.data(), C.data(), bias.data(), M, N, K);
            benchmark::DoNotOptimize(C.data());
            benchmark::ClobberMemory();
        }
        state.counters["GFLOPS"] = benchmark::Counter(
            2.0 * M * N * K * state.iterations() / 1e9, benchmark::Counter::kIsRate);
    };

    auto bm_separate = [](benchmark::State& state) {
        size_t M = state.range(0), N = state.range(1), K = state.range(2);
        auto A = make_random_vec(M * K, -1, 1);
        auto B = make_random_vec(K * N, -1, 1);
        auto bias = make_random_vec(N, -1, 1);
        std::vector<float> C(M * N);

        for (auto _ : state) {
            std::fill(C.begin(), C.end(), 0.0f);
            simd::impl::matmul_packed(A.data(), B.data(), C.data(), M, N, K);
            for (size_t i = 0; i < M; i++)
                for (size_t j = 0; j < N; j++)
                    C[i * N + j] = std::max(C[i * N + j] + bias[j], 0.0f);
            benchmark::DoNotOptimize(C.data());
            benchmark::ClobberMemory();
        }
        state.counters["GFLOPS"] = benchmark::Counter(
            2.0 * M * N * K * state.iterations() / 1e9, benchmark::Counter::kIsRate);
    };

    benchmark::RegisterBenchmark("BM_BiasReLU/fused_32", bm_bias_relu)->Args({32, 32, 64})->ArgNames({"M", "N", "K"});
    benchmark::RegisterBenchmark("BM_BiasReLU/fused_64", bm_bias_relu)->Args({64, 64, 64})->ArgNames({"M", "N", "K"});
    benchmark::RegisterBenchmark("BM_BiasReLU/fused_128", bm_bias_relu)->Args({128, 128, 128})->ArgNames({"M", "N", "K"});
    benchmark::RegisterBenchmark("BM_BiasReLU/separate_32", bm_separate)->Args({32, 32, 64})->ArgNames({"M", "N", "K"});
    benchmark::RegisterBenchmark("BM_BiasReLU/separate_64", bm_separate)->Args({64, 64, 64})->ArgNames({"M", "N", "K"});
    benchmark::RegisterBenchmark("BM_BiasReLU/separate_128", bm_separate)->Args({128, 128, 128})->ArgNames({"M", "N", "K"});

    // ── Step 14: Attention benchmarks ─────────────────────────────────────────
    auto bm_attention = [](benchmark::State& state) {
        size_t batch = state.range(0), heads = state.range(1);
        size_t T = state.range(2), S = state.range(3), D = state.range(4);
        auto Q = make_random_vec(batch * heads * T * D, -1, 1);
        auto K_mat = make_random_vec(batch * heads * S * D, -1, 1);
        auto V = make_random_vec(batch * heads * S * D, -1, 1);
        size_t total = batch * heads;
        std::vector<float> scores(total * T * S), out(total * T * D);

        for (auto _ : state) {
            std::fill(scores.begin(), scores.end(), 0.0f);
            simd::impl::attention_scores_simd(Q.data(), K_mat.data(), scores.data(),
                                               batch, heads, T, S, D);
            std::fill(out.begin(), out.end(), 0.0f);
            simd::impl::attention_values_simd(scores.data(), V.data(), out.data(),
                                               batch, heads, T, S, D);
            benchmark::DoNotOptimize(out.data());
            benchmark::ClobberMemory();
        }
        state.counters["GFLOPS"] = benchmark::Counter(
            2.0 * (T * S * D + T * D * S) * batch * heads * state.iterations() / 1e9,
            benchmark::Counter::kIsRate);
    };

    benchmark::RegisterBenchmark("BM_Attention/prefill_2x2x128x128x64", bm_attention)
        ->Args({2, 2, 128, 128, 64})->ArgNames({"batch", "heads", "T", "S", "D"});
    benchmark::RegisterBenchmark("BM_Attention/prefill_1x8x256x256x64", bm_attention)
        ->Args({1, 8, 256, 256, 64})->ArgNames({"batch", "heads", "T", "S", "D"});
    benchmark::RegisterBenchmark("BM_Attention/decode_8x8x1x128x64", bm_attention)
        ->Args({8, 8, 1, 128, 64})->ArgNames({"batch", "heads", "T", "S", "D"});

    // ── Step 15: FP16 GEMM benchmarks ─────────────────────────────────────────
    auto bm_fp16 = [](benchmark::State& state) {
        size_t M = state.range(0), N = state.range(1), K = state.range(2);
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> A_fp32(M * K), B_fp32(K * N);
        for (auto& x : A_fp32) x = dist(rng);
        for (auto& x : B_fp32) x = dist(rng);

        // Convert to FP16
        std::vector<uint16_t> A_fp16(M * K), B_fp16(K * N);
        for (size_t i = 0; i + 7 < M * K; i += 8)
            simd::impl::fp32_to_fp16_x8(&A_fp32[i], &A_fp16[i]);
        for (size_t i = M * K - (M * K % 8); i < M * K; i++) {
            float f = A_fp32[i];
            uint32_t bits; std::memcpy(&bits, &f, sizeof(bits));
            A_fp16[i] = static_cast<uint16_t>((bits >> 16) & 0xFFFF);
        }
        for (size_t i = 0; i + 7 < K * N; i += 8)
            simd::impl::fp32_to_fp16_x8(&B_fp32[i], &B_fp16[i]);
        for (size_t i = K * N - (K * N % 8); i < K * N; i++) {
            float f = B_fp32[i];
            uint32_t bits; std::memcpy(&bits, &f, sizeof(bits));
            B_fp16[i] = static_cast<uint16_t>((bits >> 16) & 0xFFFF);
        }

        std::vector<float> C(M * N);
        for (auto _ : state) {
            std::fill(C.begin(), C.end(), 0.0f);
            simd::impl::matmul_fp16_simd(A_fp16.data(), B_fp16.data(), C.data(), M, N, K);
            benchmark::DoNotOptimize(C.data());
            benchmark::ClobberMemory();
        }
        state.counters["GFLOPS"] = benchmark::Counter(
            2.0 * M * N * K * state.iterations() / 1e9, benchmark::Counter::kIsRate);
    };

    benchmark::RegisterBenchmark("BM_Fp16/packed_32", bm_fp16)->Args({32, 32, 64})->ArgNames({"M", "N", "K"});
    benchmark::RegisterBenchmark("BM_Fp16/packed_64", bm_fp16)->Args({64, 64, 64})->ArgNames({"M", "N", "K"});
    benchmark::RegisterBenchmark("BM_Fp16/packed_128", bm_fp16)->Args({128, 128, 128})->ArgNames({"M", "N", "K"});
    benchmark::RegisterBenchmark("BM_Fp16/decode_1x4096x4096", bm_fp16)->Args({1, 4096, 4096})->ArgNames({"M", "N", "K"});

    // FP16 vs FP32 comparison (same GEMM, different input formats)
    auto bm_fp32_compare = [](benchmark::State& state) {
        size_t M = state.range(0), N = state.range(1), K = state.range(2);
        auto A = make_random_vec(M * K, -1, 1);
        auto B = make_random_vec(K * N, -1, 1);
        std::vector<float> C(M * N);
        for (auto _ : state) {
            std::fill(C.begin(), C.end(), 0.0f);
            simd::impl::matmul_packed(A.data(), B.data(), C.data(), M, N, K);
            benchmark::DoNotOptimize(C.data());
            benchmark::ClobberMemory();
        }
        state.counters["GFLOPS"] = benchmark::Counter(
            2.0 * M * N * K * state.iterations() / 1e9, benchmark::Counter::kIsRate);
    };

    benchmark::RegisterBenchmark("BM_Fp32/packed_32", bm_fp32_compare)->Args({32, 32, 64})->ArgNames({"M", "N", "K"});
    benchmark::RegisterBenchmark("BM_Fp32/packed_64", bm_fp32_compare)->Args({64, 64, 64})->ArgNames({"M", "N", "K"});
    benchmark::RegisterBenchmark("BM_Fp32/packed_128", bm_fp32_compare)->Args({128, 128, 128})->ArgNames({"M", "N", "K"});
    benchmark::RegisterBenchmark("BM_Fp32/decode_1x4096x4096", bm_fp32_compare)->Args({1, 4096, 4096})->ArgNames({"M", "N", "K"});

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}
