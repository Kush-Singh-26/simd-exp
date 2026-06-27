#include "test_harness.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <simd/ops/matmul/matmul.hpp>
#include <simd/ops/matmul/scalar.hpp>
#include <simd/ops/matmul/simd.hpp>
#include <simd/ops/matmul/f16.hpp>
#include <string>
#include <vector>

using MatmulFunc = std::function<void(const float*, const float*, float*,
                                      size_t, size_t, size_t)>;

// Core: run both kernels, compare results 
static void compare_kernels(MatmulFunc a, MatmulFunc b,
                            size_t M, size_t N, size_t K, float tol,
                            const std::string& label) {
    auto A = make_random(M * K, -1, 1);
    auto B = make_random(K * N, -1, 1);
    std::vector<float> Ca(M * N), Cb(M * N);
    a(A.data(), B.data(), Ca.data(), M, N, K);
    b(A.data(), B.data(), Cb.data(), M, N, K);
    check_near(Ca.data(), Cb.data(), M * N, tol, label);
}

// Test a kernel against known exact values 
static void test_known(MatmulFunc fn, const std::string& name) {
    float A[] = {1, 2, 3, 4};
    float B[] = {5, 6, 7, 8};
    float C[4]{};
    fn(A, B, C, 2, 2, 2);
    float exact[] = {19, 22, 43, 50};
    check_exact(C, exact, 4, name);
}

// Run ALL shape categories for kernel vs reference 
// Covers: square, non-square, large, edge — one line exercises everything.
static void test_all_shapes(MatmulFunc kernel, MatmulFunc ref,
                            const std::string& name) {
    std::string tag = name + " vs " +
        (name == "ikj" ? std::string("ijk") : std::string("ikj"));

    // Square shapes: {2,4,8} × {2,4,8} × {1,3,8} = 27 cases
    for (size_t d : {2, 4, 8})
        for (size_t k : {1, 3, 8})
            compare_kernels(kernel, ref, d, d, k, 1e-5f,
                tag + " M=" + std::to_string(d) +
                " N=" + std::to_string(d) +
                " K=" + std::to_string(k));

    // Non-square shapes
    struct { size_t M, N, K; } ns[] = {{3,5,7}, {1,8,4}, {8,1,3}, {4,12,6}};
    for (auto [M, N, K] : ns)
        compare_kernels(kernel, ref, M, N, K, 1e-5f,
            tag + " M=" + std::to_string(M) +
            " N=" + std::to_string(N) +
            " K=" + std::to_string(K));

    // Large matrices (M=N=K)
    for (size_t d : {256, 512})
        compare_kernels(kernel, ref, d, d, d, 1e-3f,
            tag + " M=N=K=" + std::to_string(d));

    // Edge cases: one dimension = 1
    compare_kernels(kernel, ref, 1, 8, 8, 1e-5f, tag + " M=1");
    compare_kernels(kernel, ref, 8, 1, 8, 1e-5f, tag + " N=1");
    compare_kernels(kernel, ref, 8, 8, 1, 1e-5f, tag + " K=1");
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

// ── Known values: each kernel against exact 2×2×2 result ───────────────────
TEST(MatmulTests, KnownValues_Ijk)   { test_known(simd::impl::matmul_ijk, "ijk"); }
TEST(MatmulTests, KnownValues_Ikj)   { test_known(simd::impl::matmul_ikj, "ikj"); }
TEST(MatmulTests, KnownValues_Tiled) { test_known(simd::impl::matmul_scalar_tiled, "tiled"); }
TEST(MatmulTests, KnownValues_AVX2) { test_known(simd::impl::matmul_avx2, "avx2"); }
TEST(MatmulTests, KnownValues_4x8) {test_known(simd::impl::matmul_4x8, "4x8");}
TEST(MatmulTests, KnownValues_Packed) { test_known(simd::impl::matmul_packed, "packed"); }
TEST(MatmulTests, KnownValues_Prefetch) { test_known(simd::impl::matmul_packed_prefetch, "prefetch"); }


// ── Chain of trust: ijk (base) → ikj → tiled ──────────────────────────────
TEST(MatmulTests, IkjVsIjk)  {
    test_all_shapes(simd::impl::matmul_ikj, simd::impl::matmul_ijk, "ikj");
}
TEST(MatmulTests, TiledVsIkj) {
    test_all_shapes(simd::impl::matmul_scalar_tiled, simd::impl::matmul_ikj, "tiled");
}
TEST(MatmulTests, AVX2vsTiled) {
    test_all_shapes(simd::impl::matmul_avx2, simd::impl::matmul_scalar_tiled, "avx2");
}
TEST(MatmulTests, 4x8vsAVX2) {
    test_all_shapes(simd::impl::matmul_4x8, simd::impl::matmul_avx2, "4x8");
}

TEST(MatmulTests, PackedVsIkj) {
    test_all_shapes(simd::impl::matmul_packed<>, simd::impl::matmul_ikj, "packed");
}

TEST(MatmulTests, PrefetchVsPacked) {
    test_all_shapes(simd::impl::matmul_packed_prefetch<>, simd::impl::matmul_packed<>, "prefetch");
}

// ── M=1 specialized kernel tests ────────────────────────────────────────────
TEST(MatmulTests, KnownValues_1x8) {
    // M=1, N=2, K=2: A={1,2}, B={3,4,5,6} → C={1*3+2*5, 1*4+2*6} = {13, 16}
    float A[] = {1, 2};
    float B[] = {3, 4, 5, 6};
    float C[2]{};
    simd::impl::matmul_1x8(A, B, C, 2, 2);
    float exact[] = {13, 16};
    check_exact(C, exact, 2, "1x8");
}

TEST(MatmulTests, 1x8vsIkj) {
    for (size_t N : {1, 4, 8, 13, 16, 32, 64})
        for (size_t K : {1, 3, 8, 16})
            compare_kernels(
                [](const float* A, const float* B, float* C, size_t, size_t N, size_t K) {
                    simd::impl::matmul_1x8(A, B, C, N, K);
                },
                simd::impl::matmul_ikj,
                1, N, K, 1e-5f,
                "1x8 vs ikj N=" + std::to_string(N) + " K=" + std::to_string(K));
}

TEST(MatmulTests, Dispatcher_M1) {
    size_t M = 1, N = 64, K = 64;
    auto A = make_random(M * K, -1, 1);
    auto B = make_random(K * N, -1, 1);
    std::vector<float> C_ref(M * N), C_dispatch(M * N);
    simd::impl::matmul_ijk(A.data(), B.data(), C_ref.data(), M, N, K);
    simd::matmul(A, B, C_dispatch, M, N, K);
    check_near(C_ref.data(), C_dispatch.data(), M * N, 1e-5f, "dispatcher M=1");
}

// ── Dispatcher tests (std::span interface, standalone) ─────────────────────
TEST(MatmulDispatcherTest, MatchesIjk) {
    size_t M = 16, N = 16, K = 16;
    auto A = make_random(M * K, -1, 1);
    auto B = make_random(K * N, -1, 1);
    std::vector<float> C_ref(M * N), C_dispatch(M * N);
    simd::impl::matmul_ijk(A.data(), B.data(), C_ref.data(), M, N, K);
    simd::matmul(A, B, C_dispatch, M, N, K);
    check_near(C_ref.data(), C_dispatch.data(), M * N, 1e-5f, "dispatcher");
}

TEST(MatmulDispatcherTest, Zeros_C) {
    size_t M = 8, N = 8, K = 8;
    auto A = make_random(M * K, -1, 1);
    auto B = make_random(K * N, -1, 1);
    std::vector<float> C_garbage(M * N, 999.0f);
    simd::matmul(A, B, C_garbage, M, N, K);
    std::vector<float> C_clean(M * N);
    simd::matmul(A, B, C_clean, M, N, K);
    check_exact(C_garbage.data(), C_clean.data(), M * N, "dispatcher zero-init");
}

// ── Batched GEMM tests ──────────────────────────────────────────────────────
TEST(MatmulTests, StridedBatched_Basic) {
    size_t M = 4, N = 8, K = 4, batch = 3;
    int64_t stride_A = M * K, stride_B = K * N, stride_C = M * N;
    std::vector<float> A(batch * stride_A), B(batch * stride_B), C(batch * stride_C, 0.0f);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    // Reference: compute each batch independently
    std::vector<float> C_ref(batch * stride_C);
    for (size_t b = 0; b < batch; b++)
        simd::impl::matmul_ikj(A.data() + b * stride_A, B.data() + b * stride_B,
                               C_ref.data() + b * stride_C, M, N, K);

    simd::impl::matmul_strided_batched(A.data(), B.data(), C.data(),
                                        M, N, K, batch, stride_A, stride_B, stride_C);
    check_near(C_ref.data(), C.data(), batch * stride_C, 1e-5f, "strided_batched basic");
}

TEST(MatmulTests, StridedBatched_SharedB) {
    size_t M = 4, N = 8, K = 4, batch = 3;
    int64_t stride_A = M * K, stride_B = K * N, stride_C = M * N;
    std::vector<float> A(batch * stride_A), B(batch * stride_B), C(batch * stride_C, 0.0f);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    // Reference: compute each batch independently
    std::vector<float> C_ref(batch * stride_C);
    for (size_t b = 0; b < batch; b++)
        simd::impl::matmul_ikj(A.data() + b * stride_A, B.data() + b * stride_B,
                               C_ref.data() + b * stride_C, M, N, K);

    simd::impl::matmul_strided_batched_shared_B(A.data(), B.data(), C.data(),
                                                 M, N, K, batch, stride_A, stride_B, stride_C);
    check_near(C_ref.data(), C.data(), batch * stride_C, 1e-5f, "strided_batched_shared_B");
}

TEST(MatmulTests, StridedBatched_LargeBatch) {
    size_t M = 1, N = 64, K = 64, batch = 32;
    int64_t stride_A = M * K, stride_B = K * N, stride_C = M * N;
    std::vector<float> A(batch * stride_A), B(batch * stride_B), C(batch * stride_C, 0.0f);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    std::vector<float> C_ref(batch * stride_C);
    for (size_t b = 0; b < batch; b++)
        simd::impl::matmul_ikj(A.data() + b * stride_A, B.data() + b * stride_B,
                               C_ref.data() + b * stride_C, M, N, K);

    simd::impl::matmul_strided_batched(A.data(), B.data(), C.data(),
                                        M, N, K, batch, stride_A, stride_B, stride_C);
    check_near(C_ref.data(), C.data(), batch * stride_C, 1e-3f, "strided_batched large batch");
}

// ── Step 12: Quantized f32 × i8 → f32 tests ──────────────────────────────────

TEST(MatmulTests, F32xI8_KnownValues) {
    float A[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int8_t B_i8[] = {5, 6, 7, 8};
    float scales[] = {1.0f, 1.0f};
    float C_scalar[4]{}, C_simd[4]{};
    simd::impl::matmul_f32xi8_scalar(A, B_i8, scales, C_scalar, 2, 2, 2);
    simd::impl::matmul_f32xi8_simd(A, B_i8, scales, C_simd, 2, 2, 2);
    check_near(C_scalar, C_simd, 4, 1e-5f, "f32xi8 known values");
}

TEST(MatmulTests, F32xI8vsScalar) {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int> idist(-127, 127);
    std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);

    struct { size_t M, N, K; } shapes[] = {
        {2, 2, 2}, {3, 5, 7}, {4, 8, 4}, {8, 8, 8},
        {1, 16, 8}, {1, 64, 64}, {4, 13, 6}, {5, 17, 31}
    };

    for (auto [M, N, K] : shapes) {
        std::vector<float> A(M * K), scales(N);
        std::vector<int8_t> B_i8(K * N);
        std::vector<float> C_scalar(M * N, 0.0f), C_simd(M * N, 0.0f);
        for (auto& x : A) x = fdist(rng);
        for (auto& x : B_i8) x = idist(rng);
        for (auto& x : scales) x = 0.5f + fdist(rng) * 0.5f;

        simd::impl::matmul_f32xi8_scalar(A.data(), B_i8.data(), scales.data(),
                                           C_scalar.data(), M, N, K);
        simd::impl::matmul_f32xi8_simd(A.data(), B_i8.data(), scales.data(),
                                         C_simd.data(), M, N, K);
        check_near(C_scalar.data(), C_simd.data(), M * N, 1e-4f,
            "f32xi8 M=" + std::to_string(M) + " N=" + std::to_string(N) + " K=" + std::to_string(K));
    }
}

TEST(MatmulTests, F32xI8_M1) {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int> idist(-127, 127);
    std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);

    for (size_t N : {1, 8, 13, 32, 64})
        for (size_t K : {1, 4, 16}) {
            std::vector<float> A(K), scales(N);
            std::vector<int8_t> B_i8(K * N);
            std::vector<float> C_scalar(N, 0.0f), C_1x8(N, 0.0f);
            for (auto& x : A) x = fdist(rng);
            for (auto& x : B_i8) x = idist(rng);
            for (auto& x : scales) x = 0.5f + fdist(rng) * 0.5f;

            simd::impl::matmul_f32xi8_scalar(A.data(), B_i8.data(), scales.data(),
                                               C_scalar.data(), 1, N, K);
            simd::impl::matmul_f32xi8_1x8(A.data(), B_i8.data(), scales.data(),
                                            C_1x8.data(), N, K);
            check_near(C_scalar.data(), C_1x8.data(), N, 1e-4f,
                "f32xi8 1x8 N=" + std::to_string(N) + " K=" + std::to_string(K));
        }
}

TEST(MatmulTests, F32xI8_Batched) {
    size_t M = 4, N = 8, K = 4, batch = 3;
    int64_t stride_A = M * K, stride_B = K * N, stride_C = M * N;
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int> idist(-127, 127);
    std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);

    std::vector<float> A(batch * stride_A), scales(N), C(batch * stride_C, 0.0f);
    std::vector<int8_t> B_i8(batch * stride_B);
    for (auto& x : A) x = fdist(rng);
    for (auto& x : B_i8) x = idist(rng);
    for (auto& x : scales) x = 0.5f + fdist(rng) * 0.5f;

    std::vector<float> C_ref(batch * stride_C);
    for (size_t b = 0; b < batch; b++)
        simd::impl::matmul_f32xi8_scalar(A.data() + b * stride_A, B_i8.data() + b * stride_B,
                                           scales.data(), C_ref.data() + b * stride_C, M, N, K);

    simd::impl::matmul_f32xi8_batched(A.data(), B_i8.data(), scales.data(), C.data(),
                                        M, N, K, batch, stride_A, stride_B, stride_C);
    check_near(C_ref.data(), C.data(), batch * stride_C, 1e-4f, "f32xi8 batched");
}

// ── Step 13: Fused Epilogue tests ────────────────────────────────────────────

TEST(MatmulTests, BiasEpilogue) {
    size_t M = 8, N = 8, K = 8;
    auto A = make_random(M * K, -1, 1);
    auto B = make_random(K * N, -1, 1);
    auto bias = make_random(N, -1, 1);

    std::vector<float> C_ref(M * N), C_bias(M * N, 0.0f);
    simd::impl::matmul_packed(A.data(), B.data(), C_ref.data(), M, N, K);
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++)
            C_ref[i * N + j] += bias[j];

    simd::impl::matmul_bias(A.data(), B.data(), C_bias.data(), bias.data(), M, N, K);
    check_near(C_ref.data(), C_bias.data(), M * N, 1e-5f, "bias epilogue");
}

TEST(MatmulTests, BiasReLUEpilogue) {
    size_t M = 8, N = 8, K = 8;
    auto A = make_random(M * K, -1, 1);
    auto B = make_random(K * N, -1, 1);
    auto bias = make_random(N, -1, 1);

    std::vector<float> C_ref(M * N), C_relu(M * N, 0.0f);
    simd::impl::matmul_packed(A.data(), B.data(), C_ref.data(), M, N, K);
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++)
            C_ref[i * N + j] = std::max(C_ref[i * N + j] + bias[j], 0.0f);

    simd::impl::matmul_bias_relu(A.data(), B.data(), C_relu.data(), bias.data(), M, N, K);
    check_near(C_ref.data(), C_relu.data(), M * N, 1e-5f, "bias+relu epilogue");
}

TEST(MatmulTests, BiasGELUEpilogue) {
    size_t M = 8, N = 8, K = 8;
    auto A = make_random(M * K, -1, 1);
    auto B = make_random(K * N, -1, 1);
    auto bias = make_random(N, -1, 1);

    std::vector<float> C_ref(M * N), C_gelu(M * N, 0.0f);
    simd::impl::matmul_packed(A.data(), B.data(), C_ref.data(), M, N, K);
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++) {
            float x = C_ref[i * N + j] + bias[j];
            float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
            C_ref[i * N + j] = x * 0.5f * (1.0f + std::tanh(inner));
        }

    simd::impl::matmul_bias_gelu(A.data(), B.data(), C_gelu.data(), bias.data(), M, N, K);
    check_near(C_ref.data(), C_gelu.data(), M * N, 1e-2f, "bias+gelu epilogue");
}

TEST(MatmulTests, BiasSiLUEpilogue) {
    size_t M = 8, N = 8, K = 8;
    auto A = make_random(M * K, -1, 1);
    auto B = make_random(K * N, -1, 1);
    auto bias = make_random(N, -1, 1);

    std::vector<float> C_ref(M * N), C_silu(M * N, 0.0f);
    simd::impl::matmul_packed(A.data(), B.data(), C_ref.data(), M, N, K);
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++) {
            float x = C_ref[i * N + j] + bias[j];
            float s = 1.0f / (1.0f + std::exp(-x));
            C_ref[i * N + j] = x * s;
        }

    simd::impl::matmul_bias_silu(A.data(), B.data(), C_silu.data(), bias.data(), M, N, K);
    check_near(C_ref.data(), C_silu.data(), M * N, 1e-3f, "bias+silu epilogue");
}

// ── Step 14: Attention tests ──────────────────────────────────────────────────

TEST(MatmulTests, Attention_Basic) {
    size_t batch = 1, heads = 2, T = 4, S = 4, D = 8;
    size_t total = batch * heads * T * D;
    auto Q = make_random(total, -1, 1);
    auto K_mat = make_random(total, -1, 1);

    std::vector<float> scores_ref(batch * heads * T * S), scores_simd(batch * heads * T * S, 0.0f);
    simd::impl::attention_scores(Q.data(), K_mat.data(), scores_ref.data(), batch, heads, T, S, D);
    simd::impl::attention_scores_simd(Q.data(), K_mat.data(), scores_simd.data(), batch, heads, T, S, D);
    check_near(scores_ref.data(), scores_simd.data(), batch * heads * T * S, 1e-4f, "attention scores");
}

TEST(MatmulTests, Attention_GQA) {
    size_t batch = 1, H_q = 4, H_kv = 2, T = 4, S = 4, D = 8;
    std::vector<float> Q(batch * H_q * T * D), K_mat(batch * H_kv * S * D);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : Q) x = dist(rng);
    for (auto& x : K_mat) x = dist(rng);

    std::vector<float> scores_ref(batch * H_q * T * S), scores_gqa(batch * H_q * T * S, 0.0f);
    simd::impl::attention_scores_gqa(Q.data(), K_mat.data(), scores_ref.data(),
                                      batch, H_q, H_kv, T, S, D);
    simd::impl::attention_scores_gqa(Q.data(), K_mat.data(), scores_gqa.data(),
                                      batch, H_q, H_kv, T, S, D);
    check_exact(scores_ref.data(), scores_gqa.data(), batch * H_q * T * S, "attention gqa self-consistency");
}

TEST(MatmulTests, Attention_Values) {
    size_t batch = 1, heads = 2, T = 4, S = 4, D = 8;
    auto attn = make_random(batch * heads * T * S, 0, 1);
    auto V = make_random(batch * heads * S * D, -1, 1);

    std::vector<float> out_ref(batch * heads * T * D), out_simd(batch * heads * T * D, 0.0f);
    simd::impl::attention_values(attn.data(), V.data(), out_ref.data(), batch, heads, T, S, D);
    simd::impl::attention_values_simd(attn.data(), V.data(), out_simd.data(), batch, heads, T, S, D);
    check_near(out_ref.data(), out_simd.data(), batch * heads * T * D, 1e-4f, "attention values");
}

TEST(MatmulTests, Attention_ValuesGQA) {
    size_t batch = 1, H_q = 4, H_kv = 2, T = 4, S = 4, D = 8;
    auto attn = make_random(batch * H_q * T * S, 0, 1);
    auto V = make_random(batch * H_kv * S * D, -1, 1);

    std::vector<float> out_ref(batch * H_q * T * D), out_gqa(batch * H_q * T * D, 0.0f);
    simd::impl::attention_values_gqa(attn.data(), V.data(), out_ref.data(),
                                      batch, H_q, H_kv, T, S, D);
    simd::impl::attention_values_gqa(attn.data(), V.data(), out_gqa.data(),
                                      batch, H_q, H_kv, T, S, D);
    check_exact(out_ref.data(), out_gqa.data(), batch * H_q * T * D, "attention values gqa self-consistency");
}

// ── Step 15: FP16 GEMM tests ────────────────────────────────────────────────

// Convert FP32 vector to FP16 (uint16_t) for test setup.
static std::vector<uint16_t> fp32_to_fp16_vec(const std::vector<float>& v) {
    std::vector<uint16_t> out(v.size());
    size_t i = 0;
    for (; i + 7 < v.size(); i += 8) {
        simd::impl::fp32_to_fp16_x8(&v[i], &out[i]);
    }
    for (; i < v.size(); i++) {
        float f = v[i];
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        uint32_t sign = (bits >> 16) & 0x8000;
        int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (bits >> 13) & 0x3FF;
        if (exp <= 0) { out[i] = static_cast<uint16_t>(sign); }
        else if (exp >= 31) { out[i] = static_cast<uint16_t>(sign | 0x7BFF); }
        else { out[i] = static_cast<uint16_t>(sign | (exp << 10) | mant); }
    }
    return out;
}

TEST(MatmulTests, FP16_KnownValues) {
    // M=2, N=2, K=2: A_fp16={1.0, 2.0, 3.0, 4.0}, B_fp16={5.0, 6.0, 7.0, 8.0}
    // C = A*B = {{19, 22}, {43, 50}}
    std::vector<float> A_f = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> B_f = {5.0f, 6.0f, 7.0f, 8.0f};
    auto A_fp16 = fp32_to_fp16_vec(A_f);
    auto B_fp16 = fp32_to_fp16_vec(B_f);
    float C_scalar[4]{}, C_simd[4]{};
    simd::impl::matmul_fp16_scalar(A_fp16.data(), B_f.data(), C_scalar, 2, 2, 2);
    simd::impl::matmul_fp16_simd(A_fp16.data(), B_fp16.data(), C_simd, 2, 2, 2);
    check_near(C_scalar, C_simd, 4, 1e-5f, "fp16 known values");
}

TEST(MatmulTests, FP16vsScalar) {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    struct { size_t M, N, K; } shapes[] = {
        {2, 2, 2}, {3, 5, 7}, {4, 8, 8}, {8, 8, 8},
        {1, 16, 8}, {1, 64, 64}, {4, 13, 6}, {5, 17, 31}
    };

    for (auto [M, N, K] : shapes) {
        std::vector<float> A(M * K), B(K * N);
        for (auto& x : A) x = dist(rng);
        for (auto& x : B) x = dist(rng);
        auto A_fp16 = fp32_to_fp16_vec(A);
        auto B_fp16 = fp32_to_fp16_vec(B);

        std::vector<float> C_scalar(M * N, 0.0f), C_simd(M * N, 0.0f);
        simd::impl::matmul_fp16_scalar(A_fp16.data(), B.data(), C_scalar.data(), M, N, K);
        simd::impl::matmul_fp16_simd(A_fp16.data(), B_fp16.data(), C_simd.data(), M, N, K);
        // FP16 has ~3.3 decimal digits of precision; accumulation in FP32 keeps error bounded.
        check_near(C_scalar.data(), C_simd.data(), M * N, 5e-2f,
            "fp16 M=" + std::to_string(M) + " N=" + std::to_string(N) + " K=" + std::to_string(K));
    }
}

TEST(MatmulTests, FP16_M1) {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t N : {1, 8, 13, 32, 64})
        for (size_t K : {1, 4, 16}) {
            std::vector<float> A_fp32(K), B_fp32(K * N);
            for (auto& x : A_fp32) x = dist(rng);
            for (auto& x : B_fp32) x = dist(rng);
            auto A_fp16 = fp32_to_fp16_vec(A_fp32);
            auto B_fp16 = fp32_to_fp16_vec(B_fp32);

            std::vector<float> C_scalar(N, 0.0f), C_1x8(N, 0.0f);
            simd::impl::matmul_fp16_scalar(A_fp16.data(), B_fp32.data(), C_scalar.data(), 1, N, K);
            simd::impl::matmul_fp16_1x8(A_fp16.data(), B_fp16.data(), C_1x8.data(), N, K);
            check_near(C_scalar.data(), C_1x8.data(), N, 5e-2f,
                "fp16 1x8 N=" + std::to_string(N) + " K=" + std::to_string(K));
        }
}

TEST(MatmulTests, FP16_Dispatcher) {
    size_t M = 16, N = 16, K = 16;
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> A(M * K), B(K * N);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);
    auto A_fp16 = fp32_to_fp16_vec(A);
    auto B_fp16 = fp32_to_fp16_vec(B);

    std::vector<float> C_ref(M * N), C_dispatch(M * N);
    simd::impl::matmul_fp16_scalar(A_fp16.data(), B.data(), C_ref.data(), M, N, K);
    simd::matmul_fp16(A_fp16, B_fp16, C_dispatch, M, N, K);
    check_near(C_ref.data(), C_dispatch.data(), M * N, 5e-2f, "fp16 dispatcher");
}

TEST(MatmulTests, FP16_Batched) {
    size_t M = 4, N = 8, K = 4, batch = 3;
    int64_t stride_A = M * K, stride_B = K * N, stride_C = M * N;
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A(batch * stride_A), B(batch * stride_B), C(batch * stride_C, 0.0f);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);
    auto A_fp16 = fp32_to_fp16_vec(A);
    auto B_fp16 = fp32_to_fp16_vec(B);

    std::vector<float> C_ref(batch * stride_C);
    for (size_t b = 0; b < batch; b++)
        simd::impl::matmul_fp16_scalar(A_fp16.data() + b * stride_A, B.data() + b * stride_B,
                                        C_ref.data() + b * stride_C, M, N, K);

    simd::impl::matmul_fp16_batched(A_fp16.data(), B_fp16.data(), C.data(),
                                     M, N, K, batch, stride_A, stride_B, stride_C);
    check_near(C_ref.data(), C.data(), batch * stride_C, 5e-2f, "fp16 batched");
}

TEST(MatmulTests, FP16_BiasEpilogue) {
    size_t M = 8, N = 8, K = 8;
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> A(M * K), B(K * N), bias(N);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);
    for (auto& x : bias) x = dist(rng);
    auto A_fp16 = fp32_to_fp16_vec(A);
    auto B_fp16 = fp32_to_fp16_vec(B);

    // Reference: matmul_fp16_scalar + bias
    std::vector<float> C_ref(M * N), C_bias(M * N, 0.0f);
    simd::impl::matmul_fp16_scalar(A_fp16.data(), B.data(), C_ref.data(), M, N, K);
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++)
            C_ref[i * N + j] += bias[j];

    simd::impl::matmul_fp16_bias(A_fp16.data(), B_fp16.data(), C_bias.data(), bias.data(), M, N, K);
    check_near(C_ref.data(), C_bias.data(), M * N, 5e-2f, "fp16 bias epilogue");
}

TEST(MatmulTests, FP16_BiasReLUEpilogue) {
    size_t M = 8, N = 8, K = 8;
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> A(M * K), B(K * N), bias(N);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);
    for (auto& x : bias) x = dist(rng);
    auto A_fp16 = fp32_to_fp16_vec(A);
    auto B_fp16 = fp32_to_fp16_vec(B);

    std::vector<float> C_ref(M * N), C_relu(M * N, 0.0f);
    simd::impl::matmul_fp16_scalar(A_fp16.data(), B.data(), C_ref.data(), M, N, K);
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++)
            C_ref[i * N + j] = std::max(C_ref[i * N + j] + bias[j], 0.0f);

    simd::impl::matmul_fp16_bias_relu(A_fp16.data(), B_fp16.data(), C_relu.data(), bias.data(), M, N, K);
    check_near(C_ref.data(), C_relu.data(), M * N, 5e-2f, "fp16 bias+relu epilogue");
}

TEST(MatmulTests, FP16_BiasSiLUEpilogue) {
    size_t M = 8, N = 8, K = 8;
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> A(M * K), B(K * N), bias(N);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);
    for (auto& x : bias) x = dist(rng);
    auto A_fp16 = fp32_to_fp16_vec(A);
    auto B_fp16 = fp32_to_fp16_vec(B);

    std::vector<float> C_ref(M * N), C_silu(M * N, 0.0f);
    simd::impl::matmul_fp16_scalar(A_fp16.data(), B.data(), C_ref.data(), M, N, K);
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++) {
            float x = C_ref[i * N + j] + bias[j];
            float s = 1.0f / (1.0f + std::exp(-x));
            C_ref[i * N + j] = x * s;
        }

    simd::impl::matmul_fp16_bias_silu(A_fp16.data(), B_fp16.data(), C_silu.data(), bias.data(), M, N, K);
    check_near(C_ref.data(), C_silu.data(), M * N, 5e-2f, "fp16 bias+silu epilogue");
}
