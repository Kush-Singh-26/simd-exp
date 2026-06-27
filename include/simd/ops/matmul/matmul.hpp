#pragma once
#include "scalar.hpp"
#include "simd.hpp"
#include "f16.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

namespace simd {

inline void matmul(std::span<const float> A, std::span<const float> B, std::span<float> C, size_t M, size_t N, size_t K) {
    std::fill(C.begin(), C.end(), 0.0f);
#if defined (SIMD_AVX2_ENABLED) && defined(__FMA__)
    if (M == 1) {
        impl::matmul_1x8(A.data(), B.data(), C.data(), N, K);
        return;
    }
    impl::matmul_packed(A.data(), B.data(), C.data(), M, N, K);
#else
    impl::matmul_scalar_tiled(A.data(), B.data(), C.data(), M, N, K);
#endif
}

inline void matmul_f32xi8(std::span<const float> A, std::span<const int8_t> B_i8,
                            std::span<const float> scales, std::span<float> C,
                            size_t M, size_t N, size_t K) {
    std::fill(C.begin(), C.end(), 0.0f);
#if defined (SIMD_AVX2_ENABLED) && defined(__FMA__)
    if (M == 1) {
        impl::matmul_f32xi8_1x8(A.data(), B_i8.data(), scales.data(), C.data(), N, K);
        return;
    }
    impl::matmul_f32xi8_simd(A.data(), B_i8.data(), scales.data(), C.data(), M, N, K);
#else
    impl::matmul_f32xi8_scalar(A.data(), B_i8.data(), scales.data(), C.data(), M, N, K);
#endif
}

inline void matmul_f32xi8_batched(std::span<const float> A, std::span<const int8_t> B_i8,
                                    std::span<const float> scales, std::span<float> C,
                                    size_t M, size_t N, size_t K,
                                    size_t batch_count,
                                    int64_t stride_A, int64_t stride_B, int64_t stride_C) {
    for (size_t b = 0; b < batch_count; b++) {
        std::fill(C.begin() + b * stride_C, C.begin() + b * stride_C + M * N, 0.0f);
    }
#if defined (SIMD_AVX2_ENABLED) && defined(__FMA__)
    impl::matmul_f32xi8_batched(A.data(), B_i8.data(), scales.data(), C.data(),
                                 M, N, K, batch_count, stride_A, stride_B, stride_C);
#else
    for (size_t b = 0; b < batch_count; b++) {
        impl::matmul_f32xi8_scalar(A.data() + b * stride_A, B_i8.data() + b * stride_B,
                                    scales.data(), C.data() + b * stride_C, M, N, K);
    }
#endif
}

// FP16 dispatcher: A_fp16 (M×K), B_fp16 (K×N) → C (M×N, FP32).
// A_fp16 and B_fp16 are in IEEE 754 half-precision (uint16_t).
inline void matmul_fp16(std::span<const uint16_t> A_fp16, std::span<const uint16_t> B_fp16,
                         std::span<float> C, size_t M, size_t N, size_t K) {
    std::fill(C.begin(), C.end(), 0.0f);
#if defined(SIMD_AVX2_ENABLED) && defined(__FMA__) && defined(SIMD_F16C_ENABLED)
    if (M == 1) {
        impl::matmul_fp16_1x8(A_fp16.data(), B_fp16.data(), C.data(), N, K);
        return;
    }
    impl::matmul_fp16_simd(A_fp16.data(), B_fp16.data(), C.data(), M, N, K);
#else
    impl::matmul_fp16_scalar(A_fp16.data(), B_fp16.data(), C.data(), M, N, K);
#endif
}

// FP16 batched: batch_count GEMMs with strided A, B, C.
inline void matmul_fp16_batched(std::span<const uint16_t> A_fp16, std::span<const uint16_t> B_fp16,
                                 std::span<float> C,
                                 size_t M, size_t N, size_t K,
                                 size_t batch_count,
                                 int64_t stride_A, int64_t stride_B, int64_t stride_C) {
    for (size_t b = 0; b < batch_count; b++) {
        std::fill(C.begin() + b * stride_C, C.begin() + b * stride_C + M * N, 0.0f);
    }
#if defined(SIMD_AVX2_ENABLED) && defined(__FMA__) && defined(SIMD_F16C_ENABLED)
    impl::matmul_fp16_batched(A_fp16.data(), B_fp16.data(), C.data(),
                               M, N, K, batch_count, stride_A, stride_B, stride_C);
#else
    for (size_t b = 0; b < batch_count; b++) {
        impl::matmul_fp16_scalar(A_fp16.data() + b * stride_A,
                                  B_fp16.data() + b * stride_B,
                                  C.data() + b * stride_C, M, N, K);
    }
#endif
}

} // namespace simd
