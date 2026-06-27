#pragma once
#include "../../common.hpp"
#include "simd.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <omp.h>

namespace simd {
namespace impl {

#if defined(SIMD_AVX2_ENABLED) && defined(__FMA__) && defined(SIMD_F16C_ENABLED)

// ---------------------------------------------------------------------------
// FP16 conversion utilities
// ---------------------------------------------------------------------------

inline __m256 fp16_to_fp32_x8(const uint16_t* ptr) {
    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
}

inline void fp32_to_fp16_x8(const float* src, uint16_t* dst) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst),
                     _mm256_cvtps_ph(_mm256_loadu_ps(src), 0));
}

// Portable scalar FP16 → FP32 (for scalar fallback).
// Reinterprets uint16_t as IEEE 754 half-precision.
inline float fp16_to_scalar(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((127 - 15 + exp) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

// ---------------------------------------------------------------------------
// Scalar reference: matmul_fp16_scalar
// A is M×K in FP16 (row-major, uint16_t), B is K×N in FP32, C is M×N in FP32.
// ---------------------------------------------------------------------------
inline void matmul_fp16_scalar(const uint16_t* A_fp16, const float* B, float* C,
                                size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum = std::fma(fp16_to_scalar(A_fp16[i * K + k]),
                               B[k * N + j], sum);
            }
            C[i * N + j] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// FP16 pack functions — convert FP16 → FP32 into existing packed layout.
// The microkernel then operates entirely on FP32 packed data.
// ---------------------------------------------------------------------------

// Pack A tile: FP16 row-major → FP32 column-panel (MC-major, same as pack_A_tile).
inline void pack_A_tile_fp16(const uint16_t* A_fp16, float* packed,
                              size_t K, size_t io, size_t ko,
                              size_t mc, size_t kc) {
    for (size_t kk = 0; kk < kc; kk++) {
        size_t kk_offset = (io * K + ko + kk);  // base for this column
        for (size_t ii = 0; ii < mc; ii++) {
            packed[kk * mc + ii] = fp16_to_scalar(A_fp16[(io + ii) * K + (ko + kk)]);
        }
    }
}

// Pack B tile: FP16 row-major → FP32 row-panel (NR-padded, same as pack_B_tile).
inline void pack_B_tile_fp16(const uint16_t* B_fp16, float* packed,
                              size_t N, size_t ko, size_t jo,
                              size_t kc, size_t nr) {
    for (size_t kk = 0; kk < kc; kk++) {
        size_t jj;
        for (jj = 0; jj + 7 < nr; jj += 8) {
            __m256 b_f32 = fp16_to_fp32_x8(&B_fp16[(ko + kk) * N + (jo + jj)]);
            _mm256_storeu_ps(&packed[kk * NR + jj], b_f32);
        }
        for (; jj < nr; jj++) {
            packed[kk * NR + jj] = fp16_to_scalar(B_fp16[(ko + kk) * N + (jo + jj)]);
        }
        for (jj = nr; jj < NR; jj++) {
            packed[kk * NR + jj] = 0.0f;
        }
    }
}

// SIMD-vectorised A pack (converts 8 FP16 → 8 FP32 per step).
inline void pack_A_tile_fp16_simd(const uint16_t* A_fp16, float* packed,
                                   size_t K, size_t io, size_t ko,
                                   size_t mc, size_t kc) {
    for (size_t kk = 0; kk < kc; kk++) {
        size_t ii = 0;
        for (; ii + 7 < mc; ii += 8) {
            __m256 a_f32 = fp16_to_fp32_x8(&A_fp16[(io + ii) * K + (ko + kk)]);
            _mm256_storeu_ps(&packed[kk * mc + ii], a_f32);
        }
        for (; ii < mc; ii++) {
            packed[kk * mc + ii] = fp16_to_scalar(A_fp16[(io + ii) * K + (ko + kk)]);
        }
    }
}

// ---------------------------------------------------------------------------
// FP16 tiled GEMM — matmul_fp16_simd
// A_fp16: M×K (FP16, row-major), B_fp16: K×N (FP16, row-major), C: M×N (FP32).
// Internally: pack FP16 → FP32, then run the standard FP32 microkernel.
// ---------------------------------------------------------------------------
template <size_t MC = 32, size_t NC = 32, size_t KC = 64>
inline void matmul_fp16_simd(const uint16_t* A_fp16, const uint16_t* B_fp16,
                              float* C, size_t M, size_t N, size_t K) {
    #pragma omp parallel
    {
        alignas(32) std::array<float, MC * KC> packed_A;
        alignas(32) std::array<float, KC * NR> packed_B;

        #pragma omp for schedule(dynamic)
        for (size_t io = 0; io < M; io += MC) {
            size_t mc = std::min(MC, M - io);
            size_t mc_main = mc - (mc % MR);
            for (size_t ko = 0; ko < K; ko += KC) {
                size_t kc = std::min(KC, K - ko);
                pack_A_tile_fp16(A_fp16, packed_A.data(), K, io, ko, mc_main, kc);

                for (size_t jo = 0; jo < N; jo += NC) {
                    size_t nc = std::min(NC, N - jo);
                    for (size_t jj = 0; jj < nc; jj += NR) {
                        size_t nr = std::min(NR, nc - jj);
                        pack_B_tile_fp16(B_fp16, packed_B.data(), N, ko, jo + jj, kc, nr);
                        for (size_t ii = 0; ii < mc_main; ii += MR) {
                            microkernel_4x8_packed(packed_A.data() + ii,
                                                    packed_B.data(), C, N,
                                                    io + ii, jo + jj,
                                                    kc, mc_main, nr);
                        }
                    }
                    // Scalar tail: remaining rows (mc_main < mc)
                    for (size_t r = mc_main; r < mc; r++) {
                        for (size_t kk = 0; kk < kc; kk++) {
                            float a_val = fp16_to_scalar(A_fp16[(io + r) * K + (ko + kk)]);
                            for (size_t jj = 0; jj < nc; jj++) {
                                C[(io + r) * N + (jo + jj)] +=
                                    a_val * fp16_to_scalar(B_fp16[(ko + kk) * N + (jo + jj)]);
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FP16 M=1 specialised kernel — matmul_fp16_1x8
// ---------------------------------------------------------------------------
inline void matmul_fp16_1x8(const uint16_t* A_fp16, const uint16_t* B_fp16,
                             float* C, size_t N, size_t K) {
    size_t j = 0;
    for (; j + 7 < N; j += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (size_t k = 0; k < K; k++) {
            float a_val = fp16_to_scalar(A_fp16[k]);
            __m256 a = _mm256_broadcast_ss(&a_val);
            __m256 b = fp16_to_fp32_x8(&B_fp16[k * N + j]);
            acc = _mm256_fmadd_ps(a, b, acc);
        }
        _mm256_storeu_ps(&C[j], acc);
    }
    for (; j < N; j++) {
        float sum = 0.0f;
        for (size_t k = 0; k < K; k++) {
            sum = std::fma(fp16_to_scalar(A_fp16[k]),
                           fp16_to_scalar(B_fp16[k * N + j]), sum);
        }
        C[j] = sum;
    }
}

// ---------------------------------------------------------------------------
// FP16 batched — matmul_fp16_batched
// ---------------------------------------------------------------------------
template <size_t NC = 32>
inline void matmul_fp16_batched(const uint16_t* A_fp16, const uint16_t* B_fp16,
                                 float* C, size_t M, size_t N, size_t K,
                                 size_t batch_count,
                                 int64_t stride_A, int64_t stride_B, int64_t stride_C) {
    for (size_t b = 0; b < batch_count; b++) {
        if (M == 1) {
            matmul_fp16_1x8(A_fp16 + b * stride_A,
                            B_fp16 + b * stride_B,
                            C + b * stride_C, N, K);
        } else {
            matmul_fp16_simd<NC>(A_fp16 + b * stride_A,
                                 B_fp16 + b * stride_B,
                                 C + b * stride_C, M, N, K);
        }
    }
}

// ---------------------------------------------------------------------------
// FP16 Fused Epilogues — matmul_fp16_packed_epilogue
// Uses the same epilogue structs from simd.hpp (BiasEpilogue, etc.).
// ---------------------------------------------------------------------------
template <typename Epilogue = IdentityEpilogue, size_t MC = 32, size_t NC = 32, size_t KC = 64>
inline void matmul_fp16_packed_epilogue(const uint16_t* A_fp16, const uint16_t* B_fp16,
                                         float* C, const float* bias,
                                         size_t M, size_t N, size_t K) {
    #pragma omp parallel
    {
        alignas(32) std::array<float, MC * KC> packed_A;
        alignas(32) std::array<float, KC * NR> packed_B;

        #pragma omp for schedule(dynamic)
        for (size_t io = 0; io < M; io += MC) {
            size_t mc = std::min(MC, M - io);
            size_t mc_main = mc - (mc % MR);
            for (size_t ko = 0; ko < K; ko += KC) {
                size_t kc = std::min(KC, K - ko);
                pack_A_tile_fp16(A_fp16, packed_A.data(), K, io, ko, mc_main, kc);
                for (size_t jo = 0; jo < N; jo += NC) {
                    size_t nc = std::min(NC, N - jo);
                    for (size_t jj = 0; jj < nc; jj += NR) {
                        size_t nr = std::min(NR, nc - jj);
                        pack_B_tile_fp16(B_fp16, packed_B.data(), N, ko, jo + jj, kc, nr);
                        for (size_t ii = 0; ii < mc_main; ii += MR) {
                            microkernel_4x8_packed_epilogue<Epilogue>(
                                packed_A.data() + ii, packed_B.data(),
                                C, bias, N, io + ii, jo + jj,
                                kc, mc_main, nr);
                        }
                    }
                    // Scalar tail
                    for (size_t r = mc_main; r < mc; r++) {
                        for (size_t kk = 0; kk < kc; kk++) {
                            float a_val = fp16_to_scalar(A_fp16[(io + r) * K + (ko + kk)]);
                            for (size_t jj = 0; jj < nc; jj++) {
                                float val = C[(io + r) * N + (jo + jj)] +
                                    a_val * fp16_to_scalar(B_fp16[(ko + kk) * N + (jo + jj)]);
                                if constexpr (std::is_same_v<Epilogue, BiasReLUEpilogue>) {
                                    if (bias) val += bias[jj];
                                    val = std::max(val, 0.0f);
                                } else if constexpr (std::is_same_v<Epilogue, BiasEpilogue>) {
                                    if (bias) val += bias[jj];
                                } else if constexpr (std::is_same_v<Epilogue, BiasSiLUEpilogue>) {
                                    if (bias) val += bias[jj];
                                    float s = 1.0f / (1.0f + std::exp(-val));
                                    val = val * s;
                                } else if constexpr (std::is_same_v<Epilogue, BiasGELUEpilogue>) {
                                    if (bias) val += bias[jj];
                                    float inner = val * 0.7978845608028654f * (1.0f + 0.044715f * val * val);
                                    val = 0.5f * val * (1.0f + std::tanh(inner));
                                }
                                C[(io + r) * N + (jo + jj)] = val;
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience wrappers for fused epilogues
// ---------------------------------------------------------------------------
template <size_t MC = 32, size_t NC = 32, size_t KC = 64>
inline void matmul_fp16_bias(const uint16_t* A_fp16, const uint16_t* B_fp16,
                              float* C, const float* bias,
                              size_t M, size_t N, size_t K) {
    matmul_fp16_packed_epilogue<BiasEpilogue, MC, NC, KC>(A_fp16, B_fp16, C, bias, M, N, K);
}

template <size_t MC = 32, size_t NC = 32, size_t KC = 64>
inline void matmul_fp16_bias_relu(const uint16_t* A_fp16, const uint16_t* B_fp16,
                                   float* C, const float* bias,
                                   size_t M, size_t N, size_t K) {
    matmul_fp16_packed_epilogue<BiasReLUEpilogue, MC, NC, KC>(A_fp16, B_fp16, C, bias, M, N, K);
}

template <size_t MC = 32, size_t NC = 32, size_t KC = 64>
inline void matmul_fp16_bias_silu(const uint16_t* A_fp16, const uint16_t* B_fp16,
                                   float* C, const float* bias,
                                   size_t M, size_t N, size_t K) {
    matmul_fp16_packed_epilogue<BiasSiLUEpilogue, MC, NC, KC>(A_fp16, B_fp16, C, bias, M, N, K);
}

template <size_t MC = 32, size_t NC = 32, size_t KC = 64>
inline void matmul_fp16_bias_gelu(const uint16_t* A_fp16, const uint16_t* B_fp16,
                                   float* C, const float* bias,
                                   size_t M, size_t N, size_t K) {
    matmul_fp16_packed_epilogue<BiasGELUEpilogue, MC, NC, KC>(A_fp16, B_fp16, C, bias, M, N, K);
}

#endif // SIMD_AVX2_ENABLED && __FMA__ && SIMD_F16C_ENABLED

} // namespace impl
} // namespace simd
