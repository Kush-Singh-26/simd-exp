#pragma once
#include "../../common.hpp"
#include "../../math_utils.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <omp.h>
#include <type_traits>

namespace simd {
namespace impl {

#if defined (SIMD_AVX2_ENABLED) && defined(__FMA__)
template <size_t MC = 32, size_t NC = 32, size_t KC = 64>
inline void matmul_avx2(const float *A, const float *B, float *C, size_t M,
                        size_t N, size_t K) {
  #pragma omp parallel for schedule(dynamic) if(M >= 256)
  for (size_t io = 0; io < M; io += MC) {
    size_t mc = std::min(MC, M - io);
    for (size_t ko = 0; ko < K; ko += KC) {
      size_t kc = std::min(KC, K - ko);
      for (size_t jo = 0; jo < N; jo += NC) {
        size_t nc = std::min(NC, N - jo);
        for (size_t i = 0; i < mc; i++) {
          for (size_t k = 0; k < kc; k++) {
            float a_val = A[(io + i) * K + (ko + k)];
            __m256 a = _mm256_broadcast_ss(&a_val);
            size_t j = 0;
            for (j = 0; j + 8 <= nc; j += 8) {
              __m256 b = _mm256_loadu_ps(&B[(ko + k) * N + (jo + j)]);
              __m256 c = _mm256_loadu_ps(&C[(io + i) * N + (jo + j)]);
              c = _mm256_fmadd_ps(a, b, c);
              _mm256_storeu_ps(&C[(io + i) * N + (jo + j)], c);
            }
            for (; j < nc; j++) {
              // C[(io + i) * N + (jo + j)] += a_val * B[(ko + k) * N + (jo +
              // j)];
              C[(io + i) * N + (jo + j)] =
                  std::fma(a_val, B[(ko + k) * N + (jo + j)],
                           C[(io + i) * N + (jo + j)]);
            }
          }
        }
      }
    }
  }
}

inline void matmul_1x8(const float* A, const float* B, float* C,
                       size_t N, size_t K) {
    size_t j = 0;
    for (; j + 8 <= N; j += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (size_t k = 0; k < K; k++) {
            __m256 a = _mm256_broadcast_ss(&A[k]);
            __m256 b = _mm256_loadu_ps(&B[k * N + j]);
            acc = _mm256_fmadd_ps(a, b, acc);
        }
        _mm256_storeu_ps(&C[j], acc);
    }
    for (; j < N; j++) {
        float sum = 0.0f;
        for (size_t k = 0; k < K; k++) {
            sum = std::fma(A[k], B[k * N + j], sum);
        }
        C[j] = sum;
    }
}

template <size_t MC = 32, size_t NC = 32, size_t KC = 64>
inline void matmul_4x8(const float *A, const float *B, float *C, size_t M,
                       size_t N, size_t K) {
  #pragma omp parallel for schedule(dynamic) if(M >= 256)
  for (size_t io = 0; io < M; io += MC) {
    size_t mc = std::min(MC, M - io);
    for (size_t ko = 0; ko < K; ko += KC) {
      size_t kc = std::min(KC, K - ko);
      for (size_t jo = 0; jo < N; jo += NC) {
            size_t nc = std::min(NC, N - jo);
            size_t i = 0;
            for (; i + 4 <= mc; i += 4) {
          size_t j = 0;
          for (; j + 8 <= nc; j += 8) {
            __m256 acc0 = _mm256_setzero_ps(); // row i
            __m256 acc1 = _mm256_setzero_ps(); // row i + 1
            __m256 acc2 = _mm256_setzero_ps(); // row i + 2
            __m256 acc3 = _mm256_setzero_ps(); // row i + 3

            for (size_t k = 0; k < kc; k++) {

              __m256 b = _mm256_loadu_ps(&B[(ko + k) * N + (jo + j)]);
              
              __m256 a0 = _mm256_broadcast_ss(&A[(io + i + 0) * K + (ko + k)]);
              acc0 = _mm256_fmadd_ps(a0, b, acc0);
              __m256 a1 = _mm256_broadcast_ss(&A[(io + i + 1) * K + (ko + k)]);
              acc1 = _mm256_fmadd_ps(a1, b, acc1);
              __m256 a2 = _mm256_broadcast_ss(&A[(io + i + 2) * K + (ko + k)]);
              acc2 = _mm256_fmadd_ps(a2, b, acc2);
              __m256 a3 = _mm256_broadcast_ss(&A[(io + i + 3) * K + (ko + k)]);
              acc3 = _mm256_fmadd_ps(a3, b, acc3);
            }
            _mm256_storeu_ps(
                &C[(io + i + 0) * N + (jo + j)],
                _mm256_add_ps(_mm256_loadu_ps(&C[(io + i + 0) * N + (jo + j)]),
                              acc0));
            _mm256_storeu_ps(
                &C[(io + i + 1) * N + (jo + j)],
                _mm256_add_ps(_mm256_loadu_ps(&C[(io + i + 1) * N + (jo + j)]),
                              acc1));
            _mm256_storeu_ps(
                &C[(io + i + 2) * N + (jo + j)],
                _mm256_add_ps(_mm256_loadu_ps(&C[(io + i + 2) * N + (jo + j)]),
                              acc2));
            _mm256_storeu_ps(
                &C[(io + i + 3) * N + (jo + j)],
                _mm256_add_ps(_mm256_loadu_ps(&C[(io + i + 3) * N + (jo + j)]),
                              acc3));
          }
          for (; j < nc; j++) {
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (size_t kk = 0; kk < kc; kk++) {
              float b = B[(ko + kk) * N + (jo + j)];
              s0 = std::fma(A[(io + i + 0) * K + (ko + kk)], b, s0);
              s1 = std::fma(A[(io + i + 1) * K + (ko + kk)], b, s1);
              s2 = std::fma(A[(io + i + 2) * K + (ko + kk)], b, s2);
              s3 = std::fma(A[(io + i + 3) * K + (ko + kk)], b, s3);
            }
            C[(io + i + 0) * N + (jo + j)] += s0;
            C[(io + i + 1) * N + (jo + j)] += s1;
            C[(io + i + 2) * N + (jo + j)] += s2;
            C[(io + i + 3) * N + (jo + j)] += s3;
          }
        }
        for (; i < mc; i++) {
            for (size_t k = 0; k < kc; k++) {
                float a_val = A[(io+i)*K + (ko+k)];
                __m256 a = _mm256_broadcast_ss(&a_val);
                size_t j = 0;
                for (; j + 8 <= nc; j += 8) {
                    __m256 b = _mm256_loadu_ps(&B[(ko+k)*N + (jo+j)]);
                    __m256 c = _mm256_loadu_ps(&C[(io+i)*N + (jo+j)]);
                    c = _mm256_fmadd_ps(a, b, c);
                    _mm256_storeu_ps(&C[(io+i)*N + (jo+j)], c);
                }
                for (; j < nc; j++) {
                    C[(io+i)*N + (jo+j)] = std::fma(a_val, B[(ko+k)*N + (jo+j)], C[(io+i)*N + (jo+j)]);
                }
            }
        }
      }
    }
  }
}

/*
    For mc=4, kc=4, io=0, ko=0, original A tile:
         k=0  k=1  k=2  k=3
    i0:  1.0  2.0  3.0  4.0
    i1:  5.0  6.0  7.0  8.0
    i2:  9.0 10.0 11.0 12.0
    i3: 13.0 14.0 15.0 16.0
    Packed output (4 groups of 4):
    offset:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9] [10] [11] [12] [13] [14] [15]
    content: 1.0  5.0  9.0 13.0  2.0  6.0 10.0 14.0  3.0  7.0 11.0 15.0  4.0  8.0 12.0 16.0
             └──── kk=0 ──────┘    └── kk=1 ─────┘    └── kk=2 ─────┘    └─── kk=3 ────┘
 */
inline void pack_A_tile(const float* A, float* packed, size_t K, size_t io, size_t ko, size_t mc, size_t kc) {
    for(size_t kk = 0; kk < kc; kk++){
        for(size_t ii = 0; ii < mc; ii++){
            packed[kk * mc + ii] = A[(io + ii) * K + (ko + kk)];
        }
    }
}

/*
    For kc=4, nr=4, ko=0, jo=2, original B (say full N=8):
         j=0  j=1  j=2  j=3  j=4  j=5  j=6  j=7
    k=0:  1    2    3    4    5    6    7    8
    k=1:  9   10   11   12   13   14   15   16
    k=2: 17   18   19   20   21   22   23   24
    k=3: 25   26   27   28   29   30   31   32
    
    The tile B[0:4, 2:6] (jo=2, nr=4):
    k=0:  3    4    5    6
    k=1: 11   12   13   14
    k=2: 19   20   21   22
    k=3: 27   28   29   30
    
    Packed output (4 groups of nr=4, but allocated as NR=8 per row):
    offset:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9] [10] [11] ...
    content:  3    4    5    6    0    0    0    0   11   12   13   14  ...
              └── kk=0 (nr=4) ──┘  └── padding ──┘  └── kk=1 ──┘
 */

constexpr size_t NR = 8;
constexpr size_t MR = 4;
inline void pack_B_tile(const float* B, float* packed, size_t N, size_t ko, size_t jo, size_t kc, size_t nr) {
    for(size_t kk = 0; kk < kc; kk++){
        for(size_t jj = 0; jj < nr; jj++){
            packed[kk * NR + jj] = B[(ko + kk) * N + (jo + jj)];
        }
        for (size_t jj = nr; jj < NR; jj++) {
                    packed[kk * NR + jj] = 0.0f;
        }
    }
}

inline void microkernel_4x8_packed(const float* A_packed, const float* B_packed, float* C, size_t N, size_t i, size_t j, size_t kc, size_t mc, size_t nr) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    for (size_t kk = 0; kk < kc; kk++) {
        __m256 b = _mm256_loadu_ps(&B_packed[kk * NR]);
        __m256 a0 = _mm256_broadcast_ss(&A_packed[kk * mc + 0]);
        acc0 = _mm256_fmadd_ps(a0, b, acc0);
        __m256 a1 = _mm256_broadcast_ss(&A_packed[kk * mc + 1]);
        acc1 = _mm256_fmadd_ps(a1, b, acc1);
        __m256 a2 = _mm256_broadcast_ss(&A_packed[kk * mc + 2]);
        acc2 = _mm256_fmadd_ps(a2, b, acc2);
        __m256 a3 = _mm256_broadcast_ss(&A_packed[kk * mc + 3]);
        acc3 = _mm256_fmadd_ps(a3, b, acc3);
    }
    if (nr == 8) {
        _mm256_storeu_ps(&C[(i+0) * N + j],
            _mm256_add_ps(_mm256_loadu_ps(&C[(i+0) * N + j]), acc0));
        _mm256_storeu_ps(&C[(i+1) * N + j],
            _mm256_add_ps(_mm256_loadu_ps(&C[(i+1) * N + j]), acc1));
        _mm256_storeu_ps(&C[(i+2) * N + j],
            _mm256_add_ps(_mm256_loadu_ps(&C[(i+2) * N + j]), acc2));
        _mm256_storeu_ps(&C[(i+3) * N + j],
            _mm256_add_ps(_mm256_loadu_ps(&C[(i+3) * N + j]), acc3));
    } else {
        __m256i mask = _mm256_setr_epi32(
            nr > 0 ? -1 : 0, nr > 1 ? -1 : 0,
            nr > 2 ? -1 : 0, nr > 3 ? -1 : 0,
            nr > 4 ? -1 : 0, nr > 5 ? -1 : 0,
            nr > 6 ? -1 : 0, nr > 7 ? -1 : 0);
        __m256 c0 = _mm256_maskload_ps(&C[(i+0) * N + j], mask);
        __m256 c1 = _mm256_maskload_ps(&C[(i+1) * N + j], mask);
        __m256 c2 = _mm256_maskload_ps(&C[(i+2) * N + j], mask);
        __m256 c3 = _mm256_maskload_ps(&C[(i+3) * N + j], mask);
        _mm256_maskstore_ps(&C[(i+0) * N + j], mask, _mm256_add_ps(c0, acc0));
        _mm256_maskstore_ps(&C[(i+1) * N + j], mask, _mm256_add_ps(c1, acc1));
        _mm256_maskstore_ps(&C[(i+2) * N + j], mask, _mm256_add_ps(c2, acc2));
        _mm256_maskstore_ps(&C[(i+3) * N + j], mask, _mm256_add_ps(c3, acc3));
    }
}

template <size_t MC = 32, size_t NC = 32, size_t KC = 64>
inline void matmul_packed(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {

    #pragma omp parallel
    {
        alignas(32) std::array<float, MC * KC> packed_A;
        alignas(32) std::array<float, KC * NR> packed_B;

        #pragma omp for schedule(dynamic)
        for(size_t io = 0; io < M; io += MC){
            size_t mc = std::min(MC, M - io);
            size_t mc_main = mc - (mc % MR);
            for(size_t ko = 0; ko < K; ko += KC){
                size_t kc = std::min(KC, K - ko);
                pack_A_tile(A, packed_A.data(), K, io, ko, mc_main, kc);

                for(size_t jo = 0; jo < N; jo += NC) {
                    size_t nc = std::min(NC, N - jo);

                    for(size_t jj = 0; jj < nc; jj += NR){
                        size_t nr = std::min(NR, nc - jj);
                        pack_B_tile(B, packed_B.data(), N, ko, jo+jj, kc, nr);

                        for(size_t ii = 0; ii < mc_main; ii += MR){
                            microkernel_4x8_packed(packed_A.data() + ii, packed_B.data(), C, N, io + ii, jo + jj, kc, mc_main, nr);
                        }
                    }
                    for (size_t r = mc_main; r < mc; r++) {
                        for (size_t kk = 0; kk < kc; kk++) {
                            float a_val = A[(io + r) * K + (ko + kk)];
                            for (size_t jj = 0; jj < nc; jj++) {
                                C[(io + r) * N + (jo + jj)] +=
                                    a_val * B[(ko + kk) * N + (jo + jj)];
                            }
                        }
                    }
                }
            }
        }
    }
}

inline void microkernel_4x8_packed_prefetch(const float* A_packed, const float* B_packed, float* C, size_t N, size_t i, size_t j, size_t kc, size_t mc, size_t nr) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    constexpr size_t PREFETCH_DIST = 2;

    if (kc > PREFETCH_DIST) {
        _mm_prefetch((const char*)&B_packed[PREFETCH_DIST * NR], _MM_HINT_NTA);
        _mm_prefetch((const char*)&A_packed[PREFETCH_DIST * mc + 0], _MM_HINT_NTA);
    }

    for (size_t kk = 0; kk < kc; kk++) {
        if (kk + PREFETCH_DIST < kc) {
            _mm_prefetch((const char*)&B_packed[(kk + PREFETCH_DIST) * NR], _MM_HINT_NTA);
            _mm_prefetch((const char*)&A_packed[(kk + PREFETCH_DIST) * mc + 0], _MM_HINT_NTA);
        }

        __m256 b = _mm256_loadu_ps(&B_packed[kk * NR]);
        __m256 a0 = _mm256_broadcast_ss(&A_packed[kk * mc + 0]);
        acc0 = _mm256_fmadd_ps(a0, b, acc0);
        __m256 a1 = _mm256_broadcast_ss(&A_packed[kk * mc + 1]);
        acc1 = _mm256_fmadd_ps(a1, b, acc1);
        __m256 a2 = _mm256_broadcast_ss(&A_packed[kk * mc + 2]);
        acc2 = _mm256_fmadd_ps(a2, b, acc2);
        __m256 a3 = _mm256_broadcast_ss(&A_packed[kk * mc + 3]);
        acc3 = _mm256_fmadd_ps(a3, b, acc3);
    }
    if (nr == 8) {
        _mm256_storeu_ps(&C[(i+0) * N + j],
            _mm256_add_ps(_mm256_loadu_ps(&C[(i+0) * N + j]), acc0));
        _mm256_storeu_ps(&C[(i+1) * N + j],
            _mm256_add_ps(_mm256_loadu_ps(&C[(i+1) * N + j]), acc1));
        _mm256_storeu_ps(&C[(i+2) * N + j],
            _mm256_add_ps(_mm256_loadu_ps(&C[(i+2) * N + j]), acc2));
        _mm256_storeu_ps(&C[(i+3) * N + j],
            _mm256_add_ps(_mm256_loadu_ps(&C[(i+3) * N + j]), acc3));
    } else {
        __m256i mask = _mm256_setr_epi32(
            nr > 0 ? -1 : 0, nr > 1 ? -1 : 0,
            nr > 2 ? -1 : 0, nr > 3 ? -1 : 0,
            nr > 4 ? -1 : 0, nr > 5 ? -1 : 0,
            nr > 6 ? -1 : 0, nr > 7 ? -1 : 0);
        __m256 c0 = _mm256_maskload_ps(&C[(i+0) * N + j], mask);
        __m256 c1 = _mm256_maskload_ps(&C[(i+1) * N + j], mask);
        __m256 c2 = _mm256_maskload_ps(&C[(i+2) * N + j], mask);
        __m256 c3 = _mm256_maskload_ps(&C[(i+3) * N + j], mask);
        _mm256_maskstore_ps(&C[(i+0) * N + j], mask, _mm256_add_ps(c0, acc0));
        _mm256_maskstore_ps(&C[(i+1) * N + j], mask, _mm256_add_ps(c1, acc1));
        _mm256_maskstore_ps(&C[(i+2) * N + j], mask, _mm256_add_ps(c2, acc2));
        _mm256_maskstore_ps(&C[(i+3) * N + j], mask, _mm256_add_ps(c3, acc3));
    }
}

template <size_t MC = 32, size_t NC = 32, size_t KC = 64>
inline void matmul_packed_prefetch(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {

    #pragma omp parallel
    {
        alignas(32) std::array<float, MC * KC> packed_A;
        alignas(32) std::array<float, KC * NR> packed_B;

        #pragma omp for schedule(dynamic)
        for(size_t io = 0; io < M; io += MC){
            size_t mc = std::min(MC, M - io);
            size_t mc_main = mc - (mc % MR);
            for(size_t ko = 0; ko < K; ko += KC){
                size_t kc = std::min(KC, K - ko);
                pack_A_tile(A, packed_A.data(), K, io, ko, mc_main, kc);

                for(size_t jo = 0; jo < N; jo += NC) {
                    size_t nc = std::min(NC, N - jo);

                    for(size_t jj = 0; jj < nc; jj += NR){
                        size_t nr = std::min(NR, nc - jj);
                        pack_B_tile(B, packed_B.data(), N, ko, jo+jj, kc, nr);

                        for(size_t ii = 0; ii < mc_main; ii += MR){
                            microkernel_4x8_packed_prefetch(packed_A.data() + ii, packed_B.data(), C, N, io + ii, jo + jj, kc, mc_main, nr);
                        }
                    }
                    for (size_t r = mc_main; r < mc; r++) {
                        for (size_t kk = 0; kk < kc; kk++) {
                            float a_val = A[(io + r) * K + (ko + kk)];
                            for (size_t jj = 0; jj < nc; jj++) {
                                C[(io + r) * N + (jo + jj)] +=
                                    a_val * B[(ko + kk) * N + (jo + jj)];
                            }
                        }
                    }
                }
            }
        }
    }
}

template <size_t MC = 32, size_t NC = 32, size_t KC = 64>
inline void matmul_strided_batched(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K,
    size_t batch_count,
    int64_t stride_A, int64_t stride_B, int64_t stride_C)
{
    for (size_t b = 0; b < batch_count; b++) {
        matmul_packed<MC, NC, KC>(
            A + b * stride_A, B + b * stride_B, C + b * stride_C,
            M, N, K);
    }
}

template <size_t MC = 32, size_t NC = 32, size_t KC = 64>
inline void matmul_strided_batched_shared_B(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K,
    size_t batch_count,
    int64_t stride_A, int64_t stride_B, int64_t stride_C)
{
    for (size_t b = 0; b < batch_count; b++) {
        const float* A_b = A + b * stride_A;
        const float* B_b = B + b * stride_B;
        float* C_b = C + b * stride_C;
        std::fill(C_b, C_b + M * N, 0.0f);

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
                    pack_A_tile(A_b, packed_A.data(), K, io, ko, mc_main, kc);

                    for (size_t jo = 0; jo < N; jo += NC) {
                        size_t nc = std::min(NC, N - jo);
                        for (size_t jj = 0; jj < nc; jj += NR) {
                            size_t nr = std::min(NR, nc - jj);
                            pack_B_tile(B_b, packed_B.data(), N, ko, jo + jj, kc, nr);

                            for (size_t ii = 0; ii < mc_main; ii += MR) {
                                microkernel_4x8_packed(
                                    packed_A.data() + ii, packed_B.data(),
                                    C_b, N, io + ii, jo + jj, kc, mc_main, nr);
                            }
                        }
                        for (size_t r = mc_main; r < mc; r++) {
                            for (size_t kk = 0; kk < kc; kk++) {
                                float a_val = A_b[(io + r) * K + (ko + kk)];
                                for (size_t jj = 0; jj < nc; jj++) {
                                    C_b[(io + r) * N + (jo + jj)] +=
                                        a_val * B_b[(ko + kk) * N + (jo + jj)];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

inline void matmul_f32xi8_microkernel(const float* A, const int8_t* B_i8,
                                       const float* scales, float* C,
                                       size_t N, size_t K,
                                       size_t i, size_t j, size_t nc) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    size_t k = 0;
    for (; k + 1 <= K; k++) {
        __m128i b8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&B_i8[k * N + j]));
        __m256i b32_lo = _mm256_cvtepi8_epi32(b8);
        __m256 b_f32 = _mm256_cvtepi32_ps(b32_lo);

        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&A[(i + 0) * K + k]), b_f32, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&A[(i + 1) * K + k]), b_f32, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&A[(i + 2) * K + k]), b_f32, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&A[(i + 3) * K + k]), b_f32, acc3);
    }

    __m256 scale = _mm256_loadu_ps(&scales[j]);
    acc0 = _mm256_mul_ps(acc0, scale);
    acc1 = _mm256_mul_ps(acc1, scale);
    acc2 = _mm256_mul_ps(acc2, scale);
    acc3 = _mm256_mul_ps(acc3, scale);

    __m256 c0 = _mm256_loadu_ps(&C[(i + 0) * N + j]);
    __m256 c1 = _mm256_loadu_ps(&C[(i + 1) * N + j]);
    __m256 c2 = _mm256_loadu_ps(&C[(i + 2) * N + j]);
    __m256 c3 = _mm256_loadu_ps(&C[(i + 3) * N + j]);
    _mm256_storeu_ps(&C[(i + 0) * N + j], _mm256_add_ps(c0, acc0));
    _mm256_storeu_ps(&C[(i + 1) * N + j], _mm256_add_ps(c1, acc1));
    _mm256_storeu_ps(&C[(i + 2) * N + j], _mm256_add_ps(c2, acc2));
    _mm256_storeu_ps(&C[(i + 3) * N + j], _mm256_add_ps(c3, acc3));
}

template <size_t NC = 32>
inline void matmul_f32xi8_simd(const float* A, const int8_t* B_i8, const float* scales,
                                float* C, size_t M, size_t N, size_t K) {
    #pragma omp parallel for schedule(dynamic) if(M >= 256)
    for (size_t jo = 0; jo < N; jo += NC) {
        size_t nc = std::min(NC, N - jo);
        for (size_t i = 0; i + 3 < M; i += 4) {
            size_t j = jo;
            for (; j + 7 < jo + nc; j += 8) {
                matmul_f32xi8_microkernel(A, B_i8, scales, C, N, K, i, j, nc);
            }
            for (; j < jo + nc; j++) {
                for (size_t ii = 0; ii < 4; ii++) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < K; k++) {
                        sum = std::fma(A[(i + ii) * K + k], static_cast<float>(B_i8[k * N + j]), sum);
                    }
                    C[(i + ii) * N + j] += sum * scales[j];
                }
            }
        }
        for (size_t i = (M / 4) * 4; i < M; i++) {
            for (size_t j = jo; j < jo + nc; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; k++) {
                    sum = std::fma(A[i * K + k], static_cast<float>(B_i8[k * N + j]), sum);
                }
                C[i * N + j] += sum * scales[j];
            }
        }
    }
}

inline void matmul_f32xi8_1x8(const float* A, const int8_t* B_i8, const float* scales,
                                float* C, size_t N, size_t K) {
    size_t j = 0;
    for (; j + 7 < N; j += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (size_t k = 0; k < K; k++) {
            __m128i b8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&B_i8[k * N + j]));
            __m256i b32_lo = _mm256_cvtepi8_epi32(b8);
            __m256 b_f32 = _mm256_cvtepi32_ps(b32_lo);
            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&A[k]), b_f32, acc);
        }
        __m256 scale = _mm256_loadu_ps(&scales[j]);
        _mm256_storeu_ps(&C[j], _mm256_mul_ps(acc, scale));
    }
    for (; j < N; j++) {
        float sum = 0.0f;
        for (size_t k = 0; k < K; k++) {
            sum = std::fma(A[k], static_cast<float>(B_i8[k * N + j]), sum);
        }
        C[j] = sum * scales[j];
    }
}

template <size_t NC = 32>
inline void matmul_f32xi8_batched(const float* A, const int8_t* B_i8, const float* scales,
                                   float* C, size_t M, size_t N, size_t K,
                                   size_t batch_count,
                                   int64_t stride_A, int64_t stride_B, int64_t stride_C) {
    for (size_t b = 0; b < batch_count; b++) {
        if (M == 1) {
            matmul_f32xi8_1x8(A + b * stride_A, B_i8 + b * stride_B, scales,
                               C + b * stride_C, N, K);
        } else {
            matmul_f32xi8_simd<NC>(A + b * stride_A, B_i8 + b * stride_B, scales,
                                    C + b * stride_C, M, N, K);
        }
    }
}

// Step 13: Fused Epilogues

struct IdentityEpilogue {
    static __m256 apply(__m256 acc, const float*, size_t) { return acc; }
};

struct BiasEpilogue {
    static __m256 apply(__m256 acc, const float* bias, size_t j) {
        return _mm256_add_ps(acc, _mm256_loadu_ps(&bias[j]));
    }
};

struct BiasReLUEpilogue {
    static __m256 apply(__m256 acc, const float* bias, size_t j) {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(&bias[j]));
        return _mm256_max_ps(acc, _mm256_setzero_ps());
    }
};

inline __m256 fast_sigmoid_ps(__m256 x) {
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 exp_neg = avx2_exp_ps(neg_x);
    __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));
}

struct BiasSiLUEpilogue {
    static __m256 apply(__m256 acc, const float* bias, size_t j) {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(&bias[j]));
        __m256 sigmoid = fast_sigmoid_ps(acc);
        return _mm256_mul_ps(acc, sigmoid);
    }
};

inline __m256 fast_gelu_ps(__m256 x) {
    __m256 c1 = _mm256_set1_ps(0.7978845608028654f);
    __m256 c2 = _mm256_set1_ps(0.044715f);
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 two = _mm256_set1_ps(2.0f);

    __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
    __m256 inner = _mm256_mul_ps(_mm256_fmadd_ps(x3, c2, x), c1);
    __m256 neg_two_inner = _mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(two, inner));
    __m256 exp_neg_2inner = avx2_exp_ps(neg_two_inner);
    __m256 tanh_val = _mm256_sub_ps(_mm256_mul_ps(two,
                        _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_2inner))), one);
    return _mm256_mul_ps(x, _mm256_mul_ps(half, _mm256_add_ps(one, tanh_val)));
}

struct BiasGELUEpilogue {
    static __m256 apply(__m256 acc, const float* bias, size_t j) {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(&bias[j]));
        return fast_gelu_ps(acc);
    }
};

template <typename Epilogue = IdentityEpilogue>
inline void microkernel_4x8_packed_epilogue(const float* A_packed, const float* B_packed,
                                             float* C, const float* bias,
                                             size_t N, size_t i, size_t j,
                                             size_t kc, size_t mc, size_t nr) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    for (size_t kk = 0; kk < kc; kk++) {
        __m256 b = _mm256_loadu_ps(&B_packed[kk * NR]);
        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&A_packed[kk * mc + 0]), b, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&A_packed[kk * mc + 1]), b, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&A_packed[kk * mc + 2]), b, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&A_packed[kk * mc + 3]), b, acc3);
    }
    if (nr == 8) {
        _mm256_storeu_ps(&C[(i+0) * N + j], Epilogue::apply(acc0, bias, j));
        _mm256_storeu_ps(&C[(i+1) * N + j], Epilogue::apply(acc1, bias, j));
        _mm256_storeu_ps(&C[(i+2) * N + j], Epilogue::apply(acc2, bias, j));
        _mm256_storeu_ps(&C[(i+3) * N + j], Epilogue::apply(acc3, bias, j));
    } else {
        __m256i mask = _mm256_setr_epi32(
            nr > 0 ? -1 : 0, nr > 1 ? -1 : 0,
            nr > 2 ? -1 : 0, nr > 3 ? -1 : 0,
            nr > 4 ? -1 : 0, nr > 5 ? -1 : 0,
            nr > 6 ? -1 : 0, nr > 7 ? -1 : 0);
        __m256 c0 = _mm256_maskload_ps(&C[(i+0) * N + j], mask);
        __m256 c1 = _mm256_maskload_ps(&C[(i+1) * N + j], mask);
        __m256 c2 = _mm256_maskload_ps(&C[(i+2) * N + j], mask);
        __m256 c3 = _mm256_maskload_ps(&C[(i+3) * N + j], mask);
        _mm256_maskstore_ps(&C[(i+0) * N + j], mask, Epilogue::apply(_mm256_add_ps(c0, acc0), bias, j));
        _mm256_maskstore_ps(&C[(i+1) * N + j], mask, Epilogue::apply(_mm256_add_ps(c1, acc1), bias, j));
        _mm256_maskstore_ps(&C[(i+2) * N + j], mask, Epilogue::apply(_mm256_add_ps(c2, acc2), bias, j));
        _mm256_maskstore_ps(&C[(i+3) * N + j], mask, Epilogue::apply(_mm256_add_ps(c3, acc3), bias, j));
    }
}

template <typename Epilogue = IdentityEpilogue, size_t MC = 32, size_t NC = 32, size_t KC = 64>
inline void matmul_packed_epilogue(const float* A, const float* B, float* C,
                                    const float* bias,
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
                pack_A_tile(A, packed_A.data(), K, io, ko, mc_main, kc);
                for (size_t jo = 0; jo < N; jo += NC) {
                    size_t nc = std::min(NC, N - jo);
                    for (size_t jj = 0; jj < nc; jj += NR) {
                        size_t nr = std::min(NR, nc - jj);
                        pack_B_tile(B, packed_B.data(), N, ko, jo + jj, kc, nr);
                        for (size_t ii = 0; ii < mc_main; ii += MR) {
                            microkernel_4x8_packed_epilogue<Epilogue>(
                                packed_A.data() + ii, packed_B.data(),
                                C, bias, N, io + ii, jo + jj, kc, mc_main, nr);
                        }
                    }
                    for (size_t r = mc_main; r < mc; r++) {
                        for (size_t kk = 0; kk < kc; kk++) {
                            float a_val = A[(io + r) * K + (ko + kk)];
                            for (size_t jj = 0; jj < nc; jj++) {
                                float val = C[(io + r) * N + (jo + jj)] +
                                    a_val * B[(ko + kk) * N + (jo + jj)];
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
                                    float inner = val * 0.7978845608028654f;
                                    float t = std::tanh(inner);
                                    val = 0.5f * val * (1.0f + t);
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

inline void matmul_bias(const float* A, const float* B, float* C, const float* bias,
                          size_t M, size_t N, size_t K) {
    std::fill(C, C + M * N, 0.0f);
    matmul_packed_epilogue<BiasEpilogue>(A, B, C, bias, M, N, K);
}

inline void matmul_bias_relu(const float* A, const float* B, float* C, const float* bias,
                               size_t M, size_t N, size_t K) {
    std::fill(C, C + M * N, 0.0f);
    matmul_packed_epilogue<BiasReLUEpilogue>(A, B, C, bias, M, N, K);
}

inline void matmul_bias_silu(const float* A, const float* B, float* C, const float* bias,
                               size_t M, size_t N, size_t K) {
    std::fill(C, C + M * N, 0.0f);
    matmul_packed_epilogue<BiasSiLUEpilogue>(A, B, C, bias, M, N, K);
}

inline void matmul_bias_gelu(const float* A, const float* B, float* C, const float* bias,
                               size_t M, size_t N, size_t K) {
    std::fill(C, C + M * N, 0.0f);
    matmul_packed_epilogue<BiasGELUEpilogue>(A, B, C, bias, M, N, K);
}


inline void attention_scores(const float* Q, const float* K, float* scores,
                              size_t batch, size_t heads, size_t T, size_t S, size_t D) {
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < heads; h++) {
            const float* q = Q + (b * heads + h) * T * D;
            const float* k = K + (b * heads + h) * S * D;
            float* s = scores + (b * heads + h) * T * S;
            std::fill(s, s + T * S, 0.0f);
            for (size_t t = 0; t < T; t++) {
                for (size_t d = 0; d < D; d++) {
                    float q_val = q[t * D + d];
                    for (size_t ss = 0; ss < S; ss++) {
                        s[t * S + ss] = std::fma(q_val, k[ss * D + d], s[t * S + ss]);
                    }
                }
            }
        }
    }
}

template <size_t NC = 32>
inline void attention_scores_simd(const float* Q, const float* K, float* scores,
                                   size_t batch, size_t heads, size_t T, size_t S, size_t D) {
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < heads; h++) {
            const float* q = Q + (b * heads + h) * T * D;
            const float* k = K + (b * heads + h) * S * D;
            float* s = scores + (b * heads + h) * T * S;

            std::vector<float> K_T(D * S);
            for (size_t row = 0; row < S; row++)
                for (size_t col = 0; col < D; col++)
                    K_T[col * S + row] = k[row * D + col];

            std::fill(s, s + T * S, 0.0f);
            matmul_packed<NC, NC>(q, K_T.data(), s, T, S, D);
        }
    }
}

template <size_t NC = 32>
inline void attention_scores_gqa(const float* Q, const float* K, float* scores,
                                  size_t batch, size_t H_q, size_t H_kv,
                                  size_t T, size_t S, size_t D) {
    size_t groups = H_q / H_kv;
    for (size_t b = 0; b < batch; b++) {
        for (size_t kv_h = 0; kv_h < H_kv; kv_h++) {
            const float* k = K + (b * H_kv + kv_h) * S * D;

            std::vector<float> K_T(D * S);
            for (size_t row = 0; row < S; row++)
                for (size_t col = 0; col < D; col++)
                    K_T[col * S + row] = k[row * D + col];

            for (size_t g = 0; g < groups; g++) {
                size_t q_h = kv_h * groups + g;
                const float* q = Q + (b * H_q + q_h) * T * D;
                float* s = scores + (b * H_q + q_h) * T * S;
                std::fill(s, s + T * S, 0.0f);
                matmul_packed<NC, NC>(q, K_T.data(), s, T, S, D);
            }
        }
    }
}

inline void attention_values(const float* attn, const float* V, float* output,
                              size_t batch, size_t heads, size_t T, size_t S, size_t D) {
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < heads; h++) {
            const float* a = attn + (b * heads + h) * T * S;
            const float* v = V + (b * heads + h) * S * D;
            float* o = output + (b * heads + h) * T * D;
            std::fill(o, o + T * D, 0.0f);
            for (size_t t = 0; t < T; t++) {
                for (size_t s = 0; s < S; s++) {
                    float a_val = a[t * S + s];
                    for (size_t d = 0; d < D; d++) {
                        o[t * D + d] = std::fma(a_val, v[s * D + d], o[t * D + d]);
                    }
                }
            }
        }
    }
}

template <size_t NC = 32>
inline void attention_values_simd(const float* attn, const float* V, float* output,
                                   size_t batch, size_t heads, size_t T, size_t S, size_t D) {
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < heads; h++) {
            const float* a = attn + (b * heads + h) * T * S;
            const float* v = V + (b * heads + h) * S * D;
            float* o = output + (b * heads + h) * T * D;
            std::fill(o, o + T * D, 0.0f);
            matmul_packed<NC, NC>(a, v, o, T, D, S);
        }
    }
}

template <size_t NC = 32>
inline void attention_values_gqa(const float* attn, const float* V, float* output,
                                  size_t batch, size_t H_q, size_t H_kv,
                                  size_t T, size_t S, size_t D) {
    size_t groups = H_q / H_kv;
    for (size_t b = 0; b < batch; b++) {
        for (size_t kv_h = 0; kv_h < H_kv; kv_h++) {
            const float* v = V + (b * H_kv + kv_h) * S * D;
            for (size_t g = 0; g < groups; g++) {
                size_t q_h = kv_h * groups + g;
                const float* a = attn + (b * H_q + q_h) * T * S;
                float* o = output + (b * H_q + q_h) * T * D;
                std::fill(o, o + T * D, 0.0f);
                matmul_packed<NC, NC>(a, v, o, T, D, S);
            }
        }
    }
}

#endif
} // namespace impl
} // namespace simd
