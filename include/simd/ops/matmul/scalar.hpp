#pragma once
#include <cstddef>
#include <mdspan>

/*
 * M : no. of rows in A
 * N : no. of columns in B
 * K : no. of columns in A = no. of rows in B
 * 
 * A : M * K
 * B : K * N
 * C : M * N
 */

namespace simd {
namespace impl {

inline void matmul_ijk(const float* A, const float* B, float* C, size_t M, size_t N, size_t K){
    std::mdspan<const float, std::dextents<size_t, 2>> A_md(A, M, K);
    std::mdspan<const float, std::dextents<size_t, 2>> B_md(B, K, N);
    std::mdspan<float, std::dextents<size_t, 2>> C_md(C, M, N);
    for(size_t i = 0; i < M; i++){
        for(size_t j = 0; j < N; j++){
            float sum = 0.0f; 
            for(size_t k = 0; k < K; k++)
                sum += A_md[i, k] * B_md[k, j];
            C_md[i, j] = sum;
        }
    }
}    

inline void matmul_ikj(const float* A, const float* B, float* C, size_t M, size_t N, size_t K){
    std::mdspan<const float, std::dextents<size_t, 2>> A_md(A, M, K);
    std::mdspan<const float, std::dextents<size_t, 2>> B_md(B, K, N);
    std::mdspan<float, std::dextents<size_t, 2>> C_md(C, M, N);
    for(size_t i = 0; i < M; i++){
        for(size_t k = 0; k < K; k++){
            for(size_t j = 0; j < N; j++){
                C_md[i, j] += A_md[i, k] * B_md[k, j];
            }
        }
    }
}

/*
 * Naming Convention
 * MC : M-dimension Cache size - maximum allowable size of window
 * io : Outer / Offset
 * mc : current chunk size (fringe variables)
 */

template <size_t MC = 32, size_t NC = 32, size_t KC = 64>
inline void matmul_scalar_tiled(const float* A, const float* B, float* C, size_t M, size_t N, size_t K){
    std::mdspan<const float, std::dextents<size_t, 2>> A_md(A, M, K);
    std::mdspan<const float, std::dextents<size_t, 2>> B_md(B, K, N);
    std::mdspan<float, std::dextents<size_t, 2>> C_md(C, M, N);
    #pragma omp parallel for schedule(dynamic) if(M >= 256)
    for(size_t io = 0; io < M; io += MC){
        size_t mc = std::min(MC, M - io);
        for(size_t ko = 0; ko < K; ko += KC){
            size_t kc = std::min(KC, K - ko);
            for(size_t jo = 0; jo < N; jo += NC){
                size_t nc = std::min(NC, N - jo);
                // micro ikj loop
                for(size_t i = 0; i < mc; i++){
                    for(size_t k = 0; k < kc; k++){
                        float a = A_md[io + i, ko + k];
                        if(a == 0.0f)
                            continue;
                        for(size_t j = 0; j < nc; j++)
                            C_md[io + i, jo + j] += a * B_md[ko + k, jo + j];
                    }
                }
            }
        }
    }
}

inline void matmul_f32xi8_scalar(const float* A, const int8_t* B_i8, const float* scales,
                                  float* C, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum = std::fma(A[i * K + k], static_cast<float>(B_i8[k * N + j]), sum);
            }
            C[i * N + j] = sum * scales[j];
        }
    }
}

} // namespace impl
} // namespace simd