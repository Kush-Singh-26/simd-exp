# GEMM Implementation Workbook

Learn to build matrix multiplication kernels from scratch, progressing from naive scalar to production-grade SIMD+tiled+packed GEMM suitable for an LLM inference engine.

---

## Table of Contents

1. [GEMM Fundamentals](#1-gemm-fundamentals)
2. [Step 1: Naive ijk](#2-step-1-naive-ijk)
3. [Step 2: Register-Correct ijk](#3-step-2-register-correct-ijk)
4. [Step 3: Loop Reorder ikj](#4-step-3-loop-reorder-ikj)
5. [Step 4: Cache Tiling](#5-step-4-cache-tiling)
6. [Step 5: AVX2 Vectorized ikj](#6-step-5-avx2-vectorized-ikj)
7. [Step 6: 4×8 Microkernel (No Packing)](#7-step-6-4x8-microkernel-no-packing)
8. [Step 7: Packing — The Big Leap](#8-step-7-packing-the-big-leap)
9. [Step 8: Packed 4×8 Kernel](#9-step-8-packed-4x8-kernel)
10. [Step 9: Software Prefetch](#10-step-9-software-prefetch)
11. [Step 10: M=1 Specialized Path (Decode Hot Path)](#11-step-10-m1-specialized-path-decode-hot-path)
12. [Step 11: Batched & Strided-Batched GEMM](#12-step-11-batched-and-strided-batched-gemm)
13. [Step 12: Quantized f32 × i8 → f32](#13-step-12-quantized-f32-x-i8-to-f32)
14. [Fused Epilogues](#14-fused-epilogues)
15. [Attention-Specific Kernels](#15-attention-specific-kernels)
16. [Validation & Benchmarking](#16-validation-and-benchmarking)
17. [Inference Engine Kernel Checklist](#17-inference-engine-kernel-checklist)

---

## 1. GEMM Fundamentals

### 1.1 What is GEMM?

General Matrix Multiply (GEMM) computes:

```
C[M][N] = A[M][K] × B[K][N]
```

Where all matrices are row-major: `A[i][k] = A[i*K + k]`, `B[k][j] = B[k*N + j]`, `C[i][j] = C[i*N + j]`.

Each output element `C[i][j]` is the dot product of row `i` of A and column `j` of B:

```
C[i][j] = Σₖ A[i][k] * B[k][j]
```

This requires `2 * M * N * K` FLOPs (multiply + add per inner iteration).

### 1.2 Why GEMM Matters for Inference

Practically every operation in a transformer reduces to GEMM or a variant:

| Layer | GEMM Shape | Notes |
|---|---|---|
| Embedding lookup | — | Gather, not matmul |
| QKV projection | `[B,T,D] × [D,3D]` | Fused or separate |
| Attention scores | `[B*H,T,D/H] × [B*H,D/H,T]` | Batched batched GEMM |
| Attention values | `[B*H,T,T] × [B*H,T,D/H]` | |
| FFN gate/up/down | `[B,T,D] × [D,4D]` etc | |
| LM head | `[B,T,D] × [D,V]` | V can be 50k+ (large outer dim) |
| RoPE | Not matmul | Element-wise rotation |

### 1.3 Arithmetic Intensity & The Roofline Model

**Arithmetic intensity** = FLOPs / bytes moved to/from memory.

For GEMM:
- FLOPs = `2 * M * N * K`
- Bytes read = `M*K + K*N` (A + B in f32 = 4 bytes each)
- Bytes written = `M*N` (C)
- **Arithmetic intensity** ≈ `(2 * M * N * K) / (4*(M*K + K*N + M*N))`

For square 1024×1024: ~341 FLOPs/byte (compute bound ✅)
For M=1, N=4096, K=4096: ~170 FLOPs/byte (still compute bound)
For M=1, N=1, K=4096: ~0.5 FLOPs/byte (memory bound ❌ — this is a single dot product)

**Key insight:** The larger M and N are relative to K, the more compute-bound you are. The decode case (M=1) is always harder.

### 1.4 Memory Hierarchy (x86_64 Typical)

| Level | Size | Latency | Bandwidth |
|---|---|---|---|
| L1 cache | 32 KB/core | ~1 ns | ~200 GB/s |
| L2 cache | 256–512 KB/core | ~4 ns | ~100 GB/s |
| L3 cache | 8–32 MB shared | ~12 ns | ~50 GB/s |
| DRAM | 16–64 GB | ~80 ns | ~30–50 GB/s |

A good GEMM keeps all data movement within registers and L1 cache as much as possible.

### 1.5 Row-Major Indexing

Matrices stored flat in row-major order:

```
A[i][k] = A[i * K + k]
```

Where `K` is the **leading dimension** (stride between consecutive rows).

---

## 2. Step 1: Naive ijk

### Theory

The most direct translation of the mathematical definition:

```
for i in 0..M:
  for j in 0..N:
    sum = 0
    for k in 0..K:
      sum += A[i][k] * B[k][j]
      C[i][j] = sum        // ← BUG: store happens EVERY iteration
```

This is intentionally bad because:

1. **Store-to-load forwarding penalty**: Writing `C[i][j]` on every `k` iteration makes the CPU pipeline stall waiting for the store to retire before the next load can read from the same address. The register `sum` gets spilled to memory every cycle.

2. **Cache-inefficient access pattern**: Inner loop `k` accesses `A[i][k]` (row of A, good) and `B[k][j]` (column of B, strided by N — terrible). Each `B[k][j]` access touches a different cache line.

3. **IJK loop order** is the worst possible order for row-major storage: `B` is accessed column-wise, meaning every inner iteration is a cache miss for large N.

### Signature

```cpp
// file: include/simd/ops/matmul/scalar.hpp
// namespace: simd::impl

inline void matmul_naive(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K
);
```

### Done Criteria

- [ ] Passes known-value tests for 2×2×2, 3×3×3, 4×4×4
- [ ] Passes non-square shape: 2×3 × 3×4
- [ ] Slowest of all implementations (>5x slower than Step 2)
- [ ] Do NOT use as reference for correctness (use Step 2 instead)

### Test Cases

```cpp
// Known values
TEST(MatmulNaive, Known2x2) {
  float A[] = {1, 2,  3, 4};  // 2x2
  float B[] = {5, 6,  7, 8};  // 2x2
  float C[4];
  matmul_naive(A, B, C, 2, 2, 2);
  // C[0][0] = 1*5 + 2*7 = 19
  // C[0][1] = 1*6 + 2*8 = 22
  // C[1][0] = 3*5 + 4*7 = 43
  // C[1][1] = 3*6 + 4*8 = 50
  EXPECT_FLOAT_EQ(C[0], 19);
  EXPECT_FLOAT_EQ(C[1], 22);
  EXPECT_FLOAT_EQ(C[2], 43);
  EXPECT_FLOAT_EQ(C[3], 50);
}
```

### Observations to Record

- How many times slower is it vs Step 2 for small matrices? Large matrices?
- Does the performance gap grow with matrix size?

---

## 3. Step 2: Register-Correct ijk

### Theory

The fix is trivial: move the store *outside* the k-loop.

```
for i in 0..M:
  for j in 0..N:
    sum = 0
    for k in 0..K:
      sum += A[i][k] * B[k][j]
    C[i][j] = sum           // ← ONE store per output element
```

Why this matters:

- The variable `sum` lives in a register for the entire k-loop (no spill).
- The store to `C[i][j]` happens only once per output element.
- The CPU can fully pipeline the FMA operations.

However, the cache behavior is unchanged: `B[k][j]` still accesses columns, which is strided.

Arithmetic intensity at this point is the same regardless of loop order. But the *register pressure* and *store behavior* differ. This step removes the artificial store bottleneck from Step 1 so subsequent optimizations can be measured fairly.

### Signature

```cpp
// file: include/simd/ops/matmul/scalar.hpp
// namespace: simd::impl

inline void matmul_scalar_ijk(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K
);
```

### Done Criteria

- [ ] Passes all same tests as Step 1
- [ ] Use this as correctness reference for all future steps
- [ ] ~2–5x faster than Step 1 (the store penalty on modern CPUs is real)

### Test Cases

Identical expected values to Step 1. Use this as reference:

```cpp
void matmul_reference(const float* A, const float* B, float* C,
                      size_t M, size_t N, size_t K) {
    matmul_scalar_ijk(A, B, C, M, N, K);
}
```

### Observations to Record

- Speedup over naive for 256×256, 512×512
- Wall-clock time per matrix size

---

## 4. Step 3: Loop Reorder ikj

### Theory

The inner loop should access memory **contiguously** for best cache utilization. Row-major means consecutive elements in a row are adjacent in memory.

ijk loop order:
```
for i: for j: for k:
  C[i][j] += A[i][k] * B[k][j]
```

Inner iteration accesses:
- `A[i][k]` — contiguous row of A ✅
- `B[k][j]` — strided column of B ❌

ikj loop order:
```
for i: for k: for j:
  C[i][j] += A[i][k] * B[k][j]
```

Inner iteration accesses:
- `B[k][j]` — contiguous row of B ✅
- `C[i][j]` — contiguous row of C ✅
- `A[i][k]` — single scalar (broadcast to all j)

**B is now accessed sequentially in the inner loop**, which gives:
- Hardware prefetcher can predict the access pattern
- Cache lines are fully utilized (all 8 floats in a cache line are consumed)
- No TLB misses from strided access

The tradeoff: we must accumulate into C, so C must be zero-initialized.

```
zero C (M * N floats)
for i in 0..M:
  for k in 0..K:
    temp = A[i][k]           // scalar broadcast
    for j in 0..N:
      C[i][j] += temp * B[k][j]
```

### Math

```
C[i][j] = Σₖ A[i][k] * B[k][j]

Reorder:
For each (i,k):
  Broadcast a = A[i][k]
  For each j:
    C[i][j] += a * B[k][j]
```

This computes the same result because addition is associative and commutative. The order of accumulation does not change the final sum (modulo floating-point associativity, which is approximate — but we accept this).

### Signature

```cpp
// file: include/simd/ops/matmul/scalar.hpp
// namespace: simd::impl

inline void matmul_scalar_ikj(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K
);
```

### Implementation Notes

- **Must zero C** before accumulating (or initialize C).
- The inner loop over `j` is a simple FMA (fused multiply-add conceptually, even if scalar).
- `temp = A[i][k]` is a scalar — the compiler may auto-vectorize the inner j-loop.

### Done Criteria

- [ ] Passes correctness tests (compare to Step 2)
- [ ] ~2–4x faster than Step 2 for large matrices (256×256+)
- [ ] For M=1,N=4096,K=4096, should be clearly faster than Step 2
- [ ] Benchmark: capture speedup vs Step 2 at multiple sizes

### Observations to Record

- At what matrix size does the speedup become significant?
- Is the speedup larger for square or rectangular matrices?
- Measure GFLOPs: `2*M*N*K / time_seconds * 1e-9`

### Test Cases

```cpp
TEST(MatmulIKJ, CompareToIJK) {
  // Random data, multiple seeds
  for (auto [M, N, K] : {Dimensions{16,16,16}, {32,64,128}, {128,128,128}}) {
    std::vector<float> A(M*K), B(K*N), C1(M*N), C2(M*N);
    // fill with random data
    matmul_scalar_ijk(A.data(), B.data(), C1.data(), M, N, K);
    matmul_scalar_ikj(A.data(), B.data(), C2.data(), M, N, K);
    for (size_t i = 0; i < M*N; i++)
      EXPECT_NEAR(C1[i], C2[i], 1e-4);
  }
}
```

---

## 5. Step 4: Cache Tiling

### Theory

ikj reordering fixes the inner-loop access pattern, but the outer loops still touch large working sets. For large M, N, K, the entire matrix doesn't fit in cache. The solution: **tile/block** the loops to reuse data while it's still in cache.

**Key insight in ikj**: For each `k`, we load a row of B (`B[k][0..N]`) and broadcast `A[i][k]` across all `j`. Each row of B is read once per `i` loop. That means `B` is read `M` times from memory. For large M, this kills performance.

To fix this, we tile **all three dimensions** so that working tiles fit in L1/L2 cache:

```
for io in 0..M step MC:
  for ko in 0..K step KC:
    for jo in 0..N step NC:
      // A_tile = A[io:io+MC, ko:ko+KC]
      // B_tile = B[ko:ko+KC, jo:jo+NC]
      // C_tile = C[io:io+MC, jo:jo+NC]
      for i in io..io+MC:
        for k in ko..ko+KC:
          for j in jo..jo+NC:
            C[i][j] += A[i][k] * B[k][j]
```

### Tile Size Selection

Goal: keep tiles in cache.

**L1 cache** (32 KB):
- MC × KC (A fragment) + KC × NC (B fragment) + MC × NC (C fragment) ≤ 32 KB / 4 bytes = 8192 floats
- With MC=64, KC=64, NC=64: A=4096, B=4096, C=4096 → 12,288 floats ≈ 48 KB (too big for L1)
- With MC=32, KC=64, NC=32: A=2048, B=2048, C=1024 → 5120 floats ≈ 20 KB ✅ fits L1
- Common choice for f32: MC=64, KC=64, NC=64 (fits L2, partially L1)

**Why 64 is a common starting point**: It's a power of 2, works well with cache line boundaries (64 bytes = 16 floats), and the total tile (12K floats ≈ 48 KB) fits in L2.

### Signature

```cpp
// file: include/simd/ops/matmul/scalar.hpp
// namespace: simd::impl

inline void matmul_scalar_tiled(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K,
    size_t MC = 64, size_t NC = 64, size_t KC = 64
);
```

### Implementation Pattern

```cpp
for (size_t io = 0; io < M; io += MC) {
    size_t mc = std::min(MC, M - io);
    for (size_t ko = 0; ko < K; ko += KC) {
        size_t kc = std::min(KC, K - ko);
        for (size_t jo = 0; jo < N; jo += NC) {
            size_t nc = std::min(NC, N - jo);
            // Accumulate the mc x kc * kc x nc product
            for (size_t i = 0; i < mc; i++) {
                for (size_t k = 0; k < kc; k++) {
                    float a = A[(io + i) * K + (ko + k)];
                    if (a == 0.0f) continue; // optional sparsity skip
                    for (size_t j = 0; j < nc; j++) {
                        C[(io + i) * N + (jo + j)] +=
                            a * B[(ko + k) * N + (jo + j)];
                    }
                }
            }
        }
    }
}
```

### Important: Loop Order Within Tiles

ikj inside each tile (since it's the same row-major access pattern).

### Done Criteria

- [ ] Correctness matches Step 2 for all shapes
- [ ] Faster than Step 3 (ikj without tiling) for large matrices (≥512×512)
- [ ] Not necessarily faster for small matrices (tiling overhead)
- [ ] Experiment: try tile sizes 32, 64, 128 — note which is fastest
- [ ] Record speedup vs Step 3 for 1024×1024, 2048×2048

### Observations to Record

- Does tiling help at 256×256? If not, why? (Matrix fits in cache without tiling.)
- At what size does tiling start helping?
- How sensitive is performance to tile size? (±10%? ±50%?)
- Is MC=KC=NC=64 always best? Try unbalanced tiles (larger MC/kC for row-major).

### Roofline Analysis for This Step

Tiling increases arithmetic intensity of the inner loops because:
- A_tile: `mc * kc` floats, loaded once per `jo` iteration
- B_tile: `kc * nc` floats, loaded once per `io` iteration
- C_tile: `mc * nc` floats, loaded and stored once

For a single tile (mc, kc, nc):
- FLOPs = `2 * mc * kc * nc`
- Bytes = `4 * (mc*kc + kc*nc + 2*mc*nc)` (read A+B+C, write C)
- AI = `(2*mc*kc*nc) / (4*(mc*kc + kc*nc + 2*mc*nc))`

For 64×64×64: AI ≈ 21.3 FLOPs/byte (approaching compute bound on most CPUs).

---

## 6. Step 5: AVX2 Vectorized ikj

### Theory

The ikj inner loop over `j` is a perfect target for SIMD vectorization:

```cpp
// Scalar inner loop:
for j in 0..N:
  C[i][j] += a * B[k][j]
```

This is a **scalar broadcast × vector load + FMA into vector accumulator**:

```cpp
// AVX2 inner loop (process 8 j's at once):
__m256 a_vec = _mm256_broadcast_ss(&A[i*K + k]);  // broadcast a to 8 lanes
for j in 0..N step 8:
  __m256 b = _mm256_loadu_ps(&B[k*N + j]);
  __m256 c = _mm256_loadu_ps(&C[i*N + j]);
  c = _mm256_fmadd_ps(a_vec, b, c);
  _mm256_storeu_ps(&C[i*N + j], c);
```

### SIMD Instructions Used

| Instruction | Description |
|---|---|
| `_mm256_setzero_ps()` | Creates zero vector (for initializing C) |
| `_mm256_broadcast_ss(ptr)` | Load 1 float and broadcast to all 8 lanes |
| `_mm256_loadu_ps(ptr)` | Load 8 floats (unaligned) |
| `_mm256_fmadd_ps(a, b, c)` | `a * b + c` element-wise (8-way FMA) |
| `_mm256_storeu_ps(ptr, v)` | Store 8 floats (unaligned) |

### FMA Throughput

Modern CPUs (Zen 4, Ice Lake+) can sustain **2 FMA operations per cycle** per core. Each FMA is 8 FLOPs (4 multiplies + 4 adds conceptually, but fused as 8 FLOPs).

Theoretical peak GFLOPs = `freq * 8 * 2` (AVX2 width × dual-issue FMA).

At 3 GHz: 48 GFLOPS single-core theoretical peak.

But we need to keep the pipeline fed. Each FMA needs 2 loads + 1 store per 8 FLOPs. That's a lot of memory bandwidth. The broadcast helps: `a_vec` comes from a single scalar, so it's one load for 8 values.

### Tail Handling

When `N` is not a multiple of 8, process remaining 1–7 elements with scalar code:

```cpp
for (; j < N; j++) {
    C[i*N + j] += a * B[k*N + j];
}
```

Or use a masked load/store (AVX2 has `_mm256_maskload_ps`/`_mm256_maskstore_ps`), but these can be slower than scalar tail due to the mask generation overhead. Scalar tail is fine for small remainders.

### Combined with Tiling

The combination of tiling + AVX2 on the inner j-loop is powerful:

```
for io: for ko: for jo:
  // MC×KC × KC×NC tile
  for i in 0..mc:
    for k in 0..kc:
      a = A_row[i][k]
      AVX2 j-loop over 0..nc step 8:
        C[i][j:j+8] += a * B[k][j:j+8]
  // tail j-loop
```

### Signature

```cpp
// file: include/simd/ops/matmul/simd.hpp
// namespace: simd::impl

#if defined(SIMD_AVX2_ENABLED) && defined(__FMA__)

inline void matmul_avx2(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K
);
```

### Test Cases

```cpp
TEST(MatmulAVX2, CompareToScalar) {
  for (auto [M,N,K] : {Dimensions{1,512,512}, {1,4096,4096},
                        {64,4096,4096}, {128,128,128}}) {
    std::vector<float> A(M*K), B(K*N), C_simd(M*N), C_ref(M*N);
    fill_random(A, B);
    matmul_scalar_ikj(A.data(), B.data(), C_ref.data(), M, N, K);
    matmul_avx2(A.data(), B.data(), C_simd.data(), M, N, K);
    for (size_t i = 0; i < M*N; i++)
      EXPECT_NEAR(C_simd[i], C_ref[i], 1e-3);
  }
}
```

### Observations to Record

- Speedup over scalar ikj at 256×256, 512×512, 1024×1024
- Speedup with and without tiling
- How close to theoretical peak FMA throughput? (measure GFLOPs)
- Effect of non-power-of-2 N (tail penalty)

---

## 7. Step 6: 4×8 Microkernel (No Packing)

### Theory

Instead of processing one row of A at a time (Step 5), process **4 rows × 8 columns** of C simultaneously. This increases register reuse: we load one 8-wide vector from B and broadcast 4 scalars from A, performing 4 FMAs for the price of 1 B load.

```
for i in 0..M step 4:
  for j in 0..N step 8:
    acc00..acc33 = 0  // 4×8 = 32 accumulators
    for k in 0..K:
      b = B[k][j:j+8]   // 1 vector load
      a0 = A[i+0][k]    // 1 scalar broadcast
      acc00..acc03 += a0 * b   // 4 FMAs (well, acc0 rows)
      a1 = A[i+1][k]    // 1 scalar broadcast
      acc10..acc13 += a1 * b   // 4 FMAs
      a2 = A[i+2][k]    // 1 scalar broadcast
      acc20..acc23 += a2 * b   // 4 FMAs
      a3 = A[i+3][k]    // 1 scalar broadcast
      acc30..acc33 += a3 * b   // 4 FMAs
    store(C[i:i+4][j:j+8])  // write 4×8 block
```

### Why 4×8?

- **4 rows**: We have 16 architectural YMM registers (x86-64). Using 4 for the C accumulator tiles (4 × 8 floats = 4 YMM registers), plus working registers for A broadcast and B load, fits comfortably.
- **8 columns**: Natural AVX2 width (8 floats per vector).
- **32 FMAs per k-iteration**: Each k does 4×8 = 32 FMAs at the cost of 1 B load + 4 A broadcasts.

Register budget (YMM registers, 16 total):
- 4 accumulators (rows of C tile) ← 4 registers
- 1 B vector ← 1 register
- 1 A broadcast ← 1 register (reused)
- 12 remaining for other purposes (or nothing — the 4 accumulators already dominate)

### Comparison to Step 5 (AVX2 ikj)

Step 5: For each k, load B once → compute 1 row of C.
Step 6: For each k, load B once → compute 4 rows of C.

Step 6 does **4× the work per B load**, increasing arithmetic intensity by 4× within the k-loop.

### Implementation

```cpp
for (size_t i = 0; i + 4 <= M; i += 4) {
    for (size_t j = 0; j + 8 <= N; j += 8) {
        __m256 acc0 = _mm256_setzero_ps();  // row i
        __m256 acc1 = _mm256_setzero_ps();  // row i+1
        __m256 acc2 = _mm256_setzero_ps();  // row i+2
        __m256 acc3 = _mm256_setzero_ps();  // row i+3

        for (size_t k = 0; k < K; k++) {
            __m256 b = _mm256_loadu_ps(&B[k * N + j]);

            __m256 a0 = _mm256_broadcast_ss(&A[(i+0) * K + k]);
            acc0 = _mm256_fmadd_ps(a0, b, acc0);
            __m256 a1 = _mm256_broadcast_ss(&A[(i+1) * K + k]);
            acc1 = _mm256_fmadd_ps(a1, b, acc1);
            __m256 a2 = _mm256_broadcast_ss(&A[(i+2) * K + k]);
            acc2 = _mm256_fmadd_ps(a2, b, acc2);
            __m256 a3 = _mm256_broadcast_ss(&A[(i+3) * K + k]);
            acc3 = _mm256_fmadd_ps(a3, b, acc3);
        }

        _mm256_storeu_ps(&C[(i+0) * N + j], acc0);
        _mm256_storeu_ps(&C[(i+1) * N + j], acc1);
        _mm256_storeu_ps(&C[(i+2) * N + j], acc2);
        _mm256_storeu_ps(&C[(i+3) * N + j], acc3);
    }
}
// Handle edge cases:
// (1) remaining rows when M % 4 != 0
// (2) remaining columns when N % 8 != 0
```

### Edge Cases

**Remaining rows**: Fall back to 1-row AVX2 kernel (Step 5) for the last 1–3 rows.

**Remaining columns** (right edge): Use scalar tail or masked store.

**Remaining rows + columns** (bottom-right corner): Fully scalar.

### Masked Store for Column Tail

```cpp
// When 0 < nr < 8:
__m256i mask = _mm256_setr_epi32(
    nr > 0 ? -1 : 0, nr > 1 ? -1 : 0,
    nr > 2 ? -1 : 0, nr > 3 ? -1 : 0,
    nr > 4 ? -1 : 0, nr > 5 ? -1 : 0,
    nr > 6 ? -1 : 0, nr > 7 ? -1 : 0
);
_mm256_maskstore_ps(&C[(i+0)*N + j], mask, acc0);
// etc for rows
```

Note: `maskstore_ps` is slower than `storeu_ps` due to the non-temporal hint and write-combining behavior. Use it only for edge tiles, not main loops.

### Signature

```cpp
// file: include/simd/ops/matmul/simd.hpp
// namespace: simd::impl

#if defined(SIMD_AVX2_ENABLED) && defined(__FMA__)

inline void matmul_4x8(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K
);
```

### Done Criteria

- [ ] Passes correctness vs Step 2 for all shapes
- [ ] Faster than Step 5 (AVX2 ikj) for large matrices
- [ ] M,N tail handling is correct
- [ ] Record GFLOPs for 512×512, 1024×1024, 2048×2048

### Observations to Record

- How much faster is 4×8 vs single-row AVX2?
- What fraction of theoretical peak FLOPS are you getting?
- Try other tile sizes: 2×8, 4×4, 8×8 (8 YMM accumulators per row = 64 C values = 16 registers?? That's too many)

---

## 8. Step 7: Packing — The Big Leap

### Theory — Why Packing Matters

In the 4×8 kernel, each k-iteration loads:
- `B[k][j:j+8]` — contiguous row access ✅
- `A[i+0..3][k]` — strided within A's row (stride = K) ❌ (wait, no: A is accessed at `[(i+r)*K + k]` which is sequential in `k` for a fixed `i`, but when we move to the next tile, the access pattern changes)

Actually in the 4×8 kernel with `io`/`ko`/`jo` tiling, A and B are accessed repeatedly from memory because the same data gets loaded multiple times in the outer tile loops. The problem:

- For each `io,ko` tile of A: loaded once per `jo` iteration
- For each `ko,jo` tile of B: loaded once per `io` iteration

**Packing solves this** by reorganizing data on the fly into a format that:
1. Makes SIMD loads always contiguous (no strided access)
2. Aligns data to cache line boundaries
3. Eliminates unused elements within cache lines (padding)

### Packed Layout

**Pack A (Column-Major Packing)**:

Original A (row-major `mc × kc`): `A[i][k] = A[i*K + k]`
Packed A: `A_packed[kk * mc + ii] = A[(io+ii) * K + (ko+kk)]`

This organizes A so that for each `kk`, all `mc` values of `A[*, kk]` are contiguous. The microkernel broadcasts each one sequentially.

```
A (mc × kc, row-major):
    k0  k1  k2  k3 ...
i0  a00 a01 a02 a03
i1  a10 a11 a12 a13
i2  a20 a21 a22 a23
i3  a30 a31 a32 a33

A_packed (kc × mc, column-major within):
    [0] [mc] [2mc] ...
k0  a00 a10 a20 a30
k1  a01 a11 a21 a31
k2  a02 a12 a22 a32
```

The microkernel's inner loop:
```cpp
for kk in 0..kc:
  a_vals = A_packed[kk*mc : kk*mc+mc]
  // a_vals[0] = A[i0][k0+kk], a_vals[1] = A[i1][k0+kk], ...
  b = B_packed[kk*NR : kk*NR+NR]
  for mr in 0..MR:
    a_broadcast = a_vals[mr]
    acc[mr] += a_broadcast * b
```

**Pack B (Row-Major Packing)**:

Original B (row-major `kc × nc`): `B[k][j] = B[k*N + j]`
Packed B: `B_packed[kk * NR + jj] = B[(ko+kk) * N + (jo+jj)]`

This keeps the format that the B load already uses (contiguous in j), but ensures the data is contiguous in memory even when `N > nc`.

```
B_packed (kc × NR):
    [0]   [1]   [2]    ... [NR-1]
k0  B[j0] B[j1] B[j2]       B[j7]
k1  B[j0] B[j1] B[j2]       B[j7]
...
```

### Why Packing Helps

1. **Contiguous SIMD loads**: Both A and B are now contiguous blocks. No strided access.
2. **Better cache line utilization**: Packed data fills cache lines completely (no wasted space from stride gaps).
3. **TLB-friendly**: Packed blocks cover large strides in the original layout with few pages.
4. **Alignment**: Packed buffers can be aligned (32-byte for AVX2), enabling `_mm256_load_ps` (aligned) instead of `_mm256_loadu_ps` (unaligned). Aligned loads are slightly faster on some µarchs.

### Pack Function Signatures

```cpp
// Pack A tile: A[io:io+mc, ko:ko+kc] (row-major, stride=K)
// Into A_packed[kc][mc] (column-major within each k step)
// Only packs mc_main rows (mc - mc % MR), remaining rows handled separately
void pack_A(
    const float* A, float* packed,
    size_t K,   // stride of original A
    size_t io, size_t ko,  // tile origin
    size_t mc, size_t kc   // tile size
);

// Pack B tile: B[ko:ko+kc, jo:jo+nr] (row-major, stride=N)
// Into B_packed[kc][NR] (contiguous rows of length NR)
// Pads with 0 if jo+nr > N
void pack_B(
    const float* B, float* packed,
    size_t N,   // stride of original B
    size_t ko, size_t jo,  // tile origin
    size_t kc, size_t nr   // tile size (nr <= NR)
);
```

### Pack A Implementation

```cpp
void pack_A(const float* A, float* packed,
            size_t K, size_t io, size_t ko,
            size_t mc, size_t kc) {
    for (size_t kk = 0; kk < kc; kk++) {
        for (size_t ii = 0; ii < mc; ii++) {
            packed[kk * mc + ii] = A[(io + ii) * K + (ko + kk)];
        }
    }
}
```

### Pack B Implementation

```cpp
void pack_B(const float* B, float* packed,
            size_t N, size_t ko, size_t jo,
            size_t kc, size_t nr) {
    for (size_t kk = 0; kk < kc; kk++) {
        for (size_t jj = 0; jj < nr; jj++) {
            packed[kk * nr + jj] =
                (jo + jj < N) ? B[(ko + kk) * N + (jo + jj)] : 0.0f;
        }
    }
    // Zero out remaining slots when nr < NR (for padded B)
    for (size_t kk = 0; kk < kc; kk++) {
        for (size_t jj = nr; jj < NR; jj++) {
            packed[kk * NR + jj] = 0.0f;
        }
    }
}
```

### Where Packing Fits in the Tiled Loop

```
for io in 0..M step MC:
  for ko in 0..K step KC:
    pack_A(A, A_packed, K, io, ko, mc_main, kc)
    for jo in 0..N step NC:
      for jj in 0..nc step NR:
        pack_B(B, B_packed, N, ko, jo+jj, kc, min(NR, nc-jj))
        for ii in 0..mc_main step MR:
          microkernel(A_packed + ii, B_packed, C,
                      N, io+ii, jo+jj, kc, mc_main, nr)
      // handle mc_main..mc-1 rows (not multiples of MR)
```

### Optimal Packing Order

Notice: `pack_A` is done once per `io×ko` tile, but `pack_B` is done once per `jj` iteration within each `jo` block. This means B is packed more frequently than A.

For inference engines with static weights, B can be **pre-packed offline** (weight prepacking). The weights are packed once and reused for every inference call. A must be packed at runtime because it's the activation.

### Signature

The packing helpers are internal (not exposed in the dispatcher). They live in `simd.hpp`.

```cpp
// file: include/simd/ops/matmul/simd.hpp
// namespace: simd::impl

void pack_A_tile(const float* A, float* packed,
                 size_t K, size_t io, size_t ko,
                 size_t mc, size_t kc);

void pack_B_tile(const float* B, float* packed,
                 size_t N, size_t ko, size_t jo,
                 size_t kc, size_t nr);
```

### Pre-Packing Weights (for Inference Engine)

Since model weights are static, you can pack them offline:

```cpp
// During model loading:
std::vector<float> B_packed(K * N);  // packed layout
pack_B_whole_matrix(B, B_packed.data(), K, N);

// During inference, use B_packed directly:
// B_packed[kk * N + jj] = B[kk * N + jj], but possibly padded
```

The packed layout matches the microkernel's expected format, so the outer packing loop is eliminated at runtime.

### Done Criteria

- [ ] `pack_A` copies mc×kc tile correctly
- [ ] `pack_B` copies kc×nr tile correctly, pads with 0
- [ ] Packing functions are correct for edge tiles (near matrix boundaries)
- [ ] Use `std::vector<float>` for test packing buffers

---

## 9. Step 8: Packed 4×8 Kernel

### Theory

Now we rewrite the 4×8 microkernel to read from packed buffers:

```
A_packed[kc][mc] — column-major: kk*4 + mr selects A[io+mr][ko+kk]
B_packed[kc][NR] — row-major: kk*8 + jj selects B[ko+kk][jo+jj]
```

The microkernel becomes:

```cpp
void matmul_packed_4x8(
    const float* A_packed, const float* B_packed,
    float* C, size_t N,
    size_t i, size_t j,   // write C[i:i+4, j:j+nr]
    size_t kc, size_t mc, size_t nr
) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    for (size_t kk = 0; kk < kc; kk++) {
        // Load B[k][j:j+8] — contiguous in packed B
        __m256 b = _mm256_loadu_ps(&B_packed[kk * 8]);

        // Load and broadcast A[i+0..3][k] — contiguous in packed A
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
        _mm256_storeu_ps(&C[(i+0)*N + j], acc0);
        _mm256_storeu_ps(&C[(i+1)*N + j], acc1);
        _mm256_storeu_ps(&C[(i+2)*N + j], acc2);
        _mm256_storeu_ps(&C[(i+3)*N + j], acc3);
    } else {
        __m256i mask = _mm256_setr_epi32(/* ... mask ... */);
        _mm256_maskstore_ps(&C[(i+0)*N + j], mask, acc0);
        _mm256_maskstore_ps(&C[(i+1)*N + j], mask, acc1);
        _mm256_maskstore_ps(&C[(i+2)*N + j], mask, acc2);
        _mm256_maskstore_ps(&C[(i+3)*N + j], mask, acc3);
    }
}
```

### Differences from Step 6 (Unpacked 4×8)

| Aspect | Unpacked (Step 6) | Packed (Step 8) |
|---|---|---|
| A access | `A[i*K + k]` (stride=K) | `A_packed[kk*mc + mr]` contiguous |
| B access | `B[k*N + j:j+8]` (contiguous if N large) | `B_packed[kk*8 : kk*8+8]` contiguous |
| Outer loop overhead | B accessed each i-loop iteration | B packed once per inner tile |
| Cache behavior | Depends on stride | Fully cache-friendly |
| Alignment | Unaligned | Can be aligned |

### Full Tiled + Packed Loop

```cpp
void matmul_4x8_packed(const float* A, const float* B, float* C,
                       size_t M, size_t N, size_t K) {
    const size_t MC = 64, NC = 64, KC = 64;
    const size_t MR = 4, NR = 8;

    for (size_t i = 0; i < M*N; i++) C[i] = 0.0f;

    for (size_t io = 0; io < M; io += MC) {
        size_t mc = std::min(MC, M - io);
        size_t mc_main = mc - (mc % MR);  // only pack multiples of MR

        for (size_t ko = 0; ko < K; ko += KC) {
            size_t kc = std::min(KC, K - ko);

            // Pack A tile
            std::vector<float> packed_A(mc_main * kc);
            pack_A_tile(A, packed_A.data(), K, io, ko, mc_main, kc);

            for (size_t jo = 0; jo < N; jo += NC) {
                size_t nc = std::min(NC, N - jo);

                for (size_t jj = 0; jj < nc; jj += NR) {
                    size_t nr = std::min(NR, nc - jj);

                    // Pack B sub-tile
                    std::vector<float> packed_B(kc * NR);
                    pack_B_tile(B, packed_B.data(), N, ko, jo + jj, kc, NR);

                    // Run microkernel on MR × NR blocks
                    for (size_t ii = 0; ii < mc_main; ii += MR) {
                        matmul_packed_4x8(
                            packed_A.data() + ii,   // offset within packed A
                            packed_B.data(),
                            C, N,
                            io + ii, jo + jj,
                            kc, mc_main, nr
                        );
                    }

                    // Handle remaining rows (mc_main..mc-1)
                    for (size_t r = mc_main; r < mc; r++) {
                        for (size_t kk = 0; kk < kc; kk++) {
                            float a_val = A[(io + r) * K + (ko + kk)];
                            for (size_t c = 0; c < nr; c++) {
                                C[(io + r) * N + (jo + jj + c)] +=
                                    a_val * B[(ko + kk) * N + (jo + jj + c)];
                            }
                        }
                    }
                }
            }
        }
    }
}
```

### Signature

```cpp
// file: include/simd/ops/matmul/simd.hpp
// namespace: simd::impl

#if defined(SIMD_AVX2_ENABLED) && defined(__FMA__)

inline void matmul_4x8_packed(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K
);
```

### Done Criteria

- [ ] Correctness matches Step 2 (scalar ikj)
- [ ] Faster than Step 6 (unpacked 4×8) for large matrices
- [ ] Edge cases: M=1, M=2, M=5 (odd), N=1, N=3, N=12 (not multiple of 8)
- [ ] Benchmark GFLOPs at 1024×1024, 2048×2048
- [ ] Record the speedup that packing gives over the unpacked 4×8

### Observations to Record

- Is the speedup from packing larger for large inner dimension K?
- Does packing help more for matrices that don't fit in cache?
- Overhead of packing relative to computation time (measure packing time separately)
- What is the optimal MC/KC/NC for your CPU?

---

## 10. Step 9: Software Prefetch

### Theory

Even with packing, the microkernel still waits for memory loads from the packed buffers. Software prefetch (`_mm_prefetch`) tells the CPU to start loading cache lines *before* they're needed, hiding memory latency.

The idea: while processing packing step `kk`, issue a prefetch for data at step `kk + PREFETCH_DIST`.

```
for kk in 0..kc:
  prefetch(B_packed[(kk + dist) * NR])     // prefetch B ahead
  prefetch(A_packed[(kk + dist) * mc])     // prefetch A ahead
  // compute with B_packed[kk * NR] and A_packed[kk * mc]
```

### Prefetch Hints

| Hint | Meaning |
|---|---|
| `_MM_HINT_T0` | Prefetch into all cache levels (L1/L2/L3) |
| `_MM_HINT_T1` | Prefetch into L2/L3 only |
| `_MM_HINT_T2` | Prefetch into L3 only |
| `_MM_HINT_NTA` | Non-temporal: prefetch into L1 only, minimize cache pollution |

For GEMM, `_MM_HINT_NTA` is often best: packed data is large and streaming, so polluting L2/L3 with it is wasteful.

### Distance Tuning

Too small: prefetch arrives too late (stall).  
Too large: data may be evicted before use (prefetch wasted).

Start with distance=2, benchmark, then try 1, 3, 4. Optimal distance depends on:
- Memory latency (~80 cycles to DRAM, ~12 cycles to L3)
- Computation per kk iteration (~8 FMAs + broadcast + load ≈ 8–12 cycles)
- So distance = latency / cycles_per_iteration ≈ 80 / 10 ≈ 8 for DRAM, 12/10 ≈ 1 for L3

Since packed data should be in L2 cache, distance=2–4 is a reasonable starting point.

### Updated Microkernel

```cpp
void matmul_packed_4x8_prefetch(
    const float* A_packed, const float* B_packed,
    float* C, size_t N,
    size_t i, size_t j, size_t kc, size_t mc, size_t nr
) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    const size_t PREFETCH_DIST = 2;

    // Prefetch first iteration(s) ahead
    if (kc > PREFETCH_DIST) {
        _mm_prefetch((const char*)&B_packed[PREFETCH_DIST * 8], _MM_HINT_NTA);
        _mm_prefetch((const char*)&A_packed[PREFETCH_DIST * mc], _MM_HINT_NTA);
    }

    for (size_t kk = 0; kk < kc; kk++) {
        // Prefetch future data
        if (kk + PREFETCH_DIST < kc) {
            _mm_prefetch(
                (const char*)&B_packed[(kk + PREFETCH_DIST) * 8], _MM_HINT_NTA);
            _mm_prefetch(
                (const char*)&A_packed[(kk + PREFETCH_DIST) * mc], _MM_HINT_NTA);
        }

        // Compute
        __m256 b = _mm256_loadu_ps(&B_packed[kk * 8]);
        __m256 a0 = _mm256_broadcast_ss(&A_packed[kk * mc + 0]);
        acc0 = _mm256_fmadd_ps(a0, b, acc0);
        // ... (rows 1-3)
    }

    // Store (same as Step 8)
}
```

### When Prefetch Helps

- **Large K** (many iterations, high chance of L2/L3 miss)
- **Large matrices** (packed data doesn't fit in L2 and spills to L3/DRAM)
- **High memory latency systems** (server CPUs with many cores)

When it might not help:
- Small matrices fitting entirely in L1 cache
- Memory bandwidth already saturated (prefetch adds overhead with no benefit)
- Very fast memory (HBM, high-bandwidth systems)

### Signature

```cpp
// file: include/simd/ops/matmul/simd.hpp
// namespace: simd::impl

#if defined(SIMD_AVX2_ENABLED) && defined(__FMA__)

inline void matmul_4x8_packed_prefetch(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K
);
```

### Done Criteria

- [ ] Correctness matches Step 2
- [ ] Benchmark prefetch distance 1, 2, 3, 4 — record which is best
- [ ] Record speedup vs Step 8 (without prefetch)
- [ ] If no speedup, document why (cache already hits, or prefetch overhead)

---

## 11. Step 10: M=1 Specialized Path (Decode Hot Path)

### Theory

Inference decode (autoregressive generation) does `M=1`: a single token row-vector multiplied by a weight matrix. This is fundamentally different from the M>>1 case:

| Aspect | Prefill (M>>1) | Decode (M=1) |
|---|---|---|
| Shape | `[B*T, D] × [D, O]` | `[1, D] × [D, O]` |
| Bottleneck | Compute (high AI) | Bandwidth (low AI) |
| Reuse | A row repeated, B columns streamed | A is 1 row, B is large weight matrix |
| Strategy | Maximize FMA throughput | Minimize memory traffic |

### Why the General Kernel Is Suboptimal for M=1

The 4×8 packed kernel with M=1:
- `MC` is large (e.g. 64), but we only have 1 row
- Packing A for 1 row is a waste (it's just copying 1 element per kk)
- The outer `io` loop runs once with `mc = 1`, but the kernel is designed for `MR = 4`
- We fall into the "remaining rows" scalar path

### M=1 Optimized Strategy

Instead of the full 4×8 kernel, use a **1×N kernel**:

```
A_packed: not needed (a single scalar per k iteration)
B: can be packed or used as-is

For M=1:
  result[j] = Σₖ A[k] * B[k][j]

  for k in 0..K:
    a = A[k]
    for j in 0..N step 8:
      result[j:j+8] += a * B[k][j:j+8]
```

This is identical to Step 5 (AVX2 ikj) which already handles M=1 efficiently. But we can do better:

### Optimization 1: Pre-packed B

If B is pre-packed offline (contiguous k×8 chunks), the inner loop becomes:

```
for kk in 0..K:
  a = A[kk]
  b = B_packed[kk * NR : kk * NR + 8]
  for jj in 0..N step 8:
    result[jj:jj+8] += a * B_packed_row[kk][jj:jj+8]  // contiguous!
```

Wait — B_packed as defined in Step 7 packs B in `kc × NR` tiles. For M=1, we don't need the tile structure; we want a **fully contiguous** B in the k dimension that's already laid out so that `B[k][0..N]` is a contiguous block of `N` floats — which it already is in row-major!

The existing row-major B IS `B[k][j:j+8]` contiguous. So for M=1, there's no structural packing change for the inner j-loop.

**The real optimization for M=1**: Since we only load each element of B once (no reuse across i), there's no packing benefit for B reuse. The bottleneck is bandwidth: loading `K * N` floats from B.

### Optimization 2: Reduce Memory Traffic

The only way to speed up M=1 is to reduce the bytes moved from B. This means:
1. **Quantization**: Use INT8 weights (4x fewer bytes) — see Step 12
2. **Sparsity**: Skip zero weights (2:4 structured sparsity, etc.)
3. **Weight caching**: If the same weights are used repeatedly (e.g., FFN weights for many tokens in a batch), they stay in cache

### M=1 Implementation

For now, the M=1 path just uses the AVX2 ikj kernel (Step 5). Add a fast dispatch in the matmul dispatcher:

```cpp
if (M == 1) {
    // M=1: single row, use specialized 1×N AVX2 kernel
    matmul_vec_mat(A, B, C, N, K);
    return;
}
```

### M=1 Micro-kernel (1×8)

```cpp
void matmul_1x8(const float* A, const float* B, float* C,
                size_t N, size_t K) {
    for (size_t j = 0; j + 8 <= N; j += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (size_t k = 0; k < K; k++) {
            __m256 a = _mm256_broadcast_ss(&A[k]);
            __m256 b = _mm256_loadu_ps(&B[k * N + j]);
            acc = _mm256_fmadd_ps(a, b, acc);
        }
        _mm256_storeu_ps(&C[j], acc);
    }
    // scalar tail for N % 8
}
```

Note: This is `K` iterations of the inner kernel (no tiling/blocking in K dimension). For large K (4096+), this could be a lot of FMAs. The accumulator stays in register, so it's fine, but the B load pattern is `B[0..K-1][j:j+8]` which is `K/8` contiguous 8-float chunks — very cache friendly.

### Signature

```cpp
// file: include/simd/ops/matmul/simd.hpp
// namespace: simd::impl

inline void matmul_1x8(
    const float* A, const float* B, float* C,
    size_t N, size_t K
);
```

### Done Criteria

- [ ] M=1 path dispatches correctly
- [ ] M=1 performance is at least as fast as the general 4×8 kernel called with M=1
- [ ] Record GFLOPs for M=1,N=4096,K=4096 (decode d_model)
- [ ] Record GFLOPs for M=1,N=51200,K=4096 (vocab decode)

---

## 12. Step 11: Batched and Strided-Batched GEMM

### Theory

Multi-head attention requires many independent small GEMMs sharing the same weight matrix:

```
head h: C_h[M,N] = A_h[M,K] × B[K,N]
```

For GQA (Grouped-Query Attention), groups of query heads share a key-value head. For MQA, all query heads share.

Batched GEMM computes `batch_count` independent GEMMs:
```
for batch in 0..batch_count:
  C_batch[M,N] = A_batch[M,K] × B_batch[K,N]
```

**Strided-batched**: Each input/output is at a fixed stride in a flat buffer:
```
for batch in 0..batch_count:
  C + b*stride_C[M,N] = (A + b*stride_A)[M,K] × (B + b*stride_B)[K,N]
```

### When You Need This

| Scenario | batch_count | A stride | B stride | C stride |
|---|---|---|---|---|
| Standard attention (multi-head) | num_heads | seq_len * d_model | d_model * num_heads | seq_len * d_model |
| GQA (K,V projection) | num_kv_heads | seq_len * d_model | d_model | seq_len * d_model |
| Pointwise FFN | batch_size | seq_len * d_model | d_model | seq_len * d_model |

In practice, ML frameworks like cuBLAS expose `cublas<t>gemmStridedBatched` for exactly this.

### Strided-Batched Signature

```cpp
// C[b] = A[b] × B[b] for b in 0..batch_count
// A[b] = A_base + b * stride_A
// B[b] = B_base + b * stride_B
// C[b] = C_base + b * stride_C
void matmul_strided_batched(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K,
    size_t batch_count,
    int64_t stride_A, int64_t stride_B, int64_t stride_C
);
```

### Implementation Strategies

**Naive**:
```cpp
for (size_t b = 0; b < batch_count; b++) {
    matmul_kernel(A + b * stride_A, B + b * stride_B,
                  C + b * stride_C, M, N, K);
}
```

**Optimized**: When all GEMMs share the same B (common in attention), pack B once and reuse across all batches:

```cpp
// Pack B once
pack_B_tile(B, B_packed, K, N);  // full B matrix packed

for (size_t b = 0; b < batch_count; b++) {
    // Use packed B, process each A batch
    matmul_packed_fast(
        A + b * stride_A, B_packed,
        C + b * stride_C, M, N, K
    );
}
```

**Further optimization**: Batch the packing and tile across the batch dimension, loading multiple A rows from different batches in the same microkernel (this is what cuBLAS does).

### Shared B Optimization (for Attention)

In attention, all heads use the same K/V matrices:
- Q heads: `Q[b,h] × K^T` — each Q head has own A, shared B
- V heads: `Attn[b,h] × V` — each attention output has own A, shared V

When B is shared, pack it once:

```cpp
// Pre-packed K, V matrices at model load time
float* K_packed = pack_matrix(K, ...);
float* V_packed = pack_matrix(V, ...);

// During inference for each token:
for (size_t h = 0; h < num_heads; h++) {
    // Q_h × K^T → scores for head h
    matmul_packed_with_B_prepacked(
        Q_h, K_packed, scores_h, 1, seq_len, d_head);
}
```

### Done Criteria

- [ ] Strided-batched GEMM works correctly
- [ ] Optimized path with shared B: pack B once, call kernel N times
- [ ] Benchmark: batch_count=32, M=1, N=64, K=64 (multi-head attention per token)
- [ ] Compare naive loop vs shared-B optimization

---

## 13. Step 12: Quantized f32 × i8 → f32

### Theory

Quantization reduces memory bandwidth by storing weights in INT8 (1 byte vs 4 bytes for f32). The compute pattern becomes:

```
C_f32[i][j] = Σₖ A_f32[i][k] * (W_i8[k][j] * scale[k] + zero_point[k])
```

For simplicity, we'll implement **per-channel quantization** (each output channel has its own scale):

```
C[i][j] = Σₖ A[i][k] * (W_i8[k][j] * scale[j])
        = Σₖ A[i][k] * W_i8[k][j] * scale[j]
        = (Σₖ A[i][k] * W_i8[k][j]) * scale[j]
```

Where `scale[j]` is the scale for output channel `j`. This allows us to:
1. Compute the dot product in i32 (or f32 with i8→f32 conversion)
2. Apply the scale per output element

### INT8 Dot Product with AVX2

AVX2 has `_mm256_maddubs_epi16` (multiply unsigned bytes → add adjacent pairs → i16) and `_mm256_madd_epi16` (multiply i16 → add adjacent pairs → i32).

For signed i8 × i8:
```
vpmaddubsw: (u8 × i8) → i16  — actually this treats one arg as unsigned
```

For signed × signed, we need to handle sign bits. The sequence for i8 × i8 → i32 dot product (8 elements → 1 i32):

```cpp
// a = 8 x i8, b = 8 x i8
__m256i a = _mm256_loadu_si256(...);   // 32 x i8 loaded, but we process 16 at a time
__m256i b = _mm256_loadu_si256(...);

// Approach: convert to i16, then multiply+add
__m256i a_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a));
__m256i b_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b));
__m256i prod_lo = _mm256_madd_epi16(a_lo, b_lo);  // 4 i32 per 8 i8 pairs

// Repeat for high 128 bits
```

But this only does 8 elements per iteration (converting 8→4 i32). Not ideal.

**Better approach for inference**: Convert i8→f32 on load, then use FMA as before. This costs memory bandwidth for the i8 load but avoids complex i32 accumulation.

```cpp
// Load i8, convert to f32, FMA
__m256i b_i8 = _mm256_loadu_si256(...);  // 8 i8 values
__m128i b_i8_lo = _mm256_castsi256_si128(b_i8);
__m128i b_i8_hi = _mm256_extracti128_si256(b_i8, 1);
__m256 b_f32 = _mm256_cvtepi32_ps(
    _mm256_set_m128i(
        _mm_cvtepi8_epi32(b_i8_lo),
        _mm_cvtepi8_epi32(b_i8_hi)
    )
);
__m256 a_f32 = _mm256_loadu_ps(&A[k]);  // contiguous A
acc = _mm256_fmadd_ps(a_f32, b_f32, acc);
```

Wait — VPMADDUBSW + VPMADDWD + VPADDD for pure integer can be faster. Let me provide both approaches.

### Per-Channel Scale Dequant

After computing the i8 dot product in i32, convert to f32 and apply scale:

```cpp
// After K loop, acc_i32 contains Σ A[i][k] * W_i8[k][j]
__m256 acc_f32 = _mm256_cvtepi32_ps(acc_i32);
__m256 scale = _mm256_loadu_ps(&scales[j]);
acc_f32 = _mm256_mul_ps(acc_f32, scale);
// Add bias here if fused
_mm256_storeu_ps(&C[i * N + j], acc_f32);
```

### Per-Group Quantization (Group Size 128)

More accurate quantization: each group of 128 consecutive elements in K-dim shares a scale.

The microkernel needs to handle the group boundary. Since 128 is a multiple of 8, we can process groups of 128 K-elements, load the scale for that group, apply after accumulation.

### Approach 1: f32 accumulation (simpler)

```cpp
for (size_t k = 0; k < K; k++) {
    int8_t b_val = B_i8[k * N + j]; // scalar load — slow for large N
}
```

Better: vectorized load + conversion:

```cpp
for (size_t j = 0; j + 8 <= N; j += 8) {
    __m256 acc_f32 = _mm256_setzero_ps();
    for (size_t k = 0; k < K; k++) {
        // Broadcast A[i][k]
        __m256 a = _mm256_broadcast_ss(&A[i * K + k]);
        // Load 8 i8 values from B[k][j:j+8], convert to f32
        __m128i b8 = _mm_loadl_epi64((__m128i*)&B_i8[k * N + j]); // 8 i8
        __m256i b32 = _mm256_cvtepi8_epi32(b8);                    // 8 i32
        __m256 b_f32 = _mm256_cvtepi32_ps(b32);                   // 8 f32
        acc_f32 = _mm256_fmadd_ps(a, b_f32, acc_f32);
    }
    // Apply per-channel scale
    __m256 scale = _mm256_loadu_ps(&scales[j]);
    acc_f32 = _mm256_mul_ps(acc_f32, scale);
    _mm256_storeu_ps(&C[i * N + j], acc_f32);
}
```

### Approach 2: i32 accumulation (faster for batch)

```cpp
for (size_t j = 0; j + 8 <= N; j += 8) {
    __m256i acc_i32 = _mm256_setzero_si256();
    for (size_t k = 0; k < K; k++) {
        __m256i a = _mm256_set1_epi32(A[i * K + k]); // broadcast i32
        // Load 8 i8 values
        __m128i b8 = _mm_loadl_epi64((__m128i*)&B_i8[k * N + j]);
        __m256i b32 = _mm256_cvtepi8_epi32(b8);
        acc_i32 = _mm256_add_epi32(acc_i32, _mm256_mullo_epi32(a, b32));
    }
    __m256 acc_f32 = _mm256_cvtepi32_ps(acc_i32);
    __m256 scale = _mm256_loadu_ps(&scales[j]);
    acc_f32 = _mm256_mul_ps(acc_f32, scale);
    _mm256_storeu_ps(&C[i * N + j], acc_f32);
}
```

**Problem**: `A[i][k]` is f32. Converting to i32 for multiplication loses precision. In practice, you'd quantize A to i8 too (for W8A8), or keep A in f32 and do f32 × i8 → f32.

For W8A8 (both weights and activations i8), you'd do:
```cpp
// Both A_i8 and B_i8 are INT8
// Accumulate in i32, then convert to f32 and apply scales
__m256i acc = _mm256_setzero_si256();
for (size_t kk = 0; kk < K; kk += 16) {
    // Load 16 i8 from A, 16 i8 from B
    // .. vpmaddubsw + vpmaddwd + vpaddd
}
// acc has 8 x i32 results
// convert to f32, multiply by scale_A * scale_B
```

### Fused f32×i8 Microkernel (for this project)

Let's focus on f32×i8→f32 with f32 accumulation (simpler, matches your f32 GEMM structure):

```cpp
void matmul_f32xi8(const float* A, const int8_t* B_i8, const float* scales,
                   float* C, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j + 8 <= N; j += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (size_t k = 0; k < K; k++) {
                __m256 a = _mm256_broadcast_ss(&A[i * K + k]);
                // Load 8 i8, convert to f32
                __m128i b8 = _mm_loadl_epi64((__m128i*)&B_i8[k * N + j]);
                __m256i b32 = _mm256_cvtepi8_epi32(b8);
                __m256 b_f32 = _mm256_cvtepi32_ps(b32);
                acc = _mm256_fmadd_ps(a, b_f32, acc);
            }
            // Apply per-output-channel scale
            __m256 s = _mm256_loadu_ps(&scales[j]);
            acc = _mm256_mul_ps(acc, s);
            _mm256_storeu_ps(&C[i * N + j], acc);
        }
        // scalar tail
    }
}
```

### Done Criteria

- [ ] f32 × i8 → f32 kernel correct (compare to f32 reference with same values)
- [ ] Per-channel scaling applied correctly
- [ ] Per-group scaling correct (if implemented)
- [ ] Benchmark: speedup over f32 GEMM (expect ~2-3x from reduced memory bandwidth)

---

## 14. Fused Epilogues

### Theory

Most inference GEMM calls are followed by a **fusion** (bias add, activation, etc.). Instead of separate kernel calls, fuse the epilogue into the GEMM store:

```
C[i][j] = Σₖ A[i][k] * B[k][j] + bias[j]  // fused bias
C[i][j] = relu(Σₖ A[i][k] * B[k][j])       // fused activation
```

This saves:
1. Kernel launch overhead
2. Extra memory pass (reading C, modifying, writing back)
3. Register spilling between kernels

### How to Fuse

Modify the store path of the microkernel:

```cpp
// Instead of plain store:
// _mm256_storeu_ps(&C[i*N+j], acc);

// Fused bias:
__m256 bias = _mm256_loadu_ps(&bias[j]);
acc = _mm256_add_ps(acc, bias);

// Fused ReLU:
acc = _mm256_max_ps(acc, _mm256_setzero_ps());

// Fused SiLU (sigmoid × x):
// acc = acc * sigmoid(acc)
// But sigmoid needs exp, which is expensive. Use approximation or do separately.

// Store
_mm256_storeu_ps(&C[i*N+j], acc);
```

### Template-Based Fusion (C++)

```cpp
struct IdentityEpilogue {
    static __m256 apply(__m256 acc, const float* /*bias*/, size_t /*j*/) {
        return acc;
    }
};

struct BiasReLUEpilogue {
    static __m256 apply(__m256 acc, const float* bias, size_t j) {
        __m256 b = _mm256_loadu_ps(&bias[j]);
        acc = _mm256_add_ps(acc, b);
        return _mm256_max_ps(acc, _mm256_setzero_ps());
    }
};

template <typename Epilogue = IdentityEpilogue>
void matmul_4x8_with_epilogue(
    const float* A_packed, const float* B_packed,
    float* C, const float* bias,
    size_t N, size_t i, size_t j,
    size_t kc, size_t mc, size_t nr
) {
    // ... compute acc0..acc3 ...
    _mm256_storeu_ps(&C[(i+0)*N + j],
        Epilogue::apply(acc0, bias, j));
    // ... etc ...
}
```

### Common Fusions

| Fusion | Operation | Implementation |
|---|---|---|
| +bias | `C += bias[j]` | `_mm256_add_ps` |
| +ReLU | `max(C, 0)` | `_mm256_max_ps(C, zero)` |
| +SiLU | `C * sigmoid(C)` | Requires exp approximation |
| +GELU | `0.5*C*(1+erf(C/√2))` | Requires exp/erf approx |
| +RMS Norm | `C / sqrt(mean(C²)+ε)` | Horizontal reduction (expensive) |
| +Dequant | `C * scale + zero_point` | `_mm256_mul_ps + _mm256_add_ps` |

### Not All Fusions Are Equal

- Addition/ReLU are essentially free (fuse them always).
- GELU/SiLU require expensive math (exp, erf). Decide if the cache benefit outweighs the math cost.
- LayerNorm/RMSNorm after matmul: requires horizontal reduction (mean, variance). Often better as separate kernel due to reduction overhead.

### Signature (Example)

```cpp
void matmul_bias_relu(
    const float* A, const float* B, float* C,
    const float* bias,
    size_t M, size_t N, size_t K
);
```

### Done Criteria

- [ ] Identity epilogue matches standard matmul
- [ ] Bias epilogue matches `matmul + bias` two-step
- [ ] Bias+ReLU epilogue matches `matmul + bias + relu` three-step
- [ ] Benchmark: speedup over separate operations (expect 1.5–2x)

---

## 15. Attention-Specific Kernels

### Theory

Attention has two GEMM-like operations:

#### 1. Q × K^T (Score Computation)

```
scores[h][t][s] = Q[h][t][d] × K[h][d][s]
```

For row-major:
- `Q` is `[B, H, T, D]` → `[B*H*T, D]`
- `K` is `[B, H, S, D]` → we need `K^T` → `[B*H*D, S]`

This is a batched GEMM where each head is independent. The inner dimension `D` is typically small (64–128), so:
- Arithmetic intensity is low for T=1 (decode)
- Higher for T=S=seq_len (prefill)

**Optimization for small D**: Instead of horizontal reductions, use vertical accumulation (register blocking in the D dimension). This is what FlashAttention does.

#### 2. Attention × V (Value Aggregation)

```
output[h][t][d] = scores[h][t][s] × V[h][s][d]
```

This is also a batched GEMM with small M (= T) and moderate D.

### Strided Batched for Multi-Head

The standard approach: reshape tensors and call strided-batched GEMM.

```
Q: [B, H, T, D] → stride_A = T * D  (for strided-batched over H)
K: [B, H, S, D] → stride_B = S * D
```

For GQA: `num_query_groups > num_kv_groups`. Each query group shares a K,V group. This is:
1. Strided-batched over K,V heads
2. Broadcast K,V within each group to all query heads

### GQA Kernel Sketch

```cpp
void attention_scores_gqa(
    const float* Q,    // [B, H_q, T, D]
    const float* K,    // [B, H_kv, S, D]
    float* scores,     // [B, H_q, T, S]
    size_t B, size_t H_q, size_t H_kv,
    size_t T, size_t S, size_t D,
    size_t groups = H_q / H_kv
) {
    for (size_t b = 0; b < B; b++) {
        for (size_t kv_h = 0; kv_h < H_kv; kv_h++) {
            // Pack K[kv_h] for reuse across group
            pack_B(K + b*H_kv*S*D + kv_h*S*D, K_packed, D, ...);

            for (size_t g = 0; g < groups; g++) {
                size_t q_h = kv_h * groups + g;
                // Q[q_h] × K_packed → scores[q_h]
                matmul_packed(Q + b*H_q*T*D + q_h*T*D, K_packed,
                             scores + b*H_q*T*S + q_h*T*S,
                             T, S, D);
            }
        }
    }
}
```

### Done Criteria

- [ ] Attention score GEMM: strided-batched over heads
- [ ] GQA: grouped broadcast of K/V
- [ ] Benchmark against direct GEMM call per head

---

## 16. Validation and Benchmarking

### Correctness Testing

Each variant must be validated against the reference (Step 2, scalar ikj):

```cpp
void validate_variant(
    const char* name,
    void (*kernel)(const float*, const float*, float*, size_t, size_t, size_t),
    size_t M, size_t N, size_t K,
    float tolerance = 1e-3f
) {
    std::vector<float> A(M*K), B(K*N), C_ref(M*N), C_test(M*N);
    // Fill with random data
    fill_random(A, B);

    // Reference
    matmul_scalar_ikj(A.data(), B.data(), C_ref.data(), M, N, K);

    // Test variant
    kernel(A.data(), B.data(), C_test.data(), M, N, K);

    // Compare
    float max_diff = 0;
    for (size_t i = 0; i < M*N; i++) {
        float diff = std::abs(C_test[i] - C_ref[i]);
        max_diff = std::max(max_diff, diff);
        EXPECT_NEAR(C_test[i], C_ref[i], tolerance);
    }
    // Record max_diff for documentation
    std::cout << name << " max diff: " << max_diff << std::endl;
}
```

### Test Matrix

| Test | M | N | K | What It Tests |
|---|---|---|---|---|
| Tiny square | 2 | 2 | 2 | Basic correctness |
| Non-square | 3 | 5 | 7 | General shapes |
| Power of 2 | 64 | 64 | 64 | SIMD alignment |
| Non-power-of-2 K | 4 | 4 | 13 | Tail in K |
| Non-power-of-2 N | 4 | 13 | 4 | Tail in N |
| M=1 decode | 1 | 4096 | 4096 | Decode path |
| M=1 wide | 1 | 51200 | 4096 | Vocab decode |
| Moderate | 128 | 128 | 128 | General |
| Large | 1024 | 1024 | 1024 | Scaling |
| Odd remainders | 5 | 17 | 31 | All tails |

### Benchmarking

```cpp
void benchmark_gemm(
    const char* name,
    void (*kernel)(const float*, const float*, float*, size_t, size_t, size_t),
    size_t M, size_t N, size_t K, size_t warmup_runs, size_t timed_runs
) {
    std::vector<float> A(M*K), B(K*N), C(M*N);
    fill_random(A, B);

    double total_flops = 2.0 * M * N * K;

    // Warmup
    for (size_t w = 0; w < warmup_runs; w++) {
        kernel(A.data(), B.data(), C.data(), M, N, K);
    }

    // Timed runs
    std::vector<double> times;
    for (size_t r = 0; r < timed_runs; r++) {
        auto start = now();
        kernel(A.data(), B.data(), C.data(), M, N, K);
        auto end = now();
        times.push_back(duration_ms(end - start));
    }

    // Compute statistics
    double median = median_of(times);
    double gflops = total_flops / (median / 1000.0) / 1e9;

    std::cout << std::setw(30) << name
              << " | " << std::setw(8) << median << " ms"
              << " | " << std::setw(8) << gflops << " GFLOPs"
              << std::endl;
}
```

### Observations to Track Per Step

Record in a table:

| Step | Variant | 64² | 128² | 256² | 512² | 1024² | 2048² | M=1 D4096 |
|---|---|---|---|---|---|---|---|---|
| 1 | Naive ijk | ms | ms | ms | ms | ms | ms | ms |
| 2 | Scalar ijk | ms | ms | ms | ms | ms | ms | ms |
| 3 | Scalar ikj | ms | ms | ms | ms | ms | ms | ms |
| 4 | Tiled | ms | ms | ms | ms | ms | ms | ms |
| 5 | AVX2 ikj | ms | ms | ms | ms | ms | ms | ms |
| 6 | 4×8 unpacked | ms | ms | ms | ms | ms | ms | ms |
| 7 | 4×8 packed | ms | ms | ms | ms | ms | ms | ms |
| 8 | 4×8 prefetch | ms | ms | ms | ms | ms | ms | ms |

---

## 17. Inference Engine Kernel Checklist

### Core GEMM Kernels

- [ ] `matmul(M, N, K)` — f32 general
- [ ] `matmul(M=1, N, K)` — f32 decode-optimized
- [ ] `matmul_strided_batched` — f32 multi-head
- [ ] `matmul_f32xi8` — quantized weights
- [ ] `matmul_f32xi8_strided_batched` — quantized multi-head

### Pre-packed Weight Support

- [ ] `pack_weights(B, K, N)` — offline weight prepacking
- [ ] `matmul_with_packed_B(A, B_packed, C, M, N, K)` — runtime kernel using packed weights

### Fused Operators

- [ ] `matmul_bias` — C += bias
- [ ] `matmul_bias_relu` — C = relu(C + bias)
- [ ] `matmul_bias_gelu` — C = gelu(C + bias)
- [ ] `matmul_bias_silu` — C = silu(C + bias) (if SiLU used; common in modern LLMs)
- [ ] `matmul_residual` — C = A + C (skip connection after GEMM)

### Attention Kernels

- [ ] `attention_scores(Q, K, scores)` — strided-batched Q×K^T for MHA
- [ ] `attention_scores_gqa(Q, K, scores)` — GQA variant
- [ ] `attention_values(scores, V, output)` — strided-batched scores×V
- [ ] `attention_values_gqa(scores, V, output)` — GQA variant

### Quantization Support

- [ ] `matmul_f32xi8` — per-channel scale
- [ ] `matmul_f32xi8_grouped` — per-group scale (group size 32/64/128)
- [ ] `matmul_f32xi8_strided_batched` — quantized multi-head

### Utility

- [ ] Weight packing tool (run once at model load)
- [ ] Activation packing tool (run at runtime per layer)
- [ ] Scale pre-computation (for quantized weights)

---

## Appendix: SIMD Intrinsics Quick Reference

All AVX2+FMA intrinsics you'll need:

### Load/Store

| Intrinsic | Operation |
|---|---|
| `_mm256_loadu_ps(ptr)` | Load 8 f32 (unaligned) |
| `_mm256_storeu_ps(ptr, v)` | Store 8 f32 (unaligned) |
| `_mm256_load_ps(ptr)` | Load 8 f32 (aligned, crash if not 32-byte aligned) |
| `_mm256_stream_ps(ptr, v)` | Non-temporal store (aligned) |
| `_mm256_maskstore_ps(ptr, mask, v)` | Conditional store (mask bits = sign bits) |

### Arithmetic

| Intrinsic | Operation |
|---|---|
| `_mm256_add_ps(a, b)` | `a + b` |
| `_mm256_mul_ps(a, b)` | `a * b` |
| `_mm256_fmadd_ps(a, b, c)` | `a * b + c` (fused) |
| `_mm256_max_ps(a, b)` | `max(a, b)` |
| `_mm256_min_ps(a, b)` | `min(a, b)` |

### Broadcast/Set

| Intrinsic | Operation |
|---|---|
| `_mm256_setzero_ps()` | Zero vector |
| `_mm256_set1_ps(x)` | Broadcast scalar x to 8 lanes |
| `_mm256_broadcast_ss(ptr)` | Load scalar from memory, broadcast |
| `_mm256_setr_epi32(x0..x7)` | Set 8 i32 values (for mask creation) |

### Conversion

| Intrinsic | Operation |
|---|---|
| `_mm256_cvtepi8_epi32(v128)` | Sign-extend 4 i8 → 4 i32 (128→256) |
| `_mm256_cvtepi32_ps(v)` | Convert 8 i32 → 8 f32 |
| `_mm256_cvtps_epi32(v)` | Convert 8 f32 → 8 i32 (round to nearest) |

### Horizontal Operations (Reduction)

| Intrinsic | Operation |
|---|---|
| `_mm256_hadd_ps(a, b)` | Horizontal add (interleave) |
| `_mm256_permute2f128_ps(a, b, imm)` | Extract/insert 128-bit lanes |

### Prefetch

| Intrinsic | Operation |
|---|---|
| `_mm_prefetch(ptr, hint)` | Software prefetch (hint: `_MM_HINT_T0`, `_MM_HINT_NTA`) |

---

## Appendix: Roofline Model Reference

```
Performance = min(compute_peak, bandwidth * arithmetic_intensity)

Compute peak (GFLOPS) = freq(GHz) * simd_width * FMA_per_cycle * cores
  AVX2 f32: freq * 8 * 2 = freq * 16 GFLOPS/core

Bandwidth (GB/s): measure with STREAM benchmark

Arithmetic intensity for GEMM:
  AI = 2 * M * N * K / (4*(M*K + K*N + M*N + 4*M*N))
                          (read A) (read B) (read C) (write C)
      
  For large M,N,K, AI ≈ 2*K / 12 (ignoring C read) = K/6

  For K=4096: AI ≈ 682 FLOPs/byte → compute bound ✅
  For K=64:   AI ≈ 10.7 FLOPs/byte → bandwidth bound ❌
```

---

## Appendix: Common Pitfalls Debugging Guide

| Symptom | Likely Cause |
|---|---|
| Wrong results for odd sizes | Off-by-one in tile boundary handling |
| Wrong results near matrix edges | Tile loop doesn't stop at M/N boundary |
| Bad performance on large matrices | Packing not fitting in L2 cache (tile too large) |
| Bad performance on small matrices | Tiling/packing overhead dominates |
| Worse after packing | Packing function slow (std::vector allocation per tile) |
| No prefetch benefit | Prefetch distance wrong; already L1 resident |
| NaN/Inf results | Uninitialized C; uninitialized padding in packed B |
| Quantized wrong by factor 2 | Scale applied in wrong order (pre/post accumulation) |

---

## Final Advice

**Implement in order. Do not skip steps.** Each step teaches a specific concept. The performance gap between Step 1 (naive) and Step 8 (packed 4×8) is typically **100–500x** on large matrices.

For your inference engine:
- All kernels should be validated against scalar reference
- Attention GEMMs benefit most from strided-batched
- Decode path needs the M=1 specialization
- Quantized kernels give 2–4x memory bandwidth reduction
- Fused epilogues save significant launch/latency overhead

When you eventually implement in your engine, structure the GEMM layer as a **kernel library** with:
- A runtime dispatch mechanism (static weights pre-packed, activations packed per-call)
- Templates for fusion (bias, activation, residual)
- Support for strided-batched across heads

This workbook should let you independently implement all the kernels. Each step has "Done Criteria" to mark progress. Good luck!
