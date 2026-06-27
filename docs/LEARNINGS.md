# SIMD Experiments — Learnings & Observations

## Overview

Eight benchmarks exploring AVX2 SIMD intrinsics, Google Benchmark setup, and performance analysis. Covers element-wise ops (abs, clamp, relu), reductions (sum, dot_prod), softmax variants, and a shared SIMD math library (`math_utils`).

---

## 1. `sum_bench` — Array Sum

### What it does
Sum an array of 1M floats. Scalar vs SIMD (AVX).

### Intrinsics introduced
| Intrinsic | Purpose |
|---|---|
| `_mm256_setzero_ps()` | Create zero vector |
| `_mm256_loadu_ps(ptr)` | Load 8 unaligned floats |
| `_mm256_add_ps(a, b)` | Add 8 float pairs |
| `_mm256_permute2f128_ps(v, v, 1)` | Swap high/low 128-bit lanes |
| `_mm256_hadd_ps(a, b)` | Horizontal add (adjacent pairs) |
| `_mm256_castps256_ps128(v)` | Extract low 128 bits |
| `_mm_cvtss_f32(v)` | Extract lane 0 as scalar |

### Results (1M elements)

| Version | Time |
|---|---|
| Scalar | 5.76 ms |
| SIMD | 1.41 ms |
| **Speedup** | **~4.1x** |

### Learnings

- **Horizontal reduction** — summing a vector down to a scalar requires lane-swapping + hadd chain. The 3-step (permute → add → hadd → hadd) reduces 8 values to 1 without store/reload.
- **Memory vs compute** — sum is memory-bound for large arrays (only 1 add per 4 bytes loaded). Speedup is limited by memory bandwidth, not arithmetic throughput.

---

## 2. `dot_prod` — Dot Product

### What it does
`sum(a[i] * b[i])` over 1M elements. Three versions: scalar, SIMD (1 accumulator), SIMD (4 accumulators).

### Intrinsics introduced
| Intrinsic | Purpose |
|---|---|
| `_mm256_fmadd_ps(a, b, c)` | Fused multiply-add: `a*b + c` |

### Results (1M elements)

| Version | Time | Speedup |
|---|---|---|
| Scalar | 5.30 ms | 1x |
| SIMD (1 accumulator) | 2.67 ms | ~2x |
| SIMD (4 accumulators) | 0.60 ms | **~8.8x** |

### Learnings

- **FMA** — `_mm256_fmadd_ps` does multiply+add in one instruction with ~0.5 cycle throughput on modern CPUs. One FMA replaces two instructions (mul + add) from scalar.
- **Multiple accumulators** — the 1-accumulator version is bottlenecked by FMA latency (4 cycles). The `vsum` register must be read, multiplied, added, and written before the next iteration can use it. With 4 accumulators, each iteration chains 4 independent FMAs, hiding the 4-cycle latency. Result: **4x further improvement** over single-accumulator.
- **Law of diminishing returns** — 4 accumulators is already enough for AVX2 on most CPUs. 8 gives negligible improvement because the CPU can only schedule so many independent operations.
- **Dot product is compute-bound** — unlike sum, dot product reads 2 floats per element and does 2 ops (mul+add). The FMA instruction keeps the ALU busy, making it compute-bound. This is why multiple accumulators help so much.

---

## 3. `clamp_bench` — Clamp to `[lo, hi]`

### What it does
`clamp(x, lo, hi)` = `min(max(x, lo), hi)` over 1M floats. Three versions: scalar, SIMD (normal store), SIMD (non-temporal store).

### Intrinsics introduced
| Intrinsic | Purpose |
|---|---|
| `_mm256_set1_ps(x)` | Broadcast scalar to all 8 lanes |
| `_mm256_max_ps(a, b)` | Element-wise max |
| `_mm256_min_ps(a, b)` | Element-wise min |
| `_mm256_stream_ps(ptr, v)` | Non-temporal store (bypass cache) |
| `_mm_malloc(n, align)` | Aligned allocation for NT stores |
| `_mm_free(ptr)` | Free aligned allocation |

### Results (1M elements)

| Version | Time | Speedup vs Scalar |
|---|---|---|
| Scalar | 5.73 ms | 1x |
| SIMD (storeu) | 1.36 ms | **4.2x** |
| SIMD (stream/NT) | 0.96 ms | **5.9x** |

### Learnings

- **Memory-bound kernel** — clamp reads 1 float and writes 1 float per element. 8 bytes of memory traffic per element. The SIMD arithmetic (max+min) finishes so fast that the CPU sits idle waiting for memory. Speedup is capped at ~4x instead of the theoretical 8x.
- **Non-temporal stores** — `_mm256_stream_ps` writes directly to memory, bypassing the read-for-ownership (RFO) that normal stores require. Normal stores first pull the cache line into L1 (read), then modify it. For `dst` that's never read back, the RFO is wasted work. NT stores eliminate it.
- **NT trade-off**: helps at large sizes (1M), neutral at medium (4K), hurts at small (1K) — because bypassing cache is a penalty when data does fit in cache.

| Size | storeu | stream | NT benefit |
|---|---|---|---|
| 1K (4 KB) | 0.94 µs | 1.01 µs | **-7%** (slower) |
| 4K (16 KB) | 3.78 µs | 3.76 µs | ~same |
| 1M (4 MB) | 1.36 ms | 0.96 ms | **+41%** (faster) |

- **Benchmark setup matters** — originally the functions allocated a `std::vector` inside the timed loop (malloc + memcpy every iteration). Moving the allocation outside dropped scalar from 18.7ms to 5.8ms — the original was benchmarking allocation, not clamp.

---

## 4. `relu_bench` — ReLU Activation

### What it does
`relu(x) = max(0, x)` over 1M floats. Tests three data distributions: all-positive, all-negative, random mix (~50/50).

### Intrinsics introduced
| Intrinsic | Purpose |
|---|---|
| `_mm256_setzero_ps()` | Zero vector |
| `_mm256_max_ps(a, b)` | Element-wise max (used for ReLU) |

### Results (1M elements)

| Version | All Positive | All Negative | Random |
|---|---|---|---|
| **Scalar** (ternary) | 2.81 ms | 2.05 ms | **14.36 ms** |
| **SIMD** (max_ps) | 1.11 ms | 1.10 ms | 1.09 ms |
| **SIMD NT** | 0.70 ms | 0.72 ms | 0.73 ms |

### Learnings

- **Branch misprediction** — scalar at random (14.36ms) is **5-7x slower** than scalar at positive (2.81ms) despite doing the same arithmetic. The ternary `x > 0 ? x : 0` compiles to a real branch at `-O0`. With 50/50 random data, the branch predictor guesses wrong ~50% of the time, each mispredict costing ~15-20 cycles of pipeline flush.
- **Branchless SIMD** — `_mm256_max_ps` is unconditional. Same speed for all data distributions. This is why SIMD is critical for ML kernels where input distributions are unpredictable.
- **Predictor training at small sizes** — at 1K/4K, the scalar random penalty is smaller because the branch predictor can train on the pattern (arrays are small enough to fit in the predictor's history buffer).
- **Compiler optimization note** — with `-O3`, GCC may optimize the ternary to a branchless `maxss` instruction, eliminating the misprediction entirely. The results above are at default CMake optimization (no explicit `-O` flags).

---

## 5. `abs_bench` — Absolute Value (Bitwise)

### What it does
`abs(x)` over 1M floats using bit manipulation: `andnot(sign_mask, x)` clears the sign bit.

### Intrinsics introduced
| Intrinsic | Purpose |
|---|---|
| `_mm256_set1_ps(-0.0f)` | Create sign-bit mask (0x80000000...) |
| `_mm256_andnot_ps(a, b)` | `(~a) & b` — clear bits set in `a` |

### IEEE 754 float format
```
 31   30      23  22                0
┌────┬──────────┬────────────────────┐
│ S  │ Exponent │     Mantissa       │
└────┴──────────┴────────────────────┘
```
- Bit 31 = sign (0 = positive, 1 = negative)
- `-0.0f` = `0x80000000` (only sign bit set)
- `|x|` = clear bit 31 → `andnot_ps(sign_mask, x)`
- `-x` = flip bit 31 → `xor_ps(sign_mask, x)`
- `-|x|` = set bit 31 → `or_ps(sign_mask, x)`

### Results (1M elements)

| Version | All Positive | All Negative | Random |
|---|---|---|---|
| **Scalar** (std::abs) | 3.64 ms | 3.38 ms | 3.65 ms |
| **SIMD** (andnot) | 1.04 ms | 1.10 ms | 1.07 ms |
| **SIMD NT** | 0.83 ms | 0.80 ms | 0.79 ms |

### Learnings

- **Compiler is smart** — `std::abs` compiles to `vandps` — the same single instruction as the hand-written SIMD version (just scalar vs vector width). Scalar is flat across all data types because there's no branch. The ternary version `x < 0 ? -x : x` might also be optimized to `vandps` depending on compiler flags.
- **Scalar vs SIMD gap** — the 3.5x speedup comes purely from 8-wide processing (8 vs 1 element per instruction), not from algorithmic improvement. Both are doing the same `andnot` operation.
- **Bit manipulation insight** — treating float bits as integers using `_mm256_andnot_ps` / `or_ps` / `xor_ps` enables operations (abs, negate, copy-sign, etc.) that would otherwise require branches or comparisons. These are the cheapest SIMD operations available (~1 cycle latency, 0.5 cycle throughput).

---

## 6. `softmax_bench` — Standard 3-Pass Softmax

### What it does
`softmax(x_i) = exp(x_i - max) / Σ exp(x_j - max)` over an array. Three versions: scalar, SIMD (storeu), SIMD (NT store). Tested with positive, negative, and random data distributions.

### Algorithm (3 passes)

1. **Pass 1 — Find max**: horizontal max across all elements
2. **Pass 2 — Exp and sum**: compute `exp(x_i - max)` for each element, accumulate sum
3. **Pass 3 — Normalize**: multiply each element by `1/sum`

### Intrinsics introduced
| Intrinsic | Purpose |
|---|---|
| `_mm256_max_ps(a, b)` | Vector max (find global max) |
| `_mm256_sub_ps(a, b)` | Subtract max from each element |
| `avx2_exp_ps(x)` | SIMD exp approximation (< 3 ULP) |
| `_mm256_add_ps(a, b)` | Accumulate exp sum |
| `_mm256_mul_ps(a, b)` | Normalize by reciprocal |
| `_mm256_extractf128_ps(v, 1)` | Extract high 128 bits for horizontal reduction |
| `_mm_movehl_ps(a, b)` | Horizontal max/sum reduction |
| `_mm_movehdup_ps(v)` | Horizontal max/sum reduction |
| `_mm_cvtss_f32(v)` | Extract scalar result |
| `_mm256_stream_ps(ptr, v)` | NT store (softmax_nt) |

### Results (1M elements)

| Version | All Positive | All Negative | Random |
|---|---|---|---|
| **Scalar** | 21.53 ms | 12.23 ms | 14.09 ms |
| **SIMD** | 2.31 ms | 3.28 ms | 4.31 ms |
| **SIMD NT** | 3.35 ms | 4.28 ms | 4.96 ms |
| **Speedup (SIMD)** | **9.3x** | **3.7x** | **3.3x** |

### Learnings

- **SIMD exp is the bottleneck** — each element requires `avx2_exp_ps` (~15 instructions), making softmax **compute-bound** rather than memory-bound. This is why SIMD speedup is high (9.3x for positive data) despite reading/writing the full array 3 times.
- **3-pass vs 2-pass** — the standard softmax requires 3 passes over the data (find max, compute exp+sum, normalize). For large arrays, this means 3× memory traffic. Online softmax (below) reduces this to 2 passes.
- **Data distribution matters** — positive data is fastest because the SIMD loop processes more efficiently (no branch divergence in horizontal reductions). Negative data is slower because `exp(x - max)` produces smaller values, and the horizontal sum has more denormal-like values.
- **NT stores hurt softmax** — unlike clamp/abs where NT saves ~40%, NT softmax is **slower** because the normalization pass reads `dst` back (non-temporal write + normal read = wasted RFO). NT is only beneficial when `dst` is write-once.
- **Horizontal reductions are expensive** — the max and sum reductions use 4 instructions each (extract, movehl, movehdup, cvtss). This is unavoidable for SIMD reductions but dominates at small sizes.

---

## 7. `online_softmax_bench` — Online (2-Pass) Softmax

### What it does
Same softmax computation, but uses the **online algorithm** that computes max and sum in a single pass, reducing from 3 passes to 2.

### Algorithm (2 passes)

**Pass 1 — Online max/sum** (streaming):
```
for each chunk of 8 elements:
    new_max = max(old_max, chunk_max)
    correction = exp(old_max - new_max)
    sum = sum * correction + sum(chunk_exp)
    old_max = new_max
```

**Pass 2 — Compute exp and normalize**:
```
for each element:
    dst[i] = exp(src[i] - global_max) / global_sum
```

### Key optimization: correction factor blend

When the max doesn't change within a chunk, `correction = exp(old_max - new_max) = exp(0) = 1.0`. The blend optimization skips the correction exp:

```cpp
__m256 v_mask = _mm256_cmp_ps(v_m, v_m_prev, _CMP_NEQ_OQ);
__m256 v_correction_exp = avx2_exp_ps(_mm256_sub_ps(v_m_prev, v_m));
v_correction_factor = _mm256_blendv_ps(v_correction_factor, v_correction_exp, v_mask);
```

- `_mm256_cmp_ps` — compares old_max vs new_max per lane
- `_mm256_blendv_ps` — selects 1.0 (no change) or `exp(old - new)` (max changed) per lane

This saves an `avx2_exp_ps` call (~15 instructions) when the max is stable.

### Results (1M elements)

| Version | All Positive | All Negative | Random |
|---|---|---|---|
| **Scalar** | 34.73 ms | 20.45 ms | 26.29 ms |
| **SIMD** | 6.60 ms | 5.14 ms | 5.04 ms |
| **Speedup** | **5.3x** | **4.0x** | **5.2x** |

### Learnings

- **2-pass saves memory traffic** — online softmax reads the array twice (pass 1 + pass 2) instead of three times (max, exp+sum, normalize). For memory-bound scenarios, this is a significant improvement.
- **Online softmax is slower than plain softmax SIMD** — despite fewer passes, online softmax has higher per-element cost in pass 1: it computes `exp(old_max - new_max)` for correction, which is an `avx2_exp_ps` call per chunk. For 1M elements: plain softmax SIMD takes 2.31ms (pos) vs online takes 6.60ms (pos). The extra exp computation outweighs the pass reduction.
- **When online wins** — online softmax is designed for **fused attention** where you process tokens incrementally and can't afford to re-scan the entire sequence. In a transformer decoder, each new token's softmax can update the running max/sum without re-processing all previous tokens. The 2-pass structure enables this streaming pattern.
- **Correction factor blend is effective** — for positive data, most chunks don't update the global max, so the blend optimization skips the correction exp. This is why positive data (6.60ms) is slower than random (5.04ms) — random data has more max updates, but the blend optimization handles it efficiently.
- **Horizontal reductions in pass 1** — same cost as plain softmax (extract, shuffle, add). The online algorithm adds ~5 extra instructions per chunk (correction exp, blend, multiply-accumulate) vs the plain 3-pass approach.

---

## 8. `math_utils_test` — SIMD exp Accuracy

### What it does
Verifies `avx2_exp_ps` accuracy against `std::exp` across 7 test categories.

### Test coverage

| Test | What it verifies |
|---|---|
| KnownValues | 8 specific values (0, ±1, ±2, ±0.5, ln(2)) — < 3 ULP each |
| PositiveRange | 0 to 87 in 0.7 steps — scans entire positive domain |
| NegativeRange | 0 to -87 in 0.7 steps — scans entire negative domain |
| LargeValues | 88, -88, 88.3, -88.3 — boundary clamping behavior |
| SubnormalFlushToZero | -100, -95, -90, -89 — verifies subnormal → 0 |
| RandomData | 1024 random values in [-10, 10] — max ULP across batch |
| BatchComparison | 8 random values — relative error < 1e-6 |

### Key results

- Max ULP error across 1024 random values: **< 3 ULP**
- Max relative error: **~1.08e-07**
- Subnormal inputs (`x < -88.37`): correctly flushed to 0
- Boundary values: clamp behavior verified (88.3 → inf, -88.3 → 0)

### Learnings

- **Estrin vs Horner ILP** — the Estrin scheme reduces critical path from 7 FMAs (Horner, ~28 cycles) to 4 FMAs (~16 cycles) by exposing 3 independent FMAs at Level 0. The trade-off is 2 extra multiplies for x² and x⁴, but these issue in parallel with Level 0 FMAs.
- **Cody-Waite is essential** — without the two-step range reduction, the single-step `x - n*ln2` loses ~6 bits of precision due to catastrophic cancellation. The HI/LO split of ln2 ensures r is accurate to full float precision.
- **`cvtps_epi32` vs `floor`** — round-to-nearest gives symmetric reduction interval `[-ln2/2, ln2/2]` centered at 0, which minimizes polynomial error. Floor gives `[0, ln2)` which is asymmetric and requires different polynomial coefficients.
- **Testing strategy** — the 7-test suite covers: specific values, entire domain sweep, boundary conditions, subnormal behavior, and statistical random testing. This is more thorough than a simple random test because it catches edge cases at domain boundaries.

---

## 9. C++23 Migration — `std::mdspan` and `std::expected`

### What changed

Migrated from C++20 to C++23. Two features adopted:

1. **`std::mdspan`** — replaced hardcoded 4x4 `mat_transpose` with arbitrary NxM via multidimensional views
2. **`std::expected`** — replaced `throw std::bad_alloc` in `aligned_alloc` with type-safe error returns

### `std::mdspan` in mat_transpose

**Before** (C++20, hardcoded 4x4):
```cpp
inline void transpose_scalar(const float* src, float* dst) {
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            dst[j * 4 + i] = src[i * 4 + j];  // manual index math
        }
    }
}
```

**After** (C++23, arbitrary NxM):
```cpp
inline void transpose_scalar(const float* src, float* dst, size_t rows, size_t cols) {
    std::mdspan<const float, std::dextents<size_t, 2>> src_md(src, rows, cols);
    std::mdspan<float, std::dextents<size_t, 2>> dst_md(dst, cols, rows);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            dst_md[j, i] = src_md[i, j];  // multidimensional subscript
        }
    }
}
```

**Key points:**
- `std::mdspan` is a non-owning view — wraps a raw pointer + dimensions, no allocation or copy
- `std::dextents<size_t, 2>` = 2 dynamic dimensions (compile-time extents like `extents<size_t, 4, 4>` also available)
- `operator[](i, j)` computes the flat index internally — eliminates manual stride arithmetic
- `sizeof(mdspan)` is ~24 bytes (pointer + 2 extents) — zero overhead after inlining
- Dispatcher auto-selects SIMD 4x4 for `rows==4 && cols==4`, scalar for everything else

### `std::expected` for aligned_alloc

**Before** (C++20, exceptions):
```cpp
inline void* aligned_alloc(size_t alignment, size_t size) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        throw std::bad_alloc();  // exceptions often disabled in SIMD code
    }
    return ptr;
}
```

**After** (C++23, expected):
```cpp
inline std::expected<void*, std::errc> aligned_alloc(size_t alignment, size_t size) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return std::unexpected(std::errc::not_enough_memory);
    }
    return ptr;
}
```

**Key points:**
- `std::expected<T, E>` is C++23's replacement for `std::optional` + error info
- Callers check `has_value()` before dereferencing `.value()`
- Zero overhead — same binary layout as a union + bool flag
- Many SIMD/ML codebases disable exceptions (`-fno-exceptions`) — `expected` gives type-safe error handling without them
- Callers in bench files use early return: `if (!result.has_value()) return;`

### Other C++23 features evaluated

| Feature | Verdict | Reason |
|---------|---------|--------|
| `if consteval` | Not used | `#if defined(SIMD_AVX2_ENABLED)` preprocessor guards are the right tool for ISA detection |
| `std::unreachable` | Not used | mat_transpose SIMD is a single 4x4 block with no loops; no applicable hot paths |
| Deducing `this` | Not used | Minimal const/non-const overload duplication in current codebase |
| `std::print` | Not used | Nice-to-have for test output, not functional |

---

## Cross-Cutting Observations

### Memory-bound vs Compute-bound

| Operation | Bytes per element | Bottleneck | Max Speedup |
|---|---|---|---|
| Sum (reduction) | 4 (read) | Memory | 3-4x |
| Dot product (4 acc) | 8 (read) | Compute | 8.8x |
| Clamp (store) | 8 (read+write) | Memory | 4-6x |
| ReLU (store) | 8 (read+write) | Memory | 4-5x |
| Abs (store) | 8 (read+write) | Memory | 3.5-4.5x |
| **Softmax (3-pass)** | **24 (3×8, read+write)** | **Compute (exp)** | **3-9x** |
| **Online Softmax (2-pass)** | **16 (2×8, read+write)** | **Compute (exp+blend)** | **4-5x** |

Softmax is the only **compute-bound** operation in the library — the SIMD exp (~15 instr/element) dominates, not memory bandwidth. Online softmax has fewer passes but higher per-element cost in pass 1.

### Non-temporal stores

- **When to use**: large arrays (>L3 cache) where `dst` is written but not read back
- **When to avoid**: small arrays (<L2 cache) or when `dst` is read back (e.g., softmax normalization)
- **Mechanism**: eliminates read-for-ownership (RFO) — the cache-line load preceding every normal store
- **Trade-off**: requires 32-byte aligned destination (`simd::aligned_alloc`)
- **Softmax NT is slower** — the normalization pass reads `dst`, defeating the NT purpose

### Branch Misprediction

- Scalar ternary with unpredictable data can be 5-7x slower than predictable data
- SIMD `max`/`min`/`andnot` are unconditional — same speed for any input
- Online softmax blend optimization uses `_mm256_cmp_ps` + `_mm256_blendv_ps` — branchless max detection

### Common Patterns

| Pattern | Intrinsics |
|---|---|
| Broadcast | `_mm256_set1_ps(x)` |
| Load (unaligned) | `_mm256_loadu_ps(ptr)` |
| Load (aligned) | `_mm256_load_ps(ptr)` — requires `alignas(32)` |
| Store (normal) | `_mm256_storeu_ps(ptr, v)` |
| Store (NT) | `_mm256_stream_ps(ptr, v)` — requires aligned `ptr` |
| Horizontal max | extractf128 → max_ps → movehl → movehdup → cvtss |
| Horizontal sum | extractf128 → add_ps → movehl → movehdup → cvtss |
| FMA | `_mm256_fmadd_ps(a, b, c)` |
| Clamp | `max(min(x, hi), lo)` |
| ReLU | `max(x, 0)` |
| Abs | `andnot(sign_mask, x)` |
| Blend (branchless) | `_mm256_cmp_ps` + `_mm256_blendv_ps` |

---

## How to Run

```bash
# Build everything
make

# Run all tests
make test

# Run all benchmarks
make bench

# Single benchmark
make build && ./build/softmax_bench

# Filter specific benchmark
./build/relu_bench --benchmark_filter=rand
./build/softmax_bench --benchmark_repetitions=5 --benchmark_display_aggregates_only=true

# Run specific test
./build/math_utils_test --gtest_filter=MathUtilsTest.RandomData
```

## Tooling

```bash
# Quick perf analysis
perf stat ./build/softmax_bench

# Disassemble to check for auto-vectorization
objdump -d build/softmax_bench | grep -E '(vfmadd|vmaxps|vmovups)'

# Check compiler optimization reports
g++ -O3 -mavx2 -mfma -fopt-info-vec-optimized bench/softmax_bench.cpp -S -o /dev/null
```
