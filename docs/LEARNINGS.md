# SIMD Experiments — Learnings & Observations

## Overview

Five benchmarks exploring AVX2 SIMD intrinsics, Google Benchmark setup, and performance analysis.

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

## Cross-Cutting Observations

### Memory-bound vs Compute-bound

| Operation | Bytes per element | Bottleneck | Max Speedup |
|---|---|---|---|
| Sum (reduction) | 4 (read) | Memory | 3-4x |
| Dot product (1 acc) | 8 (read) | FMA latency | 2x |
| Dot product (4 acc) | 8 (read) | Compute | 8.8x |
| Clamp (store) | 8 (read+write) | Memory | 4-6x |
| ReLU (store) | 8 (read+write) | Memory | 4-5x |
| Abs (store) | 8 (read+write) | Memory | 3.5-4.5x |

### Non-temporal stores

- **When to use**: large arrays (>L3 cache) where `dst` is written but not read back
- **When to avoid**: small arrays (<L2 cache) where data fits in cache
- **Mechanism**: eliminates read-for-ownership (RFO) — the cache-line load preceding every normal store
- **Trade-off**: requires 32-byte aligned destination (`_mm_malloc`)

### Branch Misprediction

- Scalar ternary with unpredictable data can be 5-7x slower than predictable data
- SIMD `max`/`min`/`andnot` are unconditional — same speed for any input
- The effect is most visible at default optimization (no `-O3` auto-vectorization)

### Common Patterns

| Pattern | Intrinsics |
|---|---|
| Broadcast | `_mm256_set1_ps(x)` |
| Load (unaligned) | `_mm256_loadu_ps(ptr)` |
| Load (aligned) | `_mm256_load_ps(ptr)` — requires `alignas(32)` |
| Store (normal) | `_mm256_storeu_ps(ptr, v)` |
| Store (NT) | `_mm256_stream_ps(ptr, v)` — requires aligned `ptr` |
| Reduction (sum) | permute → hadd → hadd → extract |
| FMA | `_mm256_fmadd_ps(a, b, c)` |
| Clamp | `max(min(x, hi), lo)` |
| ReLU | `max(x, 0)` |
| Abs | `andnot(sign_mask, x)` |
| Negate | `xor(sign_mask, x)` |
| -|x| | `or(sign_mask, x)` or `andnot(sign_mask, xor(x, sign_mask))` |

---

## How to Run

```bash
# Single benchmark
cmake --build build --target sum_bench && ./build/sum_bench

# All benchmarks
cmake --build build --target run_all

# Filter specific benchmark
./build/relu_bench --benchmark_filter=rand
./build/clamp_bench --benchmark_repetitions=5 --benchmark_display_aggregates_only=true
```

## Tooling

```bash
# Quick perf analysis
perf stat ./build/sum_bench

# Disassemble to check for auto-vectorization
objdump -d build/sum_bench | grep -E '(vandps|vmaxps|vfmadd)'

# Check compiler optimization reports
g++ -O3 -mavx2 -mfma -fopt-info-vec-optimized bench/sum_bench.cpp -S -o /dev/null
```
