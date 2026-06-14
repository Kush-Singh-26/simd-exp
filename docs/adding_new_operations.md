# Developer Guide: Adding New Operations

This guide explains how to add new mathematical or vector operations to the `simd` library cleanly, adhering to the modular operation-centric architecture.

---

## 1. Directory Structure

Every new operation (e.g., `sigmoid`) should be contained in a dedicated folder under `include/simd/ops/`:

```text
include/simd/ops/sigmoid/
├── scalar.hpp    # Standard C++ scalar reference implementation
├── simd.hpp      # AVX2 / FMA optimized implementation (guarded)
└── sigmoid.hpp   # Unified public dispatch header
```

---

## 2. Implementation Template

Let's use `sigmoid` as an example to see how to structure the three header files.

### A. `scalar.hpp` (Reference Code)
Define the reference implementation inside the `simd::impl` namespace. This code is used for fallback on non-AVX2 hardware and as a baseline for correctness testing.

```cpp
#pragma once
#include <cmath>
#include <cstddef>

namespace simd {
namespace impl {

inline void sigmoid_scalar(const float* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
    }
}

} // namespace impl
} // namespace simd
```

### B. `simd.hpp` (Intrinsics Code)
Define the optimized SIMD path inside `simd::impl`, guarded by `SIMD_AVX2_ENABLED` (which is defined in `common.hpp`). Include `math_utils.hpp` if your operation needs SIMD exp (e.g., softmax, sigmoid). Provide standard unaligned stores and optional non-temporal stream implementations. Always implement a scalar loop to handle any remaining tail elements.

```cpp
#pragma once
#include "../../common.hpp"
#include "../../math_utils.hpp"  // for avx2_exp_ps (if needed)
#include <cmath>
#include <cstddef>

namespace simd {
namespace impl {

#if defined(SIMD_AVX2_ENABLED)
inline void sigmoid_simd(const float* src, float* dst, size_t n) {
    size_t i = 0;
    __m256 one = _mm256_set1_ps(1.0f);
    
    // Process chunks of 8 floats
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(src + i);
        
        // (Perform SIMD operations here...)
        // For example (using math_utils.hpp):
        // __m256 exp_neg_x = simd::impl::avx2_exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), x));
        // __m256 denom = _mm256_add_ps(one, exp_neg_x);
        // __m256 result = _mm256_div_ps(one, denom);
        
        _mm256_storeu_ps(dst + i, result);
    }
    
    // Handle remaining tail elements
    for (; i < n; ++i) {
        dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
    }
}
#endif

} // namespace impl
} // namespace simd
```

### C. `sigmoid.hpp` (Dispatcher Code)
Expose the public API `simd::sigmoid` in the root `simd` namespace. Use preprocessor checks to transparently dispatch to the fastest available target.

```cpp
#pragma once
#include "scalar.hpp"
#include "simd.hpp"
#include <cstddef>

namespace simd {

inline void sigmoid(const float* src, float* dst, size_t n) {
#if defined(SIMD_AVX2_ENABLED)
    impl::sigmoid_simd(src, dst, n);
#else
    impl::sigmoid_scalar(src, dst, n);
#endif
}

} // namespace simd
```

---

## 3. Registering the Operation

Once your headers are created, make the operation globally available by adding its dispatch header to the master library header:

**[include/simd/simd.hpp](file:///home/kush26/Projects/simd-exp/include/simd/simd.hpp)**:
```cpp
#pragma once

#include "common.hpp"
#include "ops/abs/abs.hpp"
...
#include "ops/sigmoid/sigmoid.hpp" // Add your new operation here
```

---

## 4. Benchmarking the New Operation

Create a new benchmark file under `bench/` to verify performance characteristics against the scalar reference. Use the shared data generation utilities from `bench/bench_utils.hpp`:

**`bench/sigmoid_bench.cpp`**:
```cpp
#include <benchmark/benchmark.h>
#include <simd/simd.hpp>
#include <vector>
#include <cstddef>
#include "bench_utils.hpp"

static void BM_Sigmoid_Scalar(benchmark::State& state, DataType dtype) {
  size_t n = state.range(0);
  std::vector<float> src(n);
  gen_data_random(src, dtype);
  std::vector<float> dst(n);
  for (auto _ : state) {
    simd::impl::sigmoid_scalar(src.data(), dst.data(), n);
    benchmark::DoNotOptimize(dst.data());
  }
}
BENCHMARK_CAPTURE(BM_Sigmoid_Scalar, pos, DataType::POS)->Arg(1<<20);
BENCHMARK_CAPTURE(BM_Sigmoid_Scalar, neg, DataType::NEG)->Arg(1<<20);
BENCHMARK_CAPTURE(BM_Sigmoid_Scalar, rand, DataType::RAND)->Arg(1<<20);

#if defined(SIMD_AVX2_ENABLED)
static void BM_Sigmoid_Simd(benchmark::State& state, DataType dtype) {
  size_t n = state.range(0);
  std::vector<float> src(n);
  gen_data_random(src, dtype);
  std::vector<float> dst(n);
  for (auto _ : state) {
    simd::impl::sigmoid_simd(src.data(), dst.data(), n);
    benchmark::DoNotOptimize(dst.data());
  }
}
BENCHMARK_CAPTURE(BM_Sigmoid_Simd, pos, DataType::POS)->Arg(1<<20);
BENCHMARK_CAPTURE(BM_Sigmoid_Simd, neg, DataType::NEG)->Arg(1<<20);
BENCHMARK_CAPTURE(BM_Sigmoid_Simd, rand, DataType::RAND)->Arg(1<<20);
#endif

BENCHMARK_MAIN();
```

Your benchmark is automatically picked up by the globbing pattern in `CMakeLists.txt` and compiled into a standalone executable. Use `gen_data_random` for softmax-like operations (random data) and `gen_data_const` for element-wise ops (constant values).

---

## 5. Design Best Practices

1. **Memory Alignment**: For streaming/non-temporal stores (NT), the destination pointer **must** be 32-byte aligned for AVX2. Use the library's utility:
   ```cpp
   float* dst = static_cast<float*>(simd::aligned_alloc(32, size * sizeof(float)));
   ...
   simd::aligned_free(dst);
   ```
2. **Horizontal Reductions**: Reductions (like `sum` or `dot_prod` returning a single scalar) require lane permutation and swapping. Keep vertical accumulations inside the loop, and do the horizontal reduction once at the very end.
3. **Avoid Branches inside SIMD loops**: Keep your SIMD computations branchless by using operators like `_mm256_blendv_ps`, `_mm256_max_ps`, `_mm256_min_ps`, or bitwise masking.
