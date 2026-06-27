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
Define the reference implementation inside the `simd::impl` namespace. This code is used for fallback on non-AVX2 hardware and as a baseline for correctness testing. Mark it `constexpr` to allow compile-time evaluation.

```cpp
#pragma once
#include <cmath>
#include <cstddef>

namespace simd {
namespace impl {

inline constexpr void sigmoid_scalar(const float* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
    }
}

} // namespace impl
} // namespace simd
```

### B. `simd.hpp` (Intrinsics Code)
Define the optimized SIMD path inside `simd::impl`, guarded by `SIMD_AVX2_ENABLED` (which is defined in `common.hpp`). Include `math_utils.hpp` if your operation needs SIMD exp (e.g., softmax, sigmoid). Provide standard unaligned stores and optional non-temporal stream implementations. Always implement a scalar loop to handle any remaining tail elements. Use `[[assume]]` attributes to help the compiler optimize loop bounds.

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
        [[assume(n > 0)]];
        [[assume(i + 8 <= n)]];
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
Expose the public API `simd::sigmoid` using `std::span` for type-safe buffer access and `if consteval` to select the scalar path during constant evaluation. Use preprocessor checks to transparently dispatch to the fastest available target at runtime.

```cpp
#pragma once
#include "scalar.hpp"
#include "simd.hpp"
#include <span>

namespace simd {

inline void sigmoid(std::span<const float> src, std::span<float> dst) {
    if consteval {
        impl::sigmoid_scalar(src.data(), dst.data(), src.size());
    } else {
#if defined(SIMD_AVX2_ENABLED)
        impl::sigmoid_simd(src.data(), dst.data(), src.size());
#else
        impl::sigmoid_scalar(src.data(), dst.data(), src.size());
#endif
    }
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

With the unified benchmarking harness, you do not need to write standard Google Benchmark boilerplate. Create a new benchmark file under `bench/` and use the helper macros:

**`bench/sigmoid_bench.cpp`**:
```cpp
#include "bench_harness.hpp"
#include <simd/ops/sigmoid/scalar.hpp>
#include <simd/ops/sigmoid/simd.hpp>

SIMD_BENCH_UNARY(Sigmoid,
    simd::impl::sigmoid_scalar,
    simd::impl::sigmoid_simd,
    simd::impl::sigmoid_simd_nt,
    gen_data_random)

BENCHMARK_MAIN();
```

Your benchmark is automatically discovered by `CMakeLists.txt` and compiled into a standalone executable. 

Available macros in [bench/bench_harness.hpp](file:///home/kush26/Projects/simd-exp/bench/bench_harness.hpp):
- `SIMD_BENCH_UNARY(...)`: For element-wise ops, registers Scalar, SIMD, and SIMD NT variants. Passes extra bounds/arguments to target functions if provided (e.g. clamp).
- `SIMD_BENCH_UNARY_NO_NT(...)`: Same, but omits the NT store variant.
- `SIMD_BENCH_REDUCTION(Name, ScalarFn, SimdFn, GenFn, NInputs)`: For reductions. `NInputs` is `1` (like `sum`) or `2` (like `dot_prod`).
- `SIMD_BENCH_FIXED(Name, ScalarFn, SimdFn, SimdNtFn, TileSize)`: For fixed-dimension blocks (like transpose 4x4).

---

## 5. Correctness Testing

Create a unit test file under `test/`. Use [test/test_harness.hpp](file:///home/kush26/Projects/simd-exp/test/test_harness.hpp) to avoid writing duplicate loops. Your test is automatically picked up by CMake.

**`test/sigmoid_test.cpp`**:
```cpp
#include "test_harness.hpp"
#include <simd/ops/sigmoid/scalar.hpp>
#include <simd/ops/sigmoid/simd.hpp>
#include <simd/ops/sigmoid/sigmoid.hpp>

TEST(SigmoidTest, Scalar_KnownValues) {
    float src[] = {0.0f};
    float dst[1];
    simd::impl::sigmoid_scalar(src, dst, 1);
    EXPECT_FLOAT_EQ(dst[0], 0.5f);
}

TEST(SigmoidTest, Simd_MatchesScalar) {
#if defined(SIMD_AVX2_ENABLED)
    for (size_t n : kStdSizes) {
        auto src = make_random(n, -10.f, 10.f);
        std::vector<float> scalar_dst(n), simd_dst(n);
        simd::impl::sigmoid_scalar(src.data(), scalar_dst.data(), n);
        simd::impl::sigmoid_simd(src.data(), simd_dst.data(), n);
        check_near(scalar_dst.data(), simd_dst.data(), n, 1e-5f, "n=" + std::to_string(n));
    }
#endif
}

TEST(SigmoidTest, Dispatcher_MatchesScalar) {
    size_t n = 1024;
    auto src = make_random(n, -10.f, 10.f);
    std::vector<float> scalar_dst(n), dispatch_dst(n);
    simd::impl::sigmoid_scalar(src.data(), scalar_dst.data(), n);
    simd::sigmoid(src, dispatch_dst);
    check_near(scalar_dst.data(), dispatch_dst.data(), n, 1e-5f, "sigmoid dispatcher");
}
```

Standard helpers in [test/test_harness.hpp](file:///home/kush26/Projects/simd-exp/test/test_harness.hpp):
- `kStdSizes`: `{1, 7, 8, 9, 1023, 1024, 1<<20}` - handles degenerate, tail, boundary, aligned, and large stress sizes.
- `make_random(n, lo, hi)`, `make_const(n, val)`, `make_boundary_stress(n, lo, hi)`: Data generators.
- `check_exact(a, b, n)`: Verifies floating point bitwise identity.
- `check_near(a, b, n, tol)`: Verifies values within floating point absolute tolerances.
- `check_scalar_near(a, b, tol)`: Checks single reduction outcomes.

---

## 5. Design Best Practices

1. **Memory Alignment**: For streaming/non-temporal stores (NT), the destination pointer **must** be 32-byte aligned for AVX2. Use the library's utility (returns `std::expected`, check before use):
   ```cpp
   auto result = simd::aligned_alloc(32, size * sizeof(float));
   if (!result.has_value()) return;
   float* dst = static_cast<float*>(result.value());
   ...
   simd::aligned_free(dst);
   ```
2. **Horizontal Reductions**: Reductions (like `sum` or `dot_prod` returning a single scalar) require lane permutation and swapping. Keep vertical accumulations inside the loop, and do the horizontal reduction once at the very end.
3. **Avoid Branches inside SIMD loops**: Keep your SIMD computations branchless by using operators like `_mm256_blendv_ps`, `_mm256_max_ps`, `_mm256_min_ps`, or bitwise masking.
