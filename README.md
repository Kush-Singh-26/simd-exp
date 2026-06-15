# SIMD Experiments

A header-only C++23 library and benchmark suite for AVX2 (Advanced Vector Extensions 2) SIMD intrinsics.

## Features

- Modular, operation-centric header-only library under `include/simd/`
- Safe compile-time dispatching with automatic standard C++ scalar fallbacks if AVX2 is not supported
- High-performance unrolled implementations (FMA, unaligned loads, non-temporal stores)
- Benchmarking suite powered by Google Benchmark
- Unit tests powered by Google Test with scalar fallback coverage
- C++23 features: `std::mdspan` (multidimensional views), `std::expected` (type-safe error handling)

## Operations

| Operation | Scalar | SIMD | NT Store | Description |
|-----------|--------|------|----------|-------------|
| `abs` | Yes | Yes | Yes | Element-wise absolute value |
| `clamp` | Yes | Yes | Yes | Element-wise clamp to [lo, hi] |
| `dot_prod` | Yes | Yes (FMA) | No | Dot product of two vectors |
| `mat_transpose` | Yes | Yes | Yes | Arbitrary NxM matrix transpose (`std::mdspan`) |
| `online_softmax` | Yes | Yes | No | Single-pass softmax (online max/sum) |
| `relu` | Yes | Yes | Yes | Element-wise rectified linear unit |
| `softmax` | Yes | Yes | Yes | Standard 3-pass softmax |
| `sum` | Yes | Yes | No | Sum of elements |

## Repository Structure

- `include/simd/`: Header-only library source code
  - `simd.hpp`: Master header including all operations
  - `common.hpp`: Alignments, allocators, and CPU dispatch logic
  - `math_utils.hpp`: Shared AVX2 exp approximation (Estrin polynomial, Cody-Waite range reduction, < 3 ULP / ~1e-7 accuracy — see [docs/exp_approximation.md](docs/exp_approximation.md)) and horizontal reductions (`hsum_ps`, `hmax_ps`, `hmin_ps`)
  - `ops/`: Modular directory containing per-operation headers (SIMD, scalar, and dispatcher)
- `bench/`: Source code for performance benchmarks
  - `bench_utils.hpp`: Shared data generation utilities
  - `results/`: JSON benchmark output (tracked in git)
- `test/`: Google Test unit tests for all operations
- `docs/`: In-depth explanations, math derivations, and learning journals

## Build Instructions

### Prerequisites

- C++23 compiler (GCC 13+ or Clang 17+)
- CMake >= 3.16
- Google Benchmark (system or FetchContent)
- Google Test (fetched automatically via FetchContent)

### Setup and Compilation

```bash
# Configure the build directory
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build all targets
cmake --build build
```

### Running Tests

```bash
# Run all unit tests
ctest --test-dir build --output-on-failure
```

### Running Benchmarks

```bash
# Run a specific benchmark (e.g., softmax)
./build/softmax_bench

# Run all benchmarks sequentially
cmake --build build --target run_all
```

### Test Coverage

9 test suites, 134 tests covering all operations. Scalar tests run unconditionally; SIMD tests require AVX2:

| Test | Scalar | SIMD | Dispatcher | What it verifies |
|------|--------|------|------------|------------------|
| `abs_test` | 7 tests | 4 tests | 3 tests | Bitwise abs, edge cases, public API |
| `clamp_test` | 7 tests | 4 tests | 3 tests | Clamp to [lo, hi], public API |
| `dot_prod_test` | 7 tests | 3 tests | 2 tests | Dot product accuracy (requires FMA), public API |
| `mat_transpose_test` | 15 tests | 4 tests | 2 tests | 4x4 + arbitrary NxM transpose (mdspan), public API |
| `math_utils_test` | — | 15 tests | — | `avx2_exp_ps` < 3 ULP + hreduce (hsum/hmax/hmin/hsum_sq) |
| `online_softmax_test` | 8 tests | 4 tests | 2 tests | Online softmax + plain vs online, public API |
| `relu_test` | 7 tests | 4 tests | 3 tests | ReLU activation, public API |
| `softmax_test` | 8 tests | 4 tests | 3 tests | Softmax correctness, public API |
| `sum_test` | 7 tests | 3 tests | 2 tests | Sum reduction, public API |

## Integration

### Header-Only Copy-Paste

Copy the `include/simd` directory directly into your project. Include `simd/simd.hpp` to access the APIs:

```cpp
#include <simd/simd.hpp>

// Elements will be processed using AVX2 if compiled with -mavx2,
// otherwise it will automatically fallback to standard C++ scalar loops.
simd::relu(src, dst, size);

// Matrix transpose (arbitrary NxM via std::mdspan):
simd::transpose(src, dst, rows, cols);

// For SIMD exp (used internally by softmax/online_softmax):
// avx2_exp_ps() in math_utils.hpp provides < 3 ULP accuracy via
// Estrin polynomial evaluation + Cody-Waite range reduction.
```

### C++23 Features Used

| Feature | Where | What it does |
|---------|-------|-------------|
| `std::mdspan` | `mat_transpose/scalar.hpp` | Non-owning 2D view — enables arbitrary NxM transpose (was hardcoded 4x4) |
| `std::expected` | `common.hpp` | Type-safe error handling — `aligned_alloc` returns `expected<void*, errc>` instead of throwing |
| `<mdspan>` header | `mat_transpose/scalar.hpp` | Multidimensional array view with `operator[](i, j)` indexing |
| `<expected>` header | `common.hpp` | `std::expected<T, E>` for value-or-error returns |

### CMake Integration

If using as a CMake subdirectory or FetchContent:

```cmake
add_subdirectory(simd-exp)
target_link_libraries(your_target PRIVATE simd)
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
