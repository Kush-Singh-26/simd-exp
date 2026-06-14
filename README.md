# SIMD Experiments

A header-only C++20 library and benchmark suite for AVX2 (Advanced Vector Extensions 2) SIMD intrinsics.

## Features

- Modular, operation-centric header-only library under `include/simd/`
- Safe compile-time dispatching with automatic standard C++ scalar fallbacks if AVX2 is not supported
- High-performance unrolled implementations (FMA, unaligned loads, non-temporal stores)
- Benchmarking suite powered by Google Benchmark
- Unit tests powered by Google Test with scalar fallback coverage

## Operations

| Operation | Scalar | SIMD | NT Store | Description |
|-----------|--------|------|----------|-------------|
| `abs` | Yes | Yes | Yes | Element-wise absolute value |
| `clamp` | Yes | Yes | Yes | Element-wise clamp to [lo, hi] |
| `dot_prod` | Yes | Yes (FMA) | No | Dot product of two vectors |
| `mat_transpose` | Yes | Yes | Yes | 4x4 matrix transpose |
| `online_softmax` | Yes | Yes | No | Single-pass softmax (online max/sum) |
| `relu` | Yes | Yes | Yes | Element-wise rectified linear unit |
| `softmax` | Yes | Yes | Yes | Standard 3-pass softmax |
| `sum` | Yes | Yes | No | Sum of elements |

## Repository Structure

- `include/simd/`: Header-only library source code
  - `simd.hpp`: Master header including all operations
  - `common.hpp`: Alignments, allocators, and CPU dispatch logic
  - `math_utils.hpp`: Shared AVX2 exp approximation (Estrin polynomial, Cody-Waite range reduction, < 3 ULP / ~1e-7 accuracy — see [docs/exp_approximation.md](docs/exp_approximation.md))
  - `ops/`: Modular directory containing per-operation headers (SIMD, scalar, and dispatcher)
- `bench/`: Source code for performance benchmarks
  - `bench_utils.hpp`: Shared data generation utilities
  - `results/`: JSON benchmark output (tracked in git)
- `test/`: Google Test unit tests for all operations
- `docs/`: In-depth explanations, math derivations, and learning journals

## Build Instructions

### Prerequisites

- C++20 compiler (GCC 10+ or Clang 12+)
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

9 test suites covering all operations. Scalar tests run unconditionally; SIMD tests require AVX2:

| Test | Scalar | SIMD | What it verifies |
|------|--------|------|------------------|
| `abs_test` | 7 tests | 4 tests | Bitwise abs, edge cases |
| `clamp_test` | 7 tests | 4 tests | Clamp to [lo, hi] |
| `dot_prod_test` | 7 tests | 3 tests | Dot product accuracy (requires FMA) |
| `mat_transpose_test` | 8 tests | 3 tests | 4x4 transpose |
| `math_utils_test` | — | 7 tests | `avx2_exp_ps` < 3 ULP vs `std::exp` |
| `online_softmax_test` | 8 tests | 2 tests | Online softmax + plain vs online comparison |
| `relu_test` | 7 tests | 4 tests | ReLU activation |
| `softmax_test` | 8 tests | 4 tests | Softmax correctness |
| `sum_test` | 7 tests | 3 tests | Sum reduction |

## Integration

### Header-Only Copy-Paste

Copy the `include/simd` directory directly into your project. Include `simd/simd.hpp` to access the APIs:

```cpp
#include <simd/simd.hpp>

// Elements will be processed using AVX2 if compiled with -mavx2,
// otherwise it will automatically fallback to standard C++ scalar loops.
simd::relu(src, dst, size);

// For SIMD exp (used internally by softmax/online_softmax):
// avx2_exp_ps() in math_utils.hpp provides < 3 ULP accuracy via
// Estrin polynomial evaluation + Cody-Waite range reduction.
```

### CMake Integration

If using as a CMake subdirectory or FetchContent:

```cmake
add_subdirectory(simd-exp)
target_link_libraries(your_target PRIVATE simd)
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
