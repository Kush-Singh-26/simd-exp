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
- Ninja, ccache, and mold (recommended for fast builds):
  ```bash
  # Fedora
  sudo dnf install ninja-build ccache mold

  # Ubuntu/Debian
  sudo apt install ninja-build ccache mold
  ```

### First-time Setup

```bash
cmake -B build -G Ninja \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=mold"
```

### Quick Commands

A root `Makefile` provides convenient targets. Build uses Ninja, ccache, and mold linker automatically.

| Command | What it does |
|---|---|
| `make` / `make build` | Build everything (parallel, auto-configures if needed) |
| `make test` | Build + run all unit tests |
| `make bench` | Build + run all benchmarks (JSON output to `bench/results/`) |
| `make configure` | Configure CMake (only runs if `build/` doesn't exist) |
| `make clean` | Delete build directory |
| `make rebuild` | Full clean rebuild from scratch |

Incremental builds (after the first build) take **< 1 second** thanks to ccache. Clean builds take ~75s.

### Running Tests

```bash
# All tests
make test

# Or directly
ctest --test-dir build --output-on-failure

# Run a single test suite
./build/matmul_test

# Run a specific test by name
./build/matmul_test --gtest_filter=MatmulTest.Ikj_Matches_Ijk

# Run all tests in a group
./build/sum_test --gtest_filter=SumTest.*

# List all available tests
./build/matmul_test --gtest_list_tests
```

### Running Benchmarks

```bash
# Run all benchmarks (saves JSON to bench/results/)
make bench

# Run a specific benchmark
./build/softmax_bench

# Filter specific benchmarks
./build/relu_bench --benchmark_filter=rand
./build/softmax_bench --benchmark_repetitions=5 --benchmark_display_aggregates_only=true
```

### Test Coverage

10 test suites covering all operations. Correctness is verified by comparing SIMD and dispatcher outputs directly against their scalar counterparts across 7 standard size classes (degenerate, tail-only, exact SIMD width, 1-element tail, large odd, large aligned, and very large stress sizes).

| Test Suite | What it verifies |
|------------|------------------|
| `abs_test` | Known values, SIMD vs scalar, public dispatcher API, NT stores |
| `clamp_test` | Known values, SIMD vs scalar boundary stress, public dispatcher API, NT stores |
| `dot_prod_test` | Dot product accuracy on 7 standard sizes (requires FMA), public dispatcher API |
| `matmul_test` | Known values (ijk/ikj), cross-validation ikj vs ijk, non-square matrices, large matrices (256/512), single-element dimensions (M=1, N=1, K=1), dispatcher zero-init |
| `mat_transpose_test` | Arbitrary NxM (`std::mdspan`), 4x4 SIMD kernel, NT stores, public dispatcher API |
| `math_utils_test` | `avx2_exp_ps` ULP bounds, AVX2 horizontal reductions |
| `online_softmax_test` | Single-pass SIMD vs scalar, sum-to-one invariants, equivalence to standard softmax |
| `relu_test` | SIMD vs scalar, public dispatcher API, NT stores |
| `softmax_test` | SIMD vs scalar, sum-to-one invariants, public dispatcher API |
| `sum_test` | Sum reduction on 7 standard sizes, public dispatcher API |

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
