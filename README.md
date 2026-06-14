# SIMD Experiments

A header-only C++20 library and benchmark suite for AVX2 (Advanced Vector Extensions 2) SIMD intrinsics.

## Features

- Modular, operation-centric header-only library under `include/simd/`
- Safe compile-time dispatching with automatic standard C++ scalar fallbacks if AVX2 is not supported
- High-performance unrolled implementations (FMA, unaligned loads, non-temporal stores)
- Benchmarking suite powered by Google Benchmark

## Repository Structure

- `include/simd/`: Header-only library source code
  - `simd.hpp`: Master header including all operations
  - `common.hpp`: Alignments, allocators, and CPU dispatch logic
  - `ops/`: Modular directory containing per-operation headers (SIMD, scalar, and dispatcher)
- `bench/`: Source code for performance benchmarks
- `docs/`: In-depth explanations, math derivations, and learning journals

## Build Instructions

### Prerequisites

Ensure you have a C++20 compiler, CMake (>= 3.16), and Google Benchmark installed.

### Setup and Compilation

```bash
# Configure the build directory
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build all targets
cmake --build build
```

### Running Benchmarks

```bash
# Run a specific benchmark (e.g., softmax)
./build/softmax_bench

# Run all benchmarks sequentially
cmake --build build --target run_all
```

## Integration

### Header-Only Copy-Paste

Copy the `include/simd` directory directly into your project. Include `simd/simd.hpp` to access the APIs:

```cpp
#include <simd/simd.hpp>

// Elements will be processed using AVX2 if compiled with -mavx2,
// otherwise it will automatically fallback to standard C++ scalar loops.
simd::relu(src, dst, size);
```

### CMake Integration

If using as a CMake subdirectory or FetchContent:

```cmake
add_subdirectory(simd-exp)
target_link_libraries(your_target PRIVATE simd)
```