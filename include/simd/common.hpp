#pragma once

#if defined(__AVX2__)
    #include <immintrin.h>
    #define SIMD_AVX2_ENABLED 1
    #define SIMD_WIDTH_BYTES 32
#else
    #define SIMD_SCALAR_ONLY 1
    #define SIMD_WIDTH_BYTES 4
#endif

#include <cstdlib>

namespace simd {

// Aligned allocation utility for Non-Temporal stores
inline void* aligned_alloc(size_t alignment, size_t size) {
#if defined(SIMD_AVX2_ENABLED)
    return _mm_malloc(size, alignment);
#else
    #if defined(_MSC_VER)
        return _aligned_malloc(size, alignment);
    #else
        void* ptr = nullptr;
        if (posix_memalign(&ptr, alignment, size) != 0) {
            throw std::bad_alloc();
        }
        return ptr;
    #endif
#endif
}

inline void aligned_free(void* ptr) {
#if defined(SIMD_AVX2_ENABLED)
    _mm_free(ptr);
#else
    #if defined(_MSC_VER)
        _aligned_free(ptr);
    #else
        free(ptr);
    #endif
#endif
}

} // namespace simd
