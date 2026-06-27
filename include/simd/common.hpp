#pragma once

#include <concepts>
#include <cstdlib>
#include <expected>
#include <span>
#include <system_error>
#include <utility>

#if defined(__AVX2__)
    #include <immintrin.h>
    #define SIMD_AVX2_ENABLED 1
    #define SIMD_WIDTH_BYTES 32
#else
    #define SIMD_SCALAR_ONLY 1
    #define SIMD_WIDTH_BYTES 4
#endif

#if defined(__F16C__)
    #define SIMD_F16C_ENABLED 1
#endif

namespace simd {

template <typename T>
concept Float = std::floating_point<T>;

// Aligned allocation utility for Non-Temporal stores
// Returns expected<void*, std::errc> — check has_value() before use.
inline std::expected<void*, std::errc> aligned_alloc(size_t alignment, size_t size) {
#if defined(SIMD_AVX2_ENABLED)
    void* ptr = _mm_malloc(size, alignment);
    if (!ptr) return std::unexpected(std::errc::not_enough_memory);
    return ptr;
#else
    #if defined(_MSC_VER)
        void* ptr = _aligned_malloc(size, alignment);
        if (!ptr) return std::unexpected(std::errc::not_enough_memory);
        return ptr;
    #else
        void* ptr = nullptr;
        if (posix_memalign(&ptr, alignment, size) != 0) {
            return std::unexpected(std::errc::not_enough_memory);
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
