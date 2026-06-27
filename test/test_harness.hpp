#pragma once
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>
#include <string>
#include <array>

// ── Standard sizes ──────────────────────────────────────────────────────────
// Chosen to exercise: degenerate (1), tail-only (7), exact SIMD width (8),
// 1-element tail (9), large odd (1023), large aligned (1024), very large (1<<20)
inline constexpr std::array<size_t, 7> kStdSizes = {1, 7, 8, 9, 1023, 1024, 1<<20};

// ── Data generators ─────────────────────────────────────────────────────────
inline std::vector<float> make_random(size_t n, float lo, float hi, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> res(n);
    for (size_t i = 0; i < n; ++i) {
        res[i] = dist(rng);
    }
    return res;
}

inline std::vector<float> make_const(size_t n, float val) {
    return std::vector<float>(n, val);
}

// Generates values that span lo..hi with values AT the boundaries too
inline std::vector<float> make_boundary_stress(size_t n, float lo, float hi, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> res(n);
    for (size_t i = 0; i < n; ++i) {
        if (i % 5 == 0) {
            res[i] = lo;
        } else if (i % 5 == 1) {
            res[i] = hi;
        } else {
            res[i] = dist(rng);
        }
    }
    return res;
}

// ── Array comparison helpers ─────────────────────────────────────────────────
// Wraps EXPECT_FLOAT_EQ element-by-element with a descriptive label on failure
inline void check_exact(const float* a, const float* b, size_t n, const std::string& label = "") {
    for (size_t i = 0; i < n; ++i) {
        ASSERT_FLOAT_EQ(a[i], b[i]) << (label.empty() ? "" : label + ": ") << "Mismatch at index " << i;
    }
}

// Wraps EXPECT_NEAR element-by-element
inline void check_near(const float* a, const float* b, size_t n, float tol,
                const std::string& label = "") {
    for (size_t i = 0; i < n; ++i) {
        ASSERT_NEAR(a[i], b[i], tol) << (label.empty() ? "" : label + ": ") << "Mismatch at index " << i;
    }
}

// ── Scalar result helpers (for reductions) ──────────────────────────────────
inline void check_scalar_near(float a, float b, float tol, const std::string& label = "") {
    EXPECT_NEAR(a, b, tol) << (label.empty() ? "" : label + ": ");
}
