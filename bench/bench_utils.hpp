#pragma once
#include <vector>
#include <random>
#include <algorithm>

enum class DataType { POS, NEG, RAND };

inline void gen_data_const(std::vector<float>& src, DataType dtype) {
    if (dtype == DataType::POS) {
        std::fill(src.begin(), src.end(), 1.0f);
    } else if (dtype == DataType::NEG) {
        std::fill(src.begin(), src.end(), -1.0f);
    } else {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& x : src) x = dist(rng);
    }
}

inline void gen_data_random(std::vector<float>& src, DataType dtype) {
    if (dtype == DataType::POS) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 10.0f);
        for (auto& x : src) x = dist(rng);
    } else if (dtype == DataType::NEG) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-10.0f, 0.0f);
        for (auto& x : src) x = dist(rng);
    } else {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
        for (auto& x : src) x = dist(rng);
    }
}
