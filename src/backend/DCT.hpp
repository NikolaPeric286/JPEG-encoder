#pragma once
#include <cstdint>
#include <cmath>
#include <array>
#include "tables.hpp"

constexpr int N = 8;
constexpr float PI = 3.14159265358979323846f;

inline float alpha(int k) {
    return (k == 0) ? (1.0f / std::sqrt(2.0f)) : 1.0f;
}

//creates the array of cosines to do the DCT with
const std::array<std::array<float, N>, N>& cos_table();


void DCT8x8AndQuantize(int8_t* input, int16_t* output, const uint8_t* q_table);


