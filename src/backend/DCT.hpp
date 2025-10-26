#pragma once
#include <cstdint>
#include <cmath>
#include <array>

constexpr int N = 8;
constexpr float PI = 3.14159265358979323846f;

inline float alpha(int k) {
    return (k == 0) ? (1.0f / std::sqrt(2.0f)) : 1.0f;
}

//creates the array of cosines to do the DCT with
const std::array<std::array<float, N>, N>& cos_table();


void DCT8x8(const uint8_t* input, float* output);
