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


void DCT8x8AndQuantize(const int8_t* input, int16_t* output, const uint8_t* q_table);


const uint8_t Y_q_table_50[] =     {16,	 12, 14,	 14,	 18,	 24,	 49,	 72,
                                    11,	 12, 13,	 17,	 22,	 35,	 64,	 92,
                                    10,	 14, 16,	 22,	 37,	 55,	 78,	 95,
                                    16,	 19, 24,	 29,	 56,	 64,	 87,	 98,
                                    24,	 26, 40,	 51,	 68,	 81,	103,	112,
                                    40,	 58, 57,	 87,	109,	104,	121,	100,
                                    51,	 60, 69,	 80,	103,	113,	120,	103,
                                    61,	 55, 56,	 62,	 77,	 92,	101,	 99};

