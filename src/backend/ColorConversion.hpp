#pragma once
#include <cstdint>
#include <memory>
#include <ios>
#include <cstring>
#include "Image.hpp"

constexpr int CYR = 19595, CYG = 38470, CYB =  7471;   // Y
constexpr int CBR = -11059, CBG = -21709, CBB = 32768; // Cb
constexpr int CRR = 32768, CRG = -27439, CRB = -5329;  // Cr

static inline uint8_t clamp_u8(int v) {
    return v < 0 ? 0 : (v > 255 ? 255 : (uint8_t)v);
}


void RGBToYCbCr24bit(RGBImage& input_image);
void RGBToYCbCr24bitAndDownsample(RGBImage& input_image);
