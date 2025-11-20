#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <cstring>
#include "../Image.hpp"
#include "../ColorConversion.hpp"

/*
constexpr int CYR = 19595, CYG = 38470, CYB =  7471;   // Y
constexpr int CBR = -11059, CBG = -21709, CBB = 32768; // Cb
constexpr int CRR = 32768, CRG = -27439, CRB = -5329;  // Cr
*/


void cudaConvertAndDownsample(RGBImage& src_image, YCbCrImage& output_image);

__global__ void color_convert(uint8_t* input, uint8_t* output, uint16_t width, uint16_t height);

__global__ void downsample420(uint8_t* input_Cb, uint8_t* input_Cr, uint8_t* output_Cb, uint8_t* output_Cr, uint16_t width, uint16_t height);