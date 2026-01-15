#pragma once
#include <cuda_runtime.h>
#include "CudaDCT.hpp"
#include <thread>
#include "../Image.hpp"

#include <ostream>



void CudaWriteImage(YCbCrImage& image, std::ostream& out_stream);

void get_dc_coeffs(int16_t* channel_array);