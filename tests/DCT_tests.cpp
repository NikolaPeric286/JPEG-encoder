#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <filesystem>
#include <cstdint>
#include "../src/backend/image_processing_tga.hpp"
#include "../src/backend/ColorConversion.hpp"
#include "test_helperfunctions.hpp"
#include "../src/backend/DCT.hpp"




TEST(DCT_tests, ConstBlockDC){
    constexpr int N = 8;
    constexpr int8_t dc_val = 100;
    int8_t block[N*N];

    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            block[x+y*N] = dc_val;

    int16_t output[N*N];
    
    DCT8x8AndQuantize(block, output, Y_q_table_50);
    const int16_t expectedDC = (0.25f * alpha(0)*alpha(0) * (N*N*dc_val))/Y_q_table_50[0];


    // the top left value should be equal to the dc value we set the image to
    EXPECT_NEAR(output[0], expectedDC, 1e-2);

    
    for(int x = 0; x < N; x++){
        for(int y = 0; y < N; y++){
            if(x == 0 && y == 0) continue;
            EXPECT_NEAR(output[x+y*N], 0.0f, 1e-3) << "x val: " << x << " y val : " << y; // all other values should be 0
        }
    }
    
}