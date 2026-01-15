#include "CudaDCT.hpp"

__constant__ const uint8_t d_zigzag[64];
__constant__ const uint8_t d_quant_table[64];

// input is a non level shifted array corrisponding to a channel
//
//
__global__ void CudaDCTAndQuantize(uint8_t* input, int16_t* output, int width, int height){

    __shared__ float block[8][8];
    __shared__ float temp[8][8];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = bx*8 + tx;
    int y = by*8 + ty;

    // loading block from input
    if( x < width && y < height){
        block[tx][ty] = input[x + y*width] - 128.0f; // level shifts block
    }

    __syncthreads(); // waits for all level shifting to be done per block


    // rows
    
    float sum = 0.0f;
    for(uint8_t k = 0; k < 8; k++){
        sum += block[k][ty] * dctMat[tx][k];
    }
    temp[tx][ty] = sum;

    __syncthreads();

    // columns 
    sum = 0.0f;
    for(uint8_t k = 0; k < 8; k++){
        sum += dctMat[tx][k] * temp[ty][k];
    }
    block[tx][ty] = sum ;

    float* flat_block = &block[0][0];

    output[x + y*width] = (int16_t)( flat_block[ zig_zag_table[tx+8*ty] ] / d_quant_table[tx + ty*8]);

}