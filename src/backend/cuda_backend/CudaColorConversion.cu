#include "CudaColorConversion.hpp"

__global__ void color_convert(uint8_t* input, uint8_t* output, uint16_t width, uint16_t height){


    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    constexpr int Y_start = 0;
    const int Cb_start = width*height;
    const int Cr_start = width*height*2;

    size_t i = x + width*y;
    size_t byte_index = 3*i;

    uint8_t R = input[byte_index + 2];
    uint8_t G = input[byte_index + 1];
    uint8_t B = input[byte_index + 0];

    int16_t Y = ((CYR*R + CYG*G + CYB*B + (1<<15)) >> 16);
    int16_t Cb = ((CBR*R + CBG*G + CBB*B + (1<<15)) >> 16) + 128;
    int16_t Cr = ((CRR*R + CRG*G + CRB*B + (1<<15)) >> 16) + 128;

    output[i] = (Y>255)?255:Y;
    output[Cb_start + i] = (Cb>255)?255:Cb;
    output[Cr_start + i] = (Cr>255)?255:Cr;
    


}


__global__ void downsample420(uint8_t* input_Cb, uint8_t* input_Cr, uint8_t* output_Cb, uint8_t* output_Cr, uint16_t width, uint16_t height){
    
    int cx = blockIdx.x * blockDim.x + threadIdx.x;
    int cy = blockIdx.y * blockDim.y + threadIdx.y;

    int chroma_width = (width+1)/2;
    int chroma_height = (height+1)/2;

    if( cx >= chroma_width || cy >= chroma_height) return;

    int i = cx + chroma_width*cy;

    int x0 = 2*cx;
    int y0 = 2*cy;

    int x1 = (x0 + 1 < width)  ? x0 + 1 : x0; // edge clamping
    int y1 = (y0 + 1 < height) ? y0 + 1 : y0;


    // chroma sample
    //
    //  c00 c01
    //  c10 c11

    uint8_t c00 = input_Cb[x0 + (y0*width)];
    uint8_t c01 = input_Cb[x1 + (y0*width)];
    uint8_t c10 = input_Cb[x0 + (y1*width)];
    uint8_t c11 = input_Cb[x1 + (y1*width)];

    
    output_Cb[i] = (c00 + c01 + c10 + c11 +2 ) /4; // + 2 is for rounding

    c00 = input_Cr[x0 + (y0*width)];
    c01 = input_Cr[x1 + (y0*width)];
    c10 = input_Cr[x0 + (y1*width)];
    c11 = input_Cr[x1 + (y1*width)];

    output_Cr[i] = (c00 + c01 + c10 + c11 +2 ) /4;
    
}


void cudaConvertAndDownsample(RGBImage& src_image, YCbCrImage& output_image){

    int width = src_image.width;
    int height = src_image.height;

    uint8_t* h_input_rgb = src_image.pixel_array.get();

    uint8_t* d_input_rgb;
    uint8_t* d_output_YCbCr;

    cudaMalloc(&d_input_rgb, src_image.size_bytes);
    cudaMalloc(&d_output_YCbCr, src_image.size_bytes);

    cudaMemcpy(d_input_rgb, h_input_rgb, src_image.size_bytes, cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid(
        (width  + 15) / 16,
        (height + 15) / 16
    );

    color_convert<<<grid,block>>>(d_input_rgb, d_output_YCbCr, width, height );

    size_t block_spaceing = width*height;

    
    uint8_t* d_Cb_block = d_output_YCbCr + block_spaceing;
    uint8_t* d_Cr_block = d_Cb_block + block_spaceing;


    int chroma_w = (width  + 1) / 2;
    int chroma_h = (height + 1) / 2;
    size_t chroma_size = (size_t)chroma_w * (size_t)chroma_h;

    uint8_t* d_Cb_downsampled;
    cudaMalloc(&d_Cb_downsampled, 2*chroma_size);
    uint8_t* d_Cr_downsampled = d_Cb_downsampled + chroma_size;

    
    block = dim3(16, 16);
    grid = dim3(
        (chroma_w + block.x - 1) / block.x,
        (chroma_h + block.y - 1) / block.y
    );


    downsample420<<<grid,block>>>(d_Cb_block, d_Cr_block, d_Cb_downsampled, d_Cr_downsampled, width, height );
    
    cudaMemcpy(output_image.Y.data, d_output_YCbCr, block_spaceing, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_image.Cb.data, d_Cb_downsampled, chroma_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_image.Cr.data, d_Cr_downsampled, chroma_size, cudaMemcpyDeviceToHost);
}