#include "CudaFileWrite.hpp"


void initCudaResources();

void CudaWriteImage(YCbCrImage& image, std::ostream& out_stream);



/*
uint8_t* h_Y_plane = image.Y.data;
    uint8_t* h_Cb_plane = image.Cb.data;
    uint8_t* h_Cr_plane = image.Cr.data;

    uint8_t* d_Y_in;
    uint8_t* d_Cb_in;
    uint8_t* d_Cr_in;

    size_t Y_size = image.Y.height*image.Y.width;
    size_t C_size = image.Cb.height*image.Cb.width;

    cudaMalloc(&d_Y_in,  Y_size);
    cudaMalloc(&d_Cb_in, C_size);
    cudaMalloc(&d_Cr_in, C_size);

    cudaMemcpy(d_Y_in, h_Y_plane, Y_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Cb_in, h_Cb_plane, C_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Cr_in, h_Cr_plane, C_size, cudaMemcpyHostToDevice);

    int16_t* d_Y_out;
    int16_t* d_Cb_out;
    int16_t* d_Cr_out;
 
    cudaMalloc(&d_Y_out, image.Y.height*image.Y.width * sizeof(int16_t));
    cudaMalloc(&d_Cb_out, image.Y.height*image.Y.width * sizeof(int16_t));
    cudaMalloc(&d_Cr_out, image.Y.height*image.Y.width * sizeof(int16_t));


    dim3 block(8,8);
    dim3 Y_grid(
        (image.Y.width + 7) / 8,
        (image.Y.height + 7) / 8
    );
    dim3 C_grid(
        (image.Cb.width + 7) / 8,
        (image.Cb.height + 7) / 8
    );

    CudaDCTAndQuantize<<< Y_grid, block >>>(d_Y_in, d_Y_out, image.Y.width, image.Y.height);
    CudaDCTAndQuantize<<< C_grid, block >>>(d_Cb_in, d_Cb_out, image.Cb.width, image.Cb.height);
    CudaDCTAndQuantize<<< C_grid, block >>>(d_Cr_in, d_Cr_out, image.Cr.width, image.Cr.height);


    int16_t* h_Y_out;
    int16_t* h_Cb_out;
    int16_t* h_Cr_out;


    cudaMemcpy(h_Y_out, d_Y_out, Y_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Cb_out, d_Cb_out, C_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Cr_out, d_Cr_out, C_size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    */