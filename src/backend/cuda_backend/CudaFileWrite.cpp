#include "CudaFileWrite.hpp"
#include <iostream>
#include "ThreadPool.hpp"

void initCudaResources(){
    const int device = 0;

    cudaSetDevice(device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Using GPU: " << prop.name << "\n";

}



void CudaWriteImage(YCbCrImage& image, std::ostream& out_stream){
    initCudaResources();

    ThreadPool thread_pool;

    
}