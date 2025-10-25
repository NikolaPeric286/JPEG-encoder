// main.c
//
#include <iostream>
#include <fstream>
#include "../backend/image_processing_tga.hpp"
#include "../backend/ColorConversion.hpp"
#include "../backend/Image.hpp"


int main(int arc, char** argv){


    RGBImage src_image = getImageContentTGA("../data/TGA_examples/ALL_WHITE.tga");
    std::cout << "Size of source image : " << src_image.size_bytes;

    YCbCrImage out_image = make_YCbCr420(src_image.width,src_image.height);

    convert_and_downsample(src_image, out_image);

    std::ofstream out_file("output_data", std::ios::binary);

    out_file.write( reinterpret_cast<char*>(out_image.storage.get()), out_image.size_bytes);
    out_file.close();
    
    std::cout << "\nSize of ouput image : " << out_image.size_bytes << "\n";
}