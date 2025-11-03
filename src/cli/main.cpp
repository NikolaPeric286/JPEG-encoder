// main.c
//
#include <iostream>
#include <fstream>
#include <string>
#include "../backend/image_processing_tga.hpp"
#include "../backend/ColorConversion.hpp"
#include "../backend/Image.hpp"
#include "../backend/BitBuffer.hpp"
#include "../backend/LosslessCompression.hpp"
#include "../backend/DCT.hpp"
#include "../backend/Image.hpp"
#include "../backend/tables.hpp"
#include "../backend/FileWrite.hpp"

int main(int arc, char** argv){

    //std::string file_name = "/home/nikolaperic/vscodeProjectFolders/JPEG-encoder/data/TGA_examples/EXAMPLE_part1.tga";
    std::string file_name = argv[1];
    std::ifstream input_file(file_name, std::ios::in | std::ios::binary);

    if(!input_file){
        std::cerr << "Error failed to open file " << file_name << std::endl;
        return 1;
    }

    std::cout << "getting data from tga\n";
    RGBImage input_rgb = getImageContentTGA(file_name.c_str());
    input_file.close();

    YCbCrImage color_converted_image = make_YCbCr420(input_rgb.width, input_rgb.height);;
    std::cout << "converting and downsampling\n";
    convert_and_downsample(input_rgb, color_converted_image);

    file_name.erase(file_name.length() - 4); // removes .tga 
    file_name.append(".jpg");

    std::cout << "File name -> " << file_name << "\n";

    std::ofstream out_file(file_name, std::ios::trunc | std::ios::binary);

    std::cout << "writing SOI\n";
    writeSOI(out_file);
    std::cout << "writing APP0\n";
    writeAPP0(out_file);
    std::cout << "writing quant tables\n";
    writeQuantTables(out_file);
    std::cout << "Writing SOF\n";
    writeStartOfFrame(color_converted_image, out_file);
    std::cout << "writing huff tables\n";
    writeHuffTables(out_file); 
    std::cout << "writing SOS\n";
    writeSOS(out_file);
    std::cout << "writing image\n";
    writeImage(color_converted_image, out_file);
    std::cout << "writing EOI\n";
    writeEOI(out_file);

    out_file.close();
    
}