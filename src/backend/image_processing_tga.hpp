#pragma once
#include <fstream>
#include <iostream>
#include <cstring>
#include <cstdint>
#include <limits>
#include <string>
#include <ios>
#include <memory>
#include "Image.hpp"

#pragma pack(push, 1) // tells the compiler to create structs with 0 padding so that the header can be read directly from the file



struct tgaheader{
    uint8_t  idLength;
    uint8_t  colorMapType;
    uint8_t  imageType;         // 2 = uncompressed truecolor, 3 = uncompressed grayscale (others exist)
    uint16_t colorMapFirstEntry;
    uint16_t colorMapLength;
    uint8_t  colorMapEntrySize;
    uint16_t xOrigin;
    uint16_t yOrigin;
    uint16_t width;
    uint16_t height;
    uint8_t  pixelDepth;        // 24 or 32 for color, 8 for grayscale
    uint8_t  imageDescriptor;
};
#pragma pack(pop)



RGBImage getImageContentTGA(const char* file_path);


