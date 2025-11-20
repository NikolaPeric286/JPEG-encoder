#pragma once
#include <cstdint>
#include <ostream>
#include <cstring>
#include <algorithm>
#include "LosslessCompression.hpp"
#include "Image.hpp"
#include "tables.hpp"
#include "BitBuffer.hpp"
#include "DCT.hpp"
#include "ColorConversion.hpp"
#include "Image.hpp"



void writeSOI(std::ostream& out_stream);
void writeAPP0(std::ostream& out_stream);
void writeQuantTables(std::ostream& out_stream);
void writeStartOfFrame(YCbCrImage& image, std::ostream& out_stream);
void writeHuffTables(std::ostream& out_stream); 
void writeSOS(std::ostream& out_stream);
void writeImage(YCbCrImage& image, std::ostream& out_stream);
void writeEOI(std::ostream& out_stream);



void encodeRGBImage(RGBImage& rgb_image, std::ostream& out_stream);