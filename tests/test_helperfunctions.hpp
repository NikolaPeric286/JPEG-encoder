#pragma once
#include <string>
#include <cstdint>
#include <vector>
#include "../src/backend/Image.hpp"
#include "../src/backend/image_processing_tga.hpp"


static void write_tga_fixture(const std::string& path,
                              uint16_t w, uint16_t h,
                              uint8_t  bpp,     // 24 or 32
                              bool     topLeft,
                              const std::string& idString, // can be ""
                              const std::vector<uint8_t>& pixelsBGR){

    if (!(bpp == 24 || bpp == 32)) throw std::runtime_error("Unsupported bpp in test fixture");
    const uint8_t attrBits = (bpp == 32 ? 8 : 0);
    tgaheader hdr{};
    hdr.idLength  = static_cast<uint8_t>(idString.size());
    hdr.colorMapType = 0;
    hdr.imageType = 2; // uncompressed truecolor
    hdr.colorMapFirstEntry = 0;
    hdr.colorMapLength = 0;
    hdr.colorMapEntrySize = 0;
    hdr.xOrigin = 0;
    hdr.yOrigin = 0;
    hdr.width   = w;
    hdr.height  = h;
    hdr.pixelDepth = bpp;
    hdr.imageDescriptor = (topLeft ? 0x20 : 0x00) | (attrBits & 0x0F); //imageDescriptor uses a lot of bit masking so this is kind confusing but also unimportant

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Failed to open fixture for write");

    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    if (!idString.empty()) out.write(idString.data(), static_cast<std::streamsize>(idString.size()));

    out.write(reinterpret_cast<const char*>(pixelsBGR.data()),
              static_cast<std::streamsize>(pixelsBGR.size()));
    if (!out) throw std::runtime_error("Failed to write fixture data");
}