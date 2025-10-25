#include "test_helperfunctions.hpp"

void write_tga_fixture(const std::string& path,
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

void put_bgr(uint8_t* buf, int W, int x, int y, uint8_t R, uint8_t G, uint8_t B) {
    uint8_t* p = buf + y * (W*3) + x*3;
    p[0] = B; p[1] = G; p[2] = R;
}

// Reference helpers (float for readability)
uint8_t refY(uint8_t R, uint8_t G, uint8_t B) {
    int Y = (CYR*R + CYG*G + CYB*B + (1<<15)) >> 16;
    return clamp_u8(Y);
}
uint8_t refCb_from_avg(uint8_t Ravg, uint8_t Gavg, uint8_t Bavg) {
    int Cb = ((CBR*Ravg + CBG*Gavg + CBB*Bavg + (1<<15)) >> 16) + 128;
    return clamp_u8(Cb);
}
uint8_t refCr_from_avg(uint8_t Ravg, uint8_t Gavg, uint8_t Bavg) {
    int Cr = ((CRR*Ravg + CRG*Gavg + CRB*Bavg + (1<<15)) >> 16) + 128;
    return clamp_u8(Cr);
}