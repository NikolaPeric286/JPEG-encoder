#pragma once
#include <string>
#include <cstdint>
#include <vector>
#include "../src/backend/Image.hpp"
#include "../src/backend/image_processing_tga.hpp"
#include "../src/backend/ColorConversion.hpp"

void write_tga_fixture(const std::string& path,
                              uint16_t w, uint16_t h,
                              uint8_t  bpp,     // 24 or 32
                              bool     topLeft,
                              const std::string& idString, // can be ""
                              const std::vector<uint8_t>& pixelsBGR);

void put_bgr(uint8_t* buf, int W, int x, int y, uint8_t R, uint8_t G, uint8_t B);

// Reference helpers (float for readability)
uint8_t refY(uint8_t R, uint8_t G, uint8_t B);
uint8_t refCb_from_avg(uint8_t Ravg, uint8_t Gavg, uint8_t Bavg);
uint8_t refCr_from_avg(uint8_t Ravg, uint8_t Gavg, uint8_t Bavg);