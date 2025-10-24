#pragma once
#include <cstdint>
#include <memory>

struct RGBImage{

    std::unique_ptr<uint8_t[]> pixel_array{};
    uint16_t width{};
    uint16_t height{};
    bool top_left_origin{};
    uint8_t bpp{};
    size_t size_bytes;
};

enum class Subsampling { YUV444, YUV422, YUV420 };

struct Plane {
    uint8_t* data{};
    int      width{};
    int      height{};
    int      stride{};   // bytes per row
};

struct YCbCrImage {
    // Single contiguous storage to keep things cache-friendly
    std::unique_ptr<uint8_t[]> storage;

    Plane Y;
    Plane Cb;
    Plane Cr;         // point into storage
    
    int   width{};
    int   height{};
    uint8_t bit_depth{8};
    Subsampling subsampling{Subsampling::YUV420};
};
