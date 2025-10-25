#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <filesystem>
#include <cstdint>
#include "../src/backend/image_processing_tga.hpp"
#include "../src/backend/ColorConversion.hpp"
#include "test_helperfunctions.hpp"

TEST(TGAFileInput, ReadTGAExample){
    EXPECT_NO_THROW(getImageContentTGA("../data/TGA_examples/EXAMPLE_part1.tga"));
}

TEST(TGAFileInput, IncorrectFilePath){
    ASSERT_THROW(getImageContentTGA("../data/TGA_examples/file_that_does_not_exist.tga"), std::runtime_error);
}

TEST(TGAFileInput, HeaderSize){
    ASSERT_EQ(sizeof(tgaheader), 18);
}

TEST(TGAFileInput, CheckOutputAgainstRef){
    namespace fs = std::filesystem;
    fs::path temp = fs::temp_directory_path();
    fs::path temp_file = temp / "testfile.tga";
    tgaheader header{};

    header.imageType = 2; // uncompressed truecolor
    header.width = 2;
    header.height = 2;
    header.pixelDepth = 24;
    header.imageDescriptor = 0x20; // top left origin bit set true

    EXPECT_EQ(sizeof(header), 18);
    std::vector<uint8_t> pixels = {
        //pixel 1 pixel 2
        10,20,30, 40,50,60,
        //pixel 3 pixel 4
        70,80,90, 100,110,120
    };

    std::ofstream out(temp_file, std::ios::binary | std::ios::trunc);
    

    ASSERT_TRUE(out.is_open()) << "failed to open file : " << temp.string();

    out.write(reinterpret_cast<const char*>(&header), sizeof(header)); // writes the header to the file
    ASSERT_TRUE(out.good()) << "Failed to write header";
    out.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size())); //writes the pixel data
    out.close();
    RGBImage read_image_output = getImageContentTGA(temp_file.string().c_str());

    EXPECT_EQ(read_image_output.width, 2);
    EXPECT_EQ(read_image_output.height,2);
    EXPECT_EQ(read_image_output.bpp, 24);
    EXPECT_TRUE(read_image_output.top_left_origin);
    EXPECT_EQ(read_image_output.size_bytes, pixels.size());

    for (size_t i = 0; i < pixels.size(); i++) {
        EXPECT_EQ(read_image_output.pixel_array[i], pixels[i]) << "Pixel mismatch at index " << i;
    }
}



TEST(ConvertDownsample, TwoByTwo_PrimaryColors) {
    const int W = 2, H = 2;
    RGBImage src;
    src.width = W; src.height = H;
    src.pixel_array.reset(new uint8_t[W*H*3]);

    // 2x2 block:
    // (0,0)=Red, (1,0)=Green
    // (0,1)=Blue,(1,1)=White
    put_bgr(src.pixel_array.get(), W, 0, 0, 255, 0,   0);
    put_bgr(src.pixel_array.get(), W, 1, 0, 0,   255, 0);
    put_bgr(src.pixel_array.get(), W, 0, 1, 0,   0,   255);
    put_bgr(src.pixel_array.get(), W, 1, 1, 255, 255, 255);

    YCbCrImage out = make_YCbCr420(W, H);
    convert_and_downsample(src, out);

    // Expected Y per pixel
    uint8_t expY00 = refY(255, 0,   0);
    uint8_t expY01 = refY(0,   255, 0);
    uint8_t expY10 = refY(0,   0,   255);
    uint8_t expY11 = refY(255, 255, 255);

    EXPECT_EQ(out.Y.data[0 + 0*out.Y.width], expY00);
    EXPECT_EQ(out.Y.data[1 + 0*out.Y.width], expY01);
    EXPECT_EQ(out.Y.data[0 + 1*out.Y.width], expY10);
    EXPECT_EQ(out.Y.data[1 + 1*out.Y.width], expY11);

    // Average RGB first (2x2 block)
    auto avg4 = [](int a,int b,int c,int d){ return uint8_t((a+b+c+d+2)>>2); };
    uint8_t Ravg = avg4(255, 0,   0,   255);
    uint8_t Gavg = avg4(0,   255, 0,   255);
    uint8_t Bavg = avg4(0,   0,   255, 255);

    uint8_t expCb = refCb_from_avg(Ravg, Gavg, Bavg);
    uint8_t expCr = refCr_from_avg(Ravg, Gavg, Bavg);

    // For 2x2 → Cb/Cr are 1x1
    EXPECT_EQ(out.Cb.data[0], expCb);
    EXPECT_EQ(out.Cr.data[0], expCr);
}

TEST(ConvertDownsample, ThreeByThree_OddDims_EdgeClamp) {
    const int W = 3, H = 3;
    RGBImage src;
    src.width = W; src.height = H;
    src.pixel_array.reset(new uint8_t[W*H*3]);

    // Fill with a simple pattern: R=x*40, G=y*40, B=(x+y)*20
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
          uint8_t R = uint8_t(std::min(255, x*40));
          uint8_t G = uint8_t(std::min(255, y*40));
          uint8_t B = uint8_t(std::min(255, (x+y)*20));
          put_bgr(src.pixel_array.get(), W, x, y, R, G, B);
      }

    YCbCrImage out = make_YCbCr420(W, H);
    convert_and_downsample(src, out);

    // Check a few spots:

    // Y(0,0) against reference
    {
        uint8_t R = 0*40, G = 0*40, B = 0*20;
        EXPECT_EQ(out.Y.data[0 + 0*out.Y.width], refY(R,G,B));
    }

    // Bottom-right Y(2,2) should match clamped source (2,2)
    {
        uint8_t R = uint8_t(std::min(255, 2*40));
        uint8_t G = uint8_t(std::min(255, 2*40));
        uint8_t B = uint8_t(std::min(255, (2+2)*20));
        EXPECT_EQ(out.Y.data[2 + 2*out.Y.width], refY(R,G,B));
    }

    // Cb/Cr at chroma (1,1) corresponds to Y block (2,2) anchored with clamping:
    // 2x2 block covering (x,y) = (2,2) plus clamped neighbors → effectively all (2,2)
    {
        // Average 2x2 with edge clamp at (2,2)
        int x = 2, y = 2;
        auto Rxy = [&](int X,int Y){ return uint8_t(std::min(255, X*40)); };
        auto Gxy = [&](int X,int Y){ return uint8_t(std::min(255, Y*40)); };
        auto Bxy = [&](int X,int Y){ return uint8_t(std::min(255, (X+Y)*20)); };

        uint8_t R00 = Rxy(x, y), R01 = Rxy(x, y), R10 = Rxy(x, y), R11 = Rxy(x, y);
        uint8_t G00 = Gxy(x, y), G01 = Gxy(x, y), G10 = Gxy(x, y), G11 = Gxy(x, y);
        uint8_t B00 = Bxy(x, y), B01 = Bxy(x, y), B10 = Bxy(x, y), B11 = Bxy(x, y);

        uint8_t Ravg = uint8_t((R00+R01+R10+R11+2)>>2);
        uint8_t Gavg = uint8_t((G00+G01+G10+G11+2)>>2);
        uint8_t Bavg = uint8_t((B00+B01+B10+B11+2)>>2);

        uint8_t expCb = refCb_from_avg(Ravg,Gavg,Bavg);
        uint8_t expCr = refCr_from_avg(Ravg,Gavg,Bavg);

        // Chroma coordinate (x/2, y/2) = (1,1), plane width = ceil(W/2)=2
        const int cW = out.Cb.width; // should be 2
        EXPECT_EQ(out.Cb.data[1 + 1*out.Cb.width], expCb);
        EXPECT_EQ(out.Cr.data[1 + 1*out.Cr.width], expCr);
        (void)cW;
    }
}

TEST(ConvertDownsample, IdentityOnFlatGray) {
    const int W = 4, H = 4;
    RGBImage src;
    src.width = W; src.height = H;
    src.pixel_array.reset(new uint8_t[W*H*3]);

    // Flat gray: R=G=B=128
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        put_bgr(src.pixel_array.get(), W, x, y, 128,128,128);

    YCbCrImage out = make_YCbCr420(W, H);
    convert_and_downsample(src, out);

    // Y should be ~128, Cb/Cr ~128 everywhere
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        EXPECT_NEAR(out.Y.data[x + y*out.Y.width], 128, 1);

    for (int y = 0; y < out.Cb.height; ++y)
      for (int x = 0; x < out.Cb.width; ++x) {
        EXPECT_NEAR(out.Cb.data[x + y*out.Cb.width], 128, 1);
        EXPECT_NEAR(out.Cr.data[x + y*out.Cr.width], 128, 1);
      }
}