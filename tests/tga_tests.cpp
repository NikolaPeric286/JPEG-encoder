#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <filesystem>
#include <cstdint>
#include "../src/backend/image_processing_tga.hpp"

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
