#include "image_processing_tga.hpp"


RGBImage getImageContentTGA(const char* file_path){
    
    std::ifstream file(file_path);

    if (!file){
        throw std::runtime_error("Failed to open file" + std::string(file_path));
    }

    tgaheader header{};
 
    file.read(reinterpret_cast<char*>(&header), sizeof(header)); // reads the entire header into the tga header struct
    
    if(!file) throw std::runtime_error("Failed to read TGA header");
     
    if (!(header.imageType == 2 || header.imageType == 3)) throw std::runtime_error("Only uncompressed TGA (types 2/3) supported in fast path");

    // Skip ID field using seekg
    if (header.idLength) file.seekg(header.idLength, std::ios::cur);

    // Skip color map, if any (rare for truecolor, but spec allows it)
    if (header.colorMapType) {
        const size_t cmapBytes = static_cast<size_t>(header.colorMapLength) * ((header.colorMapEntrySize + 7) / 8);
        file.seekg(cmapBytes, std::ios::cur);
    }

    const uint8_t bytesPerPixel = (header.pixelDepth + 7) / 8;
    const size_t image_size = static_cast<size_t>(header.height)*header.width*header.pixelDepth;
    
    unsigned long long size_requirement = header.width*header.height*bytesPerPixel;

    if(size_requirement == 0 || size_requirement > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())){ // checks if the array size fits into a size_t
        throw std::overflow_error("Pixel data size overflow\n");
    }   

    RGBImage output_image;
    output_image.size_bytes = size_requirement;
    output_image.width = header.width;
    output_image.height = header.height;
    output_image.bpp = header.pixelDepth;            
    output_image.top_left_origin = (header.imageDescriptor & 0x20) != 0; // 5th bit in imageDescriptor tells if the origin is top left. 0->bottom left 1->top left
    output_image.pixel_array = std::unique_ptr<uint8_t[]>(new uint8_t[size_requirement]);
    file.read(reinterpret_cast<char*>(output_image.pixel_array.get()), static_cast<std::streamsize>(size_requirement)); // cast to streamsize because that is what the funciton is expecting
    
    if(!file) throw std::runtime_error("Failed to read TGA pixel data");

    return output_image;
    
}


void RGBToYCbCr24bit(RGBImage& input_image){
    if(input_image.bpp != 24 ) throw std::invalid_argument("tried to pass 32bit image to 24bit color conversion");


    uint8_t temp_pixel[3]; // BGR
    
    for(size_t i = 0; i < input_image.size_bytes; i+=3){
        std::memcpy(temp_pixel, input_image.pixel_array.get()+i, 3); // loads the pixel into the temp array
        //                                                                          right shifting by 16 is the same as dividing
        //                                                                          by 65536 which undos the scaling factor
        int Y  = (CYR*temp_pixel[2] + CYG*temp_pixel[1] + CYB*temp_pixel[0] + 32768) >> 16;
        int Cb = ((CBR*temp_pixel[2] + CBG*temp_pixel[1] + CBB*temp_pixel[0] + 32768) >> 16) + 128;
        int Cr = ((CRR*temp_pixel[2] + CRG*temp_pixel[1] + CRB*temp_pixel[0] + 32768) >> 16) + 128;

        input_image.pixel_array[i] = clamp_u8(Y);
        input_image.pixel_array[i+1] = clamp_u8(Cb);
        input_image.pixel_array[i+2] = clamp_u8(Cr);
    }
}