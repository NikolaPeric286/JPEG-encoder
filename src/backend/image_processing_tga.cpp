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
    
    //std::cout << "\n Header width " << header.width << "\n";
    //std::cout << "Header height " << header.height << "\n";
    //std::cout << "Bytes per pixel" << bytesPerPixel << "\n";
    
    unsigned long long size_requirement = header.width*header.height*bytesPerPixel;
    //std::cout << "size requirement " << size_requirement << "\n";
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

    //std::cout << "output_image.size_bytes = " << output_image.size_bytes << "\n";
    return output_image;
    
}


