#include "ColorConversion.hpp"



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