#include "ColorConversion.hpp"





YCbCrImage make_YCbCr420(const int &W,const int &H){
    YCbCrImage output_image;

    output_image.width = W;
    output_image.height = H;
    output_image.subsampling = Subsampling::YUV420;

    const int y_width = W;
    const int y_height = H;
    
    //both Cb and Cr are gonna have the same plane sizes
    const int color_width = W/2;
    const int color_height = H/2;

    const size_t y_size = size_t(y_height)*y_width; // size of the Y section in bytes
    const size_t c_size = size_t(color_height)*color_width; // size one color plane in bytes
    output_image.size_bytes = y_size + c_size + c_size;
    output_image.storage.reset(new uint8_t[y_size + c_size + c_size]);
    //                data width data bytes_per_row
    output_image.Y = {output_image.storage.get(), y_width, y_height, y_width};
    output_image.Cb = {output_image.storage.get() + y_size, color_width, color_height, color_width};
    output_image.Cr = {output_image.storage.get() + y_size+c_size, color_width, color_height, color_width};

    return output_image;
}



void convert_and_downsample(RGBImage& src_image, YCbCrImage& output_image){
                    //BGR
    const int W = output_image.width;
    const int H = output_image.height;

    const uint8_t* src_ptr = src_image.pixel_array.get();

    uint8_t* Y_ptr = output_image.Y.data;
    uint8_t* Cb_ptr = output_image.Cb.data;
    uint8_t* Cr_ptr = output_image.Cr.data;

    const int row_stride = W*3;

    // x,y goes every 2 pixels 
    for(size_t y = 0; y < H; y+=2){
        const int y1 = (y + 1 < H) ? (y + 1) : y; // bottom edge case
        // y1 is the next pixel if it is available

        for(size_t x = 0; x < W; x+=2){
            const int x1 = (x + 1 < W) ? (x + 1) : x; // clamp at right edge
            // x1 is the next pixel if it is available

            /*
                00 01    (B00 G00 R00) (B01 G01 R01)
                10 11    (B10 G10 R10) (B11 G11 R11)
            */

            //pixel 00
            uint8_t B00 = src_ptr[3*x+0 + y*row_stride], G00 = src_ptr[3*x+1 + y*row_stride], R00 = src_ptr[3*x+2 + y*row_stride];
            //pixel 01
            uint8_t B01 = src_ptr[3*x1+0 +y*row_stride], G01 = src_ptr[3*x1+1 +y*row_stride], R01 = src_ptr[3*x1+2+y*row_stride];
            //pixel 10
            uint8_t B10 = src_ptr[3*x+0+y1*row_stride], G10 = src_ptr[3*x+1+y1*row_stride], R10 = src_ptr[3*x+2+y1*row_stride];
            //pixel 11
            uint8_t B11 = src_ptr[3*x1+0+y1*row_stride], G11 = src_ptr[3*x1+1+y1*row_stride], R11 = src_ptr[3*x1+2+y1*row_stride]; 


            //sets the Y values in the Y plane
            Y_ptr[x + y*W] = ((CYR*R00 + CYG*G00 + CYB*B00 + (1<<15)) >> 16);
            Y_ptr[x+1 + y*W] = ((CYR*R01 + CYG*G01 + CYB*B01 + (1<<15)) >> 16);
            Y_ptr[x + (y+1)*W] = ((CYR*R10 + CYG*G10 + CYB*B10 + (1<<15)) >> 16);
            Y_ptr[x+1 + (y+1)*W] = ((CYR*R11 + CYG*G11 + CYB*B11 + (1<<15)) >> 16);

            //averages the rgb values for the 4 pixels
            int RAvg = (R00 + R01 + R10 + R11 + 2) >> 2;
            int GAvg = (G00 + G01 + G10 + G11 + 2) >> 2;
            int BAvg = (B00 + B01 + B10 + B11 + 2) >> 2;

            Cb_ptr[(x>>1) + (y/2)*(W>>1)] = ((CBR*RAvg + CBG*GAvg + CBB*BAvg + (1<<15)) >> 16) + 128;
            Cr_ptr[(x>>1) + (y/2)*(W>>1)] = ((CRR*RAvg + CRG*GAvg + CRB*BAvg + (1<<15)) >> 16) + 128;
        }
    }
    

}