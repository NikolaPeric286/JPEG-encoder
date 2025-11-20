#include "FileWrite.hpp"


#if USE_CUDA
    #include "cuda_backend/cuda_backend.hpp"
#endif

void writeSOI(std::ostream& out_stream){
    out_stream.put(static_cast<char>(0xFF)); // marker for huff table 0xFFC4
    out_stream.put(static_cast<char>(0xD8));
}

void writeHuffTables(std::ostream& out_stream){
    out_stream.put(static_cast<char>(0xFF)); // marker for huff table 0xFFC4
    out_stream.put(static_cast<char>(0xC4));

    uint16_t len  = 2 + ( (1 + 16 +12) + (1+16+162) + (1+16+12) + (1+16+162));  // length of everything that follows
    
    out_stream.put(static_cast<char>(len >> 8));  // writes len big endian
    out_stream.put(static_cast<char>(len & 0xFF));

    out_stream.put(static_cast<char>(0x00)); // table header for DC Luma
    out_stream.write(reinterpret_cast<const char*>(bits_dc_luma), 16);
    out_stream.write(reinterpret_cast<const char*>(vals_dc_luma), 12);

    out_stream.put(static_cast<char>(0x10)); // table header for AC Luma
    out_stream.write(reinterpret_cast<const char*>(bits_ac_luma), 16);
    out_stream.write(reinterpret_cast<const char*>(vals_ac_luma), 162);

    out_stream.put(static_cast<char>(0x01)); // table header for DC Chroma
    out_stream.write(reinterpret_cast<const char*>(bits_dc_chroma), 16);
    out_stream.write(reinterpret_cast<const char*>(vals_dc_chroma), 12);

    out_stream.put(static_cast<char>(0x11)); // table header for AC Chroma
    out_stream.write(reinterpret_cast<const char*>(bits_ac_chroma), 16);
    out_stream.write(reinterpret_cast<const char*>(vals_ac_chroma), 162);

}

void writeAPP0(std::ostream& out_stream){
    out_stream.write(reinterpret_cast<const char*>(app0_header), 18);
}

void writeQuantTables(std::ostream& out_stream){

    // Luma quant table
    out_stream.put(static_cast<char>(0xFF));
    out_stream.put(static_cast<char>(0xDB));

    uint16_t len = 2 + 1 + 64 + 1 + 64;


    // len write
    out_stream.put(static_cast<char>(len >> 8));
    out_stream.put(static_cast<char>(len & 0xFF));
    
    // table precision and ID
    out_stream.put(static_cast<char>(0x00)); // 0 - > 8 bit values 0 -> luma quant table id

    // quant table
    uint8_t write_table[64];
    std::memcpy(write_table, Y_q_table_50, 64);
    zigZagTransform<uint8_t>(write_table);

    out_stream.write(reinterpret_cast<const char*>(write_table), 64);



    // Chroma quant table
    // table precision and ID
    out_stream.put(static_cast<char>(0x01)); // 0 - > 8 bit values 1 -> chroma quant table id

    // quant table
    std::memcpy(write_table, C_q_table_50, 64);
    zigZagTransform<uint8_t>(write_table);
    out_stream.write(reinterpret_cast<const char*>(write_table), 64);

}

void writeStartOfFrame(YCbCrImage& image, std::ostream& out_stream){

    out_stream.put(static_cast<char>(0xFF)); // SOF0 marker
    out_stream.put(static_cast<char>(0xC0));

    uint16_t len  = 2  + 1 + 2 + 2 + 1 + 3 + 3 + 3;

    out_stream.put(static_cast<char>(len >> 8)); // writes len big endian
    out_stream.put(static_cast<char>(len & 0xFF));

    uint8_t sample_precision = image.bit_depth;
    uint16_t image_height = image.height;
    uint16_t image_width = image.width;
    uint8_t image_components = 3;

    out_stream.put(static_cast<char>(sample_precision));
    
    out_stream.put(static_cast<char>(image_height >> 8));
    out_stream.put(static_cast<char>(image_height & 0xFF));
    out_stream.put(static_cast<char>(image_width >> 8));
    out_stream.put(static_cast<char>(image_width & 0xFF));

    out_stream.put(static_cast<char>(image_components));

    out_stream.put(static_cast<char>(0x01));  // Luma channel component id
    out_stream.put(static_cast<char>(0x22));  // Luma Sampling factor byte x:y 2:2
    out_stream.put(static_cast<char>(0x00));  // Luma quant table id

    out_stream.put(static_cast<char>(0x02));  // Cb channel component id
    out_stream.put(static_cast<char>(0x11));  // Chroma sampling factor 1:1
    out_stream.put(static_cast<char>(0x01));  // Chroma quant table id

    out_stream.put(static_cast<char>(0x03));  // Cr channel component id
    out_stream.put(static_cast<char>(0x11));  // Chroma sampling factor 1:1
    out_stream.put(static_cast<char>(0x01));  // Chroma quant table id

}

void writeSOS(std::ostream& out_stream){
    out_stream.put(static_cast<char>(0xFF)); // SOS marker
    out_stream.put(static_cast<char>(0xDA));

    uint16_t len = 2 + 1 + 3 + 3 + 3;

    out_stream.put(static_cast<char>(len >> 8)); // writes len big endian
    out_stream.put(static_cast<char>(len & 0xFF));

    uint8_t component_count = 3;

    out_stream.put(static_cast<char>(component_count));

    // Luma
    out_stream.put(static_cast<char>(0x01)); // id for luma
    out_stream.put(static_cast<char>(0x00)); // dc table id -> 0 ac table id -> 0

    // Cb
    out_stream.put(static_cast<char>(0x02)); // id for Cb
    out_stream.put(static_cast<char>(0x11)); // dc table id -> 1 ac table id -> 1

    //Cr
    out_stream.put(static_cast<char>(0x03)); // id for Cr
    out_stream.put(static_cast<char>(0x11)); // dc table id -> 1 ac table id -> 1

    out_stream.put(static_cast<char>(0x00)); // start of spectral selection. 0 is baseline
    out_stream.put(static_cast<char>(0x3F)); // end of spectral selection. 63 is baseline
    out_stream.put(static_cast<char>(0x00)); // successive approximation bits. 0 is baseline

}


void writeImage(YCbCrImage& image, std::ostream& out_stream){
    
    BitBuffer buffer_object;

    uint8_t temp_block[64];
    int8_t level_shifted_block[64];
    int16_t out_block[64];

    uint32_t y = 0;
    

    int16_t prev_diff_Y = 0;
    int16_t prev_diff_Cb = 0;
    int16_t prev_diff_Cr = 0;

    //printf("Image Dimensions, (%d,%d)\n", image.width, image.height);
    while(y < image.height){
        //std::cout << "y -> " << y << "\n";
        uint32_t x = 0;
        while(x < image.width){
            
            // luma blocks
            int luma_x = x;
            int luma_y = y;
            for(int n = 0; n < 4; n++){   //  n=0, n=1
                //                            n=2, n=3
                
                //gets the needed Y block into temp_block
                for( int y_b = 0; y_b < 8; y_b++){
                    for(int x_b = 0; x_b < 8; x_b++){

                        if( x_b + luma_x >= image.Y.width || y_b + luma_y >= image.Y.height){
                            temp_block[x_b + 8*y_b] = 0;
                            //printf( "Padded! block-> (%d,%d)\n", x, y);
                        }
                        else{
                            int srcY = image.Y.height - 1 - (luma_y + y_b);
                            temp_block[x_b + 8*y_b] = image.Y.data[ (x_b + luma_x) + ((srcY)*image.Y.width)];
                        }
                    }
                }

                for(int i = 0; i < 64; i++){
                    level_shifted_block[i] = temp_block[i] - 128;
                }

                //printf( "Encoding Luma section %d in block -> (%d,%d)\n", n, luma_x, luma_y);
                DCT8x8AndQuantize(level_shifted_block,out_block, Y_q_table_50 );
                zigZagTransform<int16_t>(out_block);
                huffmanEncodeBlock(out_block, buffer_object, prev_diff_Y, 0);
                
                // sequences the Y blocks
                switch(n){
                case 0:
                    luma_x +=8;
                    break;
                case 1:
                    luma_x -=8;
                    luma_y +=8;
                    break;
                case 2:
                    luma_x+=8;
                    break;
                default:
                    break;
                }
            }
            int chroma_x = x >>1;
            int chroma_y = y >>1;

            for( int y_b = 0; y_b < 8; y_b++){
                for(int x_b = 0; x_b < 8; x_b++){

                    if( x_b + chroma_x >= image.Cb.width || y_b + chroma_y >= image.Cb.height){
                        temp_block[x_b + 8*y_b] = 0;
                    }
                    else{
                        int srcCb = image.Cb.height - 1 - (chroma_y + y_b);
                        temp_block[x_b + 8*y_b] = image.Cb.data[ (x_b + chroma_x) + ((srcCb)*image.Cb.width)];
                    }
                }
            }

            for(int i = 0; i < 64; i++){
                level_shifted_block[i] = temp_block[i] - 128;
            }
            
            //( "Encoding Cb section of block -> (%d,%d)\n", x, y);
            DCT8x8AndQuantize(level_shifted_block,out_block, C_q_table_50 );
            zigZagTransform<int16_t>(out_block);
            huffmanEncodeBlock(out_block, buffer_object, prev_diff_Cb, 1);
            
            for( int y_b = 0; y_b < 8; y_b++){
                for(int x_b = 0; x_b < 8; x_b++){

                    if( x_b + chroma_x >= image.Cr.width || y_b + chroma_y >= image.Cr.height){
                        temp_block[x_b + 8*y_b] = 0;
                    }
                    else{
                        int srcCr = image.Cr.height - 1 - (chroma_y + y_b);
                        temp_block[x_b + 8*y_b] = image.Cr.data[ (x_b + chroma_x) + ((srcCr)*image.Cr.width)];
                    }
                }
            }

            for(int i = 0; i < 64; i++){
                level_shifted_block[i] = temp_block[i] - 128;
            }
            
            //printf( "Encoding Cr section of block -> (%d,%d)\n", x, y);
            DCT8x8AndQuantize(level_shifted_block,out_block, C_q_table_50 );
            zigZagTransform<int16_t>(out_block);
            huffmanEncodeBlock(out_block, buffer_object, prev_diff_Cr, 1);
            
            x+=16;
        }
        y+=16;
    }
    
    buffer_object.flush();
    //std::cout << "bit buffer size = " << buffer_object.byte_vector.size() << "\n";
    buffer_object.writeToStream(out_stream);

}

void writeEOI(std::ostream& out_stream){
    out_stream.put(static_cast<char>(0xFF));
    out_stream.put(static_cast<char>(0xD9));
}


void encodeRGBImage(RGBImage& rgb_image, std::ostream& out_stream){
    YCbCrImage color_converted_image = make_YCbCr420(rgb_image.width, rgb_image.height);;
        //std::cout << "converting and downsampling\n";

    #if USE_CUDA
        std::cout << "Using cuda\n";
        cudaConvertAndDownsample(rgb_image, color_converted_image);
    #else
        std::cout << "Not using cuda\n";
        convert_and_downsample(rgb_image, color_converted_image);
    #endif

    //std::cout << "writing SOI\n";
    writeSOI(out_stream);
    //std::cout << "writing APP0\n";
    writeAPP0(out_stream);
    //std::cout << "writing quant tables\n";
    writeQuantTables(out_stream);
    //std::cout << "Writing SOF\n";
    writeStartOfFrame(color_converted_image, out_stream);
    //std::cout << "writing huff tables\n";
    writeHuffTables(out_stream); 
    //std::cout << "writing SOS\n";
    writeSOS(out_stream);
    //std::cout << "writing image\n";
    writeImage(color_converted_image, out_stream);
    //std::cout << "writing EOI\n";
    writeEOI(out_stream);
    

}