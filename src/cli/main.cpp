// main.c
//
#include <iostream>
#include <fstream>
#include "../backend/image_processing_tga.hpp"
#include "../backend/ColorConversion.hpp"
#include "../backend/Image.hpp"
#include "../backend/BitBuffer.hpp"
#include "../backend/LosslessCompression.hpp"


int main(int arc, char** argv){

    std::ofstream out_file("cli_output");

    int16_t block[64];
    int p = 0;
    for(int i = 0; i < 64; i++){
        block[i] = p;

        if( p == 0){
            p = 50;
        }
        else{
            p = 0;
        }
        
    }
    BitBuffer buffer_object;
    int16_t prev_diff = 0;

    huffmanEncodeBlock(block, buffer_object, prev_diff, 0);
    buffer_object.flush();

    for(auto it = buffer_object.byte_vector.begin(); it != buffer_object.byte_vector.end(); it++){
        out_file << *it;
    }

    out_file.close();
    
}