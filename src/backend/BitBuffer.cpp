#include "BitBuffer.hpp"


void BitBuffer::emit_byte(uint8_t byte){
    byte_vector.push_back(byte);
    if (byte == 0xFF){ 
        byte_vector.push_back(0x00);
    } // JPEG stuffing
}

void BitBuffer::flush(){
    if (buffer_size > 0){
        int pad = 8 - buffer_size;
        
        bit_buffer = (bit_buffer << pad) | ((1u << pad) - 1u);
        buffer_size = 8;
        uint8_t out = static_cast<uint8_t>(bit_buffer);
        emit_byte(out);
        bit_buffer = 0;
        buffer_size = 0;
    }
}

void BitBuffer::writeToStream(std::ostream& out_stream){

    for (uint8_t b : byte_vector){
        out_stream.put(static_cast<char>(b));
    }

}

