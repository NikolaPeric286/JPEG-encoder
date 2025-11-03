#include "BitBuffer.hpp"




void BitBuffer::flush(){
    uint8_t buffer_room = 8 - buffer_size;

    bit_buffer <<= buffer_room;
    bit_buffer |= ((0x01u << buffer_room) - 0x01u);

    byte_vector.push_back(bit_buffer);

    if(bit_buffer == 0xFF){ // byte stuffing, this is required by the standard
        byte_vector.push_back(0x00);
    }

    bit_buffer = 0x00;
    buffer_size = 0;


}