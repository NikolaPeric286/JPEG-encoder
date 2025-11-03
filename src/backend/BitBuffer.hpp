#pragma once
#include <cstdint>
#include <vector>
#include <stdexcept>


struct BitBuffer{
    
    BitBuffer() : buffer_size(0), bit_buffer(0x00){}

    std::vector<uint8_t> byte_vector;
    uint8_t bit_buffer;
    uint8_t buffer_size;

    template<typename WRITE_TYPE, typename LEN_TYPE>
    void push(WRITE_TYPE write_bits, LEN_TYPE len);

    void flush();


};

template<typename WRITE_TYPE, typename LEN_TYPE>
void BitBuffer::push(WRITE_TYPE write_bits, LEN_TYPE len){

    if(len > sizeof(WRITE_TYPE)*8) throw std::out_of_range("len is greater than input type size");
    uint8_t buffer_room = 8 - buffer_size;
    
    if(len < 8){
        if(buffer_room == len){
            bit_buffer <<= len;
            bit_buffer |= write_bits;
            len = 0;
            byte_vector.push_back(bit_buffer);
            bit_buffer = 0x00;
            buffer_size = 0;
            return;
        }
        else if( buffer_room < len){
            bit_buffer <<= buffer_room;
            bit_buffer |= (write_bits >> (len - buffer_room));
            byte_vector.push_back(bit_buffer);
            bit_buffer = 0x00;
            buffer_size = 0;
            len -= buffer_room;
        }
        else{
            bit_buffer << len;
            bit_buffer |= write_bits;
            buffer_size += len;
            len = 0;
            return;
        }
        
    }

    while( len >= 8){
        
        uint8_t shift = len - buffer_room;
        
        bit_buffer <<= buffer_room; // shifts the buffer over to make room for the incoming data

        bit_buffer |= (write_bits >> shift) & ((0x01u << buffer_room) - 0x01u); // appends the bits from write bits according to the size in the buffer
        byte_vector.push_back(bit_buffer);
        bit_buffer = 0x00;
        buffer_size = 0;
        len -= 8;
    }
    
    bit_buffer = write_bits & ((0x01u << len) -0x01u); //puts the remaining bits (len < 8) into the buffer
    buffer_size = len;

}