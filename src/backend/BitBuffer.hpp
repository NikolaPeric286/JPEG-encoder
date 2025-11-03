#pragma once
#include <cstdint>
#include <ostream>
#include <vector>
#include <stdexcept>

struct BitBuffer{
    std::vector<uint8_t> byte_vector;

    uint32_t acc = 0; // accumulator (>= 16 bits)
    int      nbits = 0; // number of valid bits in acc [0..31]

    // Stuffed byte emit
    void emit_byte(uint8_t b){
        byte_vector.push_back(b);
        if (b == 0xFF) byte_vector.push_back(0x00); // JPEG stuffing
    }

    // Push 'len' bits of 'write_bits' (MSB-first)
    template<typename WRITE_TYPE, typename LEN_TYPE>
    void push(WRITE_TYPE write_bits, LEN_TYPE len){
        if (len < 0 || len > static_cast<int>(sizeof(WRITE_TYPE)*8))
            throw std::out_of_range("len is greater than input type size");

        // keep only 'len' LSBs of write_bits
        uint32_t v = static_cast<uint32_t>(write_bits);
        if (len < 32) v &= ((1u << len) - 1u);

        // append to the right side of acc (so MSBs of v leave first)
        acc = (acc << len) | v;
        nbits += static_cast<int>(len);

        // while we have at least one full byte, emit the top byte
        while (nbits >= 8){
            uint8_t out = static_cast<uint8_t>(acc >> (nbits - 8));
            emit_byte(out);
            nbits -= 8;
            acc &= ((nbits == 32) ? 0xFFFFFFFFu : ((1u << nbits) - 1u));
        }
    }

    // Pad with 1s to next byte boundary (JPEG requirement), then emit
    void flush(){
        if (nbits > 0){
            int pad = 8 - nbits;
            acc = (acc << pad) | ((1u << pad) - 1u);
            nbits = 8;
            uint8_t out = static_cast<uint8_t>(acc);
            emit_byte(out);
            acc = 0;
            nbits = 0;
        }
    }

    void writeToStream(std::ostream& out_stream){
        for (uint8_t b : byte_vector) out_stream.put(static_cast<char>(b));
    }
};

/*
struct BitBuffer{
    
    BitBuffer() : buffer_size(0), bit_buffer(0x00){}

    std::vector<uint8_t> byte_vector;
    uint8_t bit_buffer;
    uint8_t buffer_size;

    template<typename WRITE_TYPE, typename LEN_TYPE>
    void push(WRITE_TYPE write_bits, LEN_TYPE len);

    void flush();
    void writeToStream(std::ostream& out_stream);


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
            if( bit_buffer == 0xFF) {
                byte_vector.push_back(0x00);
            }
            bit_buffer = 0x00;
            buffer_size = 0;
            return;
        }
        else if( buffer_room < len){
            bit_buffer <<= buffer_room;
            bit_buffer |= (write_bits >> (len - buffer_room));
            byte_vector.push_back(bit_buffer);
            if( bit_buffer == 0xFF) {
                byte_vector.push_back(0x00);
            }
            bit_buffer = 0x00;
            buffer_size = 0;
            len -= buffer_room;
        }
        else{
            bit_buffer <<= len;
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
        if( bit_buffer == 0xFF) {
            byte_vector.push_back(0x00);
        }
        bit_buffer = 0x00;
        buffer_size = 0;
        len -= 8;
    }
    
    bit_buffer = write_bits & ((0x01u << len) -0x01u); //puts the remaining bits (len < 8) into the buffer
    buffer_size = len;

}
*/ 