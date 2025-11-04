#pragma once
#include <cstdint>
#include <ostream>
#include <vector>
#include <stdexcept>

struct BitBuffer{
    std::vector<uint8_t> byte_vector;

    uint32_t bit_buffer = 0; // accumulator (>= 16 bits)
    int buffer_size = 0; // number of valid bits in acc [0..31]

    // Stuffed byte emit
    void emit_byte(uint8_t byte);

    // Push 'len' bits of 'write_bits' (MSB-first)
    template<typename WRITE_TYPE, typename LEN_TYPE>
    void push(WRITE_TYPE write_bits, LEN_TYPE len);

    // Pad with 1s to next byte boundary (JPEG requirement), then emit
    void flush();

    void writeToStream(std::ostream& out_stream);
};



template<typename WRITE_TYPE, typename LEN_TYPE>
void BitBuffer::push(WRITE_TYPE write_bits, LEN_TYPE len){
    if (len < 0 || len > static_cast<int>(sizeof(WRITE_TYPE)*8)){
        throw std::out_of_range("len is greater than input type size");
    }

    // keep only 'len' LSBs of write_bits
    uint32_t v = static_cast<uint32_t>(write_bits);
    if (len < 32){
        v &= ((1u << len) - 1u);
    }

    // append to the right side of acc (so MSBs of v leave first)
    bit_buffer = (bit_buffer << len) | v;
    buffer_size += static_cast<int>(len);

    // while we have at least one full byte, emit the top byte
    while (buffer_size >= 8){
        uint8_t out = static_cast<uint8_t>(bit_buffer >> (buffer_size - 8));
        emit_byte(out);
        buffer_size -= 8;
        bit_buffer &= ((buffer_size == 32) ? 0xFFFFFFFFu : ((1u << buffer_size) - 1u));
    }
}


