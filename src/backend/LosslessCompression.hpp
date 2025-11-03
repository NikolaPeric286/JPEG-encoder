#pragma once
#include <cstdint>
#include <cstring>
#include <memory>
#include <iostream>
#include "BitBuffer.hpp"
#include "tables.hpp"





struct HuffCode{
    uint16_t code;
    uint8_t len;
};

void buildHuffTable(const uint8_t bits[16], const uint8_t* vals, int nvals, HuffCode out_code_for_symbol[256]);

template<typename T>
void zigZagTransform( T* input_array){ // this should be rewritten to use a precomputed index map, too many branch instructions now
    constexpr uint8_t N = 8;
    T buffer_copy[N*N];
    std::memcpy(&buffer_copy, input_array, sizeof(T)*N*N);
    
    for(int i = 0; i < 64; i++){
        input_array[i] = buffer_copy[zig_zag_table[i]];
    }
}

uint8_t bitlen_abs(int v);

void huffmanEncodeBlock(int16_t* input_block, BitBuffer& bit_buffer, int16_t& prev_diff, bool block_type); // 0 is luma 1 is chroma

template<typename T>
void invert_neg(T& val, uint8_t len) {
    // JPEG rule: for negative value v with size s,
    // send ((1<<s) - 1 + v)  == bitwise NOT of |v| within s bits
    if (len == 0) return;
    T mag = (val < 0) ? -val : val;                    // take magnitude
    val = static_cast<T>(((1u << len) - 1u) ^ mag);    // complement within len bits
}