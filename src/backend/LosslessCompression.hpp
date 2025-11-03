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

void zigZagTransform( int16_t* input_array);

uint8_t bitlen_abs(int v);

void huffmanEncodeBlock(int16_t* input_block, BitBuffer& bit_buffer, int16_t& prev_diff, bool block_type); // 0 is luma 1 is chroma

template<typename T>
void invert_neg(T& val, uint8_t len){
    val = (~val & ((0x01u << len) - 0x01u));
}