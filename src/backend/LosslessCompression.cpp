#include "LosslessCompression.hpp"


uint8_t bitlen_abs(int v) {
    if (v == 0 ) return 1;
    unsigned x = (v < 0 ? -v : v);
    uint8_t n = 0; 
    while (x) {
        x >>= 1; 
        ++n; 
    }
    return n; // 0 if v==0
}


void buildHuffTable(const uint8_t bits[16], const uint8_t* vals, int nvals, HuffCode out_code_for_symbol[256]){

    for (int s = 0; s < 256; ++s) out_code_for_symbol[s] = {0, 0};

    // there can be no 0 bit length
    uint8_t val_index = 0;
    uint16_t code = 0;                    // <= because 16 is the length not the array index which is len-1
    for(uint8_t bit_length = 1; bit_length <= 16; bit_length++){
        uint8_t length_count = bits[bit_length-1]; // bits[0] is the # of 1 bit long codes

        for(int j = 0; j < length_count; j++){

            uint8_t symbol = vals[val_index]; // we are calculating the index, cause its like a look up table so at this index of symbol value should the code be
            val_index++;
            out_code_for_symbol[symbol].len = bit_length;;
            out_code_for_symbol[symbol].code = code;
            code++; // this is where we increment the code for each code of that size

        }
        code <<= 1; //we left shift the code to make it one length longer to corispond with the next bit length
        // the last shift will cause an overflow but it shouldnt matter
    }

    /* this is the old chatgpt code that should be deleted, i understand how it works now.
    // init with zeros
    for (int i=0;i<256;++i){
        out_code_for_symbol[i] = {0,0}; 
    }

    int code_count[17] = {0};
    for (int l=1; l<=16; ++l){
        code_count[l] = bits[l-1];
    }

    // 2) first code per length (canonical)
    int first_code[17] = {0};
    int code = 0;
    for (int l=1; l<=16; ++l) {
        first_code[l] = code;
        code = (code + code_count[l]) << 1;
    }

    // 3) assign codes to symbols in order
    int idx = 0;
    for (int l=1; l<=16; ++l) {
        int c = first_code[l];
        for (int i=0; i<code_count[l]; ++i, ++idx, ++c) {
            const uint8_t sym = vals[idx];
            out_code_for_symbol[sym] = HuffCode{ static_cast<uint16_t>(c), static_cast<uint8_t>(l) };
        }
    }
    */
}

void zigZagTransform( int16_t* input_array){ // this should be rewritten to use a precomputed index map, too many branch instructions now
    constexpr uint8_t N = 8;
    int16_t buffer_copy[N*N];
    std::memcpy(&buffer_copy, input_array, sizeof(int16_t)*N*N);
    
    int i = 1;
    size_t x = 1;
    size_t y = 0;
    bool dir = false; // 0 is bottom left 1 is top right
    while(!(x==N-1 && y==N-1) && i < 128){
        //std::cout << i << " -> X: " << x << " Y: " << y << "\n";
        input_array[i] = buffer_copy[x+y*N];
        if(!dir){
            if(!(x == 0 || y == N-1)){
                x--;
                y++;
            }
            else if (y!=N-1){
                y++;
                dir = 1;
            }
            else{
                x++;
                dir = 1;
            }
        }
        else{
            if(!(y == 0 || x == N-1)){
                y--;
                x++;
            }
            else if(x!=N-1){
                x++;
                dir = 0;
            }
            else{
                y++;
                dir = 0;
            }
        }
        i++;
    }
}



void huffmanEncodeBlock(int16_t* input_block, BitBuffer& bit_buffer, int16_t& prev_dif, bool block_type){
    constexpr uint8_t N = 8;
    
    int bitcount = 0;

//DC section

    // builds table 
    HuffCode dc_table[256];
    buildHuffTable((block_type)?bits_dc_luma:bits_dc_chroma, (block_type)?vals_dc_luma:vals_dc_chroma, 12, dc_table);

    int16_t diff = input_block[0] - prev_dif;
    prev_dif = diff;
    std::cout << "diff -> " << diff << "\n";
    uint8_t category = bitlen_abs(diff);
    HuffCode encoded_dc = dc_table[category];
    std::cout << "encoded_dc.code -> " << encoded_dc.code << "\n";
    std::cout << "encoded_dc.len  ->" << (int)(encoded_dc.len )<< "\n";
    bit_buffer.push<uint16_t, uint8_t>(encoded_dc.code, encoded_dc.len);

    if( diff < 0){
        invert_neg(diff, category);
    }
    bit_buffer.push<int16_t,uint8_t>(diff,category);

//AC section

    HuffCode ac_table[256];
    buildHuffTable(bits_ac_luma, vals_ac_luma, 162, ac_table);



    
   
    size_t run = 0;
    int16_t val = 0;

    for( int i = 1; i < 64; i++){
        val = input_block[i];

        // increments zero run
        if(val == 0){
            run++;
            continue;
        }

        while( run >= 16){
            HuffCode zrl = ac_table[0xF0];
            bit_buffer.push<uint16_t, uint8_t>(zrl.code, zrl.len);
            run -= 16;
        }
        
        uint16_t mag = (val<0)?-val:val;
        uint8_t val_length = bitlen_abs(mag);
        uint8_t symbol = (run << 4) | val_length;

        HuffCode encoded_symbol = ac_table[symbol];

        bit_buffer.push<uint16_t, uint8_t>(encoded_symbol.code, encoded_symbol.len);

        uint8_t mag_len = bitlen_abs(mag);
        uint16_t write_val = mag;
        if (val < 0 ){ // flip the bits if its negative
            invert_neg<uint16_t>(write_val, mag_len);
        }
        
        bit_buffer.push<uint16_t, uint8_t>(write_val, mag_len);
    }

    if ( run > 0 ){
        HuffCode eob = ac_table[0x00];
        bit_buffer.push<uint16_t,uint8_t>(eob.code, eob.len);
    }

    /*
    for(int i = 1; i < 64;i++){
        val = input_block[i];
        
        if(val == 0){
            run++;

            
            continue;
        }
        while ( run >= 16){
            HuffCode zrl = acTbl[0xF0];

            bit_buffer = (bit_buffer << zrl.len) | (zrl.code & ((1u << zrl.len) - 1u));
            bitcount += zrl.len;
            // flushes if run == 16
            while (bitcount >= 8) {
                uint8_t b = (bit_buffer >> (bitcount - 8)) & 0xFF;
                bit_stream->sputc(b);
                if (b == 0xFF) bit_stream->sputc(0x00);
                bitcount -= 8;
            }
            run -=16;
        }
        // next non zero number

        //calulates the 8bit symbol
        int mag = (val<0)? -val : val;
        uint16_t val_write;
        uint8_t val_length = bitlen_abs(mag);
        if (val > 0){
            val_write = (uint16_t)mag;
        }
        else{
            uint16_t mask = (1u << val_length) - 1u;
            val_write = (~uint16_t(mag)) & mask;
        }
        uint8_t symbol = (run << 4) | val_length;
        run = 0;
        HuffCode encval = acTbl[symbol];

        uint32_t bits = encval.code & ((encval.len == 32) ? 0xFFFFFFFFu : ((1u << encval.len) - 1u)); // gets only the imporant bits with a bit mask

        // adds the huf encoded symbol to the buffer
        bit_buffer = (bit_buffer << encval.len) | bits;
        bitcount += encval.len; 
        // inverts negative vals, thats just how it works

        
        

        // adds the val to the bit_buffer
        

        //flushes the buffer 1 byte at a time
        while(bitcount >= 8){
            uint8_t write_byte = uint8_t((bit_buffer >> (bitcount - 8)) & 0xFF ); // gets the farthest 8 bits of the buffer
            bit_stream->sputc(write_byte);

            if (write_byte == 0xFF) bit_stream->sputc(0x00); // JPEG byte-stuffing, required by standard
            
            bitcount -= 8;
        }

        bit_buffer = (bit_buffer << val_length) | val_write;
        bitcount += val_length;

        if(val_length){
            while(bitcount >= 8){
                uint8_t write_byte = uint8_t((bit_buffer >> (bitcount - 8)) & 0xFF ); // gets the farthest 8 bits of the buffer
                bit_stream->sputc(write_byte);

                if (write_byte == 0xFF) bit_stream->sputc(0x00); // JPEG byte-stuffing, required by standard
                
                bitcount -= 8;
            }
        }


    }

    if (run > 0) {
        HuffCode eob = acTbl[0x00]; // (0,0)
        uint16_t code_masked = eob.code & ((1u << eob.len) - 1u);
        bit_buffer = (bit_buffer << eob.len) | code_masked;
        bitcount += eob.len;

        while (bitcount >= 8) {
            uint8_t b = (bit_buffer >> (bitcount - 8)) & 0xFF;
            bit_stream->sputc(b);
            if (b == 0xFF) bit_stream->sputc(0x00);
            bitcount -= 8;
        }
    }

    
    */
}