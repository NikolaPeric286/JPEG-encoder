#include <gtest/gtest.h>
#include <cstdint>
#include <bitset>
#include <iomanip>
#include <string>
#include "../src/backend/LosslessCompression.hpp"

static std::vector<uint8_t> bufToBytes(std::stringbuf& sb) {
    const std::string& s = sb.str();
    return std::vector<uint8_t>(s.begin(), s.end());
}

static void ExpectCodeEq(const HuffCode& hc, uint16_t code, uint8_t len) {
    ASSERT_EQ(hc.len, len) << "length mismatch";
    ASSERT_EQ(hc.code, code) << "code mismatch for length " << int(len);
}

TEST(ZigZag, test_zig_zag_order){
    constexpr uint8_t N = 8;
    int16_t test_array[N*N];
    for(int16_t i = 0; i < N*N; i++){
        test_array[i] = i;
    }
    
    zigZagTransform(test_array);

    const int16_t expected_array[]= {   0,	1,	8,	16,	9,	2,	3,	10,
                                        17,	24,	32,	25,	18,	11,	4,	5,
                                        12,	19,	26,	33,	40,	48,	41,	34,
                                        27,	20,	13,	6,	7,	14,	21,	28,
                                        35,	42,	49,	56,	57,	50,	43,	36,
                                        29,	22,	15,	23,	30,	37,	44,	51,
                                        58,	59,	52,	45,	38,	31,	39,	46,
                                        53,	60,	61,	54,	47,	55,	62,	63};
    bool comparison_result = std::memcmp(test_array, expected_array, sizeof(int16_t)*N*N);
    
    EXPECT_EQ(comparison_result, 0) << comparison_result;
}

TEST(HuffmanBuild, BuildsDefaultACLumaCanonicalCodes) {
    HuffCode table[256];
    buildHuffTable(bits_ac_luma, vals_ac_luma, 162, table);

    // Known mappings (from JPEG Annex K):
    // EOB = (RUN=0, SIZE=0) -> symbol 0x00 -> code 1010 (4 bits)
    {
        uint8_t sym = 0x00;
        // 1010b = 0xA, len=4
        ExpectCodeEq(table[sym], /*code*/0b1010, /*len*/4);
    }

    // (0,1) -> symbol 0x01 -> code 00 (2 bits)
    {
        uint8_t sym = 0x01;
        ExpectCodeEq(table[sym], 0b00, 2);
    }

    // (0,2) -> 0x02 -> 01 (2 bits)
    {
        uint8_t sym = 0x02;
        ExpectCodeEq(table[sym], 0b01, 2);
    }

    // (0,3) -> 0x03 -> 100 (3 bits)
    {
        uint8_t sym = 0x03;
        ExpectCodeEq(table[sym], 0b100, 3);
    }

    // ZRL (16 zeros) -> (15,0) -> symbol 0xF0
    // Default code is long: 11111111001 (11 bits)
    {
        uint8_t sym = 0xF0;
        ExpectCodeEq(table[sym], 0b11111111001, 11);
    }

    // A couple more spot checks from your list:
    // (1,1) -> 0x11 -> 1100 (4 bits)
    {
        uint8_t sym = 0x11;
        ExpectCodeEq(table[sym], 0b1100, 4);
    }

    // (2,2) -> 0x22 -> 11111001 (8 bits)
    {
        uint8_t sym = 0x22;
        ExpectCodeEq(table[sym], 0b11111001, 8);
    }

    // Ensure unused symbols have len==0 (pick something not in vals_ac_luma, e.g., 0xFE is in vals; choose a rare like 0x0B isn't in list)
    {
        uint8_t sym = 0x0B; // not present in default AC-luma table
        EXPECT_EQ(table[sym].len, 0) << "Unexpected code generated for unused symbol 0x0B";
    }
}

TEST(HuffmanACEncode, AllZeros_EmitsNoBytesBecauseOnlyEOBBits) {
    int16_t zz[64] = {0};
    // DC ignored by your AC-only function; all AC are zero → trailing zeros → EOB emitted (4 bits)
    std::stringbuf outbuf(std::ios::out | std::ios::binary);
    huffmanEncodeBlock(zz, &outbuf);

    auto bytes = bufToBytes(outbuf);
    EXPECT_TRUE(bytes.empty()) << "EOB is only 4 bits; encoder should not flush a byte yet.";
}

// 2) Two zeros then −3 → symbol (run=2,size=2)=0x22 → Huffman code = 11111001 (0xF9)
//    Value bits for −3 (size=2): invert(‘11’) → ‘00’ → stays buffered (2 bits)
TEST(HuffmanACEncode, Run2_Neg3_Emits0xF9ThenKeeps2BitsBuffered) {
    int16_t zz[64] = {0};
    // AC: positions 1,2 are zeros; position 3 = -3; rest zero
    zz[3] = -3;

    std::stringbuf outbuf(std::ios::out | std::ios::binary);
    huffmanEncodeBlock(zz, &outbuf);

    auto bytes = bufToBytes(outbuf);
    ASSERT_EQ(bytes.size(), 1u) << "Huffman(0x22) is 8 bits; value bits add 2 more but do not flush another byte.";
    EXPECT_EQ(bytes[0], 0xF9) << "Default AC-luma code for 0x22 should be 0xF9 (11111001).";
}

// 3) Sixteen zeros then +1 → must emit ZRL (0xF0).
//    Default AC-luma Huffman(0xF0) = 11111111001 (11 bits) → flush 0xFF and stuff 0x00, keep 3 bits.
//    Then (run=0,size=1) for +1: symbol 0x01 → code '00' (2 bits), value bit '1' → total 3+2+1=6 pending (no extra flush)
TEST(HuffmanACEncode, SixteenZerosThenPlus1_EmitsStuffedFF00) {
    int16_t zz[64] = {0};
    // Make first 16 ACs zero, then +1 afterwards.
    // AC indices 1..16 = zero; index 17 = +1 (i.e., 16 zeros then a nonzero)
    zz[17] = +1;

    std::stringbuf outbuf(std::ios::out | std::ios::binary);
    huffmanEncodeBlock(zz, &outbuf);

    auto bytes = bufToBytes(outbuf);
    ASSERT_GE(bytes.size(), 2u) << "ZRL (11 bits) must flush one 0xFF and stuff 0x00.";
    EXPECT_EQ(bytes[0], 0xFF) << "First flushed byte of ZRL code should be 0xFF.";
    EXPECT_EQ(bytes[1], 0x00) << "JPEG requires stuffing 0x00 after any 0xFF byte in entropy data.";
}