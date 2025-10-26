#include "DCT.hpp"


const std::array<std::array<float, N>, N>& cos_table() {
    static std::array<std::array<float, N>, N> tbl{};
    static bool init = false;
    if (!init) {
        for (int k = 0; k < N; ++k) {
            for (int x = 0; x < N; ++x) {
                // Note: cos argument for JPEG’s DCT-II: (2x+1)*k*pi/(2N)
                tbl[k][x] = std::cos(((2.0f * x + 1.0f) * k * PI) / (2.0f * N));
            }
        }
        init = true;
    }
    return tbl;
}


void DCT8x8( const uint8_t* input, float* output){
    //assumes level shifted 8x8 block input

    const auto& C_table = cos_table();

    // u is horisontal
    // v is vertical
    for (int u  = 0; u < N; u++){
        for(int v = 0; v < N; v++){
            float sum = 0.0f;

            for(int x = 0; x < N; x++){
                
                const float CXU = C_table[u][x]; // gets the cosine value that is gonna be used for the whole row so it doesnt have to be computed every cycle

                for(int y = 0; y < N; y++){
                    sum += input[x+y*N]*CXU*C_table[v][y];
                }
            }
            output[u + v*N] = 0.25f * alpha(u)* alpha(v) *sum;
        }
    }
}