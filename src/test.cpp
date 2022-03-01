#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include "math.h"
#include "kernel.hpp"

// unsigned int reverse_bits(unsigned int input, unsigned int bits) {
//     unsigned int rev = 0;
//     for (unsigned int i = 0; i < bits; i++) {
//         rev = (rev << 1) | (input & 1);
//         input = input >> 1;
//     }
//     return rev;
// }

// void bit_reverse(std::vector<float>& X, float* d0, float* d1, float* d2, float* d3) {
//     float temp;
//     unsigned int d0_count = 0;
//     unsigned int d1_count = 0;
//     unsigned int d2_count = 0;
//     unsigned int d3_count = 0;
//     const int bits = log2(NUM_POINTS) - 1;
//     for (unsigned int i = 0; i < NUM_POINTS; i++) {
//         if (!(i & (1 << bits)) && !(i & (1 << bits - 1))) {
//             d0[d0_count++] = X[i];
//         } else if ((i & (1 << bits)) && !(i & (1 << bits - 1))) {
//             d1[d1_count++] = X[i];
//         } else if (!(i & (1 << bits)) && (i & (1 << bits - 1))) {
//             d2[d2_count++] = X[i];
//         } else if ((i & (1 << bits)) && (i & (1 << bits - 1))) {
//             d3[d3_count++] = X[i];
//         }
//     }
// }



int main() {
    constexpr size_t num_points = 1024;

    std::vector<float2> a(num_points, {0, 0});
    for (unsigned i = 0; i < num_points; i++) {
        a[i] = {i, 0};
    }
    // a[0] = {1, 0};
    // a[1] = {1, 0};
    // a[0] = {1, 0};
    // float d0[NUM_POINTS / 4];
    // float d1[NUM_POINTS / 4];
    // float d2[NUM_POINTS / 4];
    // float d3[NUM_POINTS / 4];
    // std::iota(a.begin(), a.end(), 0);

    auto output = fft_launch<num_points>(a);

    // Print outputs
    for (unsigned j = 0; j < num_points; j++) {
        std::cout << "{" << output[j][0] << ", " << output[j][1] << "}, ";
    }
    std::cout << std::endl;

    // bit_reverse(a, d0, d1, d2, d3);

    // unsigned int i = 0;

    // for (int i = 0; i < NUM_POINTS / 4; i++) {
    //     std::cout << d0[i] << " , ";
    // }
    // std::cout << std::endl;

    // for (int i = 0; i < NUM_POINTS / 4; i++) {
    //     std::cout << d1[i] << " , ";
    // }
    // std::cout << std::endl;

    // for (int i = 0; i < NUM_POINTS / 4; i++) {
    //     std::cout << d2[i] << " , ";
    // }
    // std::cout << std::endl;

    // for (int i = 0; i < NUM_POINTS / 4; i++) {
    //     std::cout << d3[i] << " , ";
    // }
    // std::cout << std::endl;

    // for (auto v : a) {
    //     if (i == 4) {
    //         std::cout << std::endl;
    //         i = 0;
    //     }
    //     i++;
    //     std::cout << v << " , ";
    // }
    // std::cout << std::endl;

}

// 0000 1000 0100 1100
// 0001 1001 0101 1101
// 0010 1010 0110 1110
// 0011 1011 0111 1111

// 0000 0001 0010 0011     0100 0101 0110 0111 1000 1001 1010 1011 1100 1101 1110 1111
// 0000 1000 0100 1100     0001 1001 0101 1101 0010 1010 0110 1110 0011 1011 0111 1111

// void bit_reverse(std::vector<float>& input, float* d0, float* d1, float* d2, float* d3) {
//     float temp;
//     for (unsigned int i = 0; i < NUM_POINTS; i++) {
//         unsigned int bucket = i % 4;
//         unsigned int count = i / 4;
//         unsigned int count_4s = count % 4;
//         unsigned int reversed = reverse_bits(i, log2(NUM_POINTS)); // Find the bit reversed index
//         float temp = X[i];
//         unsigned int offset = 0;
//         switch (count_4s) {
//             case 0: {
//                 offset = 0;
//                 break;
//             }
//             case 1: {
//                 offset = 1;
//                 break;
//             }
//             case 2: {
//                 offset = -1;
//                 break;
//             }
//             case 3: {
//                 offset = 0;
//                 break;
//             }
//         }

//         switch (bucket) {
//             case 0: {
//                 d0[count+offset] = temp;
//                 break;
//             }
//             case 1: {
//                 d1[count+offset] = temp;
//                 break;
//             }
//             case 2: {
//                 d2[count+offset] = temp;
//                 break;
//             }
//             case 3: {
//                 d3[count+offset] = temp;
//                 break;
//             }
//         }
//     }
// }