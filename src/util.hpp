#ifndef UTIL_HPP__
#define UTIL_HPP__

#include "sycl_include.hpp"

using sycl::float2;

void butterfly(float2 input0, float2 input1, float2& output0, float2& output1) {
    output0[0] = input0[0] + input1[0];
    output0[1] = input0[1] + input1[1];

    output1[0] = input0[0] + (-1 * input1[0]);
    output1[1] = input0[1] + (-1 * input1[1]);
}

void complex_mult(float2 input0, float2 input1, float2& output) {
    float a1 = input0[0] - input0[1];
    float a2 = input1[0] - input1[1];
    float a3 = input1[0] + input1[1];

    float p1 = a1 * input1[1];
    float p2 = a2 * input0[0];
    float p3 = a3 * input0[1];

    output[0] = p1 + p2;
    output[1] = p1 + p3;

    // output[0] = (input0[0] * input1[0]) - (input0[1] * input1[1]);
    // output[1] = (input0[0] * input1[1]) + (input1[0] * input0[1]);
}

void first_stage(float2 a0, float2 a1, float2 b0, float2 b1, float2& out_a0, float2& out_a1, float2& out_b0, float2& out_b1) {
    float2 b_1_out_1;           

    butterfly(a0, a1, out_a0, out_a1);
    butterfly(b0, b1, out_b0, b_1_out_1);

    const float2 imag_neg = {0, -1};

    complex_mult(b_1_out_1, imag_neg, out_b1);
}

void last_stage(float2 a0, float2 a1, float2 b0, float2 b1, float2& out_a0, float2& out_a1, float2& out_b0, float2& out_b1) {
    butterfly(a0, a1, out_a0, out_a1);
    butterfly(b0, b1, out_b0, out_b1);
}

#endif // UTIL_HPP__