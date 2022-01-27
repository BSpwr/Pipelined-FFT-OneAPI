#ifndef KERNEL_H__
#define KERNEL_H__

#include <memory>
#include <vector>
#if defined(FPGA) || defined(FPGA_EMULATOR)
#include <sycl/ext/intel/fpga_extensions.hpp>
#else
#include <CL/sycl.hpp>
#endif

#define NUM_POINTS 64

std::vector<sycl::float2> PipelinedComplexMult4(std::vector<sycl::float2>& input, cl::sycl::queue &q);

void input_reorder(sycl::float2* X, sycl::float2* d0, sycl::float2* d1, sycl::float2* d2, sycl::float2* d3, unsigned int num_points);

void butterfly(sycl::float2 input0, sycl::float2 input1, sycl::float2* output0, sycl::float2* output1);

void complex_mult(sycl::float2 input0, sycl::float2 input1, sycl::float2* output);

void first_stage(sycl::float2 a0, sycl::float2 a1, sycl::float2 b0, sycl::float2 b1, sycl::float2& out_a0, sycl::float2& out_a1, sycl::float2& out_b0, float& out_b1);

void FFT_1d_1024_pipeline(sycl::float2 a0, sycl::float2 a1, sycl::float2 b0, sycl::float2 b1, sycl::float2& out_a0, sycl::float2& out_a1, sycl::float2& out_b0, sycl::float2& out_b1);


#endif // KERNEL_H__