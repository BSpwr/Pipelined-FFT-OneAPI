#ifndef KERNEL_H__
#define KERNEL_H__

#include <memory>
#include <vector>
#if defined(FPGA) || defined(FPGA_EMULATOR)
#include <sycl/ext/intel/fpga_extensions.hpp>
#else
#include <CL/sycl.hpp>
#endif

//template <size_t num_points>
//std::vector<sycl::float2> PipelinedFFT(std::vector<sycl::float2>& input_vec, cl::sycl::queue &q);

//template <size_t num_points>
//void input_reorder(sycl::float2* X, sycl::float2* d0, sycl::float2* d1, sycl::float2* d2, sycl::float2* d3);

void butterfly(sycl::float2 input0, sycl::float2 input1, sycl::float2& output0, sycl::float2& output1);

void complex_mult(sycl::float2 input0, sycl::float2 input1, sycl::float2& output);

void first_stage(sycl::float2 a0, sycl::float2 a1, sycl::float2 b0, sycl::float2 b1, sycl::float2& out_a0, sycl::float2& out_a1, sycl::float2& out_b0, sycl::float2& out_b1);

void last_stage(sycl::float2 a0, sycl::float2 a1, sycl::float2 b0, sycl::float2 b1, sycl::float2& out_a0, sycl::float2& out_a1, sycl::float2& out_b0, sycl::float2& out_b1);

// template<size_t delay_length, size_t pulse_length>
// void data_shuffler(sycl::float2 a, sycl::float2 b, sycl::float2& out_a, sycl::float2& out_b, sycl::float2 delay_a[delay_length], sycl::float2 delay_b[delay_length], size_t& pulse_counter, bool& mux_sel);

//void FFT_1d_1024_pipeline(sycl::float2 a0, sycl::float2 a1, sycl::float2 b0, sycl::float2 b1, sycl::float2& out_a0, sycl::float2& out_a1, sycl::float2& out_b0, sycl::float2& out_b1);

#include "kernel.tcc"


#endif // KERNEL_H__