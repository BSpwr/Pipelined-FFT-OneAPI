#ifndef KERNEL_HPP__
#define KERNEL_HPP__

#include <math.h>
#include <iostream>
#include <limits>
//#include <common.h>
#if defined(FPGA) || defined(FPGA_EMULATOR)
#include <CL/sycl/INTEL/fpga_extensions.hpp>
// Newer versions of DPCPP compiler have this include instead
// #include <sycl/ext/intel/fpga_extensions.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "shift_reg.hpp"
#include "mp_math.hpp"
#include <iostream>
#include "data_shuffler.hpp"
#include "base_inner_stage.hpp"
#include "util.hpp"
#include "fft_pipeline.hpp"

using namespace sycl;
using namespace hldutils;

template <size_t num_points>
void input_reorder(sycl::float2* X, sycl::float2* d0, sycl::float2* d1, sycl::float2* d2, sycl::float2* d3) {
    unsigned int d0_count = 0;
    unsigned int d1_count = 0;
    unsigned int d2_count = 0;
    unsigned int d3_count = 0;

    const int bits = log2(num_points) - 1;
    #pragma unroll
    for (unsigned int i = 0; i < num_points; i++) {
        if (!(i & (1 << bits)) && !(i & (1 << (bits - 1)))) {
            d0[d0_count++] = X[i];
        } else if ((i & (1 << bits)) && !(i & (1 << (bits - 1)))) {
            d1[d1_count++] = X[i];
        } else if (!(i & (1 << bits)) && (i & (1 << (bits - 1)))) {
            d2[d2_count++] = X[i];
        } else if ((i & (1 << bits)) && (i & (1 << (bits - 1)))) {
            d3[d3_count++] = X[i];
        }
    }
}

// template<size_t pulse_length>
// void data_shuffler(float2 a, float2 b, bool input_valid, float2& out_a, float2& out_b, size_t& pulse_counter, bool& mux_sel) {
//     // counter to flip mux signal every pulse_length inputs
//     if (input_valid) {
//         if (pulse_counter == pulse_length) {
//             mux_sel = !mux_sel;
//             pulse_counter = 0;
//         }
//         pulse_counter += 1;
//     }

//     // delay_length - 1 - (delay_length - index)

//     // lower mux (lower mux has inverted mux select)
//     if (mux_sel) { // 0
//         out_b = a;
//     } else { // 1
//         out_b = b;
//     }

//     // upper mux
//     float2 mux1_out;
//     if (!mux_sel) { // 0
//         mux1_out = a;
//     } else { // 1
//         mux1_out = b;
//     }

//     out_a = mux1_out;
// }

template <size_t num_points>
std::vector<sycl::float2> PipelinedFFT(std::vector<sycl::float2>& input_vec, cl::sycl::queue &q) {

   // static_assert(int(log2(num_points)) % 2 == 0, "num_points must be 2^N, where N is even");  // not allowed.

    // pad out input
    for (unsigned i = 0; i < num_points - input_vec.size(); i++) {
        input_vec.push_back({0, 0});
    }

    std::cout << "Local Memory Size: "
        << q.get_device().get_info<sycl::info::device::local_mem_size>()
        << std::endl;

    constexpr unsigned int BYTES_SIZE = num_points * sizeof(sycl::float2);

    std::vector<float2> output_ret(num_points, {0, 0});

    float2* input_data = sycl::malloc_device<float2>(num_points, q);
    auto copy_host_to_device_event = q.memcpy(input_data, input_vec.data(), BYTES_SIZE);

    // for (int i = 0; i < num_points; i++) {
    //     std::cout << input_data[i][0] << ", " << input_data[i][1] << ", ";
    // }
    // std::cout << std::endl;

    // for (int i = 0; i < num_points; i++) {
    //     std::cout << input_data2[i][0] << ", " << input_data2[i][1] << ", ";
    // }
    // std::cout << std::endl;

    float2* output_data0 = sycl::malloc_device<float2>(num_points / 4, q);
    float2* output_data1 = sycl::malloc_device<float2>(num_points / 4, q);
    float2* output_data2 = sycl::malloc_device<float2>(num_points / 4, q);
    float2* output_data3 = sycl::malloc_device<float2>(num_points / 4, q);

    std::cout << "HI" << std::endl;

    // Offload the work to kernel.
    auto kernel_event = q.submit([&](handler &h) {
        
        h.depends_on(copy_host_to_device_event);
    
    //auto a = input.get_access<access::mode::read_write>(h);
    //h.parallel_for(nd_range<1>(range<1>(size), range<1>(BLOCK_SIZE)), [=](nd_item<1> item) {
        // int i = item.get_global_id(0);

        // auto data = input.get_access<access::mode::read_write>(h);

        // auto input = input_buf.get_access<access::mode::read_write, sycl::access::target::local>(h);
        // auto output = output_buf.get_access<access::mode::read_write, sycl::access::target::local>(h);

        // accessor<float2, 1, access::mode::read, access::target::global_buffer> input(input_buf, h);
        // accessor<float2, 1, access::mode::write, access::target::global_buffer> output(output_buf, h);


        h.single_task([=] () [[intel::kernel_args_restrict]] {

            // std::array<float2, num_points> twiddle_factors = twiddle_gen<num_points>();

            device_ptr<float2> in_ptr(input_data);
            device_ptr<float2> out0_ptr(output_data0);
            device_ptr<float2> out1_ptr(output_data1);
            device_ptr<float2> out2_ptr(output_data2);
            device_ptr<float2> out3_ptr(output_data3);
            input_reorder<num_points>(in_ptr, out0_ptr, out1_ptr, out2_ptr, out3_ptr);
            float2 delay_a[4] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
            float2 delay_b[4] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
            size_t pulse_counter = 0;
            bool mux_sel = false;
            #pragma unroll
            for (int i = 0; i < num_points / 4; i++) {
                // data_shuffler<4>(out0_ptr[i], out1_ptr[i], true, out2_ptr[i], out3_ptr[i], pulse_counter, mux_sel);
                // complex_mult(out0_ptr[i], out1_ptr[i], out2_ptr[i]);

            }

        });
    });

      // copy output data back from device to host
  auto copy_device0_to_host_event = q.submit([&](handler& h) {
    // we cannot copy the output data from the device's to the host's memory
    // until the computation kernel has finished
    h.depends_on(kernel_event);
    h.memcpy(output_ret.data(), output_data0, BYTES_SIZE / 4);
  });

    auto copy_device1_to_host_event = q.submit([&](handler& h) {
    // we cannot copy the output data from the device's to the host's memory
    // until the computation kernel has finished
    h.depends_on(kernel_event);
    h.memcpy(output_ret.data() + 1 * (num_points / 4), output_data1, BYTES_SIZE / 4);
  });

    auto copy_device2_to_host_event = q.submit([&](handler& h) {
    // we cannot copy the output data from the device's to the host's memory
    // until the computation kernel has finished
    h.depends_on(kernel_event);
    h.memcpy(output_ret.data() + 2 *(num_points / 4), output_data2, BYTES_SIZE / 4);
  });

    auto copy_device3_to_host_event = q.submit([&](handler& h) {
    // we cannot copy the output data from the device's to the host's memory
    // until the computation kernel has finished
    h.depends_on(kernel_event);
    h.depends_on(copy_device0_to_host_event);
    h.depends_on(copy_device1_to_host_event);
    h.depends_on(copy_device2_to_host_event);
    h.memcpy(output_ret.data() + 3 *(num_points / 4), output_data3, BYTES_SIZE / 4);
  });

  // wait for copy back to finish
  copy_device3_to_host_event.wait();

    // // sync
    // input_buf.get_access<access::mode::read_write>();
    // output_buf.get_access<access::mode::read_write>();

    // for (int i = 0; i < input_size; i++) {
    //     std::cout << output_data[i][0] << ", " << output_data[i][1] << ", ";
    // }
    // std::cout << std::endl;

    // q.memcpy(output_ret.data(), output_data, BYTES_SIZE);
    // q.wait();

    // sycl::free(input_data, q);
    // sycl::free(input_data2, q);
    // sycl::free(output_data, q);

    // output_ret0.insert(output_ret0.end(), output_ret1.begin(), output_ret1.end());
    // AB.insert(AB.end(), B.begin(), B.end());

    return output_ret;
}

// Performs a FFT (to test this thing)
template <size_t num_points>
std::vector<float2> fft_launch(std::vector<float2> input) {
    // Zero-pad the input
    if (input.size() < num_points) {
        for (int i = num_points - input.size(); i >= 0; i--) {
            input.push_back({0,0});
        }
    }

    float2* in_a0 = new float2[num_points / 4];
    float2* in_a1 = new float2[num_points / 4];
    float2* in_b0 = new float2[num_points / 4];
    float2* in_b1 = new float2[num_points / 4];

    float2* ins[] = {in_a0, in_a1, in_b0, in_b1};

    // Reorder input into 4 arrays to perform 4-parallel FFT
    input_reorder<num_points>(input.data(), in_a0, in_a1, in_b0, in_b1);

    FFTPipeline<num_points> fft_pipeline;

    for (unsigned i = 0; i < 4; i++) {
        for (unsigned j = 0; j < num_points / 4; j++) {
            std::cout << "{" << ins[i][j][0] << ", " << ins[i][j][1] << "}, ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::vector<float2> outs_a0;
    std::vector<float2> outs_a1;
    std::vector<float2> outs_b0;
    std::vector<float2> outs_b1;

    for (unsigned i = 0; i < num_points / 4; i++) {
        float2 out_a0, out_a1, out_b0, out_b1;
        bool output_valid;
        fft_pipeline.process(in_a0[i], in_a1[i], in_b0[i], in_b1[i], true, out_a0, out_a1, out_b0, out_b1, output_valid);

        if (output_valid) {
            outs_a0.push_back(out_a0);
            outs_a1.push_back(out_a1);
            outs_b0.push_back(out_b0);
            outs_b1.push_back(out_b1);
        }
    }

    std::cout << std::endl;

    const float2 zero = {0.0, 0.0};

    while (outs_a0.size() < (num_points / 4)) {
        float2 out_a0, out_a1, out_b0, out_b1;
        bool output_valid;
        fft_pipeline.process(zero, zero, zero, zero, true, out_a0, out_a1, out_b0, out_b1, output_valid);

        if (output_valid) {
            outs_a0.push_back(out_a0);
            outs_a1.push_back(out_a1);
            outs_b0.push_back(out_b0);
            outs_b1.push_back(out_b1);
        }
    }

    std::vector<float2> ret;

    for (unsigned j = 0; j < num_points / 4; j++) {
        ret.push_back(outs_a0[j]);
        ret.push_back(outs_a1[j]);
        ret.push_back(outs_b0[j]);
        ret.push_back(outs_b1[j]);
    }

    std::cout << std::endl;

    for (unsigned j = 0; j < num_points; j++) {
        std::cout << "{" << ret[j][0] << ", " << ret[j][1] << "}, ";
    }
    std::cout << std::endl;


    // for (unsigned j = 0; j < num_points / 4; j++) {
    //     std::cout << "{" << outs_a0[j][0] << ", " << outs_a0[j][1] << "}, ";
    // }
    // std::cout << std::endl;
    // for (unsigned j = 0; j < num_points / 4; j++) {
    //     std::cout << "{" << outs_a1[j][0] << ", " << outs_a1[j][1] << "}, ";
    // }
    // std::cout << std::endl;
    //     for (unsigned j = 0; j < num_points / 4; j++) {
    //     std::cout << "{" << outs_b0[j][0] << ", " << outs_b0[j][1] << "}, ";
    // }
    // std::cout << std::endl;
    //     for (unsigned j = 0; j < num_points / 4; j++) {
    //     std::cout << "{" << outs_b1[j][0] << ", " << outs_b1[j][1] << "}, ";
    // }
    // std::cout << std::endl;

    return ret;
}


std::vector<float2> ds_test(std::vector<float2> input) {

    constexpr int num_points = 64;

    // Zero-pad the input
    if (input.size() < num_points) {
        for (int i = num_points - input.size(); i >= 0; i--) {
            input.push_back({0,0});
        }
    }

    DataShuffler<2, 2> ds__;

    float2* in_data0 = new float2[num_points / 4];
    float2* in_data1 = new float2[num_points / 4];
    float2* in_data2 = new float2[num_points / 4];
    float2* in_data3 = new float2[num_points / 4];

    float2* ins[] = {in_data0, in_data1, in_data2, in_data3};

    // Reorder input into 4 arrays to perform 4-parallel FFT
    input_reorder<num_points>(input.data(), in_data0, in_data1, in_data2, in_data3);

    // CREATE INTERNAL PIPELINE (inner_pipeline)

    // // Feed in all inputs
    // for (unsigned i = 0; i < num_points / 4; i++) {
    //     float2& a0 = in_data0[i];
    //     float2& a1 = in_data1[i];
    //     float2& b0 = in_data2[i];
    //     float2& b1 = in_data3[i];

    //     float2 fs0_out, fs1_out, fs2_out, fs3_out;
    //     first_stage(a0, a1, b0, b1, fs0_out, fs1_out, fs2_out, fs3_out);

    //     float2 is0_out, is1_out, is2_out, is3_out;
    //     bool is_out_valid;
    // }

    for (unsigned i = 0; i < 4; i++) {
        for (unsigned j = 0; j < num_points / 4; j++) {
            std::cout << "{" << ins[i][j][0] << ", " << ins[i][j][1] << "}, ";
        }
        std::cout << std::endl;
    }

    std::vector<float2> r1;
    std::vector<float2> r2;

    for (unsigned i = 0; i < num_points / 4; i++) {
        float2 out_a, out_b;
        bool output_valid;
        ds__.process(ins[2][i], ins[3][i], true, out_a, out_b, output_valid);

        if (output_valid) {
            r1.push_back(out_a);
            r2.push_back(out_b);
        }
    }

    std::cout << std::endl;

    const float2 zero = {0.0, 0.0};

    while (r1.size() < num_points / 4) {
        float2 out_a, out_b;
        bool output_valid;
        ds__.process(zero, zero, false, out_a, out_b, output_valid);
        if (output_valid) {
            r1.push_back(out_a);
            r2.push_back(out_b);
        }
    }

    for (unsigned j = 0; j < num_points / 4; j++) {
        std::cout << "{" << r1[j][0] << ", " << r1[j][1] << "}, ";
    }
    std::cout << std::endl;
    for (unsigned j = 0; j < num_points / 4; j++) {
        std::cout << "{" << r2[j][0] << ", " << r2[j][1] << "}, ";
    }
    std::cout << std::endl;

    return {};
}


#endif // KERNEL_HPP__