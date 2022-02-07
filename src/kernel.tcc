#ifndef KERNEL_TCC__
#define KERNEL_TCC__

#include <math.h>
#include <iostream>
#include <limits>
//#include <common.h>
#if defined(FPGA) || defined(FPGA_EMULATOR)
#include <sycl/ext/intel/fpga_extensions.hpp>
#else
#include <CL/sycl.hpp>
#endif

using namespace sycl;

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

// /* Data shuffler must delay inputs long enough so that pairs can align */
// template<size_t delay_length, size_t pulse_length>
// void data_shuffler(float2 a, float2 b, float2& out_a, float2& out_b, float2 delay_a[delay_length], float2 delay_b[delay_length], size_t& pulse_counter, bool& mux_sel, size_t delay) {
//     // counter to flip mux signal every pulse_length inputs
//     pulse_counter += 1;
//     if (pulse_counter == pulse_length) {
//         mux_sel = !mux_sel;
//         pulse_counter = 0;
//     }

//     // delay = (delay > delay_length) ? delay_length : delay;

//     #pragma unroll
//     for (int i = 0; i < delay_length - 1; i++)
//     {
//         delay_b[i] = delay_b[i + 1];
//     }
//     delay_b[delay_length - 1] = b;

//     // lower mux (lower mux has inverted mux select)
//     if (mux_sel) { // 0
//         out_b = a;
//     } else { // 1
//         out_b = delay_b[0];
//     }

//     // upper delay buffer
//     #pragma unroll
//     for (int i = 0; i < delay_length - 1; i++) {
//         delay_a[i] = delay_b[i + 1];
//     }

//     // upper mux
//     float2 mux1_out;
//     if (!mux_sel) { // 0
//         delay_a[delay_length - 1] = a;
//     } else { // 1
//         delay_a[delay_length - 1] = delay_b[0];
//     }

//     out_a = delay_a[0];
// }

// WORKING BUT DELAY IS BROKEN
/* Data shuffler must delay inputs long enough so that pairs can align */
template<size_t delay_length, size_t pulse_length>
void data_shuffler(float2 a, float2 b, float2& out_a, float2& out_b, float2 delay_a[delay_length], float2 delay_b[delay_length], size_t& pulse_counter, bool& mux_sel, size_t index) {
    // counter to flip mux signal every pulse_length inputs
    pulse_counter += 1;
    if (pulse_counter == pulse_length) {
        mux_sel = !mux_sel;
        pulse_counter = 0;
    }

    // lower delay buffer
    #pragma unroll
    for (int i = delay_length - 1; i >= 1; i--)
    {
        delay_b[i] = delay_b[i - 1];
    }
    delay_b[0] = b;

    // delay_length - 1 - (delay_length - index)

    // lower mux (lower mux has inverted mux select)
    if (mux_sel) { // 0
        out_b = a;
    } else { // 1
        out_b = delay_b[delay_length - 1];
    }

    // upper delay buffer
    #pragma unroll
    for (int i = delay_length - 1; i >= 1; i--) {
        delay_a[i] = delay_a[i - 1];
    }

    // upper mux
    float2 mux1_out;
    if (!mux_sel) { // 0
        delay_a[0] = a;
    } else { // 1
        delay_a[0] = delay_b[delay_length - 1];
    }

    out_a = delay_a[delay_length - 1];
}

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
                data_shuffler<4, 2>(out0_ptr[i], out1_ptr[i], out2_ptr[i], out3_ptr[i], delay_a, delay_b, pulse_counter, mux_sel, i);
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

#endif // KERNEL_TCC__