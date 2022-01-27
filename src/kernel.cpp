#include "kernel.h"

// #define float2 vec<float, 2>
// #define float4 vec<float, 4>

using namespace sycl;

std::vector<sycl::float2> PipelinedComplexMult4(std::vector<sycl::float2>& input_vec, cl::sycl::queue &q) {
  // SYCL buffer allocated for device 
//   buffer<float2, 1> input(data_gpu, size);

    // sycl::float2<

      std::cout << "Local Memory Size: "
            << q.get_device().get_info<sycl::info::device::local_mem_size>()
            << std::endl;

    const unsigned int input_size = input_vec.size();
    const unsigned int bytes_size = input_size * sizeof(sycl::float2);

    // sycl::buffer<sycl::float2, 1> input_buf(input.data(), sycl::range<1>(input_size));



    // sycl::buffer<float2, 1> input_buf(input_vec.data(), range<1>(input_size));

    std::vector<float2> output_ret(input_size, {0, 0});
    // std::vector<float2> output_ret1(input_size / 4, {0, 0});
    // std::vector<float2> output_ret2(input_size / 4, {0, 0});
    // std::vector<float2> output_ret3(input_size / 4, {0, 0});
    // sycl::buffer<float2, 1> output_buf(output_ret.data(), range<1>(input_size));

    float2* input_data = sycl::malloc_device<float2>(input_size, q);
    auto copy_host_to_device_event = q.memcpy(input_data, input_vec.data(), bytes_size);

    // for (int i = 0; i < input_size; i++) {
    //     std::cout << input_data[i][0] << ", " << input_data[i][1] << ", ";
    // }
    // std::cout << std::endl;

    // for (int i = 0; i < input_size; i++) {
    //     std::cout << input_data2[i][0] << ", " << input_data2[i][1] << ", ";
    // }
    // std::cout << std::endl;

    float2* output_data0 = sycl::malloc_device<float2>(input_size / 4, q);
    float2* output_data1 = sycl::malloc_device<float2>(input_size / 4, q);
    float2* output_data2 = sycl::malloc_device<float2>(input_size / 4, q);
    float2* output_data3 = sycl::malloc_device<float2>(input_size / 4, q);

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
            input_reorder(in_ptr, out0_ptr, out1_ptr, out2_ptr, out3_ptr, 4);
            for (int i = 0; i < input_size; i++) {
                // complex_mult(in_ptr[i], in_ptr[i], &out_ptr[i]);

            }

        });
    });

      // copy output data back from device to host
  auto copy_device0_to_host_event = q.submit([&](handler& h) {
    // we cannot copy the output data from the device's to the host's memory
    // until the computation kernel has finished
    h.depends_on(kernel_event);
    h.memcpy(output_ret.data(), output_data0, bytes_size / 4);
  });

    auto copy_device1_to_host_event = q.submit([&](handler& h) {
    // we cannot copy the output data from the device's to the host's memory
    // until the computation kernel has finished
    h.depends_on(kernel_event);
    h.memcpy(output_ret.data() + 1 * (input_size / 4), output_data1, bytes_size / 4);
  });

    auto copy_device2_to_host_event = q.submit([&](handler& h) {
    // we cannot copy the output data from the device's to the host's memory
    // until the computation kernel has finished
    h.depends_on(kernel_event);
    h.memcpy(output_ret.data() + 2 *(input_size / 4), output_data2, bytes_size / 4);
  });

    auto copy_device3_to_host_event = q.submit([&](handler& h) {
    // we cannot copy the output data from the device's to the host's memory
    // until the computation kernel has finished
    h.depends_on(kernel_event);
    h.memcpy(output_ret.data() + 3 *(input_size / 4), output_data3, bytes_size / 4);
  });

  // wait for copy back to finish
  copy_device0_to_host_event.wait();
  copy_device1_to_host_event.wait();
  copy_device2_to_host_event.wait();
  copy_device3_to_host_event.wait();

    // // sync
    // input_buf.get_access<access::mode::read_write>();
    // output_buf.get_access<access::mode::read_write>();

    // for (int i = 0; i < input_size; i++) {
    //     std::cout << output_data[i][0] << ", " << output_data[i][1] << ", ";
    // }
    // std::cout << std::endl;

    // q.memcpy(output_ret.data(), output_data, bytes_size);
    // q.wait();

    // sycl::free(input_data, q);
    // sycl::free(input_data2, q);
    // sycl::free(output_data, q);

    // output_ret0.insert(output_ret0.end(), output_ret1.begin(), output_ret1.end());
    // AB.insert(AB.end(), B.begin(), B.end());

    return output_ret;
}

void input_reorder(sycl::float2* X, sycl::float2* d0, sycl::float2* d1, sycl::float2* d2, sycl::float2* d3, unsigned int num_points) {
    unsigned int d0_count = 0;
    unsigned int d1_count = 0;
    unsigned int d2_count = 0;
    unsigned int d3_count = 0;

    const int bits = log2(NUM_POINTS) - 1;
    #pragma unroll
    for (unsigned int i = 0; i < NUM_POINTS; i++) {
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


void butterfly(float2 input0, float2 input1, float2* output0, float2* output1) {
    output0[0] = input0[0] + input1[0];
    output0[1] = input0[1] + input1[1];

    output1[0] = input0[0] + (-1 * input1[0]);
    output1[1] = input0[1] + (-1 * input1[1]);
}

void complex_mult(float2 input0, float2 input1, float2* output) {

    float a1 = input0[0] - input0[1];
    float a2 = input1[0] - input1[1];
    float a3 = input1[0] + input1[1];

    float p1 = a1 * input1[1];
    float p2 = a2 * input0[0];
    float p3 = a3 * input0[1];

    (*output)[0] = p1 + p2;
    (*output)[1] = p1 + p3;

    // (*output)[0] = (input0[0] * input1[0]) - (input0[1] * input1[1]);
    // (*output)[1] = (input0[0] * input1[1]) + (input1[0] * input0[1]);
}

void first_stage(float2 a0, float2 a1, float2 b0, float2 b1, float2& out_a0, float2& out_a1, float2& out_b0, float& out_b1) {
    float2 b_0_1_out_1;
    butterfly(a0, a1, &out_a0, &out_b0);
    butterfly(b0, b1, &out_a1, &b_0_1_out_1);

    const float2 imag_neg = {0, -1};

    // complex_mult(b_0_1_out_1, imag_neg, b_0_1_out_1);
}


// performs one iteration of the pipeline
void FFT_1d_1024_pipeline(sycl::float2 a0, sycl::float2 a1, sycl::float2 b0, sycl::float2 b1, sycl::float2& out_a0, sycl::float2& out_a1, sycl::float2& out_b0, sycl::float2& out_b1) {


}


