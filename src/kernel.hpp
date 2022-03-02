#ifndef KERNEL_HPP__
#define KERNEL_HPP__

#include <math.h>
#include <iostream>
#include <iomanip>
#include <type_traits>
#include <limits>

#include "sycl_include.hpp"

#include "shift_reg.hpp"
#include "mp_math.hpp"
#include <iostream>
#include "data_shuffler.hpp"
#include "base_inner_stage.hpp"
#include "util.hpp"
#include "fft_pipeline.hpp"

using namespace sycl;
using namespace hldutils;

template <uint16_t num_points>
void input_reorder(sycl::float2* input, sycl::float2* d0_out, sycl::float2* d1_out, sycl::float2* d2_out, sycl::float2* d3_out) {
    static_assert(IsPowerOfFour(num_points), "num_points must be a power of 4");

    uint16_t d0_count = 0;
    uint16_t d1_count = 0;
    uint16_t d2_count = 0;
    uint16_t d3_count = 0;

    constexpr uint16_t bits = Log2<uint16_t>(num_points) - 1;
    #pragma unroll
    for (uint16_t i = 0; i < num_points; i++) {
        if (!(i & (1 << bits)) && !(i & (1 << (bits - 1)))) {
            d0_out[d0_count++] = input[i];
        } else if ((i & (1 << bits)) && !(i & (1 << (bits - 1)))) {
            d1_out[d1_count++] = input[i];
        } else if (!(i & (1 << bits)) && (i & (1 << (bits - 1)))) {
            d2_out[d2_count++] = input[i];
        } else if ((i & (1 << bits)) && (i & (1 << (bits - 1)))) {
            d3_out[d3_count++] = input[i];
        }
    }
}

template <typename T, uint16_t bit_width>
constexpr T reverse_bits(T input) {
    static_assert(std::is_integral<T>::value, "input must be integral type");
    static_assert(bit_width > 0, "bit width must be greater than zero");
    T output = input;
    T left_mask = (1 << (bit_width - 1));
    T right_mask = 1;

    for (uint16_t i = 0; i < bit_width / 2; i++) {
        T l = (input & left_mask) >> (bit_width - 1 - i*2);
        T r = (input & right_mask) << (bit_width - 1 - i*2);

        output &= ~(1 << i);
        output &= ~(1 << (bit_width - 1 - i));
        output |= l;
        output |= r;

        left_mask >>= 1;
        right_mask <<= 1;
    }

    return output;
}

template <uint16_t num_points>
constexpr std::array<std::array<uint16_t, num_points / 4>, 4> output_reorder_table() {
    static_assert(IsPowerOfFour(num_points), "num_points must be a power of 4");
    
    std::array<std::array<uint16_t, num_points / 4>, 4> arr{};
    std::array<uint16_t, 4> count = {0, 1, 2, 3};

    for (uint16_t i = 0; i < num_points / 4; i++) {
        for (uint16_t j = 0; j < 4; j++) {
            arr[j][i] = count[j];
            count[j] += 4;
        }
    }

    constexpr uint16_t width = Log2<uint16_t>(num_points);

    for (uint16_t i = 0; i < 4; i++) {
        for (uint16_t j = 0; j < num_points / 4; j++) {
            arr[i][j] = reverse_bits<uint16_t, width>(arr[i][j]);
        }
    }

    // for (unsigned i = 0; i < 4; i++) {
    //     for (unsigned j = 0; j < num_points / 4; j++) {
    //         std::cout << arr[i][j] << ",";
    //     }
    //     std::cout << std::endl;
    // }

    return arr;
}

template <uint16_t num_points>
void output_reorder(sycl::float2* d0, sycl::float2* d1, sycl::float2* d2, sycl::float2* d3, sycl::float2* output) {
    static_assert(IsPowerOfFour(num_points), "num_points must be a power of 4");
    constexpr static std::array<std::array<uint16_t, num_points / 4>, 4> output_order = output_reorder_table<num_points>();
    #pragma unroll
    for (uint16_t i = 0; i < num_points / 4; i++) {
            output[output_order[0][i]] = d0[i];
            output[output_order[1][i]] = d1[i];
            output[output_order[2][i]] = d2[i];
            output[output_order[3][i]] = d3[i];
    }
}

template <uint16_t num_points>
std::vector<sycl::float2> PipelinedFFT(std::vector<sycl::float2>& input, cl::sycl::queue &q) {
    static_assert(IsPowerOfFour(num_points), "num_points must be a power of 4");

    // Zero-pad the input
    if (input.size() < num_points) {
        for (int i = num_points - input.size(); i >= 0; i--) {
            input.push_back({0,0});
        }
    }

    std::cout << "Local Memory Size: "
        << q.get_device().get_info<sycl::info::device::local_mem_size>()
        << std::endl;

    constexpr uint32_t BYTES_SIZE = num_points * sizeof(sycl::float2);

    std::vector<float2> output_ret(num_points, {0, 0});

    float2* input_data = sycl::malloc_device<float2>(num_points, q);
    float2* input_data0 = sycl::malloc_device<float2>(num_points / 4, q);
    float2* input_data1 = sycl::malloc_device<float2>(num_points / 4, q);
    float2* input_data2 = sycl::malloc_device<float2>(num_points / 4, q);
    float2* input_data3 = sycl::malloc_device<float2>(num_points / 4, q);

    float2* output_data = sycl::malloc_device<float2>(num_points, q);
    float2* output_data0 = sycl::malloc_device<float2>(num_points / 4, q);
    float2* output_data1 = sycl::malloc_device<float2>(num_points / 4, q);
    float2* output_data2 = sycl::malloc_device<float2>(num_points / 4, q);
    float2* output_data3 = sycl::malloc_device<float2>(num_points / 4, q);

    auto copy_host_to_device_event = q.memcpy(input_data, input.data(), BYTES_SIZE);

    // Offload the work to kernel.
    auto kernel_event = q.submit([&](handler &h) {
        
        h.depends_on(copy_host_to_device_event);
    
        h.single_task([=] () [[intel::kernel_args_restrict]] {

            FFTPipeline<num_points> fft_pipeline;

            device_ptr<float2> in_ptr(input_data);
            device_ptr<float2> in0_ptr(input_data0);
            device_ptr<float2> in1_ptr(input_data1);
            device_ptr<float2> in2_ptr(input_data2);
            device_ptr<float2> in3_ptr(input_data3);

            device_ptr<float2> output_ptr(output_data);
            device_ptr<float2> out0_ptr(output_data0);
            device_ptr<float2> out1_ptr(output_data1);
            device_ptr<float2> out2_ptr(output_data2);
            device_ptr<float2> out3_ptr(output_data3);

            input_reorder<num_points>(in_ptr, in0_ptr, in1_ptr, in2_ptr, in3_ptr);
            
            uint16_t count = 0;

            for (uint16_t i = 0; i < num_points / 4; i++) {
                float2 out_a0, out_a1, out_b0, out_b1;
                bool output_valid;
                fft_pipeline.process(in0_ptr[i], in1_ptr[i], in2_ptr[i], in3_ptr[i], true, out_a0, out_a1, out_b0, out_b1, output_valid);

                if (output_valid) {
                    out0_ptr[count] = out_a0;
                    out1_ptr[count] = out_a1;
                    out2_ptr[count] = out_b0;
                    out3_ptr[count] = out_b1;
                    count++;
                }
            }

            const float2 zero = {0.0, 0.0};

            while (count < (num_points / 4)) {
                float2 out_a0, out_a1, out_b0, out_b1;
                bool output_valid;
                fft_pipeline.process(zero, zero, zero, zero, true, out_a0, out_a1, out_b0, out_b1, output_valid);

                if (output_valid) {
                    out0_ptr[count] = out_a0;
                    out1_ptr[count] = out_a1;
                    out2_ptr[count] = out_b0;
                    out3_ptr[count] = out_b1;
                    count++;
                }
            }

            output_reorder<num_points>(out0_ptr, out1_ptr, out2_ptr, out3_ptr, output_ptr);

        });

    });

    // copy output data back from device to host
    auto copy_device_to_host_event = q.submit([&](handler& h) {
        // we cannot copy the output data from the device's to the host's memory
        // until the computation kernel has finished
        h.depends_on(kernel_event);
        h.memcpy(output_ret.data(), output_data, BYTES_SIZE);
    });

    // wait for copy back to finish
    copy_device_to_host_event.wait();

    sycl::free(input_data, q);
    sycl::free(input_data0, q);
    sycl::free(input_data1, q);
    sycl::free(input_data2, q);
    sycl::free(input_data3, q);

    sycl::free(output_data, q);
    sycl::free(output_data0, q);
    sycl::free(output_data1, q);
    sycl::free(output_data2, q);
    sycl::free(output_data3, q);

    return output_ret;
}

// Performs a FFT (to test this thing)
template <uint16_t num_points>
std::vector<float2> fft_launch(std::vector<float2> input) {
    static_assert(IsPowerOfFour(num_points), "num_points must be a power of 4");

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

    // Print inputs
    /*for (unsigned i = 0; i < 4; i++) {
        for (unsigned j = 0; j < num_points / 4; j++) {
            std::cout << "{" << ins[i][j][0] << ", " << ins[i][j][1] << "}, ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    */
    std::vector<float2> outs_a0;
    std::vector<float2> outs_a1;
    std::vector<float2> outs_b0;
    std::vector<float2> outs_b1;

    for (uint16_t i = 0; i < num_points / 4; i++) {
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

    // std::cout << std::endl;

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

    std::vector<float2> ret(num_points, {0.0, 0.0});

    output_reorder<num_points>(outs_a0.data(), outs_a1.data(), outs_b0.data(), outs_b1.data(), ret.data());

    // The output ordering differs from

    // for (unsigned j = 0; j < num_points / 4; j++) {
    //     ret.push_back(outs_a0[j]);
    //     ret.push_back(outs_a1[j]);
    //     ret.push_back(outs_b0[j]);
    //     ret.push_back(outs_b1[j]);
    // }




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


// std::vector<float2> ds_test(std::vector<float2> input) {

//     constexpr int num_points = 64;

//     // Zero-pad the input
//     if (input.size() < num_points) {
//         for (int i = num_points - input.size(); i >= 0; i--) {
//             input.push_back({0,0});
//         }
//     }

//     DataShuffler<2, 2> ds__;

//     float2* in_data0 = new float2[num_points / 4];
//     float2* in_data1 = new float2[num_points / 4];
//     float2* in_data2 = new float2[num_points / 4];
//     float2* in_data3 = new float2[num_points / 4];

//     float2* ins[] = {in_data0, in_data1, in_data2, in_data3};

//     // Reorder input into 4 arrays to perform 4-parallel FFT
//     input_reorder<num_points>(input.data(), in_data0, in_data1, in_data2, in_data3);

//     // CREATE INTERNAL PIPELINE (inner_pipeline)

//     // // Feed in all inputs
//     // for (unsigned i = 0; i < num_points / 4; i++) {
//     //     float2& a0 = in_data0[i];
//     //     float2& a1 = in_data1[i];
//     //     float2& b0 = in_data2[i];
//     //     float2& b1 = in_data3[i];

//     //     float2 fs0_out, fs1_out, fs2_out, fs3_out;
//     //     first_stage(a0, a1, b0, b1, fs0_out, fs1_out, fs2_out, fs3_out);

//     //     float2 is0_out, is1_out, is2_out, is3_out;
//     //     bool is_out_valid;
//     // }

//     for (unsigned i = 0; i < 4; i++) {
//         for (unsigned j = 0; j < num_points / 4; j++) {
//             std::cout << "{" << ins[i][j][0] << ", " << ins[i][j][1] << "}, ";
//         }
//         std::cout << std::endl;
//     }

//     std::vector<float2> r1;
//     std::vector<float2> r2;

//     for (unsigned i = 0; i < num_points / 4; i++) {
//         float2 out_a, out_b;
//         bool output_valid;
//         ds__.process(ins[2][i], ins[3][i], true, out_a, out_b, output_valid);

//         if (output_valid) {
//             r1.push_back(out_a);
//             r2.push_back(out_b);
//         }
//     }

//     std::cout << std::endl;

//     const float2 zero = {0.0, 0.0};

//     while (r1.size() < num_points / 4) {
//         float2 out_a, out_b;
//         bool output_valid;
//         ds__.process(zero, zero, false, out_a, out_b, output_valid);
//         if (output_valid) {
//             r1.push_back(out_a);
//             r2.push_back(out_b);
//         }
//     }

//     for (unsigned j = 0; j < num_points / 4; j++) {
//         std::cout << "{" << r1[j][0] << ", " << r1[j][1] << "}, ";
//     }
//     std::cout << std::endl;
//     for (unsigned j = 0; j < num_points / 4; j++) {
//         std::cout << "{" << r2[j][0] << ", " << r2[j][1] << "}, ";
//     }
//     std::cout << std::endl;

//     return {};
// }


#endif // KERNEL_HPP__