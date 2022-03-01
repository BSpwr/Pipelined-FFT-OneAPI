#include <math.h>
#include <iostream>
#include <limits>
//#include <common.h>

#include "kernel.hpp"

using namespace sycl;

int main(int argc, char *argv[]) {

    // create device selector for the device of your interest
#ifdef FPGA_EMULATOR
    // DPC++ extension: FPGA emulator selector on systems without FPGA card
    sycl::ext::intel::fpga_emulator_selector dev_ct1;
#elif FPGA
    // DPC++ extension: FPGA selector on systems with FPGA card
    sycl::ext::intel::fpga_selector dev_ct1;
#elif GPU
    sycl::gpu_selector dev_ct1;
#else
    sycl::default_selector dev_ct1;
#endif

    cl::sycl::queue q(dev_ct1);

    constexpr size_t num_points = 1024;

    std::vector<float2> input(num_points, {0, 0});
    for (unsigned i = 0; i < num_points; i++) {
        input[i] = {i, 0};
    }

    // start timer
    auto start = std::chrono::system_clock::now();

    std::vector<float2> output = PipelinedFFT<num_points>(input, q);

    // std::vector<float2> output(left.size(), {0, 0});

    // std::vector<float2> output;

    // for (int i = 0; i < left.size(); i++) {
    //     complex_mult(left[i], left[i], &output[i]);
    // }

    // int i = 0;
    // for (auto a : output) {
    //     std::cout << a[0] << ", ";
    //     i++;
    //     if (i % (input.size() / 4) == 0) {
    //         std::cout << std::endl;
    //     }
    // }
    // std::cout << std::endl;
    // i = 0;
    // for (auto a : output) {
    //     std::cout << a[1] << ", ";
    //     i++;
    //     if (i % (input.size() / 4) == 0) {
    //         std::cout << std::endl;
    //     }
    // }
    // std::cout << std::endl;

    auto end = std::chrono::system_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double ms = ((double)ns.count()) / 1000000;
    printf("Milliseconds elapsed: %f\n", ms);

    std::cout << std::endl;

    // Print outputs
    for (unsigned j = 0; j < num_points; j++) {
        std::cout << "{" << output[j][0] << ", " << output[j][1] << "}, ";
    }
    std::cout << std::endl;

  }