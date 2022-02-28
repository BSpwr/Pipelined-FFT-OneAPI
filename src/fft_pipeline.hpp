#ifndef FFT_PIPELINE_HPP__
#define FFT_PIPELINE_HPP__

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
#include "util.hpp"
#include "twiddle_lut.hpp"
#include "twiddle_sel.hpp"
#include "inner_stage.hpp"

using namespace sycl;
using namespace hldutils;

template <size_t num_points>
class FFTPipeline {
public:
    InnerStage<num_points, Log2<size_t>(num_points) / Log2<size_t>(4) - 1, 0> inner_stage;

    FFTPipeline() {
        static_assert(IsPowerOfFour(num_points), "num_points must be a power of 4");
    }

    void process(float2 a0, float2 a1, float2 b0, float2 b1, bool input_valid, float2& out_a0, float2& out_a1, float2& out_b0, float2& out_b1, bool& output_valid) {
        float2 a0_fs_out, a1_fs_out, b0_fs_out, b1_fs_out;
        first_stage(a0, a1, b0, b1, a0_fs_out, a1_fs_out, b0_fs_out, b1_fs_out);

        float2 a0_is_out, a1_is_out, b0_is_out, b1_is_out;
        // NOTE: a1_fs_out and b0_fs_out are flipped between first stage and inner stage
        inner_stage.process(a0_fs_out, b0_fs_out, a1_fs_out, b1_fs_out, input_valid, a0_is_out, a1_is_out, b0_is_out, b1_is_out, output_valid);
        
        last_stage(a0_is_out, a1_is_out, b0_is_out, b1_is_out, out_a0, out_a1, out_b0, out_b1);
    }
};

#endif // FFT_PIPELINE_HPP__