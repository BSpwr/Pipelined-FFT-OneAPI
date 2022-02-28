#ifndef INNER_STAGE_HPP__
#define INNER_STAGE_HPP__

#if defined(FPGA) || defined(FPGA_EMULATOR)
#include <CL/sycl/INTEL/fpga_extensions.hpp>
// Newer versions of DPCPP compiler have this include instead
// #include <sycl/ext/intel/fpga_extensions.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "optional"

template <size_t num_points, size_t num_stage_pairs, int twiddle_idx_shift_left_amt>
class InnerStage {
public:
    BaseInnerStage<num_points, Pow<size_t>(4, num_stage_pairs), Pow<size_t>(4, num_stage_pairs) / 2, twiddle_idx_shift_left_amt> base_inner_stage;
    InnerStage<num_points, num_stage_pairs - 1, twiddle_idx_shift_left_amt + 2> recurse_inner_stage;

    InnerStage() {}

    void process(float2 a0, float2 a1, float2 b0, float2 b1, bool input_valid, float2& out_a0, float2& out_a1, float2& out_b0, float2& out_b1, bool& output_valid) {
        if constexpr (num_stage_pairs == 1) {
            base_inner_stage.process(a0, a1, b0, b1, input_valid, out_a0, out_a1, out_b0, out_b1, output_valid);
        } else {
            float2 a0_bis_out, a1_bis_out, b0_bis_out, b1_bis_out;
            bool bis_output_valid;
            base_inner_stage.process(a0, a1, b0, b1, input_valid, a0_bis_out, a1_bis_out, b0_bis_out, b1_bis_out, bis_output_valid);
            recurse_inner_stage.process(a0_bis_out, a1_bis_out, b0_bis_out, b1_bis_out, bis_output_valid, out_a0, out_a1, out_b0, out_b1, output_valid);
        }
    }

};

template <size_t num_points, int twiddle_idx_shift_left_amt>
class InnerStage<num_points, 0, twiddle_idx_shift_left_amt> {};

#endif // INNER_STAGE_HPP__