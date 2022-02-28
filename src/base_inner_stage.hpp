#ifndef BASE_INNER_STAGE_HPP__
#define BASE_INNER_STAGE_HPP__

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

using namespace sycl;
using namespace hldutils;

template <size_t num_points, size_t rot_length, size_t ds_length, size_t twiddle_idx_shift_left_amt>
class BaseInnerStage {
public:
    TwiddleLUT<num_points> twiddle_lut;
    DataShuffler<ds_length, ds_length> ds_0_0;
    DataShuffler<ds_length, ds_length> ds_0_1;
    DataShuffler<ds_length / 2, ds_length / 2> ds_1_0;
    DataShuffler<ds_length / 2, ds_length / 2> ds_1_1;
    TwiddleSel<rot_length, 2, twiddle_idx_shift_left_amt> ts_0;
    TwiddleSel<rot_length, 1, twiddle_idx_shift_left_amt> ts_1;
    TwiddleSel<rot_length, 3, twiddle_idx_shift_left_amt> ts_2;
    
    BaseInnerStage() {
        static_assert(ds_length % 2 == 0, "ds_length must be divisible by two");
    }

    void process(float2 a0, float2 a1, float2 b0, float2 b1, bool input_valid, float2& out_a0, float2& out_a1, float2& out_b0, float2& out_b1, bool& output_valid) {
        //stage 1
        float2 b_1_0_out_0, b_1_0_out_1;
        float2 b_1_0_db_out_0;
        float2 b_1_0_cm_out_1;
        
        float2 b_1_0_ds_out_0, b_1_0_ds_out_1;
        
        float2 b_1_1_out_0, b_1_1_out_1;
        float2 b_1_1_cm_out_0, b_1_1_cm_out_1;
        float2 b_1_1_ds_out_0, b_1_1_ds_out_1;
        
        // NOTE: some of these might be redundanct and will need to be passed into function
        // Since they must be persistent
        // See note below at data shuffler
        // bool cm_valid, ds_0_valid, ds_1_valid, ds_0_select, ds_1_select;

        //stage 2
        float2 b_2_0_out_0, b_2_0_out_1;
        float2 b_2_0_ds_out_0, b_2_0_ds_out_1;

        float2 b_2_1_out_0, b_2_1_out_1;
        float2 b_2_1_ds_out_0, b_2_1_ds_out_1;
        
        float2 b_2_1_tm_out_1;


        /***************************
                STAGE 1
        ****************************/

        //b_1_0
        butterfly(a0, a1, b_1_0_out_0, b_1_0_out_1);

        // b_1_1
        butterfly(b0, b1, b_1_1_out_0, b_1_1_out_1);

        // get twiddle factors
        float tf_0_idx = ts_0.process(input_valid);
        float2 tf_0 = twiddle_lut[tf_0_idx];
        float tf_1_idx = ts_1.process(input_valid);
        float2 tf_1 = twiddle_lut[tf_0_idx];
        float tf_2_idx = ts_2.process(input_valid);
        float2 tf_2 = twiddle_lut[tf_2_idx];
        // std::cout << tf_0_idx << "," << tf_1_idx << "," << tf_2_idx << std::endl;
        // std::cout << tf_0[0] << "," << tf_0[1] << std::endl;
        // std::cout << tf_1[0] << "," << tf_1[1] << std::endl;
        // std::cout << tf_2[0] << "," << tf_2[1] << std::endl;

        // cm_0
        complex_mult(b_1_0_out_1, tf_0, b_1_0_cm_out_1);
        // cm_1
        complex_mult(b_1_1_out_0, tf_1, b_1_1_cm_out_0);
        // cm_2
        complex_mult(b_1_1_out_1, tf_2, b_1_1_cm_out_1);

        // Ignored db_3 and data shuffler control
        // db_3

        // The data shuffler **does not** have it's own delay buffers, so we need to create these and manually hook them up
        // WARNING: we are currently declaring the integers here to pass into data shuffer... this won't work since these values will be reset every function call
        // We should probably do the thing that Bittware's FFT did, namely, count all these variables and buffers we need, define them as [register] and then
        // pass them into the invocation of this function.
        // ds_0_0
        // size_t pulse_counter_ds_0_0 = 0;
        // bool input_valid_ds_0_0 = false;
        // data_shuffler<ds_length>(b_1_0_out_0, b_1_1_cm_out_0, input_valid_ds_0_0, b_1_0_ds_out_0, b_1_0_ds_out_1, pulse_counter_ds_0_0, ds_0_select);
        // data_shuffler(float2 a, float2 b, bool input_valid, float2& out_a, float2& out_b, size_t& pulse_counter, bool& mux_sel) {
        bool ds_0_0_output_valid;
        ds_0_0.process(b_1_0_out_0, b_1_1_cm_out_0, input_valid, b_1_0_ds_out_0, b_1_0_ds_out_1, ds_0_0_output_valid);


        // WARNING: see above
        // ds_0_1
        // size_t pulse_counter_ds_0_1 = 0;
        // bool input_valid_ds_0_1 = false;
        // data_shuffler<ds_length>(b_1_0_cm_out_1, b_1_1_cm_out_1, input_valid_ds_0_1, b_1_1_ds_out_0, b_1_1_ds_out_1, pulse_counter_ds_0_1, ds_0_select);
        bool ds_0_1_output_valid;
        ds_0_1.process(b_1_0_cm_out_1, b_1_1_cm_out_1, input_valid, b_1_1_ds_out_0, b_1_1_ds_out_1, ds_0_1_output_valid);

        // TODO: Data shuffler delays go here ^^ (note: should be built in now?)

        /***************************
                STAGE 2
        ****************************/

        // b_2_0
        butterfly(b_1_0_ds_out_0, b_1_0_ds_out_1, b_2_0_out_0, b_2_0_out_1);

        // b_2_1
        butterfly(b_1_1_ds_out_0, b_1_1_ds_out_1, b_2_1_out_0, b_2_1_out_1);

        // TODO: ds_ctrl_1

        // WARNING: see above
        // ds_1_0
        // size_t pulse_counter_ds_1_0 = 0;
        // bool input_valid_ds_1_0 = false;
        // data_shuffler<ds_length / 2>(b_2_0_out_0, b_2_1_out_0, input_valid_ds_1_0, b_2_0_ds_out_0, b_2_0_ds_out_1, pulse_counter_ds_1_0, ds_1_select);
        bool ds_1_0_output_valid;
        ds_1_0.process(b_2_0_out_0, b_2_1_out_0, ds_0_0_output_valid, b_2_0_ds_out_0, b_2_0_ds_out_1, ds_1_0_output_valid);
        
        // WARNING: see above
        // ds_1_1
        // size_t pulse_counter_ds_1_1 = 0;
        // bool input_valid_ds_1_1 = false;
        // data_shuffler<ds_length / 2>(b_2_0_out_1, b_2_1_out_1, input_valid_ds_1_1, b_2_1_ds_out_0, b_2_1_ds_out_1, pulse_counter_ds_1_1, ds_1_select);
        bool ds_1_1_output_valid;
        ds_1_1.process(b_2_0_out_1, b_2_1_out_1, ds_0_1_output_valid, b_2_1_ds_out_0, b_2_1_ds_out_1, ds_1_1_output_valid);

        // tm_1
        const float2 imag_neg = {0, -1};
        complex_mult(b_2_1_ds_out_1, imag_neg, b_2_1_tm_out_1);
        out_b1 = b_2_1_tm_out_1;

        //db_4
        out_a0 = b_2_0_ds_out_0;
        // db_5
        out_a1 = b_2_0_ds_out_1;
        // db_6
        out_b0 = b_2_1_ds_out_0;

        output_valid = ds_1_0_output_valid && ds_1_1_output_valid;
    }

};

#endif // BASE_INNER_STAGE_HPP__