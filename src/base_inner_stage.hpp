#ifndef BASE_INNER_STAGE_HPP__
#define BASE_INNER_STAGE_HPP__

#include "sycl_include.hpp"

#include "shift_reg.hpp"
#include "mp_math.hpp"
#include <iostream>
#include "data_shuffler.hpp"
#include "util.hpp"
#include "twiddle_lut.hpp"
#include "twiddle_sel.hpp"

using namespace sycl;
using namespace hldutils;

template <uint16_t num_points, uint16_t rot_length, uint16_t ds_length, uint16_t twiddle_idx_shift_left_amt>
class BaseInnerStage {
public:
    constexpr static TwiddleLUT<num_points> twiddle_lut;
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
        /***************************
                STAGE 1
        ****************************/
        float2 b_1_0_out_0, b_1_0_out_1;
        float2 b_1_0_cm_out_1;
        float2 b_1_0_ds_out_0, b_1_0_ds_out_1;
        
        float2 b_1_1_out_0, b_1_1_out_1;
        float2 b_1_1_cm_out_0, b_1_1_cm_out_1;
        float2 b_1_1_ds_out_0, b_1_1_ds_out_1;

        //b_1_0
        butterfly(a0, a1, b_1_0_out_0, b_1_0_out_1);

        // b_1_1
        butterfly(b0, b1, b_1_1_out_0, b_1_1_out_1);

        // get twiddle factors
        uint16_t tf_0_idx = ts_0.process(input_valid);
        float2 tf_0 = twiddle_lut[tf_0_idx];
        uint16_t tf_1_idx = ts_1.process(input_valid);
        float2 tf_1 = twiddle_lut[tf_1_idx];
        uint16_t tf_2_idx = ts_2.process(input_valid);
        float2 tf_2 = twiddle_lut[tf_2_idx];
        //std::cout << ds_length << ": " << tf_0_idx << "," << tf_1_idx << "," << tf_2_idx << std::endl;

        // cm_0
        complex_mult(b_1_0_out_1, tf_0, b_1_0_cm_out_1);
        // cm_1
        complex_mult(b_1_1_out_0, tf_1, b_1_1_cm_out_0);
        // cm_2
        complex_mult(b_1_1_out_1, tf_2, b_1_1_cm_out_1);

        // ds_0_0
        bool ds_0_0_output_valid;
        ds_0_0.process(b_1_0_out_0, b_1_1_cm_out_0, input_valid, b_1_0_ds_out_0, b_1_0_ds_out_1, ds_0_0_output_valid);

        // ds_0_1
        bool ds_0_1_output_valid;
        ds_0_1.process(b_1_0_cm_out_1, b_1_1_cm_out_1, input_valid, b_1_1_ds_out_0, b_1_1_ds_out_1, ds_0_1_output_valid);

        /***************************
                STAGE 2
        ****************************/
        float2 b_2_0_out_0, b_2_0_out_1;
        float2 b_2_0_ds_out_0, b_2_0_ds_out_1;

        float2 b_2_1_out_0, b_2_1_out_1;
        float2 b_2_1_ds_out_0, b_2_1_ds_out_1;
        float2 b_2_1_tm_out_1;

        // b_2_0
        butterfly(b_1_0_ds_out_0, b_1_0_ds_out_1, b_2_0_out_0, b_2_0_out_1);

        // b_2_1
        butterfly(b_1_1_ds_out_0, b_1_1_ds_out_1, b_2_1_out_0, b_2_1_out_1);
       
        // ds_1_0
        bool ds_1_0_output_valid;
        ds_1_0.process(b_2_0_out_0, b_2_1_out_0, ds_0_0_output_valid, b_2_0_ds_out_0, b_2_0_ds_out_1, ds_1_0_output_valid);
        
        // ds_1_1
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