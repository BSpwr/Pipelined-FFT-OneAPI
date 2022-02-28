#ifndef DATA_SHUFFLER_HPP__
#define DATA_SHUFFLER_HPP__

#include "CL/sycl.hpp"
#include "shift_reg.hpp"

using namespace hldutils;
using sycl::float2;

template<size_t delay_length, size_t pulse_length>
class DataShuffler {
private:
    [[intel::fpga_register]] ShiftReg<float2, delay_length> lower_delay;
    [[intel::fpga_register]] ShiftReg<float2, delay_length> upper_delay;
    [[intel::fpga_register]] ShiftReg<bool, delay_length> input_valid_delay;
    [[intel::fpga_register]] bool mux_sel = false;
    [[intel::fpga_register]] size_t pulse_counter = 0;
public:
    DataShuffler() {
        bool default_input_valid = false;
        input_valid_delay.Init(default_input_valid);
    }

    void process(float2 a, float2 b, bool input_valid, float2& out_a, float2& out_b, bool& output_valid) {
        // counter to flip mux signal every pulse_length inputs
        if (input_valid) {
            if (pulse_counter == pulse_length) {
                mux_sel = !mux_sel;
                pulse_counter = 0;
            }
            pulse_counter += 1;
        }

        // lower mux -> output
        out_b = (mux_sel) ? a : lower_delay[0];

        // upper mux -> shift_reg -> output
        float2 upper_mux_out = (mux_sel) ? lower_delay[0] : a;
        out_a = upper_delay[0];

        upper_delay.Shift(upper_mux_out);

        lower_delay.Shift(b);

        output_valid = input_valid_delay[0];
        input_valid_delay.Shift(input_valid);
    }
};

#endif // DATA_SHUFFLER_HPP__