#ifndef TWIDDLE_SEL_HPP__
#define TWIDDLE_SEL_HPP__

#include "sycl_include.hpp"

#include "shift_reg.hpp"
#include "mp_math.hpp"
#include <iostream>
#include "data_shuffler.hpp"
#include "kernel.hpp"
#include "twiddle_lut.hpp"

using namespace sycl;
using namespace hldutils;

template <size_t len_sequence, size_t increment_amt, size_t idx_shift_left_amt>
class TwiddleSel {
public:
    [[intel::fpga_register]] size_t twiddle_idx;

    TwiddleSel() {
        twiddle_idx = 0;
    }

    size_t process(bool count_en) {
        size_t prev_twiddle_idx = twiddle_idx;

        if (count_en) {
            if (twiddle_idx == (len_sequence - 1) * increment_amt) {
                twiddle_idx = 0;
            } else {
                twiddle_idx += increment_amt;
            }
        }

        return prev_twiddle_idx << idx_shift_left_amt;
    }
};

#endif // TWIDDLE_SEL_HPP__