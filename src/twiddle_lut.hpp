#ifndef TWIDDLE_LUT_HPP__
#define TWIDDLE_LUT_HPP__

#include "rom_base.hpp"
#include "CL/sycl.hpp"
#include "mp_math.hpp"

static constexpr int taylor_series_terms = 40;
static_assert(taylor_series_terms > 3);

//
// A LUT for computing { cos(-2 * PI * x / num_points), sin(-2 * PI * x / num_points) }
//

template <size_t num_points>
struct TwiddleLUT : ROMBase<sycl::float2, num_points> {
    constexpr TwiddleLUT() : ROMBase<sycl::float2, num_points>(
    [](int x) {
        float real = hldutils::Cos(-2 * M_PI * x / num_points, taylor_series_terms);
        float imag = hldutils::Sin(-2 * M_PI * x / num_points, taylor_series_terms);
        return {real, imag}; 
    }) {}
};

#endif // TWIDDLE_LUT_HPP__
