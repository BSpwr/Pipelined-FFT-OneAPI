#ifndef TWIDDLE_LUT_HPP__
#define TWIDDLE_LUT_HPP__

#include "rom_base.hpp"
#include "CL/sycl.hpp"
#include "mp_math.hpp"

static constexpr unsigned taylor_series_terms = 40;
static_assert(taylor_series_terms > 3);

//
// A LUT for computing { cos(-2 * PI * x / num_points), sin(-2 * PI * x / num_points) }
//

// sycl::float2 does not have a constexpr constructor, doing this as a workaround
struct float_complex {
        float real;
        float imag;

        constexpr float_complex(): real(0), imag(0) {}
        constexpr float_complex(float real, float imag): real(real), imag(imag) {}
    };

template <size_t num_points>
struct TwiddleLUT : ROMBase<float_complex, num_points> {

    constexpr TwiddleLUT() : ROMBase<float_complex, num_points>(
    [](int x) {
        float real = hldutils::Cos(-2 * M_PI * x / num_points, taylor_series_terms);
        float imag = hldutils::Sin(-2 * M_PI * x / num_points, taylor_series_terms);
        return float_complex{real, imag}; 
    }) {}

    // only define a const operator[], since this is a ROM
    const float2 operator[](int i) const { 
        const float_complex& tf = this->data_[i];
        return {tf.real, tf.imag};
    }
};

#endif // TWIDDLE_LUT_HPP__
