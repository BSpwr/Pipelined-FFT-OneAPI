#ifndef KERNEL_TCC__
#define KERNEL_TCC__

#include <math.h>
#include <iostream>
#include <limits>
//#include <common.h>
#if defined(FPGA) || defined(FPGA_EMULATOR)
#include <CL/sycl/INTEL/fpga_extensions.hpp>
// Newer versions of DPCPP compiler have this include instead
// #include <sycl/ext/intel/fpga_extensions.hpp>
#else
#include <CL/sycl.hpp>
#endif

using namespace sycl;

template <typename T>
constexpr T const_pow(T num, unsigned int pow)
{
    return (pow >= sizeof(unsigned int)*8) ? 0 :
        pow == 0 ? 1 : num * const_pow(num, pow-1);
}

template <size_t num_points>
void input_reorder(sycl::float2* X, sycl::float2* d0, sycl::float2* d1, sycl::float2* d2, sycl::float2* d3) {
    unsigned int d0_count = 0;
    unsigned int d1_count = 0;
    unsigned int d2_count = 0;
    unsigned int d3_count = 0;

    const int bits = log2(num_points) - 1;
    #pragma unroll
    for (unsigned int i = 0; i < num_points; i++) {
        if (!(i & (1 << bits)) && !(i & (1 << (bits - 1)))) {
            d0[d0_count++] = X[i];
        } else if ((i & (1 << bits)) && !(i & (1 << (bits - 1)))) {
            d1[d1_count++] = X[i];
        } else if (!(i & (1 << bits)) && (i & (1 << (bits - 1)))) {
            d2[d2_count++] = X[i];
        } else if ((i & (1 << bits)) && (i & (1 << (bits - 1)))) {
            d3[d3_count++] = X[i];
        }
    }
}

// /* Data shuffler must delay inputs long enough so that pairs can align */
// template<size_t delay_length, size_t pulse_length>
// void data_shuffler(float2 a, float2 b, float2& out_a, float2& out_b, float2 delay_a[delay_length], float2 delay_b[delay_length], size_t& pulse_counter, bool& mux_sel, size_t delay) {
//     // counter to flip mux signal every pulse_length inputs
//     pulse_counter += 1;
//     if (pulse_counter == pulse_length) {
//         mux_sel = !mux_sel;
//         pulse_counter = 0;
//     }

//     // delay = (delay > delay_length) ? delay_length : delay;

//     #pragma unroll
//     for (int i = 0; i < delay_length - 1; i++)
//     {
//         delay_b[i] = delay_b[i + 1];
//     }
//     delay_b[delay_length - 1] = b;

//     // lower mux (lower mux has inverted mux select)
//     if (mux_sel) { // 0
//         out_b = a;
//     } else { // 1
//         out_b = delay_b[0];
//     }

//     // upper delay buffer
//     #pragma unroll
//     for (int i = 0; i < delay_length - 1; i++) {
//         delay_a[i] = delay_b[i + 1];
//     }

//     // upper mux
//     float2 mux1_out;
//     if (!mux_sel) { // 0
//         delay_a[delay_length - 1] = a;
//     } else { // 1
//         delay_a[delay_length - 1] = delay_b[0];
//     }

//     out_a = delay_a[0];
// }

// // WORKING BUT DELAY IS BROKEN
// /* Data shuffler must delay inputs long enough so that pairs can align */
// template<size_t delay_length, size_t pulse_length>
// void data_shuffler(float2 a, float2 b, float2& out_a, float2& out_b, float2 delay_a[delay_length], float2 delay_b[delay_length], size_t& pulse_counter, bool& mux_sel, size_t index) {
//     // counter to flip mux signal every pulse_length inputs
//     pulse_counter += 1;
//     if (pulse_counter == pulse_length) {
//         mux_sel = !mux_sel;
//         pulse_counter = 0;
//     }

//     // lower delay buffer
//     #pragma unroll
//     for (int i = delay_length - 1; i >= 1; i--)
//     {
//         delay_b[i] = delay_b[i - 1];
//     }
//     delay_b[0] = b;

//     // delay_length - 1 - (delay_length - index)

//     // lower mux (lower mux has inverted mux select)
//     if (mux_sel) { // 0
//         out_b = a;
//     } else { // 1
//         out_b = delay_b[delay_length - 1];
//     }

//     // upper delay buffer
//     #pragma unroll
//     for (int i = delay_length - 1; i >= 1; i--) {
//         delay_a[i] = delay_a[i - 1];
//     }

//     // upper mux
//     float2 mux1_out;
//     if (!mux_sel) { // 0
//         delay_a[0] = a;
//     } else { // 1
//         delay_a[0] = delay_b[delay_length - 1];
//     }

//     out_a = delay_a[delay_length - 1];
// }

template<size_t pulse_length>
void data_shuffler(float2 a, float2 b, bool input_valid, float2& out_a, float2& out_b, size_t& pulse_counter, bool& mux_sel) {
    // counter to flip mux signal every pulse_length inputs
    if (input_valid) {
        if (pulse_counter == pulse_length) {
            mux_sel = !mux_sel;
            pulse_counter = 0;
        }
        pulse_counter += 1;
    }

    // delay_length - 1 - (delay_length - index)

    // lower mux (lower mux has inverted mux select)
    if (mux_sel) { // 0
        out_b = a;
    } else { // 1
        out_b = b;
    }

    // upper mux
    float2 mux1_out;
    if (!mux_sel) { // 0
        mux1_out = a;
    } else { // 1
        mux1_out = b;
    }

    out_a = mux1_out;
}

void butterfly(float2 input0, float2 input1, float2& output0, float2& output1) {
    output0[0] = input0[0] + input1[0];
    output0[1] = input0[1] + input1[1];

    output1[0] = input0[0] + (-1 * input1[0]);
    output1[1] = input0[1] + (-1 * input1[1]);
}

void complex_mult(float2 input0, float2 input1, float2& output) {

    float a1 = input0[0] - input0[1];
    float a2 = input1[0] - input1[1];
    float a3 = input1[0] + input1[1];

    float p1 = a1 * input1[1];
    float p2 = a2 * input0[0];
    float p3 = a3 * input0[1];

    output[0] = p1 + p2;
    output[1] = p1 + p3;

    // (*output)[0] = (input0[0] * input1[0]) - (input0[1] * input1[1]);
    // (*output)[1] = (input0[0] * input1[1]) + (input1[0] * input0[1]);
}

void first_stage(float2 a0, float2 a1, float2 b0, float2 b1, float2& out_a0, float2& out_a1, float2& out_b0, float2& out_b1) {
    float2 b_1_out_1;           

    butterfly(a0, a1, out_a0, out_a1);
    butterfly(b0, b1, out_b0, b_1_out_1);

    const float2 imag_neg = {0, -1};

    complex_mult(b_1_out_1, imag_neg, out_b1);
}

void last_stage(float2 a0, float2 a1, float2 b0, float2 b1, float2& out_a0, float2& out_a1, float2& out_b0, float2& out_b1) {
    butterfly(a0, a1, out_a0, out_a1);
    butterfly(b0, b1, out_b0, out_b1);
}

template <size_t num_points>
std::vector<sycl::float2> PipelinedFFT(std::vector<sycl::float2>& input_vec, cl::sycl::queue &q) {

   // static_assert(int(log2(num_points)) % 2 == 0, "num_points must be 2^N, where N is even");  // not allowed.

    // pad out input
    for (unsigned i = 0; i < num_points - input_vec.size(); i++) {
        input_vec.push_back({0, 0});
    }

    std::cout << "Local Memory Size: "
        << q.get_device().get_info<sycl::info::device::local_mem_size>()
        << std::endl;

    constexpr unsigned int BYTES_SIZE = num_points * sizeof(sycl::float2);

    std::vector<float2> output_ret(num_points, {0, 0});

    float2* input_data = sycl::malloc_device<float2>(num_points, q);
    auto copy_host_to_device_event = q.memcpy(input_data, input_vec.data(), BYTES_SIZE);

    // for (int i = 0; i < num_points; i++) {
    //     std::cout << input_data[i][0] << ", " << input_data[i][1] << ", ";
    // }
    // std::cout << std::endl;

    // for (int i = 0; i < num_points; i++) {
    //     std::cout << input_data2[i][0] << ", " << input_data2[i][1] << ", ";
    // }
    // std::cout << std::endl;

    float2* output_data0 = sycl::malloc_device<float2>(num_points / 4, q);
    float2* output_data1 = sycl::malloc_device<float2>(num_points / 4, q);
    float2* output_data2 = sycl::malloc_device<float2>(num_points / 4, q);
    float2* output_data3 = sycl::malloc_device<float2>(num_points / 4, q);

    std::cout << "HI" << std::endl;

    // Offload the work to kernel.
    auto kernel_event = q.submit([&](handler &h) {
        
        h.depends_on(copy_host_to_device_event);
    
    //auto a = input.get_access<access::mode::read_write>(h);
    //h.parallel_for(nd_range<1>(range<1>(size), range<1>(BLOCK_SIZE)), [=](nd_item<1> item) {
        // int i = item.get_global_id(0);

        // auto data = input.get_access<access::mode::read_write>(h);

        // auto input = input_buf.get_access<access::mode::read_write, sycl::access::target::local>(h);
        // auto output = output_buf.get_access<access::mode::read_write, sycl::access::target::local>(h);

        // accessor<float2, 1, access::mode::read, access::target::global_buffer> input(input_buf, h);
        // accessor<float2, 1, access::mode::write, access::target::global_buffer> output(output_buf, h);


        h.single_task([=] () [[intel::kernel_args_restrict]] {
            device_ptr<float2> in_ptr(input_data);
            device_ptr<float2> out0_ptr(output_data0);
            device_ptr<float2> out1_ptr(output_data1);
            device_ptr<float2> out2_ptr(output_data2);
            device_ptr<float2> out3_ptr(output_data3);
            input_reorder<num_points>(in_ptr, out0_ptr, out1_ptr, out2_ptr, out3_ptr);
            float2 delay_a[4] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
            float2 delay_b[4] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
            size_t pulse_counter = 0;
            bool mux_sel = false;
            #pragma unroll
            for (int i = 0; i < num_points / 4; i++) {
                data_shuffler<4>(out0_ptr[i], out1_ptr[i], true, out2_ptr[i], out3_ptr[i], pulse_counter, mux_sel);
                // complex_mult(out0_ptr[i], out1_ptr[i], out2_ptr[i]);

            }

        });
    });

      // copy output data back from device to host
  auto copy_device0_to_host_event = q.submit([&](handler& h) {
    // we cannot copy the output data from the device's to the host's memory
    // until the computation kernel has finished
    h.depends_on(kernel_event);
    h.memcpy(output_ret.data(), output_data0, BYTES_SIZE / 4);
  });

    auto copy_device1_to_host_event = q.submit([&](handler& h) {
    // we cannot copy the output data from the device's to the host's memory
    // until the computation kernel has finished
    h.depends_on(kernel_event);
    h.memcpy(output_ret.data() + 1 * (num_points / 4), output_data1, BYTES_SIZE / 4);
  });

    auto copy_device2_to_host_event = q.submit([&](handler& h) {
    // we cannot copy the output data from the device's to the host's memory
    // until the computation kernel has finished
    h.depends_on(kernel_event);
    h.memcpy(output_ret.data() + 2 *(num_points / 4), output_data2, BYTES_SIZE / 4);
  });

    auto copy_device3_to_host_event = q.submit([&](handler& h) {
    // we cannot copy the output data from the device's to the host's memory
    // until the computation kernel has finished
    h.depends_on(kernel_event);
    h.depends_on(copy_device0_to_host_event);
    h.depends_on(copy_device1_to_host_event);
    h.depends_on(copy_device2_to_host_event);
    h.memcpy(output_ret.data() + 3 *(num_points / 4), output_data3, BYTES_SIZE / 4);
  });

  // wait for copy back to finish
  copy_device3_to_host_event.wait();

    // // sync
    // input_buf.get_access<access::mode::read_write>();
    // output_buf.get_access<access::mode::read_write>();

    // for (int i = 0; i < input_size; i++) {
    //     std::cout << output_data[i][0] << ", " << output_data[i][1] << ", ";
    // }
    // std::cout << std::endl;

    // q.memcpy(output_ret.data(), output_data, BYTES_SIZE);
    // q.wait();

    // sycl::free(input_data, q);
    // sycl::free(input_data2, q);
    // sycl::free(output_data, q);

    // output_ret0.insert(output_ret0.end(), output_ret1.begin(), output_ret1.end());
    // AB.insert(AB.end(), B.begin(), B.end());

    return output_ret;
}


template <size_t rot_length, size_t ds_length, size_t twiddle_idx_shift_left_amt>
void base_inner_stage(float2 a0, float2 a1, float2 b0, float2 b1, bool input_valid, float2& out_a0, float2& out_a1, float2& out_b0, float2& out_b1, bool& output_valid) {

    //stage 1
    float2 b_1_0_out_0, b_1_0_out_1;
    float2 b_1_0_db_out_0;
    float2 b_1_0_cm_out_1;
    
    float2 b_1_0_ds_out_0, b_1_0_ds_out_1;
    float2 ts0;
    
    float2 b_1_1_out_0, b_1_1_out_1;
    float2 b_1_1_cm_out_0, b_1_1_cm_out_1;
    float2 b_1_1_ds_out_0, b_1_1_ds_out_1;
    float2 ts1, ts2;
    
    // NOTE: some of these might be redundanct and will need to be passed into function
    // Since they must be persistent
    // See note below at data shuffler
    bool cm_valid, ds_0_valid, ds_1_valid, ds_0_select, ds_1_select;

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

    // TODO: figure out twiddle factors and how to index into it
    // For now let's leave twiddle factors out
    // ts{0-2} are currently zero... TODO...

    // cm_0
    complex_mult(b_1_0_out_1, ts0, b_1_0_cm_out_1);
    
    // cm_1
    complex_mult(b_1_1_out_0, ts1, b_1_1_cm_out_0);

    // cm_2
    complex_mult(b_1_1_out_1, ts2, b_1_1_cm_out_1);

    // Ignored db_3 and data shuffler control
    // db_3

    // The data shuffler **does not** have it's own delay buffers, so we need to create these and manually hook them up
    // WARNING: we are currently declaring the integers here to pass into data shuffer... this won't work since these values will be reset every function call
    // We should probably do the thing that Bittware's FFT did, namely, count all these variables and buffers we need, define them as [register] and then
    // pass them into the invocation of this function.
    // ds_0_0
    size_t pulse_counter_ds_0_0 = 0;
    bool input_valid_ds_0_0 = false;
    data_shuffler<ds_length>(b_1_0_out_0, b_1_1_cm_out_0, input_valid_ds_0_0, b_1_0_ds_out_0, b_1_0_ds_out_1, pulse_counter_ds_0_0, ds_0_select);
    // data_shuffler(float2 a, float2 b, bool input_valid, float2& out_a, float2& out_b, size_t& pulse_counter, bool& mux_sel) {


    // WARNING: see above
    // ds_0_1
    size_t pulse_counter_ds_0_1 = 0;
    bool input_valid_ds_0_1 = false;
    data_shuffler<ds_length>(b_1_0_cm_out_1, b_1_1_cm_out_1, input_valid_ds_0_1, b_1_1_ds_out_0, b_1_1_ds_out_1, pulse_counter_ds_0_1, ds_0_select);
    

    // TODO: Data shuffler delays go here

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
    size_t pulse_counter_ds_1_0 = 0;
    bool input_valid_ds_1_0 = false;
    data_shuffler<ds_length / 2>(b_2_0_out_0, b_2_1_out_0, input_valid_ds_1_0, b_2_0_ds_out_0, b_2_0_ds_out_1, pulse_counter_ds_1_0, ds_1_select);
    
    // WARNING: see above
    // ds_1_1
    size_t pulse_counter_ds_1_1 = 0;
    bool input_valid_ds_1_1 = false;
    data_shuffler<ds_length / 2>(b_2_0_out_1, b_2_1_out_1, input_valid_ds_1_1, b_2_1_ds_out_0, b_2_1_ds_out_1, pulse_counter_ds_1_1, ds_1_select);

    // tm_1
    const float2 imag_neg = {0, -1};
    complex_mult(b_2_1_ds_out_1, imag_neg, b_2_1_tm_out_1);
    b1 = b_2_1_tm_out_1;

    //db_4
    a0 = b_2_0_ds_out_0;
    // db_5
    a1 = b_2_0_ds_out_1;
    // db_6
    b0 = b_2_1_ds_out_0;

    

}

template <size_t num_stage_pairs, int twiddle_idx_shift_left_amt>
void inner_stage(float2 a0, float2 a1, float2 b0, float2 b1, bool input_valid, float2& out_a0, float2& out_a1, float2& out_b0, float2& out_b1, bool& output_valid) {

    float2 front_stage_0_output, front_stage_1_output, front_stage_2_output, front_stage_3_output;
    bool front_stage_output_valid;
    
    //Base case
    if constexpr (num_stage_pairs == 1){
        //Base inner stage
        constexpr size_t rot_length = const_pow<size_t>(4, num_stage_pairs);
        constexpr size_t ds_length = const_pow<size_t>(4, num_stage_pairs) / 2;
        base_inner_stage<rot_length, ds_length, twiddle_idx_shift_left_amt>(
            a0, a1, b0, b1, 
            input_valid,
            front_stage_0_output, front_stage_1_output, front_stage_2_output, front_stage_3_output,
            front_stage_output_valid);
    }
    //Recursive case
    else {
        //New inner stage
        constexpr size_t rot_length = const_pow<size_t>(4, num_stage_pairs);
        constexpr size_t ds_length = const_pow<size_t>(4, num_stage_pairs) / 2;
        base_inner_stage<rot_length, ds_length, twiddle_idx_shift_left_amt>(
            a0, a1, b0, b1, 
            input_valid,
            front_stage_0_output, front_stage_1_output, front_stage_2_output, front_stage_3_output,
            front_stage_output_valid
            );
        //Recursive inner stage
        inner_stage<num_stage_pairs - 1, twiddle_idx_shift_left_amt + 2>(
            front_stage_0_output, front_stage_1_output, front_stage_2_output, front_stage_3_output, 
            front_stage_output_valid, 
            out_a0, out_a1, out_b0, out_b1, 
            output_valid);
    }
}


void fft_pipeline(float2 a0, float2 a1, float2 b0, float2 b1, bool input_valid, float2& out_a0, float2& out_a1, float2& out_b0, float2& out_b1, bool& output_valid) {
    // We should consider the delay through the pipeline and keep track of where the inputs to the pipeline are valid
    // Since we need to know if the pipeline contains valid data for the data shuffer and twiddle generator

    // First stage doesn't have a delay associated with it, so let's ignore the input_valid variable for this one
    float2 fs0_out, fs1_out, fs2_out, fs3_out;
    first_stage(a0, a1, b0, b1, fs0_out, fs1_out, fs2_out, fs3_out);

    // How should we allocate the space for this pipeline?
    // One possible approach -- allocate all the space as one long register, and just pass pointers into this space
    float2 is0_out, is1_out, is2_out, is3_out;
    bool is_out_valid;
    // Note: this is a pipeline, so there is hidden state in there, as such, we might have the input be valid (valid element coming in), 
    // but the output will still be invalid, since samples will take multiple cycles (calls of this function) to output
    // NOTE: for now: num_stages = 16 (16-point FFT)
    inner_stage<1, 0>(fs0_out, fs1_out, fs2_out, fs3_out, input_valid, is0_out, is1_out, is2_out, is3_out, output_valid);
    // Last stage doesn't have a delay associated with it, so let's ignore the input_valid variable for this one
    last_stage(is0_out, is1_out, is2_out, is3_out, out_a0, out_a1, out_b0, out_b1);
}


#endif // KERNEL_TCC__