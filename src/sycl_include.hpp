#ifndef SYCL_INCLUDE_HPP__
#define SYCL_INCLUDE_HPP__

#if defined(FPGA) || defined(FPGA_EMULATOR)
    #if __SYCL_COMPILER_VERSION >= 20211123
        #include <sycl/ext/intel/fpga_extensions.hpp>  //For version 2022.0.0 (build date 2021/11/23)
    #elif __SYCL_COMPILER_VERSION <= BETA09
        #include <CL/sycl/intel/fpga_extensions.hpp>
        namespace INTEL = sycl::intel;  // Namespace alias for backward compatibility
    #else
        #include <CL/sycl/INTEL/fpga_extensions.hpp>
    #endif
#else
#include <CL/sycl.hpp>
#endif

#endif // SYCL_INCLUDE_HPP__