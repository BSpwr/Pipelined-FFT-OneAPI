emulator: 
	dpcpp -fintelfpga -DFPGA_EMULATOR main.cpp kernel.cpp -o main.fpga_emu

s10: 
	dpcpp -fintelfpga -Xshardware -Xsboard=intel_s10sx_pac:pac_s10 -DFPGA main.cpp kernel.cpp -o main.fpga_s10