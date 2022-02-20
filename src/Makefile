none: 
	dpcpp -std=c++17 main.cpp -o main.fpga_emu

emulator: 
	dpcpp -std=c++17 -fintelfpga -DFPGA_EMULATOR main.cpp -o main.fpga_emu

s10: 
	dpcpp -std=c++17 -fintelfpga -Xshardware -Xsboard=intel_s10sx_pac:pac_s10 -DFPGA main.cpp -o main.fpga_s10