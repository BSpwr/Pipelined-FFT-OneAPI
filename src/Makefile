TARGET_EMU    = main.fpga_emu
TARGET_HW_A10 = main.fpga_a10
TARGET_HW_S10 = main.fpga_s10
TARGET_REPORT = fpga_report.a

SRCS     = main.cpp
OBJS     = $(SRCS:.cpp=.o)
ETS      = $(SRCS:.cpp=.d)

CXX      = dpcpp
CXXFLAGS = -std=c++17

.PHONY: build build_emu build_hw report run_emu run_hw clean run
.DEFAULT_GOAL := build_emu


# Intel-supported FPGA cards 
FPGA_DEVICE_A10 = intel_a10gx_pac:pac_a10
FPGA_DEVICE_S10 = intel_s10sx_pac:pac_s10

# Compile flags
EMULATOR_FLAGS		= -DFPGA_EMULATOR -fintelfpga
HARDWARE_FLAGS_A10	= -DFPGA -fintelfpga -Xshardware -Xsboard=$(FPGA_DEVICE_A10)
HARDWARE_FLAGS_S10	= -DFPGA -fintelfpga -Xshardware -Xsboard=$(FPGA_DEVICE_S10)
REPORT_FLAGS		= $(HARDWARE_FLAGS_S10) -fsycl-link

# Build test file (for use to check if compilation succeeds)
# on systems that have DPCPP installed but no valid accelerator
test: test.cpp
	dpcpp $(CXXFLAGS) test.cpp -o test

build_check: $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ 

# Build for FPGA emulator
build: build_emu
build_emu: $(TARGET_EMU)

$(TARGET_EMU): $(SRCS)
	$(CXX) $(CXXFLAGS) $(EMULATOR_FLAGS) -o $@ $^ 

# Generate FPGA optimization report (without compiling all the way to hardware)
report: $(TARGET_REPORT)

$(TARGET_REPORT): $(SRCS)
	$(CXX) $(CXXFLAGS) $(REPORT_FLAGS) -o $@ $^ 

# Build for FPGA hardware
build_hw_a10: $(TARGET_HW_A10)
build_hw_s10: $(TARGET_HW_S10)

$(TARGET_HW_A10): $(SRCS)
	$(CXX) $(CXXFLAGS) $(HARDWARE_FLAGS_A10) -fintelfpga -o $@ $^

$(TARGET_HW_S10): $(SRCS)
	$(CXX) $(CXXFLAGS) $(HARDWARE_FLAGS_S10) -fintelfpga -o $@ $^ 

# Run on the FPGA emulator
run: run_emu
run_emu: $(TARGET_EMU)
	./$(TARGET_EMU)

# Run on the FPGA card
run_hw: $(TARGET_HW)
	./$(TARGET_HW)

# Clean all
clean:
	-$(RM) test $(OBJS) $(TARGET_EMU) $(TARGET_HW_A10) $(TARGET_HW_S10) $(TARGET_REPORT) *.d
	-$(RM) -R *.prj
