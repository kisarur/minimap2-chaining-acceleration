# Minimap2's Chaining Step Accelerator

This is a heterogeneous computing system that is designed to accelerate compute-intensive chaining step in the third-generation DNA sequence analysis tool - [Minimap2](https://github.com/lh3/minimap2). The system combines an OpenCL HLS-based FPGA (Intel Arria 10 GX) hardware accelerator and a multi-threaded software framework to accelerate the hotspot of the tool. The system can efficiently split a given set of chaining tasks between hardware accelerator (FPGA-based) and software (CPU) platforms based on each chaining task’s computational complexity, and process the split tasks on both the platforms in parallel. Further details about this work is available in the following publication. 

> K. Liyanage, H. Gamaarachchi, R. Ragel and S. Parameswaran, "[Cross Layer Design Using HW/SW Co-Design and HLS to Accelerate Chaining in Genomic Analysis](https://ieeexplore.ieee.org/document/10015864)," in IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, doi: 10.1109/TCAD.2023.3236559.

# Getting Started 

1. Install "Intel® PAC with Intel® Arria® 10 GX FPGA Acceleration Stack Version 1.2.1" on the system 

2. Use the commands below to download the GitHub repo, setup the environment (you may need to update the variables defined in `opencl/init_env.sh`, if they're not already pointing to the correct paths in your system), and build the host application
```
git clone --recurse-submodules https://github.com/kisarur/minimap2-chaining-acceleration.git
cd minimap2-chaining-acceleration
source opencl/init_env.sh
make
```

3. To build the hardware binary (.aocx) for the hardware accelerator from OpenCL source (.cl), use the given `generate_fpga_binary.sh` script. Please note that this build process can take hours to complete. Skip this step if you want to use the already built FPGA binary (for "Intel Arria 10 GX" device) located at `bin/minimap2_chain.aocx`.

2. Download the testbed from https://github.com/UCLA-VAST/minimap2-acceleration and generate input data for chaining function (it also generates the expected output used for accuracy comparisons) according to the instructions provided under "Build testbed and generate test data" section.

3. Use the commands below to convert the input data to a version compatible with this work and then run the chaining computation on FPGA+CPU system.
```
cd ext/input-converter
g++ -O2 -o converter main.cpp
./converter <original_input_data_file> > <converted_input_data_file>
cd ../../
bin/host <converted_input_data_file>
```

# Acknowledgement

1. The software chaining algorithm was obtained from [Minimap2](https://github.com/lh3/minimap2).
2. Testbed to generate input/output data to/from minimap2's chaining function was obtained from [Minimap2-acceleration](https://github.com/UCLA-VAST/minimap2-acceleration).
3. SIMD-accelerated minimap2's chaining function was obtained from [mm2-fast](https://github.com/bwa-mem2/mm2-fast).
