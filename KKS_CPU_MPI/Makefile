CXX = mpicxx
PKG_CONFIG ?= pkg-config
HDF5_PKG ?= hdf5
PYTHON ?= python3
CPPFLAGS += -Iinclude $(shell $(PKG_CONFIG) --cflags fftw3 $(HDF5_PKG))
CXXFLAGS ?= -O3 -g -std=c++17 -Wall -Wextra -Wpedantic
LDLIBS += -lfftw3_mpi -lfftw3_threads $(shell $(PKG_CONFIG) --libs fftw3 $(HDF5_PKG)) -pthread
CORE = src/elasticity.cpp src/phasefield.cpp src/io.cpp
OBJECTS = $(CORE:src/%.cpp=build/%.o)
.PHONY: all clean test
all: build/microsim_cpu build/test_solver build/test_cuda_oracle
build:
	mkdir -p build
build/%.o: src/%.cpp include/cpu_mpi.hpp | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
build/libmicrosim_cpu.a: $(OBJECTS)
	ar rcs $@ $^
build/microsim_cpu: build/main.o build/libmicrosim_cpu.a
	$(CXX) $(CXXFLAGS) $^ $(LDLIBS) -o $@
build/test_solver: tests/test_solver.cpp build/libmicrosim_cpu.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(LDLIBS) -o $@
build/test_cuda_oracle: tests/test_cuda_oracle.cpp tests/original_kernel_reference.hpp build/libmicrosim_cpu.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -Wno-unused-parameter tests/test_cuda_oracle.cpp build/libmicrosim_cpu.a $(LDLIBS) -o $@
test: all
	$(PYTHON) tests/run_validation.py
clean:
	rm -rf build

# Pure-C calling interface backed by the same tested C++ elasticity engine.
build/elasticity_c.o: src/elasticity_c.cpp include/elasticity_c.h include/cpu_mpi.hpp | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
build/libelasticity_c.a: build/elasticity.o build/elasticity_c.o
	ar rcs $@ $^
build/test_c_api.o: tests/test_c_api.c include/elasticity_c.h | build
	mpicc $(CPPFLAGS) -std=c11 -O2 -Wall -Wextra -c $< -o $@
build/test_c_api: build/test_c_api.o build/libelasticity_c.a
	$(CXX) $(CXXFLAGS) $^ $(LDLIBS) -o $@

all: build/test_c_api

# Fixed cubic benchmark reference for comparison with CUDA diagnostic fields.
build/cuda_field_reference.o: tests/cuda_field_reference.c include/elasticity_c.h | build
	mpicc $(CPPFLAGS) -std=c11 -O2 -Wall -Wextra -c $< -o $@
build/cuda_field_reference: build/cuda_field_reference.o build/libelasticity_c.a
	$(CXX) $(CXXFLAGS) $^ $(LDLIBS) -o $@
