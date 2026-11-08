#!/usr/bin/env bash
# Read-only inventory. No installation, driver changes, or privileged commands.
set -u
for tool in nvidia-smi nvcc mpicxx mpiexec h5pcc pkg-config compute-sanitizer; do
    if command -v "$tool" >/dev/null 2>&1; then
        command -v "$tool"
    else
        printf '%s is unavailable on PATH\n' "$tool"
    fi
done
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi
if command -v nvcc >/dev/null 2>&1; then nvcc --version; fi
if command -v mpicxx >/dev/null 2>&1; then mpicxx --version | head -n 2; fi
if command -v h5pcc >/dev/null 2>&1; then h5pcc -show; fi
if command -v pkg-config >/dev/null 2>&1; then
    for lib in gsl fftw3 hdf5-openmpi hdf5; do
        pkg-config --modversion "$lib" 2>/dev/null || true
    done
fi
printf '%s\n' 'Also identify the NVHPC/cuFFTMp, NCCL and cuBLAS library directories before building.'
