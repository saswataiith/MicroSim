#!/usr/bin/env bash
set -euo pipefail
MODULE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$MODULE/nvhpc_env.sh"
if [[ -z ${CUDA_ARCH:-} ]]; then
    CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '. ')
fi
[[ $CUDA_ARCH =~ ^[0-9]+$ ]] || { echo 'Set CUDA_ARCH to the target SM number, e.g. 86.' >&2; exit 1; }
exec make -C "$MODULE" -j"${BUILD_JOBS:-4}" build \
    HPCSDK_VERSION="$HPCSDK_VERSION" NVHPC_CUDA_HOME="$NVHPC_CUDA_HOME" MPI_HOME="$MPI_HOME" \
    CUFFT_LIB="$MATH_HOME/lib64" CUFFT_INC="$MATH_HOME/include/cufftmp" NCCL_HOME="$NCCL_HOME" \
    ENABLE_HDF5=0 ENABLE_CUFFTMP=1 "GSLFLAGS=$(pkg-config --cflags --libs gsl)" \
    "CUDACFLAGS=-I$NVHPC_CUDA_HOME/include -I$MATH_HOME/include" "MPICFLAGS=-I$MPI_HOME/include" \
    "GENCODE_FLAGS=-gencode arch=compute_$CUDA_ARCH,code=sm_$CUDA_ARCH" "$@"
