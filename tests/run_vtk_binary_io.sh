#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CHECK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/microsim-vtk.XXXXXX")
trap 'rm -rf -- "$CHECK_DIR"' EXIT
MODULE="$ROOT/KKS_FD_CUDA_MPI"
"${MPICXX:-mpicxx}" -std=c++17 -O2 -ffunction-sections -fdata-sections -DENABLE_HDF5=0 \
    -I"$MODULE/solverloop" -I"$MODULE/functions" \
    -c "$MODULE/solverloop/fileWriter.c" -o "$CHECK_DIR/fileWriter.o"
"${MPICXX:-mpicxx}" -std=c++17 -O2 -Wl,--gc-sections -DENABLE_HDF5=0 \
    -I"$MODULE/solverloop" -I"$MODULE/functions" \
    "$ROOT/tests/test_vtk_binary_io.cpp" "$CHECK_DIR/fileWriter.o" -o "$CHECK_DIR/test"
read -r -a LAUNCHER <<< "${MPIEXEC:-mpiexec}"
cd "$CHECK_DIR"
"${LAUNCHER[@]}" -n "${TEST_MPI_RANKS:-4}" ./test
