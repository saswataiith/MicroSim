#!/usr/bin/env bash
set -euo pipefail
REPOSITORY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPOSITORY/KKS_FD_CUDA_MPI/nvhpc_env.sh"
# Compatible local single-GPU MPI transport for the tested HPC-X distribution.
export OMPI_MCA_pml=${OMPI_MCA_pml:-ob1} OMPI_MCA_btl=${OMPI_MCA_btl:-self,sm,tcp}
export OMPI_MCA_coll=${OMPI_MCA_coll:-^ucc} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
exec "${PYTHON:-python3}" "$REPOSITORY/tests/run_2d_f1_validation.py" --launcher "$MPI_HOME/bin/mpiexec -n 1" "$@"
