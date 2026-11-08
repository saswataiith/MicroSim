# Inhomogeneous elasticity branch

Local development is in `/Users/saswata/MicroSim`, on
`MicroScim-CUDA-InhomElast`, based on upstream v6.0. The original local `main`
branch was preserved. The destination for this branch is the user's personal fork,
`saswataiith/MicroSim`; no upstream merge is requested.

## Implementations

- `KKS_CPU_MPI`: the complete deterministic CPU MPI phase-field and heterogeneous
  elasticity program recovered from the MicroSimCPU work, with MPI FFTW, full
  tensor eigenstrain, applied stress or strain, parallel HDF5 and restart. It has
  one independent chemical composition field. General variant count does not
  imply general multicomponent chemistry.
- `KKS_FD_CUDA_MPI`: the July two-phase CUDA inhomogeneous port, integrated into
  v6 with corrected coupling, anisotropic eigenstress, max norm, cap handling,
  input checks and spectral helpers. Unsupported tensor and phase configurations
  are rejected. GPU build and runtime validation remain pending.

The CPU module is an independent executable and benchmark source; it is not a
replacement implementation silently substituted into the GPU program. Its
elasticity model is broader than the currently integrated GPU port. The latter
still uses full-complex cuFFTMp storage. General GPU tensor support, an independent
mechanical-equilibrium residual criterion and more elastic phases remain separate
development tasks.

## Local checks

Run `python tests/run_branch_checks.py` after installing the test requirements and
GSL with a C/C++ compiler and pkg-config. The test extracts actual source routines
for host execution. It checks the elastic bridge through both chemical branches
and phase permutations, the anisotropic eigenstress, scaled acoustic inverse,
odd/even frequency convention, and the CSV reader with 2/3/5 components under
AddressSanitizer. No GPU execution is implied by these checks.

Build and test the CPU module with `make -C KKS_CPU_MPI test PYTHON=python`.
See its README for MPI-enabled HDF5 selection. On the development Mac all CPU
acceptance checks passed again on 8 November 2026, including 1/2/4-rank tests,
restart, original local CUDA kernels and independent one-step calculations.
The generated `KKS_CPU_MPI/VALIDATION.json` and logs remain local build evidence.

## Thermodynamic table fixes

The v6 FUNCTION_F=4 cross-component accumulator in the CUDA reader is reset for
each component. Two-endpoint spline allocations are corrected in the CUDA reader,
GP MPI F04/F06 and OpenCL F04. The TC-Python exporter now joins phase data by actual
temperature, writes per-phase composition CSVs, supports repeated output-directory
creation, and evaluates Hessians at the corresponding phase mole compositions.
It no longer deletes calculation caches automatically. The helper/mocked API
tests pass; a licensed Thermo-Calc integration run is still required.

Selecting a kinetic database does not mean this exporter emits mobility matrices.
That functionality, WC line-compound treatment, interface data and quinary
production validation remain requirements of the separate binder-design project.

## Next hardware step

The user plans to upgrade the CUDA laptop to Ubuntu 24.04 before connecting it.
SSH and GPU validation are deferred. Once available, run `tools/check_cuda_host.sh`
to identify the actual driver, CUDA/NVHPC, cuFFTMp, MPI, NCCL, cuBLAS, GSL and
parallel HDF5 installation. Confirm compatibility for those exact versions.
The synthetic input pair in `KKS_FD_CUDA_MPI` is only an integration smoke case.
Follow its integration notes for single-GPU checks, compute-sanitizer, matched CPU
benchmarks and later multi-GPU validation. No GPU validation is claimed yet.
