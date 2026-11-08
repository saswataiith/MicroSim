# CPU and CUDA inhomogeneous elasticity development

This branch starts from MicroSim v6.0 (365aa4137e922db883bc3a91643964e2131fe0c2)
and integrates the July 2026 CUDA port with corrections from the November review.
The CPU implementation is in `../KKS_CPU_MPI` and is a standalone MPI simulator.
It has one independent composition field and full tensor eigenstrain. It is not
an already validated quinary WC binder model.

## Current validation status

CPU acceptance tests run on this Mac, including 1/2/4 ranks, analytical and energy
checks, original CUDA local-kernel comparisons, conservation and restart.
`../tests/run_branch_checks.py` checks actual extracted CUDA host/local routines,
including both chemical coupling branches, phase permutations, tie-line equality
for 2/3/5 components and AddressSanitizer. These checks are not GPU execution.
The new CUDA integration still requires an NVIDIA build, device-link, memory
checks, independent mechanical residual checks and coupled reference comparisons.

## CUDA scope

The new path requires `ENABLE_CUFFTMP=1`, `ELASTICITY=1`, `ELASTICITY_INHOM=1`,
a three-dimensional periodic domain, padding one, and exactly two elastic phases.
One phase has zero eigenstrain; the other must have finite diagonal eigenstrain
with nonzero xx and equal yy and zz. Unsupported shear or unequal yy/zz is
rejected. Stiffness is aligned cubic and must be positive definite. The variable
loading is prescribed mean stress. General eigenstrains, more elastic phases,
independent equilibrium-residual acceptance and 2D are further development.

`INHOM_STRICT=1` is required. A cap, invalid residual or backend failure aborts
before returning forces. The context records successful iteration count and
residual. The stopping residual still monitors displacement increments; physical
mechanical-equilibrium checks must be verified on the target GPU before production.

## Corrections

- The bridge exports an unchained elastic partial; both chemical kernels apply
  the interpolation derivative once. MicroSim v6 has opposite binary off-diagonal
  derivatives. The host test checks the constrained phase-force pair against a
  finite-difference energy derivative. The old review assumption of zero binary
  off-diagonal derivatives does not describe v6.
- The anisotropic xx eigenstress includes the transverse eigenstrain factor.
- The max norm uses the largest of all three component maxima.
- Iteration caps and nonfinite norms cannot be labelled converged or return forces.
- Unsupported tensors, unstable stiffness, invalid tolerance/cap/norm, invalid
  spacing, malformed applied stress and unsupported empty slabs fail explicitly.
- Odd-grid frequencies and real-field Nyquist odd derivatives use a consistent
  convention. The acoustic inverse uses dimensionless scaling instead of a
  dimensional determinant cutoff. Host inverse/scaling tests cover this helper.

The CUDA spectral storage remains full complex cuFFTMp. Warm starts are retained;
restart recomputes the initial mechanical seed. No GPU timing or scaling result
is claimed. Elastic constants, chemical energy density, distance and time must
use a consistent physical unit system.

## Build and laptop verification

Run `tools/check_cuda_host.sh` from the repository root on Ubuntu first. It only
reports tools and GPU information; it installs nothing. Check the NVHPC layout,
CUDA-aware MPI, GSL, parallel HDF5, cuFFTMp, NCCL and cuBLAS. Adjust the Makefile
paths with explicit overrides. Ordinary CUDA Toolkit does not necessarily include
cuFFTMp. Set the compilation architecture for the actual laptop GPU.

Build in this module, then run the synthetic `Input_inhom_smoke.in` and
`Filling_inhom_smoke.in` with a single MPI process. These Al-Zn demo thermodynamic
tables are a smoke test, not a calibrated WC example. Run compute-sanitizer before
physical comparisons. Compare frozen fields/forces with the CPU solver at matched
parameters before comparing evolved microstructures. Then test identical-modulus
and heterogeneous-modulus cases, applied stress, odd/noncubic grids, restarts and
mean-stress/equilibrium residuals. Multiple GPUs are required for multi-GPU checks.

The existing homogeneous Bpq path remains selected when `ELASTICITY_INHOM=0`.
Its GPU regression is still required; source routing alone does not establish
numerical equivalence.
