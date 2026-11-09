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
Local Linux validation now includes a CUDA 12.9 NVIDIA build/device link,
single-GPU coupled runs, and independent dense mechanical/energy-derivative
checks for both 2D models and all loading choices. Whole-program sanitizer
and genuine multi-GPU hardware validation remain pending; an extracted
reference-kernel harness passed device memcheck. See the audit evidence.

## CUDA scope

The new path requires `ENABLE_CUFFTMP=1`, `ELASTICITY=1`, `ELASTICITY_INHOM=1`,
a periodic domain, padding one, and exactly two elastic phases.
The local 2D extension supports `DIMENSION=2`, `MESH_Z=1`, and
`INHOM_2D_MODE=1` (generalized plane strain) or `2` (intrinsic 2D).
Under stress-free loading, model 1 solves uniform Ezz so that mean sigma_zz=0,
retains full 3D normal constitutive coupling, and fixes xz/yz strains to zero.
Model 2 requires
zero zz eigenstrain and has no out-of-plane energy or mean Ezz unknown.
Neither option imposes local plane stress. Loading is independent:
`INHOM_LOADING=STRESS_FREE` evaluates E from zero mean stress;
`INHOM_LOADING=CONSTRAINED` holds `HOMOGENEOUS_STRAIN={xx,yy,zz,yz,xz,xy}`
fixed (tensor shear); `INHOM_LOADING=APPLIED_STRESS` evaluates E for
`APPLIED_STRESS`. The latter is the backward-compatible default.
Intrinsic 2D excludes zz, and both 2D models exclude xz/yz. Mixed mean
loading is currently available through the CPU engine/C API, not this CUDA input.
The 3D constitutive route remains unchanged.
One phase has zero eigenstrain; the other must have finite diagonal eigenstrain
with nonzero xx; 3D requires equal yy and zz. Unsupported shear or unequal yy/zz is
rejected. Stiffness is aligned cubic and must be positive definite. The variable
loading is prescribed mean stress. General eigenstrains, more elastic phases,
independent equilibrium-residual stopping criteria remain further development.

Small single-GPU 2D F1 tests pass against an independent dense variational
reference, including applied normal/shear stress. Local kernels from the
3 November 2026 Atchyut archive were compared separately; its `Pstrain`
numbering and mean-loading conditions differ from these modes. Full legacy
trajectory equivalence, multi-GPU runs and production validation are not claimed.

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


## Reproducible 2D Function F1 verification

See `examples/2d_f1/README.md`. `tests/run_2d_f1_validation.py` independently
checks compatible-strain equilibrium, phase and composition timesteps, KKS
equalities, mean loading, energy derivatives, conservation, and temporal
refinement. It uses smooth known initial fields, unequal parabolic curvatures,
non-proportional elastic constants, unequal spacings and odd/even grids.

This branch corrects the 2D isotropic rectangular-grid stencil (zero row sum
and quadratic moments), the F1 anti-trapping positive-x composition sampling
(shared face flux), and roundoff-level gradient normalization in 2D. Explicit
`ANTI_TRAPPING=0` is respected; the historical enabled default remains 1.
ASCII fields preserve 17-digit precision for restart and verification.

## Large-volume diagnostics and binary checkpoints

`INHOM_DIAGNOSTICS = 0` keeps ordinary runs free of diagnostic copies.
`INHOM_DIAGNOSTICS = 1` retains the full per-rank CSV diagnostic.
`INHOM_DIAGNOSTICS = 2` writes one physical z plane instead:

```bash
export MICROSIM_DIAGNOSTIC_Z_INDEX=64  # zero-based index; defaults to MESH_Z/2
mpiexec -n 1 ./microsim_kks_fd_cuda_mpi input.in fill.in state
```

The slice file is named `elastic_rank<R>_step<S>_slice_z<Z>.csv`, with the same
`h,exx,eyy,ezz,eyz,exz,exy,dfdh` columns as the full diagnostic. Strains are total
tensor strains, including the evaluated homogeneous strain. Rows cover the local
physical x slab in `[local_x][global_y]` order. The slice uses strided copies of
only the requested plane; it does not allocate full-volume host strain arrays.
Invalid z indices abort explicitly. This path is opt-in and outside performance
measurements.

For large-volume output, `WRITEFORMAT = BINARY` avoids decimal conversion and
preserves double precision. Binary restart uses physical cell counts, consumes
only the ASCII header terminator before each payload, and rejects truncated
fields. Endian conversion and I/O operate on contiguous physical rows without
writing halo cells. The regression test covers 2D/3D, F1/F2 field layouts, multiple
MPI ranks, unchanged halos, and payload bytes resembling text whitespace/digits:

```bash
bash tests/run_vtk_binary_io.sh
```

The elastic iteration now swaps accepted/workspace displacement descriptors
instead of copying three complex volumes after every iteration. Box views are
rebuilt from the current descriptors at each iteration, including across warm
starts. This changes neither the convergence norm nor tolerance, interpolation,
constitutive law, frequency convention, or precision.

The 256-cubed execution benchmark uses eight particles, two different cubic
stiffness tensors, diagonal isotropic eigenstrain, F1 chemistry and zero mean
stress. It exercises the full 3D constitutive law, with no plane-strain reduction.
The grids and spacings in that benchmark are cubic. Comparisons with the
independent FFTW-MPI CPU solver are performed on a smaller matching field.
Execution and timing checks are not a spatial-convergence or long-production
validation, and one physical GPU does not validate multi-GPU scaling.
