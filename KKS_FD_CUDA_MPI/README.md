# C/CUDA KKS with inhomogeneous elasticity

This branch runs the coupled phase, composition and mechanical solves in compiled
C/CUDA. Python prepares synthetic inputs, verifies solutions and plots results;
it is not the production phase-field solver.

- **2D generalized plane strain:** retain the 3D normal constitutive law and
  uniform total Ezz. Stress-free loading evaluates Ezz from zero mean sigma_zz;
  neither local elastic epsilon_zz nor local sigma_zz is generally zero.
- **Intrinsic 2D:** xx, yy and xy only, with no out-of-plane energy contribution.
- **3D:** the full cubic constitutive law; no plane-strain reduction.
- **Loading is independent of dimension:** stress-free evaluates homogeneous E,
  constrained loading specifies E, and applied mean stress evaluates E for that
  stress. A specified homogeneous strain is a constrained homogenization solve.

Current CUDA scope is two elastic phases, one nonzero diagonal eigenstrain and
aligned cubic stiffnesses. The 3D path currently requires yy=zz eigenstrain.
The MPI/cuFFTMp implementation supports distributed execution, but the new GPU
validation was on **one physical RTX A3000**. CPU MPI checks are not multi-GPU
validation.

## Build

Install GSL and a compatible NVIDIA HPC SDK with CUDA, cuFFTMp, NCCL and MPI.
The tested environment was NVHPC 26.9, CUDA 12.9, SM86, double precision.

```bash
bash KKS_FD_CUDA_MPI/build_nvhpc.sh
source KKS_FD_CUDA_MPI/nvhpc_env.sh
```

The wrapper detects the local GPU architecture. Set `CUDA_ARCH`,
`HPCSDK_VERSION`, `NVHPC_CUDA_VERSION` or the documented paths in
[nvhpc_env.sh](nvhpc_env.sh) for another installation. The wrapper builds without
HDF5; output examples below use binary VTK. The older [README](README) retains
legacy installation and cluster instructions.

## Reproduce the studies

From the repository root, with a NumPy-capable Python environment:

```bash
python tests/prepare_elasticity_case.py --case nucleus2d --output run70
# After sourcing the module environment, use an absolute executable path:
CUDA_EXE="$PWD/KKS_FD_CUDA_MPI/microsim_kks_fd_cuda_mpi"
(cd run70 && mpiexec -n 1 "$CUDA_EXE" input.in fill.in state)

# Continue from 200 ns to 2 us, preserving physical time when changing dt:
python tests/restart_elasticity_case.py --source run70 --step 2000 \
  --dt 2e-10 --until 2e-6 --output run70long
(cd run70long && mpiexec -n 1 "$CUDA_EXE" input.in fill.in state)

# Eight particles with different cubic stiffnesses on a 256-cubed grid:
python tests/prepare_elasticity_case.py --case cubic3d --output cubic256
(cd cubic256 && mpiexec -n 1 "$CUDA_EXE" input.in fill.in state)
```

The default 3D fixture is **20 coupled steps (2 ns)**, an execution/performance
check, not a long growth study. Its measured GPU allocation was about 9 GiB.
Start with `--grid 64` to check your setup. The initializer refuses an existing
output directory. `--radius-nm`, `--supersaturation`, `--steps`, `--save`, `--dt`,
`--epsilon-nm`, and spacing/box arguments make the numerical parameters explicit.
All supplied material parameters are synthetic.

On the tested workstation, the MPI transport overrides below avoided local
UCX/UCC compatibility problems. They are environment-specific, not physics:

```bash
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,sm,tcp OMPI_MCA_coll=^ucc
export OPENBLAS_NUM_THREADS=1
```

A terminal-detached run can hold a Linux sleep inhibitor with
`systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block ...` around the
job. Screen-off is compatible; actual forced suspend pauses computation.
Checkpoints support restart. No solver can compute while the GPU is suspended.

## Tests and diagnostics

```bash
python tests/run_branch_checks.py                 # host algebra/export checks
bash tests/run_vtk_binary_io.sh                   # MPI binary roundtrip/errors
bash tests/run_2d_f1_nvhpc.sh --output /path/to/new-results # GPU coupled F1 suite
```

See the wrapper's `--help`/argument usage before choosing an output directory.
[Integration details](INHOM_ELASTICITY_INTEGRATION.md) describe controls, tensor
shear, unchained elastic forces, precision, and volume/slice diagnostics.
`INHOM_DIAGNOSTICS=2` with `MICROSIM_DIAGNOSTIC_Z_INDEX` exports a bounded-memory
stress-reconstruction plane. Invalid indices abort explicitly.

## Results and limits

The [elasticity study guide](../docs/elasticity/README.md) contains benchmark
parameters, results, figures and numerical limits. Highlights:

- Critical-size C/CUDA sweep near 59 nm; 50 nm shrinks with elasticity but grows
  without it at the same initial chemical potential.
- A 70 nm seed continued to 2 us, with measured chemical/elastic/capillary/kinetic
  balance and independent mechanics comparisons.
- A 256³ cubic-elasticity execution check; identical fields and convergence
  histories after optimization. Three timings per version gave a 5.93% lower
  median end-to-end runtime on this workstation.

These tests do not establish mesh-converged physical kinetics, long 3D growth,
licensed thermodynamic integration, or multi-GPU scaling. The test meshes have
uniform cubic spacing in 3D; unequal-spacing 3D accuracy is not established.
