# Two-dimensional Function F1 elasticity

Two independent choices define a case:

| Input | Meaning |
|---|---|
| `INHOM_2D_MODE=1` | Generalized plane strain: full 3D normal law on a 2D domain, uniform Ezz evaluated under mean-stress control. |
| `INHOM_2D_MODE=2` | Intrinsic 2D: xx, yy, xy only, with no out-of-plane eigenstrain or energy. |
| `INHOM_LOADING=STRESS_FREE` | Evaluate homogeneous strain from zero mean stress. |
| `INHOM_LOADING=CONSTRAINED` | Specify `HOMOGENEOUS_STRAIN={xx,yy,zz,yz,xz,xy}`; evaluate reaction stresses. |
| `INHOM_LOADING=APPLIED_STRESS` | Evaluate homogeneous strain for `APPLIED_STRESS={xx,yy,zz,yz,xz,xy}`. |

All strain inputs use **tensor shear**, epsilon_xy, not engineering gamma_xy.
Both 2D models exclude xz/yz. The intrinsic model also excludes zz. Under
generalized stress-free loading only the **mean** sigma_zz vanishes; local
sigma_zz need not vanish. These examples are synthetic F1 benchmarks, not
calibrated Al–Zn or WC material data.

## Build and verify

From the repository root, with NVIDIA HPC SDK, GCC/G++, GSL/pkg-config and NumPy:

```bash
bash KKS_FD_CUDA_MPI/build_nvhpc.sh
bash tests/run_2d_f1_nvhpc.sh --output /absolute/path/to/a/new/test-directory
```

The wrappers default to NVHPC 26.9 and CUDA 12.9; override `NVHPC_HOME`,
`HPCSDK_VERSION`, `NVHPC_CUDA_VERSION`, `MPI_HOME`, `MATH_HOME`, `NCCL_HOME`
as needed. `CUDA_ARCH` is detected from the first GPU unless explicitly set.
Set `PYTHON` to a Python interpreter containing NumPy. This tested build uses
VTK output and disables HDF5. Each test runs in its own directory and preserves
its input, initial/output fields, diagnostics and log. Existing case directories
are never overwritten. The runner rejects an executable older than the sources.

For an individual geometry-filled example, after setting the same library/MPI
environment as `nvhpc_env.sh`, run the executable with these three arguments:

```text
examples/2d_f1/generalized.in examples/2d_f1/fill.in output_name
```

Alternatively choose `intrinsic.in`. Use MPI with one rank on the local GPU.
The test runner is the recommended verification route: it supplies known smooth
restart fields, unequal parabolic curvatures, non-proportional phase stiffnesses
and unequal x/y grid spacings, rather than relying only on a sharp filling shape.

## Independent verification

The mechanical oracle constructs and minimizes a dense compatible-strain energy
on small periodic grids. It shares neither the CUDA Green tensor nor the
fixed-point/compliance implementation. Checks cover strain, mean loading,
equilibrium, and the finite-difference elastic-energy derivative. A Richardson
combination removes the cubic truncation term in zero-force energy checks.

The F1 oracle solves the two-parabola KKS equations analytically, then predicts
phase and composition fields at every timestep. With p=phi0 and phi1=1-p,
its scalar energy is

```text
F = h(p)*f0(c0)/V + (1-h(p))*f1(c1)/V + W_el
    + theta*p^2*(1-p)^2 + kappa*|grad p|^2.
```

The kinetic coefficient is evaluated from MicroSim's supplied thin-interface
parameters. Chemical/elastic interpolation derivatives follow MicroSim's
constrained binary convention; they must not be chained twice. Composition
transport uses one arithmetic-mobility flux per face. Enabled anti-trapping is
also compared against an independent face-flux expression, including the
zero-gradient limit. Total composition and phi0+phi1 are checked separately.

The suite covers both models and all three loading cases, uniform analytical
solutions, odd/even grids, homogeneous stiffness, solved Ezz with zero physical
zz eigenstrain, anti-trapping, temporal refinement, and refusal to advance when
mechanics does not converge. The selected no-anti-trapping cases also check
monotonic loading potential. Anti-trapping is not assumed to be a variational
energy descent.

Additional stencil checks:

```bash
c++ -std=c++17 -O1 -g -fsanitize=address,undefined tests/test_2d_stencil.cpp -o /tmp/test_2d_stencil
/tmp/test_2d_stencil
```

The corrected rectangular nine-point stencil preserves constants and quadratic
moments and converges at second order for Fourier modes. It retains the prior
square-grid stencil. ASCII field output now retains 17-digit precision for
verification and restart. `ANTI_TRAPPING=0` is respected; its default remains 1.

CUDA currently supports two aligned-cubic elastic phases, diagonal eigenstrain,
one phase with zero eigenstrain, and nonzero xx eigenstrain in the other. Mixed
mean loading and shear eigenstrain are outside this CUDA path. The distributed
cuFFTMp/NCCL route compiles, but these execution checks use one physical GPU;
genuine multi-GPU and cluster scaling remain to be validated.
