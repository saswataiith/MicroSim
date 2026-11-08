# 3DKKSMP CPU MPI

A runnable C++17 CPU implementation of the **coupled phase-field and elasticity model** in the user-supplied July 2026 `3DKKSMP_OPT` CUDA code. Uses Open MPI, FFTW-MPI and parallel HDF5. The executable, reusable library, original-kernel checks and complete acceptance suite are included.

This is a standalone CPU MPI module included in MicroSim. Its current chemical model has one independent composition field; it is not yet a quinary WC binder solver. See [PORTING.md](docs/PORTING.md) for the equation mapping, deliberate corrections, feature coverage and integration interface.

## Build and run

The development Mac already has the required C++ libraries. On another machine install an MPI C++ compiler, double-precision FFTW with MPI/thread support, parallel HDF5 and pkg-config through the machine's normal package manager or cluster modules. No CUDA, NCCL, cuBLAS or PETSc is required.

```sh
make -j4
mpiexec -n 4 build/microsim_cpu examples/small.cfg
```

On systems where the parallel HDF5 pkg-config name is `hdf5-openmpi`, use `make HDF5_PKG=hdf5-openmpi`. Override `CXX=...` for a different MPI wrapper. FFTW and HDF5 must use the same MPI installation as the compiler and launcher. The build rejects serial HDF5. Do not combine libraries from incompatible MPI installations.

The example takes five small steps with anisotropic eigenstrain and unequal phase stiffness. It writes `run-small/diagnostics.csv`, parallel HDF5 phase fields and a complete elastic field output. Output arrays are double precision in `[x,y,z]` order, with `z` contiguous.

```sh
mpiexec -n 4 build/microsim_cpu examples/four_variants.cfg
mpiexec -n 2 build/microsim_cpu examples/elasticity.cfg
```

For tests only, install Python, NumPy and h5py in an environment of your choice:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install numpy h5py
make test PYTHON=.venv/bin/python
```

`MPIEXEC` can include launcher options, e.g. `MPIEXEC='mpiexec --oversubscribe' make test PYTHON=.venv/bin/python`. Do not request more CPU slots than your allocation. `fft_threads=2` uses two FFTW threads per MPI rank; reserve `ranks × threads` CPU cores. The pointwise loops are currently single-threaded.

## Physics and conventions

- Periodic 3D coherent elasticity with spatially varying **aligned cubic** stiffness, `C(x)=sum_p h_p C_p`, and full six-component `eps*(x)=sum_p h_p eps*_p`. All phase stiffnesses must be finite and strictly positive definite. Exact voids/rigid phases are not supported.
- Tensor ordering is `xx,yy,zz,yz,xz,xy`; shear strain is **tensor shear**, so `sigma_yz=2*C44*eps_yz` and tensor contractions count shear twice.
- Mean strain, mean stress, or a componentwise mixture can be prescribed. `fixed_strain = 0,0,0,0,0,0` means all mean stresses are prescribed. A `1` selects the corresponding `applied_strain` component instead.
- KKS quadratic chemical free energies, quintic interpolation, one/four order variants (runtime count is also supported), quartic and optional sixth-order variant penalties, semi-implicit Allen-Cahn phase evolution and conservative composition evolution.
- For phase evolution, `h_p=h(phi_p)` for precipitate variants and `h_matrix=1-sum_p h_p`. The matrix is phase 0; order variant `phi0` uses material phase 1, etc. Negative fractions or excessive overlap are rejected rather than clipped.
- `nz=1` gives a **z-invariant 3D problem**. It is not automatically intrinsic 2D or plane stress. Plane strain requires the appropriate prescribed out-of-plane strain and zero out-of-plane shear eigenstrain/loading for this aligned-cubic model.
- Every input is already in a consistent unit system. Stress, stiffness, chemical curvature and barrier height use energy/volume; `kappa` uses energy/length; `diffusivity` uses length²/time; `kinetic` uses volume/(energy·time). No unit inference or hidden SI conversion occurs.

The mathematical definition and stopping criteria are in [MODEL.md](docs/MODEL.md).

## Configuration

Plain `key = value` text. `#` starts a comment. Unknown/duplicate keys, malformed tensors, nonfinite values, invalid enums, unstable stiffness and invalid grids fail explicitly. Booleans are `0` or `1`. There are no implicit profile or unit conversions.

| Keys | Defaults / meaning |
|---|---|
| `mode` | `phasefield`; alternatively `elasticity` |
| `nx,ny,nz`; `dx,dy,dz` | `16,16,16`; `1,1,1` |
| `fft_threads`; `fft_measure` | `1`; `0` (reusable FFTW_ESTIMATE plans) |
| `variants`; `phases` | `1`; `variants+1` |
| `phaseP.c11,c12,c44` | Matrix: `300,100,80`; variants: `450,150,120` |
| `phaseP.eigenstrain` | Matrix zero; variants `-.005,-.005,-.005,0,0,0` |
| `applied_stress`, `applied_strain` | Six zeros each |
| `fixed_strain` | Six zeros: all mean stresses prescribed |
| `elastic_max_iterations` | `1000`; a cap is a failure |
| `elastic_change_tolerance` | `1e-10` |
| `elastic_equilibrium_tolerance` | `1e-8` |
| `elastic_mean_stress_tolerance` | `1e-10` |
| `elastic_norm` | `rms` or `max`; both include homogeneous-strain change |
| `dt`; `steps`; `output_every` | `.01`; `10` additional steps; checkpoint every `10` |
| `kinetic,kappa,omega,theta,theta2` | `1,.2,1,10,0` |
| `a_matrix,a_precip` | `1,1`, energy-density parabola coefficients |
| `c_matrix,c_precip`; `composition` | `.15,.23`; initial total composition `.19` |
| `diffusivity`; `mobility_method` | `.01`; `0` |
| `evolve_composition`; `elasticity` | `1`; `1` |
| `initial` | `sphere`, `sine`, or `uniform` |
| `radius`; `interface_width` | `.13*shortest active box length`; `1` |
| `phi_uniform` | `.3`, divided between variants |
| `phi_background` | `0`; optional explicit background for generated spheres |
| `initial_file` | Optional HDF5 with `composition`, `phi0`... in `[x,y,z]` |
| `restart` | Optional CPU checkpoint; restores fields, absolute step and time |
| `fraction_file` | Elasticity-only HDF5 with `h0`... for every material phase |
| `output` | `run`; files in this destination are replaced on repeated runs |
| `allow_restart_parameter_change` | `0`; use `1` only for an intentional change of experiment |

Diffusion method 0 uses constant mobility `D/(2*a_matrix)`; method 1 computes `div(M(h) grad(mu))` with `M=D*((1-h)/(2*a_matrix)+h/(2*a_precip))`; method 3 uses the volume average of that mobility. All preserve the zero Fourier mode of composition. Legacy method 2 is rejected because diffusing phase compositions independently and recombining them does not generally conserve total solute. No phase/composition clipping is performed. Use a smaller `dt` if a proposed state becomes invalid. Explicit nonlinear terms still impose a timestep restriction; the semi-implicit gradient update does not guarantee energy decrease for arbitrary steps.

## Outputs and restart

`state_final.h5` and `checkpoint_STEP.h5` contain `composition`, `phi0`..., shape/spacing, step/time and a fingerprint of the physical and numerical model. Checkpoints are written to `.partial`, closed collectively, then renamed. A different MPI rank count is allowed on restart. A different model or timestep is rejected unless explicitly allowed.

`fields_final.h5` contains fractions, displacement `u0,u1,u2`, total strain, stress, elastic energy density and **unchained** `dFdhP` partials. It records convergence status and residuals. Failed standalone elastic runs save their last fields with `converged=0`, omit forces and return nonzero. Coupled runs refuse to evolve with an unconverged mechanical solution.

`diagnostics.csv` reports absolute step/time, mean composition, total energy density, stress-controlled thermodynamic potential and elastic residuals. `input.cfg` records the supplied keys; `model.txt` records the effective model parameters, including defaults, used for restart checks. Keep these with the source manifest and validation report.

To continue a five-step run for five more steps, copy the original configuration, set a new `output`, set `restart = run-small/state_final.h5`, and set `steps = 5`. Preserve the other physical/numerical parameters.

## Bring in your actual CUDA data

The original archive is preserved in `reference/3DKKSMP.tar.gz`. The input converter supports its documented `Non_dim_calc=1` path and derives explicit density units from the original scales. It refuses ambiguous legacy branches rather than guessing.

```sh
.venv/bin/python tools/import_cuda_h5.py cuda_snapshot.h5 cpu_initial.h5 --variants 4
.venv/bin/python tools/convert_cuda_input.py InputParams cpu.cfg --initial-file cpu_initial.h5
mpiexec -n 4 build/microsim_cpu cpu.cfg
```

The importer reads global floating-point `COMP,PHI0,...` CUDA datasets. Quantized integer or processor-local snapshots need their original scale/layout metadata and are rejected. The imported file is an initial condition, not a fabricated CPU restart history.

`--example-initialization` explicitly substitutes a demonstration geometry when testing conversion without original fields. It does not reproduce the old nucleus placement. Converted legacy files may request a large grid and many steps; set those deliberately before running. Old stochastic nucleation/noise, smoothing, GPU tuning and quantized output switches are not emulated. Arbitrary original microstructures can be supplied through `initial_file`.

## Evidence and limits

`make test` creates [VALIDATION.json](VALIDATION.json) and per-run logs under `build/validation/`. Checks include original CUDA local kernels, independent FFT/analytical/variational checks, a NumPy one-step oracle, rank invariance, empty slabs, threaded FFTs, restart across rank counts and error paths. Host kernel checks do not constitute GPU execution.

This is a tested research implementation, not a claim of validated production kinetics or large-cluster scaling. No NVIDIA end-to-end reference trajectory was available. Exact agreement with the old program is not expected where its bugs or discretization conventions were corrected. The initial acceptance runs use small grids, with a 32³ execution smoke check. Validate your actual material parameters, grid convergence, interface resolution and timestep on your CPU cluster before scientific production.

Source attribution remains with the original authors and user-supplied code. No new license or publication claim is made here.
