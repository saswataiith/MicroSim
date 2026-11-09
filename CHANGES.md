# Changes on the inhomogeneous-elasticity feature branch

## 2026-11-09 — verified nuclei, 3D execution and solver/I/O optimization

### Physics, examples and evidence

- Document dimensional models separately from loading: stress-free evaluates
  homogeneous strain, constrained loading specifies it, and applied mean stress
  evaluates strain for that stress. Generalized plane strain evaluates uniform
  Ezz using the 3D normal law; intrinsic 2D excludes out-of-plane energy.
- Add reproducible synthetic F1 preparation and physical-time-preserving restart
  tools for 2D nuclei and eight-particle cubic 3D cases. These tools write inputs;
  compiled C/CUDA performs all physical evolution.
- Publish compact results for the critical-size sweep near 59 nm, 70 nm growth
  to 2 µs, the growing-interface elastic Gibbs–Thomson/kinetic balance, and the
  256³ cubic-elasticity execution/performance study.
- Record numerical limits alongside results. The 3D large-grid test covers
  20 coupled steps (2 ns), not long production evolution or multi-GPU scaling.

### CUDA and binary I/O

- Swap accepted/workspace displacement descriptors after elastic iterations,
  rebuilding box views and preserving the final accepted warm start. Remove
  three complete complex-volume copies per iteration without changing the
  tolerance, norm, interpolation, constitutive law, FFT convention or precision.
- Correct binary restart's inclusive-count mismatch, prevent formatted scanning
  from consuming payload bytes, reject truncated/missing fields, and avoid
  aliasing in byte conversion. Batch binary input/output by physical rows,
  preserve full-precision spacing, and exclude halo cells.
- Add `INHOM_DIAGNOSTICS=2` for bounded-memory z-plane export, controlled by
  `MICROSIM_DIAGNOSTIC_Z_INDEX`; preserve default-off and full-volume modes.
  Reject invalid modes and slice indices explicitly.
- Three matched 256³ runs per version: median 100.752 → 94.777 s (5.93% lower
  wall time); identical phase/composition arrays and convergence histories.

### CPU C interface and tests

- Publish `elasticity_c.h`, backed by the existing CPU/FFTW-MPI C++ elasticity
  engine. C callers own MPI and collective solver handles. Interface fields use
  tensor shear and expose 3D, generalized-plane-strain and intrinsic-2D models,
  stress/strain/mixed loading, diagnostics, energy and unchained fraction forces.
- Add C caller tests and independent dense/finite-difference checks, including
  MPI empty slabs and threaded FFTs. This is a C ABI, not a separate pure-C
  elasticity implementation.
- Add binary roundtrip/error tests to host CI. Recheck the 20 GPU F1 acceptance
  cases and 9 host checks after optimization; retain the CPU acceptance suite.
- Compare matching 64³ cubic mechanics against the CPU C API on 1/4 MPI ranks;
  stress difference below 1 Pa. Independently check two 3D coupled local updates.

### Earlier changes retained on this branch

- Add CUDA 2D generalized-plane-strain and intrinsic-2D inhomogeneous elasticity,
  with independent stress-free/constrained/applied-stress loading controls.
- Correct the rectangular 2D isotropic stencil, anti-trapping positive-x face
  composition sampling and roundoff-level gradient-normal limit. Respect the
  anti-trapping input switch and preserve 17-digit ASCII field precision.
- Preserve strict failure on unconverged mechanics and correct intrinsic-2D
  diagnostics to exclude out-of-plane stress.

See [the study guide](docs/elasticity/README.md),
[CUDA README](KKS_FD_CUDA_MPI/README.md), and
[CPU README](KKS_CPU_MPI/README.md) for commands, conventions and evidence.
