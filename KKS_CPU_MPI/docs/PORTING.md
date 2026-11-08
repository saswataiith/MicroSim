# CUDA source mapping and corrections

Original: `reference/3DKKSMP.tar.gz`, uploaded in the conversation [Code and CUDA algorithms slides](https://claude.ai/chat/d27c2168-515b-8785-abaa-8f53b6d5a798). This is the original coupled spectral program, not the incomplete July MicroSim FD-module patch.

| Original operation | CPU implementation |
|---|---|
| cuFFTMp plans/descriptors and `BoxIterator` | `FFT`, FFTW-MPI R2C/C2R, actual FFTW slab metadata |
| CUDA pointwise kernels | Local C++ loops, compact x-slab arrays |
| CUB/cuBLAS/NCCL reductions | Local sums/maxima plus MPI_Allreduce |
| `get_omega`, `Compute_ts`, `Update_disp`, periodic strain | `Elasticity::project`, `reconstruct`, constitutive iteration |
| `Compute_homstr`, average stiffness/eigenstress/stress | `Elasticity::balance` using full tensors |
| `Compute_dfeldphi_inhomsv` | Fraction partials and one explicit chain rule |
| `Calc_hsum_dhdphi_sv`, multivariant interpolation | `interpolation`, `interpolation_derivative`, `PhaseField::fractions` |
| `Calc_phase_comp`, phase energies, `Calc_mu`, chemical phase force | KKS closure in `PhaseField::evaluate` |
| `Calc_dgdphi`, `Calc_dgdphi2` | Quartic/sixth-order barrier and matching energy |
| `Update_phi` | Same semi-implicit Fourier update |
| Composition methods 0/1 | Conservative constant/variable-mobility spectral updates |
| Method 3 | Conservative volume-averaged mobility with corrected factor |
| Parallel HDF5 field output | Collective hyperslabs, double precision, empty slabs supported |

## Deliberate fixes

1. General constitutive averaging includes the anisotropy factor missing from the original xx eigenstress average. Full tensor eigenstrain, including `eps_xx=0` and shear, is supported.
2. The max displacement norm uses the maximum across **all** components. The original signed reduction followed by a component minimum could declare convergence with two components still changing.
3. A cap is a failure. Mechanical equilibrium and mean loading are checked in addition to displacement/mean-strain change. Failed forces are unavailable to the phase-field integrator.
4. The phase derivative is applied exactly once. The extra derivative in the July MicroSim bridge/chemical-kernel combination is absent here.
5. Normalized reference acoustic inversion removes the dimensionful `abs(det)>1e-6` cutoff. Reference inverses are cached.
6. Odd-grid wave numbers and even-grid first-derivative Nyquist conventions are explicit and consistent.
7. Diagnostic gradient energy is `kappa/2*|grad phi|^2`, consistent with the live phase update. The CUDA diagnostic had a conflicting prefactor.
8. Weighted mobility includes the `1/2` required by `mu=2*a*(c-c_eq)`. The CPU computes its current volume average rather than freezing an initial value.
9. Nonconservative legacy method 2 is rejected. Use the conservative flux update instead; no after-the-fact mass projection is hidden in the code.
10. Input validation, collective failures, zero-sized slabs, complete HDF5 checkpoint metadata and model-consistent restart checks are implemented.

The live shear derivative factor `0.5*[...+8*Chet44*sum(e_shear^2)]` is **preserved**, not “fixed” to the commented old coefficient. It follows from the implemented tensor-shear energy. `tests/test_cuda_oracle.cpp` checks actual extracted CUDA local kernels against the CPU force (one and four variants). Original attribution is retained in the test header; the untouched archive is packaged for traceability.

## Coverage and deliberate boundaries

Implemented: complete deterministic evolution of composition and phase order fields, coupled mechanical equilibrium, homogeneous and heterogeneous cubic materials, applied/mixed mean loading, arbitrary full eigenstrain, one/four variants, runtime variant count, standalone N-material elasticity, energy/stress/strain/force outputs, small built-in initial profiles, external microstructure input, parallel checkpoints and restart.

This is not a byte-for-byte replacement for every historical utility switch. Original random-nucleus placement, stochastic noise injection, smoothing passes, quantized output variants and GPU performance switches are not emulated. Supply the original floating-point state through `initial_file` to use the actual microstructure. The converter rejects unsupported delayed activation; staged runs with intentional loading changes can represent such experiments. Intrinsic 2D physics, rotated general anisotropic stiffness, plasticity, exact void/rigid constraints and GPU execution are outside this implementation.

The CPU phase-field chemistry retains the two-chemical-phase KKS model. Having several mechanical phases/order variants does not silently introduce an N-chemical-phase thermodynamic model.

## Future MicroSim integration

This program can run independently now. The static library is a starting point for CPU MicroSim integration, not a pre-applied repository modification.

1. Use the library's actual FFTW `local_nx,x_start`, or redistribute MicroSim physical fields onto these slabs. **Do not assume** cuFFTMp's remainder partition and FFTW's default block partition match on uneven grids.
2. Strip the caller's halo using its actual padding/strides. Pack fractions/eigenstrain with the same phase conventions and tensor-shear ordering.
3. Run `Elasticity::solve` collectively, check status and use either the unchained `fraction_derivative` or an explicitly chained phase force. Define that interface once; the old FD chemical routine must not chain an already chained force.
4. Align energy-density units and loading conventions. Restore caller halos after the returned force is inserted.
5. Re-run force/energy, homogeneous-limit, rank, restart and phase-evolution regression checks against that specific MicroSim branch before merging.

## Verification boundary

Independent analytical laminate and compatible-eigenstrain cases, finite-difference relaxed energy derivatives and coupled phase-field derivatives verify the corrected equations. NumPy checks the complete one-step Fourier update. MPI tests verify rank changes, empty slabs, threading and HDF5/restart behavior. Sanitizer evidence is recorded separately.

No NVIDIA trajectory was run in this environment, so end-to-end CUDA/CPU trajectory agreement is not claimed. Long-time calibrated kinetics, large-contrast convergence and large-cluster scaling remain application-specific validation work.
