# Equations, storage and acceptance

Tensor components are `(xx,yy,zz,yz,xz,xy)` with tensor shear. All calculations use double precision. `dot(a,b)` includes twice the off-diagonal products.

For material fractions summing to one,

```
C(x) = sum_p h_p(x) C_p
eps*(x) = sum_p h_p(x) eps*_p
eps(x) = E + sym grad u(x)
e(x) = eps(x) - eps*(x)
sigma = C:e
f_el = 1/2 e:C:e
```

The mean displacement is fixed to zero. For fixed mean stress, the appropriate potential is `<f_el> - Sigma:E`; mixed loading subtracts work only on stress-controlled components, with the tensor contraction's factor of two for shears.

The iteration is the displacement perturbation scheme in the original `Compute_ts` and `Update_disp`:

```
tau = C:eps* - C:E - (C-Cref):eps_periodic
u_hat = -i inverse(Cref_ijkl k_j k_l) (tau_hat . k)
eps_periodic_hat = sym(i k tensor u_hat)
```

After reconstruction, a small exact linear solve enforces every unconstrained mean-stress component using the current `<C>` and `<C:(eps* - eps_periodic)>`. This directly fixes the original anisotropic eigenstress averaging issue: all six eigenstrain components go through the constitutive map instead of separate special-case average formulas.

For one/two distinct cubic stiffnesses, `Cref` is their arithmetic mean (repeated identical variants do not change it). With more distinct stiffnesses, a positive isotropic reference bounds their maximum tensor-space stiffness eigenvalue. All phases must be strictly positive definite; high contrasts can still require many fixed-point iterations and must pass the cap/tolerances.

The inverse reference acoustic tensor is precomputed on the local Fourier slab. The small matrix inversion uses a relative conditioning test, with the wavevector normalized first. There is no absolute dimensional determinant cutoff.

The local material-fraction partial is

```
g_p = d f_el/d h_p = 1/2 e:C_p:e - sigma:eps*_p.
```

The order-variable force is `h'(phi_p)*(g_p-g_matrix)`. The matrix fraction is dependent. Apply this chain rule exactly once. At a converged equilibrium, these partials also give the derivative of the relaxed thermodynamic potential.

## Phase-field model

The KKS closure is the original two-chemical-phase parabolic model, even when multiple order variants are used:

```
h = sum_p h(phi_p),  h(phi)=phi^3*(10-15phi+6phi^2)
f_m = a_m*(c_m-c_m_eq)^2, f_p = a_p*(c_p-c_p_eq)^2
mu = 2*a_m*(c_m-c_m_eq) = 2*a_p*(c_p-c_p_eq)
c = (1-h)*c_m + h*c_p
c_m = [c-h*(c_p_eq-(a_m/a_p)*c_m_eq)] / [1-h+h*a_m/a_p]
c_p = c_p_eq + (a_m/a_p)*(c_m-c_m_eq)
df_chem/dh = f_p-f_m+(c_m-c_p)*mu
```

Order variants are not separate chemical compositions. They can have different mechanical stiffness/eigenstrain tensors in the CPU implementation, but all precipitate variants share this chemical free energy.

Barrier energy is `omega*[sum(phi_p^2*(1-phi_p)^2) + theta*sum_{p<q}phi_p^2 phi_q^2 + theta2*sum_{p<q<r}phi_p^2 phi_q^2 phi_r^2]`. Gradient energy is `kappa/2*sum_p |grad phi_p|^2`, matching the live CUDA update denominator. The CPU diagnostics correct the original inconsistent gradient-energy prefactor.

```
phi_hat_new = (phi_hat - dt*L*local_force_hat)/(1+dt*L*kappa*k^2)
c_hat_new = c_hat - dt*M*k^2*mu_hat            [constant/averaged mobility]
c_hat_new = c_hat + dt*FFT(div(M(h)*grad(mu))) [variable mobility]
```

All explicit forces use the same old state. Material balance, common chemical potential, phase barrier and elastic force are evaluated before either evolution equation is committed. No interpolation derivative is applied twice. No global mass correction or clipping is used to make a bad update pass.

## FFT distribution and discrete derivatives

FFTW-MPI determines `local_nx,x_start` for compact physical fields. Its R2C/C2R scratch array is padded on the last dimension even out of place. Forward output is transposed to `[local_y][global_x][kz]`; inverse plans consume that layout. Only reduced complex spectra are stored. Fields are never gathered to rank zero for solving or output. MPI reductions communicate scalars or six-component tensors; distributed FFTs handle the global transpose.

The layout follows the [FFTW MPI real-data documentation](https://www.fftw.org/fftw3_doc/Multi_002ddimensional-MPI-DFTs-of-Real-Data.html) and [transposed-distribution documentation](https://www.fftw.org/fftw3_doc/Transposed-distributions.html).

Odd-grid frequency indexing is corrected. For first derivatives on even grids, each axis's Nyquist derivative is explicitly zero, yielding a real, skew-adjoint collocation derivative. Elastic strain and equilibrium use the same derivative. Constant-coefficient diffusion and the phase-gradient Laplacian use the full scalar `k^2`, including Nyquist, as spectral second derivatives. Consequently the variable-mobility and constant-mobility schemes have different highest-frequency treatment on even grids, but agree on resolved non-Nyquist modes. This convention is explicit and tested; it does not reproduce the old implicit imaginary-part discard.

## Stopping rule

Let `Lbox` be the largest box length, `e_scale` the largest phase eigenstrain, prescribed mean strain or `|Sigma|/Cscale` (with a `1e-15` floor), and `s_scale=max(|Sigma|,Cscale*e_scale)`.

- Change: maximum of `rms(delta u)/(Lbox*e_scale)` (or max component norm) and every `|delta E_a|/e_scale`.
- Equilibrium: `rms(div sigma)/[(2*pi/Lbox)*s_scale]`, evaluated from independently transformed constitutive stress with correct reduced-spectrum Parseval weights.
- Mean stress: maximum stress-controlled component error divided by `s_scale`.

All three checks must pass. At the cap, `converged=false`, reason is `iteration_limit`, and no force is returned. Nonfinite iterates fail collectively. Coupled evolution refuses a mechanical failure. A displacement update alone does not establish equilibrium.

## Library use

`FFT` owns plans and scratch storage. Initialize MPI and FFTW-MPI before constructing it; destroy the solver/FFT objects before MPI finalization. Calls are collective on the supplied communicator, including empty-slab ranks.

`Elasticity::solve` accepts one compact local fraction array per phase and a `Loading`. Its returned fields are owned copies. Solver buffers retain the last displacement/periodic strain for the next call's warm start; `reset()` clears them. After a failed solve, call `reset()` before an intentional retry with different controls/model. `solve` does not alter the input fractions.

`PhaseField::evaluate` does not modify the phase/composition state, but updates the elasticity warm-start workspace. `advance` evaluates old-state forces and constructs a proposed state, validates it, then commits it. It returns diagnostics for the **old** state. `evaluate` can then obtain diagnostics/fields for the new state.

`build/libmicrosim_cpu.a` is reusable. Link it with the MPI compiler, FFTW-MPI, FFTW threads, FFTW double precision and parallel HDF5. See `include/cpu_mpi.hpp` for the complete public API.
