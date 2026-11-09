#ifndef COMPUTE_ELASTIC_INHOM_CUH
#define COMPUTE_ELASTIC_INHOM_CUH

/*
 * Inhomogeneous (heterogeneous-modulus) microelasticity solver for
 * KKS_FD_CUDA_MPI, ported from the standalone 3DKKSMP spectral solver
 * (inhomelast.cu / calc_uzero.cu).
 *
 * Physics: iterative Khachaturyan-type solver.  Local stiffness
 *   C(r) = Chom + Chet*(2*h(phi) - 1),
 *   Chom = (C_precip + C_matrix)/2,  Chet = (C_precip - C_matrix)/2,
 * eigenstrain is diagonal; intrinsic 2D requires its zz component to be zero,
 * applied stress supported through the homogeneous-strain balance
 *   Ebar = inverse(<C>) ( sigma_appl + <sig*> - <dsig> ).
 *
 * Current scope (deliberate, matching the source solver):
 *   - 3-D or 2-D: generalized plane strain / intrinsic 2-D (MESH_Z == 1)
 *   - exactly one precipitate phase with non-zero EIGEN_STRAIN and one
 *     matrix phase with zero EIGEN_STRAIN (NUMPHASES == 2)
 *   - diagonal eigenstrain; 3D retains yy==zz, 2D permits separate zz
 *     (generalized plane strain retains zz eigenstrain energy; intrinsic 2D requires zz=0)
 *   - cubic stiffness from VOIGT_ISOTROPIC {C11, C12, C44} per phase
 *
 * Reductions: CUB (intra-GPU) + NCCL (inter-GPU) + cuBLAS (L2 norm),
 * NCCL communicator bootstrapped over the module's MPI communicator.
 */

#ifndef ENABLE_CUFFTMP
#define ENABLE_CUFFTMP 0
#endif

#if ENABLE_CUFFTMP == 1

#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftMp.h>
#include <nccl.h>
#include <cublas_v2.h>

#include "structures.h"
#include "defines.h"

/*
 * All solver state lives in this context so that no file-scope globals
 * are introduced (MicroSim convention: state travels through structs).
 */
typedef struct InhomElasticCtx
{
    /* distributed cuFFTMp fields (Z2Z, x-slab INPLACE at rest) */
    cudaLibXtDesc *ux, *uy, *uz;          /* converged displacement (persistent, warm start) */
    cudaLibXtDesc *unewx, *unewy, *unewz; /* iteration workspace */
    cudaLibXtDesc *str_v[6];              /* periodic strain / source tau, Voigt 11,22,33,23,13,12 */

    /* real-space slab-sized double arrays (local physical cells, no halo) */
    double *hphisum;                      /* h(phi_precip) */
    double *dfelas;                       /* scalar elastic driving force */
    double *scr1, *scr2, *scr3;           /* scratch: C-averages, stress averages, residuals */

    /* small device arrays */
    double *Chom_d, *Chet_d;              /* [3] : C11, C12, C44 */
    double *Cavg, *S_d;                   /* [3] */
    double *avgeigsts;                    /* [3] */
    double *avgpersts;                    /* [6] */
    double *hom_strain_v;                 /* [6] */
    double *sigappl_v;                    /* [6] */
    double *disperr;                      /* [3] */

    /* reduction machinery */
    void   *cubTemp;
    size_t  cubTempBytes;
    cublasHandle_t blas;
    ncclComm_t     nccl;

    /* host-side parameters */
    double epszero;                       /* eps0 = eigen_strain[precip].xx */
    double misfit_z;                      /* eigen_strain[precip].zz / .xx */
    int loading;                         /* 0 stress-free, 1 constrained, 2 applied stress */
    int model2d;                         /* 0: full 3D, 1: generalized plane strain, 2: intrinsic */
    double misfit_aniso;                  /* eigen_strain[precip].yy / .xx  */
    double Tol;                           /* convergence tolerance          */
    long   nelast;                        /* max solver iterations          */
    int    norm_calc;                     /* 0: L2 (cuBLAS+MPI), 1: max (CUB+NCCL) */
    int    strict;                        /* 1: abort if not converged in nelast   */
    long   precipPhase, matrixPhase;
    long   num_items;                     /* local physical cells = numX*MESH_Y*MESH_Z */
    int    converged;
    long   iterations;
    double residual;
    int    firstCall;                     /* 1 until the zeroth-order seed has run     */
    int    rank;
} InhomElasticCtx;

/*
 * Allocate all state, set device constants, derive Chom/Chet and the
 * eigenstrain parameters from simParams, bootstrap NCCL over comm.
 * Must be called after the cuFFTMp plan has been created.
 */
void elasticInhomInit(InhomElasticCtx *ctx, cufftHandle plan,
                      domainInfo simDomain, controls simControls,
                      simParameters simParams, subdomainInfo subdomain,
                      MPI_Comm comm);

/*
 * One mechanical-equilibrium solve + driving-force accumulation.
 *   phi    : module phase fields (halo layout, double**, device pointers)
 *   dfdphi : module driving-force arrays; unchained elastic partials are ADDED; the chemical stage chains once
 * Call once per time step in place of the Bpq elastic path.
 */
void computeDrivingForce_ElasticInhom(double **phi, double **dfdphi,
                                      InhomElasticCtx *ctx, cufftHandle plan,
                                      domainInfo simDomain, controls simControls,
                                      simParameters simParams, subdomainInfo subdomain,
                                      cudaStream_t stream,
                                      dim3 gridSize, dim3 blockSize,
                                      MPI_Comm comm);

void elasticInhomFree(InhomElasticCtx *ctx);

#endif /* ENABLE_CUFFTMP */

#endif /* COMPUTE_ELASTIC_INHOM_CUH */
