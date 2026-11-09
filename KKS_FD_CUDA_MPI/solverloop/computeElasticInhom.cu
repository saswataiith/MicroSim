#include "computeElasticInhom.cuh"

#if ENABLE_CUFFTMP == 1

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iterator>
#include <initializer_list>
#include <climits>
#include <utility>
#include <vector>
#include <cub/cub.cuh>

#include "box_iterator.hpp"
#include "error_checks.hpp"
#include "functionH.cuh"
#include "inhom_math.hpp"

#define NCCL_CHECK(call)                                                    \
    do {                                                                    \
        ncclResult_t r__ = (call);                                          \
        if (r__ != ncclSuccess) {                                           \
            fprintf(stderr, "NCCL error %s at %s:%d\n",                     \
                    ncclGetErrorString(r__), __FILE__, __LINE__);           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* ------------------------------------------------------------------ */
/* device-constant grid metadata (set once in elasticInhomInit)        */
/* ------------------------------------------------------------------ */
__constant__ int    inh_nx, inh_ny, inh_nz;
__constant__ double inh_dx, inh_dy, inh_dz;

/* ------------------------------------------------------------------ */
/* device helpers (ported from calc_uzero.cu)                          */
/* ------------------------------------------------------------------ */
static __device__ void inh_get_k(double *kx, double *ky, double *kz,
                                 BoxIterator<cufftDoubleComplex> it)
{
    const long x = (long)it.x();
    const long y = (long)it.y();
    const long z = (long)it.z();

    *kx = inhom_wave_number(x, inh_nx, inh_dx);
    *ky = inhom_wave_number(y, inh_ny, inh_dy);
    *kz = inhom_wave_number(z, inh_nz, inh_dz);
}

/* cubic Green function Omega_ik(k) = (Chom_ijkl k_j k_l)^-1, 6 Voigt slots */
static __device__ void inh_get_omega(double *omega, const double* __restrict__ n,
                                     const double* __restrict__ Chom)
{
    inhom_green_inverse(omega, n, Chom);
}

/* ------------------------------------------------------------------ */
/* bridge kernels: module halo layout  <->  cuFFTMp slab layout        */
/* 2-D/3-D, padding == 1 in active axes; singleton z has no halo.      */
/* ------------------------------------------------------------------ */
__global__
void inh_bridge_hphi(double **phi, double *hphisum, long precip,
                     long NUMPHASES, long xStep, long yStep,
                     long sizeX, long sizeY, long sizeZ)
{
    const long x = threadIdx.x + blockIdx.x*blockDim.x;
    const long y = threadIdx.y + blockIdx.y*blockDim.y;
    const long z = threadIdx.z + blockIdx.z*blockDim.z;

    if (x >= sizeX || y >= sizeY || z >= sizeZ)
        return;

    if (x != 0 && x != sizeX-1 && y != 0 && y != sizeY-1 &&
        (sizeZ == 1 || (z != 0 && z != sizeZ-1)))
    {
        const long idx    = x*xStep + y*yStep + z;
        const long nz = sizeZ == 1 ? 1 : sizeZ-2;
        const long fftIdx = (x-1)*(sizeY-2)*nz + (y-1)*nz + (sizeZ == 1 ? 0 : z-1);
        hphisum[fftIdx] = calcInterp5th(phi, precip, idx, NUMPHASES);
    }
}

__global__
void inh_bridge_dfeldphi(double **phi, double **dfdphi, const double* __restrict__ dfelas,
                         long precip, long NUMPHASES, long xStep, long yStep,
                         long sizeX, long sizeY, long sizeZ)
{
    const long x = threadIdx.x + blockIdx.x*blockDim.x;
    const long y = threadIdx.y + blockIdx.y*blockDim.y;
    const long z = threadIdx.z + blockIdx.z*blockDim.z;

    if (x >= sizeX || y >= sizeY || z >= sizeZ)
        return;

    if (x != 0 && x != sizeX-1 && y != 0 && y != sizeY-1 &&
        (sizeZ == 1 || (z != 0 && z != sizeZ-1)))
    {
        const long idx    = x*xStep + y*yStep + z;
        const long nz = sizeZ == 1 ? 1 : sizeZ-2;
        const long fftIdx = (x-1)*(sizeY-2)*nz + (y-1)*nz + (sizeZ == 1 ? 0 : z-1);
        const double df   = dfelas[fftIdx];

        // Match the Bpq interface: chemical kernels apply dh/dphi once.
        // dF/dh_precip is the constrained precipitate-minus-matrix partial.
        dfdphi[precip][idx] += df;
    }
}

/* ------------------------------------------------------------------ */
/* field kernels (1-D launches over the local cuFFTMp box)             */
/* ported from inhomelast.cu / calc_uzero.cu                           */
/* ------------------------------------------------------------------ */

/* eigenstress sigma* = Chom : eps0*h(phi), stored in u arrays (seed)  */
__global__
void inh_eigsts_hom(BoxIterator<cufftDoubleComplex> ux, BoxIterator<cufftDoubleComplex> uxEnd,
                    BoxIterator<cufftDoubleComplex> uy, BoxIterator<cufftDoubleComplex> uz,
                    const double* __restrict__ hphisum, const double* __restrict__ Chom,
                    double epszero, double misfit_aniso, double misfit_z)
{
    const long tid = threadIdx.x + (long)blockIdx.x*blockDim.x;
    ux += tid; uy += tid; uz += tid;
    if (ux < uxEnd)
    {
        const double eig11 = epszero * hphisum[tid];
        const double eig22 = eig11 * misfit_aniso;
        const double eig33 = eig11 * misfit_z;
        const double C11 = Chom[0], C12 = Chom[1];

        *ux = { fma(C11, eig11, C12*(eig22 + eig33)), 0.0 };
        *uy = { fma(C11, eig22, C12*(eig11 + eig33)), 0.0 };
        *uz = { fma(C11, eig33, C12*(eig11 + eig22)), 0.0 };
    }
}

/* zeroth-order displacement u0 = -i Omega sigma* k (SHUFFLED layout)  */
__global__
void inh_uzero(BoxIterator<cufftDoubleComplex> ux, BoxIterator<cufftDoubleComplex> uxEnd,
               BoxIterator<cufftDoubleComplex> uy, BoxIterator<cufftDoubleComplex> uz,
               const double* __restrict__ Chom)
{
    const long tid = threadIdx.x + (long)blockIdx.x*blockDim.x;
    ux += tid; uy += tid; uz += tid;
    double n[3], omega[6];
    if (ux < uxEnd)
    {
        inh_get_k(&n[0], &n[1], &n[2], ux);
        inh_get_omega(omega, n, Chom);

        const cufftDoubleComplex fk11 = { ux->x * n[0], ux->y * n[0] };
        const cufftDoubleComplex fk22 = { uy->x * n[1], uy->y * n[1] };
        const cufftDoubleComplex fk33 = { uz->x * n[2], uz->y * n[2] };

        *ux = {  (omega[0]*fk11.y + omega[5]*fk22.y + omega[4]*fk33.y),
                -(omega[0]*fk11.x + omega[5]*fk22.x + omega[4]*fk33.x) };
        *uy = {  (omega[5]*fk11.y + omega[1]*fk22.y + omega[3]*fk33.y),
                -(omega[5]*fk11.x + omega[1]*fk22.x + omega[3]*fk33.x) };
        *uz = {  (omega[4]*fk11.y + omega[3]*fk22.y + omega[2]*fk33.y),
                -(omega[4]*fk11.x + omega[3]*fk22.x + omega[2]*fk33.x) };
    }
}

/* periodic strain from displacement, 6 Voigt comps (SHUFFLED layout)  */
__global__
void inh_perstr(BoxIterator<cufftDoubleComplex> ux, BoxIterator<cufftDoubleComplex> uy,
                BoxIterator<cufftDoubleComplex> uz,
                BoxIterator<cufftDoubleComplex> s0, BoxIterator<cufftDoubleComplex> s0End,
                BoxIterator<cufftDoubleComplex> s1, BoxIterator<cufftDoubleComplex> s2,
                BoxIterator<cufftDoubleComplex> s3, BoxIterator<cufftDoubleComplex> s4,
                BoxIterator<cufftDoubleComplex> s5)
{
    const long tid = threadIdx.x + (long)blockIdx.x*blockDim.x;
    ux += tid; uy += tid; uz += tid;
    s0 += tid; s1 += tid; s2 += tid; s3 += tid; s4 += tid; s5 += tid;
    double nk[3];
    if (s0 < s0End)
    {
        const cufftDoubleComplex vx = *ux, vy = *uy, vz = *uz;
        inh_get_k(&nk[0], &nk[1], &nk[2], s0);

        *s0 = { -vx.y*nk[0], vx.x*nk[0] };                                    /* XX */
        *s1 = { -vy.y*nk[1], vy.x*nk[1] };                                    /* YY */
        *s2 = { -vz.y*nk[2], vz.x*nk[2] };                                    /* ZZ */
        *s3 = { -0.5*(vy.y*nk[2] + vz.y*nk[1]), 0.5*(vy.x*nk[2] + vz.x*nk[1]) }; /* YZ */
        *s4 = { -0.5*(vx.y*nk[2] + vz.y*nk[0]), 0.5*(vx.x*nk[2] + vz.x*nk[0]) }; /* XZ */
        *s5 = { -0.5*(vx.y*nk[1] + vy.y*nk[0]), 0.5*(vx.x*nk[1] + vy.x*nk[0]) }; /* XY */
    }
}

/* post-iFFT normalisation: scale real part, zero imaginary part       */
__global__
void inh_normalize3(BoxIterator<cufftDoubleComplex> a, BoxIterator<cufftDoubleComplex> aEnd,
                    BoxIterator<cufftDoubleComplex> b, BoxIterator<cufftDoubleComplex> c,
                    double sizescale)
{
    const long tid = threadIdx.x + (long)blockIdx.x*blockDim.x;
    a += tid; b += tid; c += tid;
    if (a < aEnd)
    {
        *a = { a->x * sizescale, 0.0 };
        *b = { b->x * sizescale, 0.0 };
        *c = { c->x * sizescale, 0.0 };
    }
}

__global__
void inh_normalize6(BoxIterator<cufftDoubleComplex> s0, BoxIterator<cufftDoubleComplex> s0End,
                    BoxIterator<cufftDoubleComplex> s1, BoxIterator<cufftDoubleComplex> s2,
                    BoxIterator<cufftDoubleComplex> s3, BoxIterator<cufftDoubleComplex> s4,
                    BoxIterator<cufftDoubleComplex> s5, double sizescale)
{
    const long tid = threadIdx.x + (long)blockIdx.x*blockDim.x;
    s0 += tid; s1 += tid; s2 += tid; s3 += tid; s4 += tid; s5 += tid;
    if (s0 < s0End)
    {
        *s0 = { s0->x * sizescale, 0.0 };
        *s1 = { s1->x * sizescale, 0.0 };
        *s2 = { s2->x * sizescale, 0.0 };
        *s3 = { s3->x * sizescale, 0.0 };
        *s4 = { s4->x * sizescale, 0.0 };
        *s5 = { s5->x * sizescale, 0.0 };
    }
}

/* local stiffness fields C11(r), C12(r), C44(r) for the volume average */
__global__
void inh_Ctotal(double *C11, double *C12, double *C44,
                const double* __restrict__ hphisum,
                const double* __restrict__ Chom, const double* __restrict__ Chet,
                long n)
{
    const long tid = threadIdx.x + (long)blockIdx.x*blockDim.x;
    if (tid < n)
    {
        const double t = 2.0*hphisum[tid] - 1.0;
        C11[tid] = Chom[0] + Chet[0]*t;
        C12[tid] = Chom[1] + Chet[1]*t;
        C44[tid] = Chom[2] + Chet[2]*t;
    }
}

/* average compliance from average stiffness (scalar work, on device)   */
__global__
void inh_Sij(double *Cavg, double *S_d, double sizescale, int model2d)
{
    Cavg[0] *= sizescale; Cavg[1] *= sizescale; Cavg[2] *= sizescale;

    inhom_average_compliance(S_d, Cavg, model2d);
}

/* All three diagonal eigenstress averages, including plane-strain zz. */
__global__
void inh_eigsts_avg_diag(double *s0, double *s1, double *s2, const double *hphi,
                         const double *ch, const double *dc, double e0,
                         double ratio_y, double ratio_z, long n)
{
    const long i = threadIdx.x + (long)blockIdx.x*blockDim.x;
    if (i < n) {
        const double h=hphi[i], c11=ch[0]+dc[0]*(2*h-1), c12=ch[1]+dc[1]*(2*h-1);
        const double ex=e0*h, ey=ex*ratio_y, ez=ex*ratio_z;
        s0[i]=c11*ex+c12*(ey+ez);
        s1[i]=c11*ey+c12*(ex+ez);
        s2[i]=c11*ez+c12*(ex+ey);
    }
}

/* periodic stress (normal / shear) with LOCAL stiffness, real space    */
__global__
void inh_persts_N(double *p11, double *p22, double *p33,
                  BoxIterator<cufftDoubleComplex> s0, BoxIterator<cufftDoubleComplex> s0End,
                  BoxIterator<cufftDoubleComplex> s1, BoxIterator<cufftDoubleComplex> s2,
                  const double* __restrict__ hphisum,
                  const double* __restrict__ Chom, const double* __restrict__ Chet)
{
    const long tid = threadIdx.x + (long)blockIdx.x*blockDim.x;
    s0 += tid; s1 += tid; s2 += tid;
    if (s0 < s0End)
    {
        const double t = 2.0*hphisum[tid] - 1.0;
        const double Ct11 = Chom[0] + Chet[0]*t;
        const double Ct12 = Chom[1] + Chet[1]*t;

        p11[tid] = fma(Ct11, s0->x, Ct12*(s1->x + s2->x));
        p22[tid] = fma(Ct11, s1->x, Ct12*(s0->x + s2->x));
        p33[tid] = fma(Ct11, s2->x, Ct12*(s0->x + s1->x));
    }
}

__global__
void inh_persts_S(double *p23, double *p13, double *p12,
                  BoxIterator<cufftDoubleComplex> s3, BoxIterator<cufftDoubleComplex> s3End,
                  BoxIterator<cufftDoubleComplex> s4, BoxIterator<cufftDoubleComplex> s5,
                  const double* __restrict__ hphisum,
                  const double* __restrict__ Chom, const double* __restrict__ Chet)
{
    const long tid = threadIdx.x + (long)blockIdx.x*blockDim.x;
    s3 += tid; s4 += tid; s5 += tid;
    if (s3 < s3End)
    {
        const double t = 2.0*hphisum[tid] - 1.0;
        const double Ct44 = Chom[2] + Chet[2]*t;

        p23[tid] = 2.0*Ct44*(s3->x);
        p13[tid] = 2.0*Ct44*(s4->x);
        p12[tid] = 2.0*Ct44*(s5->x);
    }
}

/* homogeneous (macroscopic) strain, initial version: scales avgeigsts  */
__global__
void inh_homstr(double *hom_strain_v, const double *S_d, const double *sigappl,
                double *avgeigsts, double *avgpersts,
                double sizescale, int model2d)
{
    avgpersts[0] *= sizescale; avgpersts[1] *= sizescale; avgpersts[2] *= sizescale;
    avgpersts[3] *= sizescale; avgpersts[4] *= sizescale; avgpersts[5] *= sizescale;

    avgeigsts[0] *= sizescale;
    avgeigsts[1] *= sizescale;
    avgeigsts[2] *= sizescale;

    const double S11 = S_d[0], S12 = S_d[1], S44 = S_d[2];

    const double t0 = sigappl[0] - avgpersts[0] + avgeigsts[0];
    const double t1 = sigappl[1] - avgpersts[1] + avgeigsts[1];
    const double t2 = sigappl[2] - avgpersts[2] + avgeigsts[2];
    const double t3 = sigappl[3] - avgpersts[3];
    const double t4 = sigappl[4] - avgpersts[4];
    const double t5 = sigappl[5] - avgpersts[5];

    hom_strain_v[0] = S11*t0 + S12*t1 + (model2d == 2 ? 0.0 : S12*t2);
    hom_strain_v[1] = S12*t0 + S11*t1 + (model2d == 2 ? 0.0 : S12*t2);
    hom_strain_v[2] = model2d == 2 ? 0.0 : S12*t0 + S12*t1 + S11*t2;
    hom_strain_v[3] = model2d ? 0.0 : 0.5*S44*t3;
    hom_strain_v[4] = model2d ? 0.0 : 0.5*S44*t4;
    hom_strain_v[5] = 0.5*S44*t5;
}

/* in-loop version: avgeigsts already scaled, only avgpersts is fresh   */
__global__
void inh_homstr2(double *hom_strain_v, const double *S_d, const double *sigappl,
                 const double *avgeigsts, double *avgpersts, double sizescale, int model2d)
{
    avgpersts[0] *= sizescale; avgpersts[1] *= sizescale; avgpersts[2] *= sizescale;
    avgpersts[3] *= sizescale; avgpersts[4] *= sizescale; avgpersts[5] *= sizescale;

    const double S11 = S_d[0], S12 = S_d[1], S44 = S_d[2];

    const double t0 = sigappl[0] + avgeigsts[0] - avgpersts[0];
    const double t1 = sigappl[1] + avgeigsts[1] - avgpersts[1];
    const double t2 = sigappl[2] + avgeigsts[2] - avgpersts[2];
    const double t3 = sigappl[3] - avgpersts[3];
    const double t4 = sigappl[4] - avgpersts[4];
    const double t5 = sigappl[5] - avgpersts[5];

    hom_strain_v[0] = S11*t0 + S12*t1 + (model2d == 2 ? 0.0 : S12*t2);
    hom_strain_v[1] = S12*t0 + S11*t1 + (model2d == 2 ? 0.0 : S12*t2);
    hom_strain_v[2] = model2d == 2 ? 0.0 : S12*t0 + S12*t1 + S11*t2;
    hom_strain_v[3] = model2d ? 0.0 : 0.5*S44*t3;
    hom_strain_v[4] = model2d ? 0.0 : 0.5*S44*t4;
    hom_strain_v[5] = 0.5*S44*t5;
}

/* effective source tau = Ct:(eps0 - Ebar) - DeltaC:eps_per (real space) */
__global__
void inh_ts(BoxIterator<cufftDoubleComplex> s0, BoxIterator<cufftDoubleComplex> s0End,
            BoxIterator<cufftDoubleComplex> s1, BoxIterator<cufftDoubleComplex> s2,
            BoxIterator<cufftDoubleComplex> s3, BoxIterator<cufftDoubleComplex> s4,
            BoxIterator<cufftDoubleComplex> s5,
            const double* __restrict__ hom_strain_v,
            const double* __restrict__ hphisum,
            const double* __restrict__ Chom, const double* __restrict__ Chet,
            double epszero, double misfit_aniso, double misfit_z)
{
    const long tid = threadIdx.x + (long)blockIdx.x*blockDim.x;
    s0 += tid; s1 += tid; s2 += tid; s3 += tid; s4 += tid; s5 += tid;
    if (s0 < s0End)
    {
        const double hsum  = hphisum[tid];
        const double t     = 2.0*hsum - 1.0;
        const double eig11 = epszero*hsum;
        const double eig22 = eig11*misfit_aniso;
        const double eig33 = eig11*misfit_z;

        const double Ct0 = Chom[0] + Chet[0]*t;    /* C11(r) */
        const double Ct1 = Chom[1] + Chet[1]*t;    /* C12(r) */
        const double Ct2 = Chom[2] + Chet[2]*t;    /* C44(r) */
        const double Cin0 = Ct0 - Chom[0];
        const double Cin1 = Ct1 - Chom[1];
        const double Cin2 = Ct2 - Chom[2];

        const double h0 = hom_strain_v[0], h1 = hom_strain_v[1], h2 = hom_strain_v[2];
        const double h3 = hom_strain_v[3], h4 = hom_strain_v[4], h5 = hom_strain_v[5];

        const double e0 = s0->x, e1 = s1->x, e2 = s2->x;
        const double e3 = s3->x, e4 = s4->x, e5 = s5->x;

        *s0 = { ( Ct0*(eig11-h0) - Cin0*e0
                + Ct1*(eig22-h1) - Cin1*e1
                + Ct1*(eig33-h2) - Cin1*e2 ), 0.0 };

        *s1 = { ( Ct1*(eig11-h0) - Cin1*e0
                + Ct0*(eig22-h1) - Cin0*e1
                + Ct1*(eig33-h2) - Cin1*e2 ), 0.0 };

        *s2 = { ( Ct1*(eig11-h0) - Cin1*e0
                + Ct1*(eig22-h1) - Cin1*e1
                + Ct0*(eig33-h2) - Cin0*e2 ), 0.0 };

        *s3 = { 2.0*(Ct2*(-h3) - Cin2*e3), 0.0 };
        *s4 = { 2.0*(Ct2*(-h4) - Cin2*e4), 0.0 };
        *s5 = { 2.0*(Ct2*(-h5) - Cin2*e5), 0.0 };
    }
}

/* new displacement u_new = -i Omega(Chom) (tau . k) (SHUFFLED layout)  */
__global__
void inh_update_disp(BoxIterator<cufftDoubleComplex> ux, BoxIterator<cufftDoubleComplex> uxEnd,
                     BoxIterator<cufftDoubleComplex> uy, BoxIterator<cufftDoubleComplex> uz,
                     BoxIterator<cufftDoubleComplex> s0, BoxIterator<cufftDoubleComplex> s1,
                     BoxIterator<cufftDoubleComplex> s2, BoxIterator<cufftDoubleComplex> s3,
                     BoxIterator<cufftDoubleComplex> s4, BoxIterator<cufftDoubleComplex> s5,
                     const double* __restrict__ Chom)
{
    const long tid = threadIdx.x + (long)blockIdx.x*blockDim.x;
    ux += tid; uy += tid; uz += tid;
    s0 += tid; s1 += tid; s2 += tid; s3 += tid; s4 += tid; s5 += tid;

    cufftDoubleComplex fk11, fk22, fk33;
    double n[3], om[6];

    if (ux < uxEnd)
    {
        inh_get_k(&n[0], &n[1], &n[2], s0);
        inh_get_omega(om, n, Chom);

        const cufftDoubleComplex t0 = *s0, t1 = *s1, t2 = *s2;
        const cufftDoubleComplex t3 = *s3, t4 = *s4, t5 = *s5;

        fk11.x = t0.x*n[0] + t5.x*n[1] + t4.x*n[2];
        fk22.x = t5.x*n[0] + t1.x*n[1] + t3.x*n[2];
        fk33.x = t4.x*n[0] + t3.x*n[1] + t2.x*n[2];
        fk11.y = t0.y*n[0] + t5.y*n[1] + t4.y*n[2];
        fk22.y = t5.y*n[0] + t1.y*n[1] + t3.y*n[2];
        fk33.y = t4.y*n[0] + t3.y*n[1] + t2.y*n[2];

        *ux = {  (om[0]*fk11.y + om[5]*fk22.y + om[4]*fk33.y),
                -(om[0]*fk11.x + om[5]*fk22.x + om[4]*fk33.x) };
        *uy = {  (om[5]*fk11.y + om[1]*fk22.y + om[3]*fk33.y),
                -(om[5]*fk11.x + om[1]*fk22.x + om[3]*fk33.x) };
        *uz = {  (om[4]*fk11.y + om[3]*fk22.y + om[2]*fk33.y),
                -(om[4]*fk11.x + om[3]*fk22.x + om[2]*fk33.x) };
    }
}

/* residual field(s) for the convergence norm                           */
__global__
void inh_diff_disp(BoxIterator<cufftDoubleComplex> nx_, BoxIterator<cufftDoubleComplex> nxEnd,
                   BoxIterator<cufftDoubleComplex> ny_, BoxIterator<cufftDoubleComplex> nz_,
                   BoxIterator<cufftDoubleComplex> ox, BoxIterator<cufftDoubleComplex> oy,
                   BoxIterator<cufftDoubleComplex> oz,
                   double *d1, double *d2, double *d3, int norm_calc)
{
    const long tid = threadIdx.x + (long)blockIdx.x*blockDim.x;
    nx_ += tid; ny_ += tid; nz_ += tid;
    ox  += tid; oy  += tid; oz  += tid;
    if (nx_ < nxEnd)
    {
        const double ddx = nx_->x - ox->x;
        const double ddy = ny_->x - oy->x;
        const double ddz = nz_->x - oz->x;
        if (norm_calc == 0)
            d1[tid] = ddx*ddx + ddy*ddy + ddz*ddz;
        else
        {
            d1[tid] = fabs(ddx);
            d2[tid] = fabs(ddy);
            d3[tid] = fabs(ddz);
        }
    }
}

__global__
void inh_copy_new(BoxIterator<cufftDoubleComplex> nx_, BoxIterator<cufftDoubleComplex> nxEnd,
                  BoxIterator<cufftDoubleComplex> ny_, BoxIterator<cufftDoubleComplex> nz_,
                  BoxIterator<cufftDoubleComplex> ox, BoxIterator<cufftDoubleComplex> oy,
                  BoxIterator<cufftDoubleComplex> oz)
{
    const long tid = threadIdx.x + (long)blockIdx.x*blockDim.x;
    nx_ += tid; ny_ += tid; nz_ += tid;
    ox  += tid; oy  += tid; oz  += tid;
    if (nx_ < nxEnd)
    {
        *ox = *nx_;
        *oy = *ny_;
        *oz = *nz_;
    }
}

/* scalar elastic driving force df_el (real space); the bridge kernel   */
/* distributes it as dfdphi_b += dh_precip/dphi_b * df_el               */
__global__
void inh_dfelas(double *dfelas,
                BoxIterator<cufftDoubleComplex> s0, BoxIterator<cufftDoubleComplex> s0End,
                BoxIterator<cufftDoubleComplex> s1, BoxIterator<cufftDoubleComplex> s2,
                BoxIterator<cufftDoubleComplex> s3, BoxIterator<cufftDoubleComplex> s4,
                BoxIterator<cufftDoubleComplex> s5,
                const double* __restrict__ hom_strain_v,
                const double* __restrict__ hphisum,
                const double* __restrict__ Chom, const double* __restrict__ Chet,
                double epszero, double misfit_aniso, double misfit_z)
{
    const long tid = threadIdx.x + (long)blockIdx.x*blockDim.x;
    s0 += tid; s1 += tid; s2 += tid; s3 += tid; s4 += tid; s5 += tid;
    if (s0 < s0End)
    {

        const double eigen[3] = {epszero, epszero*misfit_aniso, epszero*misfit_z};
        const double h = hphisum[tid];
        const double e[6] = {hom_strain_v[0] + s0->x - eigen[0]*h,
                             hom_strain_v[1] + s1->x - eigen[1]*h,
                             hom_strain_v[2] + s2->x - eigen[2]*h,
                             hom_strain_v[3] + s3->x,
                             hom_strain_v[4] + s4->x,
                             hom_strain_v[5] + s5->x};
        dfelas[tid] = inhom_elastic_partial(e, eigen, Chom, Chet, h);
    }
}

/* ------------------------------------------------------------------ */
/* host-side helpers                                                   */
/* ------------------------------------------------------------------ */

static inline long launch1D(long n) { return (n + 255) / 256; }

#define ITERS(fmt, desc)                                                        \
    BoxIterators(fmt, CUFFT_Z2Z, subdomain.rank, subdomain.size,                \
                 (size_t)simDomain.MESH_X, (size_t)simDomain.MESH_Y,            \
                 (size_t)simDomain.MESH_Z,                                      \
                 (cufftDoubleComplex*)(desc)->descriptor->data[0])

/* ------------------------------------------------------------------ */
/* init                                                                */
/* ------------------------------------------------------------------ */
void elasticInhomInit(InhomElasticCtx *ctx, cufftHandle plan,
                      domainInfo simDomain, controls simControls,
                      simParameters simParams, subdomainInfo subdomain,
                      MPI_Comm comm)
{
    ctx->rank = subdomain.rank;

    if (simDomain.DIMENSION != 3 && simDomain.DIMENSION != 2)
    {
        if (ctx->rank == 0)
            fprintf(stderr, "ELASTICITY_INHOM: only DIMENSION = 2 or 3 is supported.\n");
        MPI_Abort(comm, EXIT_FAILURE);
    }

    ctx->loading = simControls.inhomLoading;
    ctx->model2d = simDomain.DIMENSION == 2 ? simControls.inhom2DMode : 0;
    if (ctx->model2d && (simDomain.MESH_Z != 1 ||
                        (ctx->model2d != 1 && ctx->model2d != 2)))
    {
        if (ctx->rank == 0) fprintf(stderr, "2D inhomogeneous elasticity requires MESH_Z=1 and INHOM_2D_MODE=1 or 2.\n");
        MPI_Abort(comm, EXIT_FAILURE);
    }
    for (int face = 0; face < 2*simDomain.DIMENSION; ++face)
    {
        if (simControls.boundary[face][0].type != 3 ||
            simControls.boundary[face][2].type != 3 ||
            (simControls.FUNCTION_F == 2 && simControls.boundary[face][1].type != 3))
        {
            if (ctx->rank == 0)
                fprintf(stderr, "ELASTICITY_INHOM requires periodic phi and composition boundaries on all active faces (and mu for FUNCTION_F=2).\n");
            MPI_Abort(comm, EXIT_FAILURE);
        }
    }

    if (simDomain.numPhases != 2)
    {
        if (ctx->rank == 0)
            fprintf(stderr, "ELASTICITY_INHOM: this solver currently supports "
                            "exactly NUMPHASES = 2 (found %ld).\n", simDomain.numPhases);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    /* ---- identify matrix / precipitate from eigenSwitch ---- */
    long precip = -1, matrix = -1;
    for (long p = 0; p < simDomain.numPhases; p++)
    {
        if (simControls.eigenSwitch[p] == 1)
        {
            if (precip != -1)
            {
                if (ctx->rank == 0)
                    fprintf(stderr, "ELASTICITY_INHOM: exactly ONE phase may have a "
                                    "non-zero EIGEN_STRAIN (found several).\n");
                MPI_Abort(comm, EXIT_FAILURE);
            }
            precip = p;
        }
        else
            matrix = p;
    }
    if (precip == -1 || matrix == -1)
    {
        if (ctx->rank == 0)
            fprintf(stderr, "ELASTICITY_INHOM: need one phase with non-zero "
                            "EIGEN_STRAIN (precipitate) and one with zero (matrix).\n");
        MPI_Abort(comm, EXIT_FAILURE);
    }
    ctx->precipPhase = precip;
    ctx->matrixPhase = matrix;

    /* ---- eigenstrain parameters ---- */
    const symmetric_tensor es = simParams.eigen_strain[precip];
    ctx->epszero      = es.xx;
    ctx->misfit_aniso = (es.xx != 0.0) ? (es.yy / es.xx) : 1.0;
    ctx->misfit_z = (es.xx != 0.0) ? (es.zz / es.xx) : 1.0;
    if (!isfinite(es.xx) || !isfinite(es.yy) || !isfinite(es.zz) ||
        es.xx == 0.0 || es.yz != 0.0 || es.xz != 0.0 || es.xy != 0.0 ||
        (!ctx->model2d && es.zz != es.yy) || (ctx->model2d == 2 && es.zz != 0.0))
    {
        if (ctx->rank == 0)
            fprintf(stderr, "ELASTICITY_INHOM requires finite diagonal eigenstrain with xx != 0 (3D: yy == zz; intrinsic 2D: zz == 0); shear eigenstrain is unsupported.\n");
        MPI_Abort(comm, EXIT_FAILURE);
    }

    /* ---- Chom / Chet from the two phases' cubic constants ---- */
    const Stiffness_cubic Cp = simParams.Stiffness_c[precip];
    const Stiffness_cubic Cm = simParams.Stiffness_c[matrix];
    for (const Stiffness_cubic C : {Cp, Cm})
        if (!isfinite(C.C11) || !isfinite(C.C12) || !isfinite(C.C44) ||
            C.C44 <= 0.0 || C.C11-C.C12 <= 0.0 ||
            (ctx->model2d == 2 ? C.C11+C.C12 <= 0.0 : C.C11+2.0*C.C12 <= 0.0))
        {
            if (ctx->rank == 0) fprintf(stderr, "ELASTICITY_INHOM requires positive definite finite cubic stiffness.\n");
            MPI_Abort(comm, EXIT_FAILURE);
        }
    double Chom_h[3] = { 0.5*(Cp.C11 + Cm.C11), 0.5*(Cp.C12 + Cm.C12), 0.5*(Cp.C44 + Cm.C44) };
    double Chet_h[3] = { 0.5*(Cp.C11 - Cm.C11), 0.5*(Cp.C12 - Cm.C12), 0.5*(Cp.C44 - Cm.C44) };

    double sig_h[6] = { simParams.sigma_applied.xx, simParams.sigma_applied.yy,
                        simParams.sigma_applied.zz, simParams.sigma_applied.yz,
                        simParams.sigma_applied.xz, simParams.sigma_applied.xy };

    double prescribed[6]={simParams.homogeneous_strain.xx,simParams.homogeneous_strain.yy,
        simParams.homogeneous_strain.zz,simParams.homogeneous_strain.yz,
        simParams.homogeneous_strain.xz,simParams.homogeneous_strain.xy};
    bool invalid=ctx->loading<0 || ctx->loading>2;
    for(int j=0;j<6;++j){
        invalid=invalid || !isfinite(prescribed[j]);
        if(ctx->loading!=2)invalid=invalid || sig_h[j]!=0.;
        if(ctx->loading!=1)invalid=invalid || prescribed[j]!=0.;
    }
    if(ctx->model2d){
        invalid=invalid || sig_h[3]!=0. || sig_h[4]!=0. || prescribed[3]!=0. || prescribed[4]!=0.;
        if(ctx->model2d==2)invalid=invalid || sig_h[2]!=0. || prescribed[2]!=0.;
    }
    if(invalid){
        if(ctx->rank==0)fprintf(stderr,"Inconsistent dimensional model and mean-loading inputs: prescribe stress OR homogeneous strain; intrinsic 2D excludes zz and both 2D models exclude xz/yz.\n");
        MPI_Abort(comm,EXIT_FAILURE);
    }

    /* ---- solver controls ---- */
    ctx->Tol       = simControls.inhomTol;
    ctx->nelast    = simControls.inhomMaxIter;
    ctx->norm_calc = simControls.inhomNorm;
    ctx->strict    = simControls.inhomStrict;
    if (!isfinite(ctx->misfit_aniso) || !isfinite(ctx->misfit_z) || !isfinite(ctx->Tol) || ctx->Tol <= 0 || ctx->nelast <= 0 ||
        (ctx->norm_calc != 0 && ctx->norm_calc != 1) || !ctx->strict ||
        simControls.padding != 1 || simDomain.MESH_X <= 0 || simDomain.MESH_Y <= 0 || simDomain.MESH_Z <= 0 ||
        !isfinite(simDomain.DELTA_X) || simDomain.DELTA_X <= 0 ||
        !isfinite(simDomain.DELTA_Y) || simDomain.DELTA_Y <= 0 ||
        !isfinite(simDomain.DELTA_Z) || simDomain.DELTA_Z <= 0)
    {
        if (ctx->rank == 0) fprintf(stderr, "Invalid inhomogeneous solver controls, spacing or padding; INHOM_STRICT must be 1.\n");
        MPI_Abort(comm, EXIT_FAILURE);
    }
    for (double value : sig_h)
        if (!isfinite(value)) MPI_Abort(comm, EXIT_FAILURE);
    ctx->firstCall = 1;
    ctx->num_items = (long)subdomain.numX * simDomain.MESH_Y * simDomain.MESH_Z;

    if (ctx->num_items <= 0 || ctx->num_items > INT_MAX)
    {
        if (ctx->rank == 0) fprintf(stderr, "ELASTICITY_INHOM requires a nonempty slab fitting the cuBLAS integer count.\n");
        MPI_Abort(comm, EXIT_FAILURE);
    }

    /* ---- device constants ---- */
    const int    nxc = (int)simDomain.MESH_X, nyc = (int)simDomain.MESH_Y, nzc = (int)simDomain.MESH_Z;
    const double dxc = simDomain.DELTA_X, dyc = simDomain.DELTA_Y, dzc = simDomain.DELTA_Z;
    CUDA_CHECK(cudaMemcpyToSymbol(inh_nx, &nxc, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(inh_ny, &nyc, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(inh_nz, &nzc, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(inh_dx, &dxc, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(inh_dy, &dyc, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(inh_dz, &dzc, sizeof(double)));

    /* ---- distributed fields ---- */
    CUFFT_CHECK(cufftXtMalloc(plan, &ctx->ux,    CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan, &ctx->uy,    CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan, &ctx->uz,    CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan, &ctx->unewx, CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan, &ctx->unewy, CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan, &ctx->unewz, CUFFT_XT_FORMAT_INPLACE));
    for (int v = 0; v < 6; v++)
        CUFFT_CHECK(cufftXtMalloc(plan, &ctx->str_v[v], CUFFT_XT_FORMAT_INPLACE));

    /* ---- slab-sized double arrays ---- */
    const size_t nb = (size_t)ctx->num_items * sizeof(double);
    CUDA_CHECK(cudaMalloc((void**)&ctx->hphisum, nb));
    CUDA_CHECK(cudaMalloc((void**)&ctx->dfelas,  nb));
    CUDA_CHECK(cudaMalloc((void**)&ctx->scr1,    nb));
    CUDA_CHECK(cudaMalloc((void**)&ctx->scr2,    nb));
    CUDA_CHECK(cudaMalloc((void**)&ctx->scr3,    nb));
    CUDA_CHECK(cudaMemset(ctx->hphisum, 0, nb));
    CUDA_CHECK(cudaMemset(ctx->dfelas,  0, nb));

    /* ---- small device arrays ---- */
    CUDA_CHECK(cudaMalloc((void**)&ctx->Chom_d,       3*sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&ctx->Chet_d,       3*sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&ctx->Cavg,         3*sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&ctx->S_d,          3*sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&ctx->avgeigsts,    3*sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&ctx->avgpersts,    6*sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&ctx->hom_strain_v, 6*sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&ctx->sigappl_v,    6*sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&ctx->disperr,      3*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(ctx->Chom_d,    Chom_h, 3*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->Chet_d,    Chet_h, 3*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->sigappl_v, sig_h,  6*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->hom_strain_v, prescribed, 6*sizeof(double), cudaMemcpyHostToDevice));

    /* ---- CUB temp storage (max of Sum and Max requirements) ---- */
    size_t bytesSum = 0, bytesMax = 0;
    cub::DeviceReduce::Sum((void*)nullptr, bytesSum, ctx->scr1, ctx->Cavg, ctx->num_items);
    cub::DeviceReduce::Max((void*)nullptr, bytesMax, ctx->scr1, ctx->disperr, ctx->num_items);
    ctx->cubTempBytes = (bytesSum > bytesMax) ? bytesSum : bytesMax;
    CUDA_CHECK(cudaMalloc(&ctx->cubTemp, ctx->cubTempBytes));

    /* ---- cuBLAS ---- */
    if (cublasCreate(&ctx->blas) != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "ELASTICITY_INHOM: cublasCreate failed\n");
        MPI_Abort(comm, EXIT_FAILURE);
    }

    /* ---- NCCL bootstrap over MPI ---- */
    int size;
    MPI_Comm_size(comm, &size);
    ncclUniqueId id;
    if (ctx->rank == 0)
        NCCL_CHECK(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm);
    NCCL_CHECK(ncclCommInitRank(&ctx->nccl, size, id, ctx->rank));

    if (ctx->rank == 0)
    {
        if (ctx->model2d)
            printf("ELASTICITY_INHOM: 2D %s, MESH_Z=1; total out-of-plane strain is zero.\n",
                   ctx->model2d == 1 ? "generalized plane strain" : "intrinsic");
        printf("ELASTICITY_INHOM: matrix phase %ld, precipitate phase %ld\n",
               ctx->matrixPhase, ctx->precipPhase);
        printf("ELASTICITY_INHOM: eps0 = %le, ratios yy/xx = %le, zz/xx = %le\n", ctx->epszero, ctx->misfit_aniso, ctx->misfit_z);
        printf("ELASTICITY_INHOM: Chom = {%le, %le, %le}, Chet = {%le, %le, %le}\n",
               Chom_h[0], Chom_h[1], Chom_h[2], Chet_h[0], Chet_h[1], Chet_h[2]);
        printf("ELASTICITY_INHOM: Tol = %le, max iters = %ld, norm = %s\n",
               ctx->Tol, ctx->nelast, ctx->norm_calc == 0 ? "L2" : "max");
    }
}

/* ------------------------------------------------------------------ */
/* one solve + driving force                                           */
/* ------------------------------------------------------------------ */
void computeDrivingForce_ElasticInhom(double **phi, double **dfdphi,
                                      InhomElasticCtx *ctx, cufftHandle plan,
                                      domainInfo simDomain, controls simControls,
                                      simParameters simParams, subdomainInfo subdomain,
                                      cudaStream_t stream,
                                      dim3 gridSize, dim3 blockSize,
                                      MPI_Comm comm)
{
    const long   N        = ctx->num_items;
    const double szTotal  = (double)simDomain.MESH_X * simDomain.MESH_Y * simDomain.MESH_Z;
    const double invTotal = 1.0 / szTotal;

    /* iterator views: r = real-space (INPLACE), f = Fourier (SHUFFLED) */
    auto [uxR,  uxRe]  = ITERS(CUFFT_XT_FORMAT_INPLACE,          ctx->ux);
    auto [uyR,  uyRe]  = ITERS(CUFFT_XT_FORMAT_INPLACE,          ctx->uy);
    auto [uzR,  uzRe]  = ITERS(CUFFT_XT_FORMAT_INPLACE,          ctx->uz);
    auto [uxF,  uxFe]  = ITERS(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, ctx->ux);
    auto [uyF,  uyFe]  = ITERS(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, ctx->uy);
    auto [uzF,  uzFe]  = ITERS(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, ctx->uz);
    auto [nxR,  nxRe]  = ITERS(CUFFT_XT_FORMAT_INPLACE,          ctx->unewx);
    auto [nyR,  nyRe]  = ITERS(CUFFT_XT_FORMAT_INPLACE,          ctx->unewy);
    auto [nzR,  nzRe]  = ITERS(CUFFT_XT_FORMAT_INPLACE,          ctx->unewz);
    auto [nxF,  nxFe]  = ITERS(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, ctx->unewx);
    auto [nyF,  nyFe]  = ITERS(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, ctx->unewy);
    auto [nzF,  nzFe]  = ITERS(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, ctx->unewz);

    /* BoxIterator has no default constructor: build pair arrays directly */
    using BIPair = std::pair<BoxIterator<cufftDoubleComplex>, BoxIterator<cufftDoubleComplex>>;
    BIPair sRp[6] = { ITERS(CUFFT_XT_FORMAT_INPLACE, ctx->str_v[0]),
                      ITERS(CUFFT_XT_FORMAT_INPLACE, ctx->str_v[1]),
                      ITERS(CUFFT_XT_FORMAT_INPLACE, ctx->str_v[2]),
                      ITERS(CUFFT_XT_FORMAT_INPLACE, ctx->str_v[3]),
                      ITERS(CUFFT_XT_FORMAT_INPLACE, ctx->str_v[4]),
                      ITERS(CUFFT_XT_FORMAT_INPLACE, ctx->str_v[5]) };
    BIPair sFp[6] = { ITERS(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, ctx->str_v[0]),
                      ITERS(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, ctx->str_v[1]),
                      ITERS(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, ctx->str_v[2]),
                      ITERS(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, ctx->str_v[3]),
                      ITERS(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, ctx->str_v[4]),
                      ITERS(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, ctx->str_v[5]) };
    #define sR(v)  (sRp[v].first)
    #define sRe(v) (sRp[v].second)
    #define sF(v)  (sFp[v].first)
    #define sFe(v) (sFp[v].second)

    const long nbR = launch1D(std::distance(uxR, uxRe));   /* real-space box   */
    const long nbF = launch1D(std::distance(uxF, uxFe));   /* Fourier-space box */
    const long nbN = launch1D(N);                          /* plain double arrays */

    /* ---- 1. bridge: h(phi_precip) from the FD arrays ---- */
    inh_bridge_hphi<<<gridSize, blockSize, 0, stream>>>(
        phi, ctx->hphisum, ctx->precipPhase, simDomain.numPhases,
        subdomain.xStep, subdomain.yStep,
        subdomain.sizeX, subdomain.sizeY, subdomain.sizeZ);

    /* ---- 2. first call: zeroth-order (homogeneous) seed ---- */
    if (ctx->firstCall)
    {
        inh_eigsts_hom<<<nbR, 256, 0, stream>>>(uxR, uxRe, uyR, uzR,
                                                ctx->hphisum, ctx->Chom_d,
                                                ctx->epszero, ctx->misfit_aniso, ctx->misfit_z);

        ctx->ux->subFormat = CUFFT_XT_FORMAT_INPLACE;
        ctx->uy->subFormat = CUFFT_XT_FORMAT_INPLACE;
        ctx->uz->subFormat = CUFFT_XT_FORMAT_INPLACE;
        CUFFT_CHECK(cufftXtExecDescriptor(plan, ctx->ux, ctx->ux, CUFFT_FORWARD));
        CUFFT_CHECK(cufftXtExecDescriptor(plan, ctx->uy, ctx->uy, CUFFT_FORWARD));
        CUFFT_CHECK(cufftXtExecDescriptor(plan, ctx->uz, ctx->uz, CUFFT_FORWARD));

        inh_uzero<<<nbF, 256, 0, stream>>>(uxF, uxFe, uyF, uzF, ctx->Chom_d);

        /* periodic strain of the seed, then everything back to real space */
        inh_perstr<<<nbF, 256, 0, stream>>>(uxF, uyF, uzF,
                                            sF(0), sFe(0), sF(1), sF(2),
                                            sF(3), sF(4), sF(5));

        for (int v = 0; v < 6; v++)
        {
            ctx->str_v[v]->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
            CUFFT_CHECK(cufftXtExecDescriptor(plan, ctx->str_v[v], ctx->str_v[v], CUFFT_INVERSE));
        }
        CUFFT_CHECK(cufftXtExecDescriptor(plan, ctx->ux, ctx->ux, CUFFT_INVERSE));
        CUFFT_CHECK(cufftXtExecDescriptor(plan, ctx->uy, ctx->uy, CUFFT_INVERSE));
        CUFFT_CHECK(cufftXtExecDescriptor(plan, ctx->uz, ctx->uz, CUFFT_INVERSE));

        inh_normalize6<<<nbR, 256, 0, stream>>>(sR(0), sRe(0), sR(1), sR(2),
                                                sR(3), sR(4), sR(5), invTotal);
        inh_normalize3<<<nbR, 256, 0, stream>>>(uxR, uxRe, uyR, uzR, invTotal);

        ctx->firstCall = 0;
    }

    /* ---- 3. volume averages: <C>, <sigma*>, <S> ---- */
    inh_Ctotal<<<nbN, 256, 0, stream>>>(ctx->scr1, ctx->scr2, ctx->scr3,
                                        ctx->hphisum, ctx->Chom_d, ctx->Chet_d, N);
    cub::DeviceReduce::Sum(ctx->cubTemp, ctx->cubTempBytes, ctx->scr1, &ctx->Cavg[0], N, stream);
    cub::DeviceReduce::Sum(ctx->cubTemp, ctx->cubTempBytes, ctx->scr2, &ctx->Cavg[1], N, stream);
    cub::DeviceReduce::Sum(ctx->cubTemp, ctx->cubTempBytes, ctx->scr3, &ctx->Cavg[2], N, stream);
    NCCL_CHECK(ncclAllReduce(ctx->Cavg, ctx->Cavg, 3, ncclDouble, ncclSum, ctx->nccl, stream));

    inh_eigsts_avg_diag<<<nbN, 256, 0, stream>>>(ctx->scr1, ctx->scr2, ctx->scr3,
        ctx->hphisum, ctx->Chom_d, ctx->Chet_d, ctx->epszero,
        ctx->misfit_aniso, ctx->misfit_z, N);
    cub::DeviceReduce::Sum(ctx->cubTemp, ctx->cubTempBytes, ctx->scr1, &ctx->avgeigsts[0], N, stream);
    cub::DeviceReduce::Sum(ctx->cubTemp, ctx->cubTempBytes, ctx->scr2, &ctx->avgeigsts[1], N, stream);
    cub::DeviceReduce::Sum(ctx->cubTemp, ctx->cubTempBytes, ctx->scr3, &ctx->avgeigsts[2], N, stream);
    NCCL_CHECK(ncclAllReduce(ctx->avgeigsts, ctx->avgeigsts, 3, ncclDouble, ncclSum, ctx->nccl, stream));

    inh_Sij<<<1, 1, 0, stream>>>(ctx->Cavg, ctx->S_d, invTotal, ctx->model2d);

    /* ---- 4. initial <dsigma> and homogeneous strain ---- */
    inh_persts_N<<<nbR, 256, 0, stream>>>(ctx->scr1, ctx->scr2, ctx->scr3,
                                          sR(0), sRe(0), sR(1), sR(2),
                                          ctx->hphisum, ctx->Chom_d, ctx->Chet_d);
    cub::DeviceReduce::Sum(ctx->cubTemp, ctx->cubTempBytes, ctx->scr1, &ctx->avgpersts[0], N, stream);
    cub::DeviceReduce::Sum(ctx->cubTemp, ctx->cubTempBytes, ctx->scr2, &ctx->avgpersts[1], N, stream);
    cub::DeviceReduce::Sum(ctx->cubTemp, ctx->cubTempBytes, ctx->scr3, &ctx->avgpersts[2], N, stream);
    inh_persts_S<<<nbR, 256, 0, stream>>>(ctx->scr1, ctx->scr2, ctx->scr3,
                                          sR(3), sRe(3), sR(4), sR(5),
                                          ctx->hphisum, ctx->Chom_d, ctx->Chet_d);
    cub::DeviceReduce::Sum(ctx->cubTemp, ctx->cubTempBytes, ctx->scr1, &ctx->avgpersts[3], N, stream);
    cub::DeviceReduce::Sum(ctx->cubTemp, ctx->cubTempBytes, ctx->scr2, &ctx->avgpersts[4], N, stream);
    cub::DeviceReduce::Sum(ctx->cubTemp, ctx->cubTempBytes, ctx->scr3, &ctx->avgpersts[5], N, stream);
    NCCL_CHECK(ncclAllReduce(ctx->avgpersts, ctx->avgpersts, 6, ncclDouble, ncclSum, ctx->nccl, stream));

    if (ctx->loading != 1) inh_homstr<<<1, 1, 0, stream>>>(ctx->hom_strain_v, ctx->S_d, ctx->sigappl_v,
                                    ctx->avgeigsts, ctx->avgpersts, invTotal, ctx->model2d);

    /* ---- 5. fixed-point iteration ---- */
    int  converge = 1;
    long iter = 0;
    double disperror = 0.0;

    while (converge)
    {
        /* tau (overwrites strain in str_v, real space) */
        inh_ts<<<nbR, 256, 0, stream>>>(sR(0), sRe(0), sR(1), sR(2), sR(3), sR(4), sR(5),
                                        ctx->hom_strain_v, ctx->hphisum,
                                        ctx->Chom_d, ctx->Chet_d,
                                        ctx->epszero, ctx->misfit_aniso, ctx->misfit_z);

        for (int v = 0; v < 6; v++)
        {
            ctx->str_v[v]->subFormat = CUFFT_XT_FORMAT_INPLACE;
            CUFFT_CHECK(cufftXtExecDescriptor(plan, ctx->str_v[v], ctx->str_v[v], CUFFT_FORWARD));
        }

        ctx->unewx->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
        ctx->unewy->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
        ctx->unewz->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED;

        inh_update_disp<<<nbF, 256, 0, stream>>>(nxF, nxFe, nyF, nzF,
                                                 sF(0), sF(1), sF(2), sF(3), sF(4), sF(5),
                                                 ctx->Chom_d);

        inh_perstr<<<nbF, 256, 0, stream>>>(nxF, nyF, nzF,
                                            sF(0), sFe(0), sF(1), sF(2),
                                            sF(3), sF(4), sF(5));

        CUFFT_CHECK(cufftXtExecDescriptor(plan, ctx->unewx, ctx->unewx, CUFFT_INVERSE));
        CUFFT_CHECK(cufftXtExecDescriptor(plan, ctx->unewy, ctx->unewy, CUFFT_INVERSE));
        CUFFT_CHECK(cufftXtExecDescriptor(plan, ctx->unewz, ctx->unewz, CUFFT_INVERSE));
        for (int v = 0; v < 6; v++)
            CUFFT_CHECK(cufftXtExecDescriptor(plan, ctx->str_v[v], ctx->str_v[v], CUFFT_INVERSE));

        inh_normalize3<<<nbR, 256, 0, stream>>>(nxR, nxRe, nyR, nzR, invTotal);
        inh_normalize6<<<nbR, 256, 0, stream>>>(sR(0), sRe(0), sR(1), sR(2),
                                                sR(3), sR(4), sR(5), invTotal);

        /* fresh <dsigma> and homogeneous strain */
        inh_persts_N<<<nbR, 256, 0, stream>>>(ctx->scr1, ctx->scr2, ctx->scr3,
                                              sR(0), sRe(0), sR(1), sR(2),
                                              ctx->hphisum, ctx->Chom_d, ctx->Chet_d);
        cub::DeviceReduce::Sum(ctx->cubTemp, ctx->cubTempBytes, ctx->scr1, &ctx->avgpersts[0], N, stream);
        cub::DeviceReduce::Sum(ctx->cubTemp, ctx->cubTempBytes, ctx->scr2, &ctx->avgpersts[1], N, stream);
        cub::DeviceReduce::Sum(ctx->cubTemp, ctx->cubTempBytes, ctx->scr3, &ctx->avgpersts[2], N, stream);
        inh_persts_S<<<nbR, 256, 0, stream>>>(ctx->scr1, ctx->scr2, ctx->scr3,
                                              sR(3), sRe(3), sR(4), sR(5),
                                              ctx->hphisum, ctx->Chom_d, ctx->Chet_d);
        cub::DeviceReduce::Sum(ctx->cubTemp, ctx->cubTempBytes, ctx->scr1, &ctx->avgpersts[3], N, stream);
        cub::DeviceReduce::Sum(ctx->cubTemp, ctx->cubTempBytes, ctx->scr2, &ctx->avgpersts[4], N, stream);
        cub::DeviceReduce::Sum(ctx->cubTemp, ctx->cubTempBytes, ctx->scr3, &ctx->avgpersts[5], N, stream);
        NCCL_CHECK(ncclAllReduce(ctx->avgpersts, ctx->avgpersts, 6, ncclDouble, ncclSum, ctx->nccl, stream));

        if (ctx->loading != 1) inh_homstr2<<<1, 1, 0, stream>>>(ctx->hom_strain_v, ctx->S_d, ctx->sigappl_v,
                                         ctx->avgeigsts, ctx->avgpersts, invTotal, ctx->model2d);

        /* convergence norm */
        inh_diff_disp<<<nbR, 256, 0, stream>>>(nxR, nxRe, nyR, nzR, uxR, uyR, uzR,
                                               ctx->scr1, ctx->scr2, ctx->scr3,
                                               ctx->norm_calc);
        if (ctx->norm_calc == 0)
        {
            if (cublasSetStream(ctx->blas, stream) != CUBLAS_STATUS_SUCCESS ||
                cublasDasum(ctx->blas, (int)N, ctx->scr1, 1, &disperror) != CUBLAS_STATUS_SUCCESS)
                MPI_Abort(comm, EXIT_FAILURE);
            MPI_Allreduce(MPI_IN_PLACE, &disperror, 1, MPI_DOUBLE, MPI_SUM, comm);
            disperror = sqrt(disperror);
        }
        else
        {
            cub::DeviceReduce::Max(ctx->cubTemp, ctx->cubTempBytes, ctx->scr1, &ctx->disperr[0], N, stream);
            cub::DeviceReduce::Max(ctx->cubTemp, ctx->cubTempBytes, ctx->scr2, &ctx->disperr[1], N, stream);
            cub::DeviceReduce::Max(ctx->cubTemp, ctx->cubTempBytes, ctx->scr3, &ctx->disperr[2], N, stream);
            NCCL_CHECK(ncclAllReduce(ctx->disperr, ctx->disperr, 3, ncclDouble, ncclMax, ctx->nccl, stream));

            double d_h[3];
            CUDA_CHECK(cudaMemcpyAsync(d_h, ctx->disperr, 3*sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            disperror = d_h[0];
            disperror = inhom_max3(d_h[0], d_h[1], d_h[2]);
        }

        if (!isfinite(disperror)) MPI_Abort(comm, EXIT_FAILURE);
        if (disperror < ctx->Tol) converge = 0;

        /* u_old <- u_new */
        inh_copy_new<<<nbR, 256, 0, stream>>>(nxR, nxRe, nyR, nzR, uxR, uyR, uzR);

        iter++;
        if (iter >= ctx->nelast && converge)
        {
            ctx->converged = 0;
            ctx->iterations = iter;
            ctx->residual = disperror;
            if (ctx->rank == 0)
                fprintf(stderr, "ELASTICITY_INHOM failed after %ld iterations; residual %le. No force returned.\n", iter, disperror);
            MPI_Abort(comm, EXIT_FAILURE);
        }
    }
    ctx->converged = 1;
    ctx->iterations = iter;
    ctx->residual = disperror;

    if (simControls.count % simControls.trackProgress == 0 && ctx->rank == 0)
        printf("ELASTICITY_INHOM: converged in %ld iterations, residual %le\n",
               iter, disperror);

    /* ---- 6. driving force back to the phase-field solver ---- */
    inh_dfelas<<<nbR, 256, 0, stream>>>(ctx->dfelas,
                                        sR(0), sRe(0), sR(1), sR(2), sR(3), sR(4), sR(5),
                                        ctx->hom_strain_v, ctx->hphisum,
                                        ctx->Chom_d, ctx->Chet_d,
                                        ctx->epszero, ctx->misfit_aniso, ctx->misfit_z);

    inh_bridge_dfeldphi<<<gridSize, blockSize, 0, stream>>>(
        phi, dfdphi, ctx->dfelas, ctx->precipPhase, simDomain.numPhases,
        subdomain.xStep, subdomain.yStep,
        subdomain.sizeX, subdomain.sizeY, subdomain.sizeZ);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Opt-in validation output: raw h, total tensor strain and unchained dF/dh.
    // No diagnostics or host copies are performed in ordinary simulations.
    if (simControls.inhomDiagnostics)
    {
        std::vector<double> h(N), df(N);
        std::vector<cufftDoubleComplex> strain[6];
        double mean[6];
        CUDA_CHECK(cudaMemcpy(h.data(), ctx->hphisum, N*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(df.data(), ctx->dfelas, N*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mean, ctx->hom_strain_v, sizeof(mean), cudaMemcpyDeviceToHost));
        for (int v=0; v<6; ++v) {
            strain[v].resize(N);
            CUDA_CHECK(cudaMemcpy(strain[v].data(), ctx->str_v[v]->descriptor->data[0],
                                  N*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));
        }
        char name[128];
        snprintf(name, sizeof(name), "elastic_rank%d_step%ld.csv", ctx->rank, simControls.count);
        FILE *fp=fopen(name, "w");
        if (!fp) { perror(name); MPI_Abort(comm, EXIT_FAILURE); }
        fprintf(fp, "h,exx,eyy,ezz,eyz,exz,exy,dfdh\n");
        for (long i=0; i<N; ++i) {
            fprintf(fp, "%.17g", h[i]);
            for (int v=0; v<6; ++v) fprintf(fp, ",%.17g", mean[v]+strain[v][i].x);
            fprintf(fp, ",%.17g\n", df[i]);
        }
        if (fclose(fp)) { perror(name); MPI_Abort(comm, EXIT_FAILURE); }
    }

    #undef sR
    #undef sRe
    #undef sF
    #undef sFe
}

/* ------------------------------------------------------------------ */
/* teardown                                                            */
/* ------------------------------------------------------------------ */
void elasticInhomFree(InhomElasticCtx *ctx)
{
    CUFFT_CHECK(cufftXtFree(ctx->ux));
    CUFFT_CHECK(cufftXtFree(ctx->uy));
    CUFFT_CHECK(cufftXtFree(ctx->uz));
    CUFFT_CHECK(cufftXtFree(ctx->unewx));
    CUFFT_CHECK(cufftXtFree(ctx->unewy));
    CUFFT_CHECK(cufftXtFree(ctx->unewz));
    for (int v = 0; v < 6; v++)
        CUFFT_CHECK(cufftXtFree(ctx->str_v[v]));

    CUDA_CHECK(cudaFree(ctx->hphisum));
    CUDA_CHECK(cudaFree(ctx->dfelas));
    CUDA_CHECK(cudaFree(ctx->scr1));
    CUDA_CHECK(cudaFree(ctx->scr2));
    CUDA_CHECK(cudaFree(ctx->scr3));
    CUDA_CHECK(cudaFree(ctx->Chom_d));
    CUDA_CHECK(cudaFree(ctx->Chet_d));
    CUDA_CHECK(cudaFree(ctx->Cavg));
    CUDA_CHECK(cudaFree(ctx->S_d));
    CUDA_CHECK(cudaFree(ctx->avgeigsts));
    CUDA_CHECK(cudaFree(ctx->avgpersts));
    CUDA_CHECK(cudaFree(ctx->hom_strain_v));
    CUDA_CHECK(cudaFree(ctx->sigappl_v));
    CUDA_CHECK(cudaFree(ctx->disperr));
    CUDA_CHECK(cudaFree(ctx->cubTemp));

    cublasDestroy(ctx->blas);
    ncclCommDestroy(ctx->nccl);
}

#endif /* ENABLE_CUFFTMP */
