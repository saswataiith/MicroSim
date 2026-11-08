// Extracted function bodies (trailing whitespace normalized) from the user-supplied 3DKKSMP CUDA
// source, 23 July 2026. Host-only local-kernel oracle; no GPU execution.
// The scaffolding below replaces CUDA coordinates with one CPU grid point.
#pragma once
namespace legacy_cuda {
struct Dim { long x=0,y=0,z=0; };
inline Dim threadIdx, blockIdx, blockDim{1,1,1};
inline int nx_d=1, ny_d=1, nz_d=1;
struct cufftDoubleComplex { double x,y; };
template<class T> using BoxIterator=T*;
void Calc_hsum_dhdphi_sv(double* __restrict__ hphisum, double* __restrict__ dhdphi0,
                                    double*  __restrict__ ftotal, double* __restrict__ volfrac,
                                    BoxIterator<cufftDoubleComplex> phi0_begin,
                                    BoxIterator<cufftDoubleComplex> phi0_end,
                                    BoxIterator<cufftDoubleComplex> dfdphi0_begin,
                                    double omega, int engs_out, int output, int TRACK_OBS, int ny, int nz)
{
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    long j = threadIdx.y + blockIdx.y * blockDim.y;
    long k = threadIdx.z + blockIdx.z * blockDim.z;

    long idx = k + nz*(j + i*ny);

    phi0_begin += idx;
    dfdphi0_begin += idx;

    if (phi0_begin >= phi0_end)
        return;
    double p0 = phi0_begin->x;
    double p0p0 = p0*p0;

    hphisum[idx] = (p0*p0p0 * (6.0*p0p0 - 15.0*p0 + 10.0));

    dhdphi0[idx] = (30.0*p0p0 * (p0p0 - 2.0*p0 + 1.0));

    *dfdphi0_begin = {(omega*(2.0*p0 * (1.0-p0)*(1.0-2.0*p0))), 0.0};

    if ((engs_out != 0 && output == 1) || TRACK_OBS == 1)
        ftotal[idx] = omega*(p0p0 * (1.0-p0)*(1.0-p0));

    if(TRACK_OBS == 1)
    {
        volfrac[idx] = (p0 > 0.5) ? 1.0 : 0.0;
    }
}
void Calc_hphisum_dhdphi(double *hphisum, double *dhdphi0, double *dhdphi1,
                                   double *dhdphi2, double *dhdphi3, double *vol_frac,
                                  BoxIterator<cufftDoubleComplex> phi0_begin,
                                  BoxIterator<cufftDoubleComplex> phi0_end,
                                  BoxIterator<cufftDoubleComplex> phi1_begin,
                                  BoxIterator<cufftDoubleComplex> phi2_begin,
                                  BoxIterator<cufftDoubleComplex> phi3_begin,
                                  int TRACK_OBS, int ny, int nz)
{
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    long j = threadIdx.y + blockIdx.y * blockDim.y;
    long k = threadIdx.z + blockIdx.z * blockDim.z;

    long idx = k + (nz * (j + (i*ny)));
    phi0_begin += idx; phi1_begin += idx; phi2_begin += idx; phi3_begin += idx;

    if (phi0_begin < phi0_end)
    {
        double p0 = phi0_begin->x;
        double p1 = phi1_begin->x;
        double p2 = phi2_begin->x;
        double p3 = phi3_begin->x;

        double p0p0 = p0*p0;
        double p1p1 = p1*p1;
        double p2p2 = p2*p2;
        double p3p3 = p3*p3;

        hphisum[idx] = (p0*p0p0 * (6.0*p0p0 - 15.0*p0 + 10.0) +
                        p1*p1p1 * (6.0*p1p1 - 15.0*p1 + 10.0) +
                        p2*p2p2 * (6.0*p2p2 - 15.0*p2 + 10.0) +
                        p3*p3p3 * (6.0*p3p3 - 15.0*p3 + 10.0));

        dhdphi0[idx] = (30.0*p0p0 * (p0p0 - 2.0*p0 + 1.0));
        dhdphi1[idx] = (30.0*p1p1 * (p1p1 - 2.0*p1 + 1.0));
        dhdphi2[idx] = (30.0*p2p2 * (p2p2 - 2.0*p2 + 1.0));
        dhdphi3[idx] = (30.0*p3p3 * (p3p3 - 2.0*p3 + 1.0));

        if(TRACK_OBS == 1)
            vol_frac[idx] = (p0 > 0.5 || p1 > 0.5 || p2 > 0.5 || p3 > 0.5) ? 1.0 : 0.0;
    }
    //__syncthreads();

}
void Calc_dgdphi(BoxIterator<cufftDoubleComplex> dgdphi0_begin,
                            BoxIterator<cufftDoubleComplex> dgdphi0_end,
                            BoxIterator<cufftDoubleComplex> dgdphi1_begin,
                            BoxIterator<cufftDoubleComplex> dgdphi2_begin,
                            BoxIterator<cufftDoubleComplex> dgdphi3_begin,
                            BoxIterator<cufftDoubleComplex> phi0_begin,
                            BoxIterator<cufftDoubleComplex> phi1_begin,
                            BoxIterator<cufftDoubleComplex> phi2_begin,
                            BoxIterator<cufftDoubleComplex> phi3_begin,
                            double omega, double theta, double* __restrict__ ftotal,
                            int engs_out, int TRACK_OBS, int output, int ny, int nz)
{

    long i = threadIdx.x + blockIdx.x*blockDim.x;
    long j = threadIdx.y + blockIdx.y*blockDim.y;
    long k = threadIdx.z + blockIdx.z*blockDim.z;
    long idx = k + (nz * (j + (i*ny)));

    dgdphi0_begin += idx; dgdphi1_begin += idx; dgdphi2_begin += idx; dgdphi3_begin += idx;
    phi0_begin += idx; phi1_begin += idx; phi2_begin += idx; phi3_begin += idx;

    if (dgdphi0_begin < dgdphi0_end)
    {
        /*
        ////Type-1 :
        const double p0 = phi0_begin->x;
        const double p1 = phi1_begin->x;
        const double p2 = phi2_begin->x;
        const double p3 = phi3_begin->x;

        const double p0p0 = p0*p0;
        const double p1p1 = p1*p1;
        const double p2p2 = p2*p2;
        const double p3p3 = p3*p3;

        const double omg = omega;
        const double tmp = 2.0*theta;

        *dgdphi0_begin = {(omg*((2.0*p0 * (1.0-p0)*(1.0-2.0*p0)) + tmp*p0 * (p1p1 + p2p2 + p3p3))), 0.0};
        *dgdphi1_begin = {(omg*((2.0*p1 * (1.0-p1)*(1.0-2.0*p1)) + tmp*p1 * (p0p0 + p2p2 + p3p3))), 0.0};
        *dgdphi2_begin = {(omg*((2.0*p2 * (1.0-p2)*(1.0-2.0*p2)) + tmp*p2 * (p0p0 + p1p1 + p3p3))), 0.0};
        *dgdphi3_begin = {(omg*((2.0*p3 * (1.0-p3)*(1.0-2.0*p3)) + tmp*p3 * (p0p0 + p1p1 + p2p2))), 0.0};*/

        ////Type-2 : Fastest
        const double p0 = phi0_begin->x;
        const double p1 = phi1_begin->x;
        const double p2 = phi2_begin->x;
        const double p3 = phi3_begin->x;

        const double p0p0 = p0*p0;
        const double p1p1 = p1*p1;
        const double p2p2 = p2*p2;
        const double p3p3 = p3*p3;
        const double psum = p0p0 + p1p1 + p2p2 + p3p3;

        const double omg = omega;
        const double tmp = 2.0*theta;

        *dgdphi0_begin = {(omg*((2.0*p0 * (1.0-p0)*(1.0-2.0*p0)) + tmp*p0 * (psum - p0p0))), 0.0};
        *dgdphi1_begin = {(omg*((2.0*p1 * (1.0-p1)*(1.0-2.0*p1)) + tmp*p1 * (psum - p1p1))), 0.0};
        *dgdphi2_begin = {(omg*((2.0*p2 * (1.0-p2)*(1.0-2.0*p2)) + tmp*p2 * (psum - p2p2))), 0.0};
        *dgdphi3_begin = {(omg*((2.0*p3 * (1.0-p3)*(1.0-2.0*p3)) + tmp*p3 * (psum - p3p3))), 0.0};

        const double fterm = (p0p0*(1.0-p0)*(1.0-p0) + p1p1*(1.0-p1)*(1.0-p1) +
                 p2p2*(1.0-p2)*(1.0-p2) + p3p3*(1.0-p3)*(1.0-p3));

        const double sterm = (p0p0*(p1p1 +p2p2 + p3p3) + p1p1*(p2p2 + p3p3) + p2p2*p3p3);

        if ((engs_out != 0 && output == 1) || TRACK_OBS == 1)
            ftotal[idx] += omg * (fterm + theta*sterm);

        /*
        ////Type-3 :
        const double p0 = phi0_begin->x;
        const double p1 = phi1_begin->x;
        const double p2 = phi2_begin->x;
        const double p3 = phi3_begin->x;

        const double omg = omega;
        const double tmp = 2.0*theta;

        const double fterm0 = (2.0*p0 * (1.0-p0)*(1.0-2.0*p0));
        const double fterm1 = (2.0*p1 * (1.0-p1)*(1.0-2.0*p1));
        const double fterm2 = (2.0*p2 * (1.0-p2)*(1.0-2.0*p2));
        const double fterm3 = (2.0*p3 * (1.0-p3)*(1.0-2.0*p3));

        const double sterm0 = fma(p1, p1, fma(p2, p2, p3*p3));
        const double sterm1 = fma(p0, p0, fma(p2, p2, p3*p3));
        const double sterm2 = fma(p1, p1, fma(p0, p0, p3*p3));
        const double sterm3 = fma(p1, p1, fma(p2, p2, p0*p0));

        dgdphi0_begin->x = (omg*(fterm0 + tmp*p0*sterm0));
        dgdphi1_begin->x = (omg*(fterm1 + tmp*p1*sterm1));
        dgdphi2_begin->x = (omg*(fterm2 + tmp*p2*sterm2));
        dgdphi3_begin->x = (omg*(fterm3 + tmp*p3*sterm3));

        dgdphi0_begin->y = 0.0;
        dgdphi1_begin->y = 0.0;
        dgdphi2_begin->y = 0.0;
        dgdphi3_begin->y = 0.0; */
    }
}
void Calc_dgdphi2(BoxIterator<cufftDoubleComplex> dgdphi0_begin,
                            BoxIterator<cufftDoubleComplex> dgdphi0_end,
                            BoxIterator<cufftDoubleComplex> dgdphi1_begin,
                            BoxIterator<cufftDoubleComplex> dgdphi2_begin,
                            BoxIterator<cufftDoubleComplex> dgdphi3_begin,
                            BoxIterator<cufftDoubleComplex> phi0_begin,
                            BoxIterator<cufftDoubleComplex> phi1_begin,
                            BoxIterator<cufftDoubleComplex> phi2_begin,
                            BoxIterator<cufftDoubleComplex> phi3_begin,
                            double omega, double theta, double theta2,
                            int ny, int nz)
{

    long i = threadIdx.x + blockIdx.x*blockDim.x;
    long j = threadIdx.y + blockIdx.y*blockDim.y;
    long k = threadIdx.z + blockIdx.z*blockDim.z;
    long idx = k + (nz * (j + (i*ny)));

    dgdphi0_begin += idx; dgdphi1_begin += idx; dgdphi2_begin += idx; dgdphi3_begin += idx;
    phi0_begin += idx; phi1_begin += idx; phi2_begin += idx; phi3_begin += idx;

    if (dgdphi0_begin < dgdphi0_end)
    {
        const double p0 = phi0_begin->x;
        const double p1 = phi1_begin->x;
        const double p2 = phi2_begin->x;
        const double p3 = phi3_begin->x;

        const double p0p0 = p0*p0;
        const double p1p1 = p1*p1;
        const double p2p2 = p2*p2;
        const double p3p3 = p3*p3;
        const double psum = p0p0 + p1p1 + p2p2 + p3p3;

        const double omg = omega;
        const double tmp = 2.0*theta;
        const double tmp2 = 2.0*theta2;

        *dgdphi0_begin = {(omg*((2.0*p0 * (1.0-p0)*(1.0-2.0*p0)) + tmp*p0 * (psum - p0p0) + tmp2*p0*(p1p1*p2p2 + p1p1*p3p3 + p2p2*p3p3))), 0.0};
        *dgdphi1_begin = {(omg*((2.0*p1 * (1.0-p1)*(1.0-2.0*p1)) + tmp*p1 * (psum - p1p1) + tmp2*p1*(p0p0*p2p2 + p0p0*p3p3 + p2p2*p3p3))), 0.0};
        *dgdphi2_begin = {(omg*((2.0*p2 * (1.0-p2)*(1.0-2.0*p2)) + tmp*p2 * (psum - p2p2) + tmp2*p2*(p0p0*p1p1 + p0p0*p3p3 + p1p1*p3p3))), 0.0};
        *dgdphi3_begin = {(omg*((2.0*p3 * (1.0-p3)*(1.0-2.0*p3)) + tmp*p3 * (psum - p3p3) + tmp2*p3*(p0p0*p1p1 + p0p0*p2p2 + p1p1*p2p2))), 0.0};

    }
}
void Calc_phase_comp(BoxIterator<cufftDoubleComplex> Calpha,
                                BoxIterator<cufftDoubleComplex> Cbeta,
                                BoxIterator<cufftDoubleComplex> comp_begin,
                                BoxIterator<cufftDoubleComplex> comp_end, double *hphisum,
                                double *c_eq, double *f0, long nx, long ny, long nz)
{
    long i = threadIdx.x + blockIdx.x*blockDim.x;
    long j = threadIdx.y + blockIdx.y*blockDim.y;
    long k = threadIdx.z + blockIdx.z*blockDim.z;
    long idx = k + (nz * (j + (i*ny)));
    comp_begin += idx;
    Calpha += idx;
    Cbeta += idx;

    if (comp_begin < comp_end)
    {
        double hsum = hphisum[idx];
        double frac = f0[0]/f0[1];
        double calpha_eq = c_eq[0];
        double cbeta_eq = c_eq[1];

        ////Matrix phase comp.
        double C1  = (comp_begin->x - (hsum*(cbeta_eq - (frac*calpha_eq))));
        C1 /= ((hsum*frac) + (1.0 - hsum));

        /////Precipitate phase comp.
        double C2  = cbeta_eq + frac*(C1 - calpha_eq);

        //*comp_begin = {((hsum*C2) + ((1.0 - hsum)*C1)), 0.0};

        Calpha->x = C1;
        Calpha->y = 0.0;

        Cbeta->x = C2;
        Cbeta->y = 0.0;

    }
    //__syncthreads();
}
void Calc_phase_engs(BoxIterator<cufftDoubleComplex> phase_engy_begin,
                                 BoxIterator<cufftDoubleComplex> phase_engy_end,
                                 BoxIterator<cufftDoubleComplex> Calpha,
                                 BoxIterator<cufftDoubleComplex> Cbeta,
                                double *hphisum, double *ftotal, double *c_eq,
                                double *f0, double Am, double Ap, double NV,
                                int FE_form, int engs_out, int output,
                                int TRACK_OBS, long nx, long ny, long nz)
{
    long i = threadIdx.x + blockIdx.x*blockDim.x;
    long j = threadIdx.y + blockIdx.y*blockDim.y;
    long k = threadIdx.z + blockIdx.z*blockDim.z;
    long idx = k + (nz * (j + (i*ny)));
    phase_engy_begin += idx;
    Calpha += idx;
    Cbeta += idx;

    if (phase_engy_begin < phase_engy_end)
    {
        if (FE_form == 0)
        {
            double c1 = Calpha->x - c_eq[0];
            double c2 = Cbeta->x - c_eq[1];

            ////Using a fft pointer at the moment real part to store matrix and imag for precipitate.
            phase_engy_begin->x = Am * c1*c1;
            phase_engy_begin->y = Ap * c2*c2;

        }
        else if (FE_form == 1)
        {
            double calpha_eq = c_eq[0];
            double cbeta_eq = c_eq[1];

            double c1 = Calpha->x;
            double c2 = Cbeta->x;

            phase_engy_begin->x = (Am * c1*c1);

            phase_engy_begin->y = ((Ap*c2*c2) + (2.0*(Am*calpha_eq - Ap*cbeta_eq)*c2) +
                                    (Ap*cbeta_eq*cbeta_eq - Am*calpha_eq*calpha_eq));

        }

        if (TRACK_OBS == 1 || (engs_out != 0 && output == 1))
            ftotal[idx] += NV*(phase_engy_begin->x * (1.0-hphisum[idx]) + (phase_engy_begin->y * hphisum[idx]));
    }
    //__syncthreads();
}
void Calc_mu(BoxIterator<cufftDoubleComplex> mu_begin,
                        BoxIterator<cufftDoubleComplex> mu_end,
                        BoxIterator<cufftDoubleComplex> Calpha,
                        double *c_eq, double Am, double Ap, int FE_form,
                        long nx, long ny, long nz)
{

    long i = threadIdx.x + blockIdx.x*blockDim.x;
    long j = threadIdx.y + blockIdx.y*blockDim.y;
    long k = threadIdx.z + blockIdx.z*blockDim.z;
    long idx = k + (nz * (j + (i*ny)));
    mu_begin += idx;
    Calpha += idx;

    if (mu_begin < mu_end)
    {
        if (FE_form == 0)
            *mu_begin = {(2.0*Am*(Calpha->x - c_eq[0])), 0.0};

        else if (FE_form == 1)
            *mu_begin = {(2.0*Am*Calpha->x), 0.0};

    }
    //__syncthreads();
}
void Calc_dfchdphi_sv(BoxIterator<cufftDoubleComplex> dfdphi_begin,
                               BoxIterator<cufftDoubleComplex> dfdphi_end,
                               BoxIterator<cufftDoubleComplex> dfdc_begin,
                               BoxIterator<cufftDoubleComplex> Calpha,
                               BoxIterator<cufftDoubleComplex> Cbeta,
                               BoxIterator<cufftDoubleComplex> phase_eng,
                               double *dhdphi,
                               double Nv, long nx, long ny, long nz)
{
    long i = threadIdx.x + blockIdx.x*blockDim.x;
    long j = threadIdx.y + blockIdx.y*blockDim.y;
    long k = threadIdx.z + blockIdx.z*blockDim.z;
    long idx = k + (nz * (j + (i*ny)));

    dfdphi_begin += idx;
    dfdc_begin += idx;
    Calpha += idx;
    Cbeta += idx;
    phase_eng += idx;

    if (dfdphi_begin < dfdphi_end)
    {
        const double tmp = dfdphi_begin->x;
        const double C1 = Calpha->x;
        const double C2 = Cbeta->x;
        const cufftDoubleComplex tmp2 = *phase_eng;

        ////imag part has precip value and real has matrix value.
        dfdphi_begin->x = tmp + Nv*(dhdphi[idx] * ((tmp2.y - tmp2.x) + (C1 - C2)*dfdc_begin->x));
        dfdphi_begin->y = 0.0;

    }
    //__syncthreads();
}
void Calc_dfchdphi(BoxIterator<cufftDoubleComplex> dfdphi0_begin,
                               BoxIterator<cufftDoubleComplex> dfdphi0_end,
                               BoxIterator<cufftDoubleComplex> dfdphi1_begin,
                               BoxIterator<cufftDoubleComplex> dfdphi2_begin,
                               BoxIterator<cufftDoubleComplex> dfdphi3_begin,
                               BoxIterator<cufftDoubleComplex> dfdc_begin,
                               BoxIterator<cufftDoubleComplex> Calpha,
                               BoxIterator<cufftDoubleComplex> Cbeta,
                               BoxIterator<cufftDoubleComplex> phase_eng,
                               double *dhdphi0, double *dhdphi1,
                               double *dhdphi2, double *dhdphi3,
                               double Nv, long nx, long ny, long nz)
{
    long i = threadIdx.x + blockIdx.x*blockDim.x;
    long j = threadIdx.y + blockIdx.y*blockDim.y;
    long k = threadIdx.z + blockIdx.z*blockDim.z;
    long idx = k + (nz * (j + (i*ny)));

    dfdphi0_begin += idx;
    dfdphi1_begin += idx;
    dfdphi2_begin += idx;
    dfdphi3_begin += idx;
    dfdc_begin += idx;
    Calpha += idx;
    Cbeta += idx;
    phase_eng += idx;

    if (dfdphi0_begin < dfdphi0_end)
    {
        const double tmp0 = dfdphi0_begin->x;
        const double tmp1 = dfdphi1_begin->x;
        const double tmp2 = dfdphi2_begin->x;
        const double tmp3 = dfdphi3_begin->x;

        const cufftDoubleComplex ph_eng = *phase_eng;
        const double c_alpha = Calpha->x;
        const double c_beta = Cbeta->x;

        double fcd = Nv*((ph_eng.y - ph_eng.x) + (c_alpha - c_beta)*dfdc_begin->x);

        *dfdphi0_begin = {(tmp0 + dhdphi0[idx]*fcd), 0.0};
        *dfdphi1_begin = {(tmp1 + dhdphi1[idx]*fcd), 0.0};
        *dfdphi2_begin = {(tmp2 + dhdphi2[idx]*fcd), 0.0};
        *dfdphi3_begin = {(tmp3 + dhdphi3[idx]*fcd), 0.0};

    }
    //__syncthreads();
}
void Compute_dfeldphi_inhomsv(BoxIterator<cufftDoubleComplex> dfdphi0_begin,
                                         BoxIterator<cufftDoubleComplex> dfdphi0_end,
                                         BoxIterator<cufftDoubleComplex> str_v0_begin,
                                         BoxIterator<cufftDoubleComplex> str_v1_begin,
                                         BoxIterator<cufftDoubleComplex> str_v2_begin,
                                         BoxIterator<cufftDoubleComplex> str_v3_begin,
                                         BoxIterator<cufftDoubleComplex> str_v4_begin,
                                         BoxIterator<cufftDoubleComplex> str_v5_begin,
                                         double* __restrict__ felas, const double* __restrict__ hphisum,
                                         const double* __restrict__ dhdphi, const double* __restrict__ hom_strain_v,
                                         double Chom11_d, double Chom12_d, double Chom44_d,
                                         double Chet11_d, double Chet12_d, double Chet44_d,
                                         double epszero_d, double misfit_aniso,
                                         int engs_out, int output, int TRACK_OBS, long nx, long ny, long nz)
{

    long i = threadIdx.x + blockDim.x*blockIdx.x;
    long j = threadIdx.y + blockDim.y*blockIdx.y;
    long k = threadIdx.z + blockDim.z*blockIdx.z;

    long idx = k + (nz*(j + i*ny));

    dfdphi0_begin += idx;
    str_v0_begin += idx; str_v1_begin += idx; str_v2_begin += idx;
    str_v3_begin += idx; str_v4_begin += idx; str_v5_begin += idx;

    if (dfdphi0_begin < dfdphi0_end)
    {
        double df = dfdphi0_begin->x;
        double temp = 2.0*hphisum[idx] - 1.0;
        double Ct11 = Chom11_d + Chet11_d * temp;
        double Ct12 = Chom12_d + Chet12_d * temp;
        double Ct44 = Chom44_d + Chet44_d * temp;

        double hphi = hphisum[idx];
        double eig11 = epszero_d;
        double eig22 = eig11 * misfit_aniso;

        double e11 = hom_strain_v[0] + str_v0_begin->x - eig11*hphi;
        double e22 = hom_strain_v[1] + str_v1_begin->x - eig22*hphi;
        double e33 = hom_strain_v[2] + str_v2_begin->x - eig22*hphi; ////eig33 since misfit_aniso is same so using eig22.
        double e23 = hom_strain_v[3] + str_v3_begin->x;
        double e13 = hom_strain_v[4] + str_v4_begin->x;
        double e12 = hom_strain_v[5] + str_v5_begin->x;

        *dfdphi0_begin = {df + (dhdphi[idx]*(0.5*(2.0*Chet11_d*(e11*e11 + e22*e22 + e33*e33) +
                                                    4.0*Chet12_d*(e11*e22 + e11*e33 + e22*e33) +
                                                    8.0*Chet44_d*(e23*e23 + e13*e13 + e12*e12)) -

                                               1.0*(Ct11*(e11*eig11 + e22*eig22 + e33*eig22) +
                                                    Ct12*(e11*eig22 + e22*eig11 + e11*eig22 +
                                                    e33*eig11 + e22*eig22 + e33*eig22)))), 0.0};

        if (TRACK_OBS == 1 || (engs_out != 0 && output == 1))
        {
            felas[idx] = 0.5*(Ct11*(e11*e11 + e22*e22 + e33*e33)+
                              Ct12*2.0*(e11*e22 + e11*e33 + e22*e33)+
                              Ct44*4.0*(e23*e23 + e13*e13 + e12*e12));
        }
    }
}
}
