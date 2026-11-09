#ifndef MICROSIM_INHOM_MATH_HPP
#define MICROSIM_INHOM_MATH_HPP
#include <math.h>
#include <float.h>
#ifdef __CUDACC__
#define INHOM_HD __host__ __device__
#else
#define INHOM_HD
#endif

// Real-field odd derivatives use a zero Nyquist multiplier on even grids.
INHOM_HD static inline double inhom_wave_number(long index, long n, double spacing)
{
    if (n % 2 == 0 && index == n/2) return 0.0;
    const long mode = index < (n+1)/2 ? index : index-n;
    return mode * (6.283185307179586476925286766559 / (spacing*n));
}

INHOM_HD static inline double inhom_max3(double x, double y, double z)
{
    return fmax(x, fmax(y, z));
}

// Symmetric cubic acoustic tensor inverse; slots xx yy zz yz xz xy.
// Scaling removes the former dimensional determinant cutoff.
INHOM_HD static inline void inhom_green_inverse(double *out, const double *k, const double *c)
{
    const double a=c[0], b=c[1]+c[2], s=c[2];
    double v[6]={a*k[0]*k[0]+s*(k[1]*k[1]+k[2]*k[2]),
                 a*k[1]*k[1]+s*(k[0]*k[0]+k[2]*k[2]),
                 a*k[2]*k[2]+s*(k[0]*k[0]+k[1]*k[1]),
                 b*k[1]*k[2], b*k[0]*k[2], b*k[0]*k[1]};
    double scale=0.0;
    for (int i=0;i<6;i++) scale=fmax(scale,fabs(v[i]));
    if (scale==0.0) {for(int i=0;i<6;i++)out[i]=0.0; return;}
    for (int i=0;i<6;i++)v[i]/=scale;
    const double det=v[0]*(v[1]*v[2]-v[3]*v[3])-v[5]*(v[5]*v[2]-v[4]*v[3])+v[4]*(v[5]*v[3]-v[4]*v[1]);
    if (!isfinite(det) || det<=64.0*DBL_EPSILON) {
        for(int i=0;i<6;i++)out[i]=NAN; return;
    }
    const double inv=1.0/(scale*det);
    out[0]=inv*(v[1]*v[2]-v[3]*v[3]);
    out[1]=inv*(v[0]*v[2]-v[4]*v[4]);
    out[2]=inv*(v[0]*v[1]-v[5]*v[5]);
    out[3]=-inv*(v[0]*v[3]-v[4]*v[5]);
    out[4]=inv*(v[5]*v[3]-v[4]*v[1]);
    out[5]=-inv*(v[5]*v[2]-v[4]*v[3]);
}
// In-plane compliance uses tensor shear: epsilon_xy = sigma_xy/(2*C44).
// model2d=1: generalized plane strain (free mean Ezz); 2: intrinsic 2D.
INHOM_HD static inline void inhom_average_compliance(double *s, const double *c, int model2d)
{
    if (model2d == 2) {
        const double den = (c[0]-c[1])*(c[0]+c[1]);
        s[0] = c[0]/den;
        s[1] = -c[1]/den;
    } else {
        const double den = (c[0]-c[1])*(c[0]+2*c[1]);
        s[0] = (c[0]+c[1])/den;
        s[1] = -c[1]/den;
    }
    s[2] = 1.0/c[2];
}

// Derivative of 1/2 (epsilon-h*eigen):C(h):(epsilon-h*eigen), at fixed epsilon.
// e stores elastic tensor strain, in xx yy zz yz xz xy order.
INHOM_HD static inline double inhom_elastic_partial(const double *e, const double *eigen,
                                                    const double *ch, const double *dc, double h)
{
    const double c11=ch[0]+dc[0]*(2*h-1), c12=ch[1]+dc[1]*(2*h-1);
    const double stiffness_part = dc[0]*(e[0]*e[0]+e[1]*e[1]+e[2]*e[2])
        +2*dc[1]*(e[0]*e[1]+e[0]*e[2]+e[1]*e[2])
        +4*dc[2]*(e[3]*e[3]+e[4]*e[4]+e[5]*e[5]);
    return stiffness_part
        -(c11*e[0]+c12*(e[1]+e[2]))*eigen[0]
        -(c11*e[1]+c12*(e[0]+e[2]))*eigen[1]
        -(c11*e[2]+c12*(e[0]+e[1]))*eigen[2];
}
#undef INHOM_HD
#endif
