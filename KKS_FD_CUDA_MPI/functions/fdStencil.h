#ifndef MICROSIM_FD_STENCIL_H
#define MICROSIM_FD_STENCIL_H
#include <float.h>
#ifdef __CUDACC__
#define MS_FD_HD __host__ __device__
#else
#define MS_FD_HD
#endif
/* Symmetric, conservative nine-point Laplacian on rectangular 2D grids.
 * Positive edge weights, zero row sum, exact quadratic moments.
 * At dx=dy this preserves the original axial=1/2, diagonal=1/4 stencil.
 */
MS_FD_HD static inline double microsim_laplacian_2d(double center,
 double value_xminus,double value_xplus,double value_yminus,double value_yplus,double diagonals,double dx,double dy)
{
 const double wd=0.5/(dx*dx+dy*dy);
 const double wx=1.0/(dx*dx)-2.0*wd,wy=1.0/(dy*dy)-2.0*wd;
 return wx*(value_xminus+value_xplus-2.0*center)+wy*(value_yminus+value_yplus-2.0*center)+wd*(diagonals-4.0*center);
}
/* Do not turn roundoff-level phase differences into a unit normal. */
MS_FD_HD static inline int microsim_2d_gradient_resolved(double magnitude,double dx,double dy)
{
 return magnitude*(dx<dy?dx:dy)>64.0*DBL_EPSILON;
}
#undef MS_FD_HD
#endif
