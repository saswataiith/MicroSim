#ifndef MICROSIM_ELASTICITY_C_H
#define MICROSIM_ELASTICITY_C_H
#include <stddef.h>
#include <mpi.h>
#ifdef __cplusplus
extern "C" {
#endif
/* C ABI for the shared CPU/MPI implementation; tensor shear, xx yy zz yz xz xy.
 * Models: 0=3D, 1=generalized plane strain, 2=intrinsic 2D.
 * All create/solve calls are collective on comm. The application owns MPI.
 * Initialize once after MPI_Init, destroy handles before cleanup/MPI_Finalize.
 */
typedef struct ms_elastic ms_elastic;
typedef struct {double c11,c12,c44;double eigenstrain[6];} ms_elastic_phase;
typedef struct {
 int converged,iterations;
 double change,equilibrium,mean_stress_error,mean_energy,mean_potential;
 double mean_strain[6],mean_stress[6];
} ms_elastic_status;
int ms_elastic_initialize(int fft_threads);
void ms_elastic_cleanup(void);
ms_elastic *ms_elastic_create(MPI_Comm comm,const ptrdiff_t shape[3],const double spacing[3],
 int model,int phase_count,const ms_elastic_phase *phases);
void ms_elastic_destroy(ms_elastic *solver);
/* Compact local x slab, indexed [local_x][global_y][global_z]. */
size_t ms_elastic_local_size(const ms_elastic *solver);
ptrdiff_t ms_elastic_x_start(const ms_elastic *solver);
/* Loading: fixed[a]=0 evaluates E[a] from prescribed mean stress[a].
 * fixed[a]=1 prescribes E[a]=strain[a]. All zero stress/free flags => stress-free.
 * In 2D yz/xz are constrained to zero; intrinsic additionally excludes zz.
 * fractions are [phase][local_site], sum=1. Output fields are [component][site].
 * Force is returned only for converged equilibrium, [phase][site], dF/dh_phase;
 * chain through interpolation exactly once in the calling phase-field model.
 * Returns 0 success, 1 failed equilibrium, -1 invalid input/runtime exception.
 */
int ms_elastic_solve(ms_elastic *solver,const double *fractions,const int fixed[6],
 const double stress[6],const double strain[6],ms_elastic_status *status,
 double *total_strain,double *constitutive_stress,double *energy,double *fraction_force);
const char *ms_elastic_error(void);
#ifdef __cplusplus
}
#endif
#endif
