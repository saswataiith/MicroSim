#include "elasticity_c.h"
#include "cpu_mpi.hpp"
#include <algorithm>
#include <memory>
#include <stdexcept>
using namespace microsim;
static thread_local std::string last_error;
struct ms_elastic {
 int model,count;
 FFT fft;
 Elasticity solver;
 ms_elastic(MPI_Comm comm,Grid grid,int mode,std::vector<Phase> phases)
 :model(mode),count(int(phases.size())),fft(grid,comm),solver(fft,std::move(phases)){}
};
extern "C" {
int ms_elastic_initialize(int threads){
 if(threads<1){last_error="FFT thread count must be positive";return -1;}
 if(!fftw_init_threads()){last_error="FFTW thread initialization failed";return -1;}
 fftw_plan_with_nthreads(threads);fftw_mpi_init();return 0;
}
void ms_elastic_cleanup(void){fftw_mpi_cleanup();fftw_cleanup_threads();}
const char *ms_elastic_error(void){return last_error.c_str();}
ms_elastic *ms_elastic_create(MPI_Comm comm,const ptrdiff_t shape[3],const double spacing[3],int mode,int count,const ms_elastic_phase *input){
 try{
  require(shape&&spacing&&input&&count>0&&mode>=0&&mode<=2,"Invalid C elasticity model/phase parameters",comm);
  require(mode==0||shape[2]==1,"2D elasticity requires nz=1",comm);
  Grid g;g.nx=shape[0];g.ny=shape[1];g.nz=shape[2];g.dx=spacing[0];g.dy=spacing[1];g.dz=spacing[2];
  std::vector<Phase> p(count);bool valid=true;
  for(int i=0;i<count;++i){
   p[i].stiffness={input[i].c11,input[i].c12,input[i].c44};
   std::copy_n(input[i].eigenstrain,6,p[i].eigenstrain.begin());
   if(mode)valid=valid&&p[i].eigenstrain[3]==0&&p[i].eigenstrain[4]==0;
   if(mode==2)valid=valid&&p[i].eigenstrain[2]==0;
  }
  require(valid,"Unsupported out-of-plane eigenstrain for the selected 2D model",comm);
  return new ms_elastic(comm,g,mode,std::move(p));
 }catch(const std::exception &e){last_error=e.what();return nullptr;}
}
void ms_elastic_destroy(ms_elastic *s){delete s;}
size_t ms_elastic_local_size(const ms_elastic *s){return s?s->fft.real_size:0;}
ptrdiff_t ms_elastic_x_start(const ms_elastic *s){return s?s->fft.x_start:0;}
int ms_elastic_solve(ms_elastic *s,const double *h,const int fixed[6],const double stress[6],const double strain[6],ms_elastic_status *status,double *eout,double *sout,double *energy,double *force){
 try{
  if(!s){last_error="Null elasticity handle";return -1;}
  const size_t n=s->fft.real_size;
  require((h||!n)&&fixed&&stress&&strain&&status,"Missing C elasticity buffers/loading",s->fft.comm);
  Loading load;bool valid=true;
  for(int a=0;a<6;++a){load.fixed_strain[a]=fixed[a]!=0;load.stress[a]=stress[a];load.strain[a]=strain[a];}
  if(s->model){
   for(int a:{3,4}){valid=valid&&stress[a]==0&&strain[a]==0;load.fixed_strain[a]=true;}
   if(s->model==2){valid=valid&&stress[2]==0&&strain[2]==0;load.fixed_strain[2]=true;}
  }
  require(valid,"Excluded out-of-plane loading is nonzero",s->fft.comm);
  std::vector<Real> fractions(s->count,Real(n));
  for(int p=0;p<s->count;++p)if(n)std::copy_n(h+p*n,n,fractions[p].begin());
  auto r=s->solver.solve(fractions,load);*status={};
  auto &st=r.status;status->converged=st.converged;status->iterations=st.iterations;
  status->change=st.change;status->equilibrium=st.equilibrium;status->mean_stress_error=st.mean_stress_error;
  status->mean_energy=st.mean_energy;status->mean_potential=st.mean_potential;
  std::copy(st.mean_strain.begin(),st.mean_strain.end(),status->mean_strain);
  std::copy(st.mean_stress.begin(),st.mean_stress.end(),status->mean_stress);
  if(s->model==2){status->mean_stress[2]=0;r.stress[2].assign(n,0.);}
  if(!st.converged){last_error="Elasticity did not converge; output/force buffers were not written";return 1;}
  for(int a=0;a<6;++a){
   if(eout)std::copy(r.strain[a].begin(),r.strain[a].end(),eout+a*n);
   if(sout)std::copy(r.stress[a].begin(),r.stress[a].end(),sout+a*n);
  }
  if(energy)std::copy(r.energy.begin(),r.energy.end(),energy);
  if(force)for(int p=0;p<s->count;++p)std::copy(r.fraction_derivative[p].begin(),r.fraction_derivative[p].end(),force+p*n);
  return 0;
 }catch(const std::exception &e){last_error=e.what();return -1;}
}
}
