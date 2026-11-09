#include "elasticity_c.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc,char **argv){
 MPI_Init(&argc,&argv);int rank;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
 int threads=argc>1?atoi(argv[1]):1;
 if(ms_elastic_initialize(threads))MPI_Abort(MPI_COMM_WORLD,2);
 ptrdiff_t grid[3]={9,7,1};double spacing[3]={1.,1.3,1.};
 for(int mode=1;mode<=2;++mode){
  ms_elastic_phase phase[2]={{4.5,1.5,1.2,{.005,.01,0,0,0,.002}},{3.,1.,.8,{0}}};
  if(mode==1)phase[0].eigenstrain[2]=.003;
  ms_elastic *s=ms_elastic_create(MPI_COMM_WORLD,grid,spacing,mode,2,phase);
  if(!s){fprintf(stderr,"%s\n",ms_elastic_error());MPI_Abort(MPI_COMM_WORLD,3);}
  size_t n=ms_elastic_local_size(s);ptrdiff_t start=ms_elastic_x_start(s);
  double *h=calloc(2*(n?n:1),sizeof(double)),*e=calloc(6*(n?n:1),sizeof(double));
  double *sig=calloc(6*(n?n:1),sizeof(double)),*w=calloc(n?n:1,sizeof(double)),*force=calloc(2*(n?n:1),sizeof(double));
  if(!h||!e||!sig||!w||!force)MPI_Abort(MPI_COMM_WORLD,9);
  for(size_t i=0;i<n;++i){double x=(double)(start+(ptrdiff_t)(i/7)),y=(double)(i%7);h[i]=.4+.15*cos(2*3.141592653589793*x/9)*cos(2*3.141592653589793*y/7);h[n+i]=1-h[i];}
  for(int loading=0;loading<3;++loading){
   int fixed[6]={0};double stress[6]={0},strain[6]={0};ms_elastic_status out;
   if(loading==1){stress[0]=.01;stress[1]=-.02;stress[5]=.005;}
   if(loading==2)for(int a=0;a<6;++a)fixed[a]=1;
   int rc=ms_elastic_solve(s,h,fixed,stress,strain,&out,e,sig,w,force);
   if(rc){fprintf(stderr,"%s\n",ms_elastic_error());MPI_Abort(MPI_COMM_WORLD,4);}
   if(!out.converged||out.equilibrium>1e-8||out.mean_stress_error>1e-10)MPI_Abort(MPI_COMM_WORLD,5);
   if(loading!=2)for(int a=0;a<6;++a)if(fabs(out.mean_stress[a]-stress[a])>1e-10)MPI_Abort(MPI_COMM_WORLD,6);
   if(mode==2&&out.mean_strain[2]!=0)MPI_Abort(MPI_COMM_WORLD,7);
   if(loading==2)for(int a=0;a<6;++a)if(out.mean_strain[a]!=0)MPI_Abort(MPI_COMM_WORLD,8);
   if(argc>2){
    char filename[512];snprintf(filename,sizeof(filename),"%s-model%d-loading%d-rank%d.csv",argv[2],mode,loading,rank);
    FILE *f=fopen(filename,"w");if(!f)MPI_Abort(MPI_COMM_WORLD,10);
    fprintf(f,"x,y,h,exx,eyy,ezz,eyz,exz,exy,sxx,syy,szz,syz,sxz,sxy,W,dfdh\n");
    for(size_t i=0;i<n;++i){
     fprintf(f,"%td,%zu,%.17g",start+(ptrdiff_t)(i/7),i%7,h[i]);
     for(int a=0;a<6;++a)fprintf(f,",%.17g",e[a*n+i]);
     for(int a=0;a<6;++a)fprintf(f,",%.17g",sig[a*n+i]);
     fprintf(f,",%.17g,%.17g\n",w[i],force[i]-force[n+i]);
    }
    fclose(f);
   }
   if(!rank)printf("model=%d loading=%d Ezz=%.15g meanW=%.15g equilibrium=%.3g\n",mode,loading,out.mean_strain[2],out.mean_energy,out.equilibrium);
  }
  free(h);free(e);free(sig);free(w);free(force);ms_elastic_destroy(s);
 }
 ms_elastic_cleanup();MPI_Finalize();return 0;
}
