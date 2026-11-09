#include "elasticity_c.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* Fixed cubic benchmark reference; h input is native double [x][y][z]. */
int main(int argc,char **argv){
 MPI_Init(&argc,&argv);int rank;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
 if(argc<4){if(!rank)fprintf(stderr,"Usage: %s N h.bin output_prefix [spacing_m]\n",argv[0]);MPI_Finalize();return 2;}
 int N=atoi(argv[1]);double dx=argc>4?strtod(argv[4],NULL):1.28e-6/N;ptrdiff_t shape[3]={N,N,N};double spacing[3]={dx,dx,dx};
 if(ms_elastic_initialize(1))MPI_Abort(MPI_COMM_WORLD,2);
 ms_elastic_phase phases[2]={{4.5e11,1.5e11,1.2e11,{.001,.001,.001,0,0,0}},{3.2e11,.7e11,.85e11,{0}}};
 ms_elastic *solver=ms_elastic_create(MPI_COMM_WORLD,shape,spacing,0,2,phases);if(!solver){fprintf(stderr,"%s\n",ms_elastic_error());MPI_Abort(MPI_COMM_WORLD,3);}
 size_t n=ms_elastic_local_size(solver);ptrdiff_t start=ms_elastic_x_start(solver);
 double *h=calloc(2*n,sizeof(double)),*e=calloc(6*n,sizeof(double)),*s=calloc(6*n,sizeof(double)),*W=calloc(n,sizeof(double)),*force=calloc(2*n,sizeof(double));
 FILE *f=fopen(argv[2],"rb");if(!f||fseek(f,(long)(start*N*N*sizeof(double)),SEEK_SET)||fread(h,sizeof(double),n,f)!=n)MPI_Abort(MPI_COMM_WORLD,4);fclose(f);for(size_t i=0;i<n;++i)h[n+i]=1-h[i];
 int fixed[6]={0};double stress[6]={0},strain[6]={0};ms_elastic_status out;
 if(ms_elastic_solve(solver,h,fixed,stress,strain,&out,e,s,W,force)){fprintf(stderr,"%s\n",ms_elastic_error());MPI_Abort(MPI_COMM_WORLD,5);}
 char filename[1024];snprintf(filename,sizeof(filename),"%s-rank%d.bin",argv[3],rank);f=fopen(filename,"wb");if(!f)MPI_Abort(MPI_COMM_WORLD,6);
 uint64_t header[2]={(uint64_t)start,(uint64_t)n};fwrite(header,sizeof(uint64_t),2,f);fwrite(e,sizeof(double),6*n,f);fwrite(s,sizeof(double),6*n,f);fwrite(W,sizeof(double),n,f);for(size_t i=0;i<n;++i)force[i]-=force[n+i];fwrite(force,sizeof(double),n,f);fclose(f);
 if(!rank)printf("{\"converged\":%d,\"iterations\":%d,\"equilibrium\":%.17g,\"mean_stress_error\":%.17g,\"mean_energy\":%.17g}\n",out.converged,out.iterations,out.equilibrium,out.mean_stress_error,out.mean_energy);
 free(h);free(e);free(s);free(W);free(force);ms_elastic_destroy(solver);ms_elastic_cleanup();MPI_Finalize();return 0;
}
