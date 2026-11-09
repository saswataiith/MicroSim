#include "fileWriter.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <sys/stat.h>
#include <unistd.h>
int main(int argc,char **argv){
 MPI_Init(&argc,&argv);int rank;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
 mkdir("DATA",0777);char dir[128];snprintf(dir,sizeof(dir),"DATA/Processor_%d",rank);mkdir(dir,0777);
 for(int nz:{1,6}) for(int func:{1,2}){
  domainInfo d={};controls ctl={};subdomainInfo sd={};d.numPhases=2;d.numComponents=2;d.DELTA_X=2.5e-9;d.DELTA_Y=3.3e-9;d.DELTA_Z=4.4e-9;
  char alpha[]="alpha",liquid[]="liquid",solute[]="Al",solvent[]="Zn";char *phases[]={alpha,liquid},*components[]={solute,solvent};d.phaseNames=phases;d.componentNames=components;ctl.FUNCTION_F=func;ctl.count=nz*10+func;
  sd.numX=4+rank;sd.numY=5;sd.numZ=nz;sd.xS=rank*sd.numX;sd.xE=sd.xS+sd.numX;sd.yS=0;sd.yE=sd.numY;sd.zS=0;sd.zE=nz;
  sd.sizeX=sd.numX+2;sd.sizeY=sd.numY+2;sd.sizeZ=nz==1?1:nz+2;sd.xS_r=sd.yS_r=1;sd.xE_r=sd.numX+1;sd.yE_r=sd.numY+1;sd.zS_r=nz==1?0:1;sd.zE_r=nz==1?1:nz+1;
  sd.xStep=sd.sizeY*sd.sizeZ;sd.yStep=sd.sizeZ;sd.numCompCells=sd.sizeX*sd.sizeY*sd.sizeZ;sd.numCells=sd.numX*sd.numY*nz;
  size_t n=sd.numCompCells;std::vector<double> phi(2*n,-7),c(n,-7),mu(n,-7),outp(2*n,-7),outc(n,-7),outmu(n,-7);
  for(long x=sd.xS_r;x<sd.xE_r;++x)for(long y=sd.yS_r;y<sd.yE_r;++y)for(long z=sd.zS_r;z<sd.zE_r;++z){size_t i=x*sd.xStep+y*sd.yStep+z;phi[i]=.1+.001*i;phi[n+i]=1-phi[i];c[i]=.82+.00001*i;mu[i]=19608.+i;}
  // Valid finite doubles whose first payload bytes look like ASCII whitespace
  // and a digit; a formatted floating scan must never inspect these bytes.
  size_t first=sd.xS_r*sd.xStep+sd.yS_r*sd.yStep+sd.zS_r;uint64_t whitespace=0x2000000000000000ULL,digit=0x3000000000000000ULL;memcpy(&phi[first],&whitespace,8);memcpy(&c[first],&digit,8);phi[n+first]=1-phi[first];
  char name[]="roundtrip";char *args[]={nullptr,nullptr,nullptr,name};writeVTK_BINARY(phi.data(),c.data(),mu.data(),d,sd,ctl,rank,MPI_COMM_WORLD,args);
  char file[256];snprintf(file,sizeof(file),"%s/roundtrip_%ld.vtk",dir,ctl.count);FILE *f=fopen(file,"rb");assert(f);assert(readVTK_BINARY(f,outp.data(),outc.data(),outmu.data(),d,sd,ctl)==1);fclose(f);
  assert(memcmp(phi.data(),outp.data(),2*n*sizeof(double))==0);assert(memcmp(c.data(),outc.data(),n*sizeof(double))==0);if(func==2)assert(memcmp(mu.data(),outmu.data(),n*sizeof(double))==0);
  // A truncated final field must be rejected, not silently accepted.
  f=fopen(file,"rb");fseek(f,0,SEEK_END);long size=ftell(f);fclose(f);assert(truncate(file,size-16)==0);f=fopen(file,"rb");assert(f);assert(readVTK_BINARY(f,outp.data(),outc.data(),outmu.data(),d,sd,ctl)==0);fclose(f);
 }
 if(!rank)puts("PASS binary I/O: 2D/3D, F1/F2, endian payload bytes, halos, truncation");MPI_Finalize();return 0;
}
