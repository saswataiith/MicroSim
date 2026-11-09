#include <cmath>
#include <cstdio>
#include <initializer_list>
#include "../KKS_FD_CUDA_MPI/functions/fdStencil.h"
int main(){
 for(double scale:{1.,1e-8})for(double ratio:{.5,1.,1.3,2.}){
  double dx=scale,dy=scale*ratio;
  if(microsim_laplacian_2d(.37,.37,.37,.37,.37,4*.37,dx,dy)!=0)return 1;
  auto eval=[&](auto f){return microsim_laplacian_2d(f(0.,0.),f(-dx,0.),f(dx,0.),f(0.,-dy),f(0.,dy),f(-dx,-dy)+f(-dx,dy)+f(dx,-dy)+f(dx,dy),dx,dy);};
  if(std::abs(eval([](double x,double y){return x*x+y*y;})-4)>2e-14)return 2;
  if(std::abs(eval([](double x,double y){return x*y;}))>1e-14)return 3;
 }
 if(microsim_2d_gradient_resolved(1e-9,1e-8,1.3e-8))return 5;
 if(!microsim_2d_gradient_resolved(1e5,1e-8,1.3e-8))return 6;
 double previous=1e10;
 for(int n:{16,32,64}){
  double dx=1./n,dy=1.3/n,kx=2.,ky=3.;
  double lap=microsim_laplacian_2d(1.,cos(kx*dx),cos(kx*dx),cos(ky*dy),cos(ky*dy),4*cos(kx*dx)*cos(ky*dy),dx,dy);
  double error=std::abs(lap+kx*kx+ky*ky);
  if(error>previous*.27)return 4;previous=error;
 }
 puts("PASS rectangular 2D constant/quadratic moments and second-order Fourier convergence");
}
