#include <cmath>
#include <cstdio>
#include <random>
#include "../KKS_FD_CUDA_MPI/solverloop/inhom_math.hpp"

// Independent energy from the full symmetric tensor contraction.
static double energy(const double *total, const double *eigen, const double *ch,
                     const double *dc, double h) {
    double e[6], stress[6];
    for(int i=0;i<6;++i)e[i]=total[i]-(i<3?h*eigen[i]:0);
    double c11=ch[0]+dc[0]*(2*h-1),c12=ch[1]+dc[1]*(2*h-1),c44=ch[2]+dc[2]*(2*h-1);
    for(int i=0;i<3;++i)stress[i]=c11*e[i]+c12*(e[(i+1)%3]+e[(i+2)%3]);
    for(int i=3;i<6;++i)stress[i]=2*c44*e[i];
    double ans=0;
    for(int i=0;i<6;++i)ans+=0.5*e[i]*stress[i]*(i<3?1:2);
    return ans;
}
int main(){
    std::mt19937 gen(193);std::uniform_real_distribution<double> rand(-1,1);
    double worst=0;
    for(int mode=0;mode<3;++mode)for(int n=0;n<1000;++n){
        double ch[3]={4.0,1.1,1.2},dc[3]={.4,.15,.1},total[6],eig[3]={.005,.009,mode==2?0:.013},e[6];
        for(int i=0;i<6;++i)total[i]=.02*rand(gen);
        if(mode){total[3]=total[4]=0;}
        if(mode==2)total[2]=0;
        double h=.5+.45*rand(gen),delta=1e-6;
        for(int i=0;i<6;++i)e[i]=total[i]-(i<3?h*eig[i]:0);
        double actual=inhom_elastic_partial(e,eig,ch,dc,h);
        double ref=(energy(total,eig,ch,dc,h+delta)-energy(total,eig,ch,dc,h-delta))/(2*delta);
        double err=fabs(actual-ref)/fmax(1e-6,fabs(actual));worst=fmax(worst,err);
        if(err>1e-6)return 1;
        double s[3];inhom_average_compliance(s,ch,mode);
        double stress[3]={.3,-.1,.2};
        double ex=s[0]*stress[0]+s[1]*stress[1],ey=s[1]*stress[0]+s[0]*stress[1];
        if(mode==2 && (fabs(ch[0]*ex+ch[1]*ey-stress[0])>1e-14 || fabs(ch[1]*ex+ch[0]*ey-stress[1])>1e-14))return 2;
    }
    // A prescribed total out-of-plane eigenstrain stores energy; intrinsic 2D has none.
    double total[6]={},eig[3]={0,0,.01},ch[3]={4,1,1},dc[3]={};
    if(fabs(energy(total,eig,ch,dc,1)-.0002)>1e-15)return 3;
    printf("3000 fixed-strain energy derivatives passed; max relative error %.3g\n",worst);
}
