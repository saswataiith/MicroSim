#!/usr/bin/env python3
"""Host checks for the v6 table path and CUDA local algebra; no GPU emulation claim."""
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / 'generate_csv_files_from_tdbs/from_Thermocalc'
sys.path.insert(0, str(EXPORT))
from table_utils import output_directory, phase_frame, common_phase_frame, write_composition_tables


def function(text, signature):
    start=text.index(signature);end=text.index('{',start)+1;depth=1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    return text[start:end]


class Tables(unittest.TestCase):
    def test_repeated_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=output_directory('test',Path(tmp));self.assertEqual(output_directory('test',Path(tmp)),p)
            with self.assertRaises(ValueError):output_directory('../other',Path(tmp))

    def test_temperature_alignment_and_export(self):
        def g(element,t,y):return types.SimpleNamespace(label=element+' in phase',x=t,y=y)
        a=phase_frame({'1':g('W',[1000,1001,1002,1003],[.1,.2,.3,.4])},'WC',['W'])
        b=phase_frame({'1':g('W',[1001,1002,1003,1004],[.5,.6,.7,.8])},'LIQUID',['W'])
        both=common_phase_frame([a,b]);self.assertEqual(list(both.Temperature),[1001,1002,1003])
        self.assertEqual(both.iloc[0]['W@WC'],.2)
        with tempfile.TemporaryDirectory() as tmp:
            write_composition_tables(both,['WC'],'LIQUID',['W'],Path(tmp))
            self.assertTrue((Path(tmp)/'Composition_WC.csv').exists())
            self.assertEqual((Path(tmp)/'Composition_WC.csv').read_text().splitlines()[1],'1001,0.2,0.5')

    def test_short_or_nonfinite_tables_rejected(self):
        import pandas as pd
        for x in [[1000,1001],[1000,1001,1002]]:
            f=pd.DataFrame({'Temperature':x,'W@WC':[float('nan')]*len(x)})
            with self.assertRaises(ValueError):common_phase_frame([f])

    def test_hessian_uses_phase_mole_composition(self):
        old=sys.modules.get('tc_python');sys.modules['tc_python']=types.ModuleType('tc_python')
        try:
            tc=importlib.import_module('tcsetup')
            obj=tc.setup(None,'D','M',['W','C'],'FE',[.8,.05],'mass',1500,1400,'case')
            class Calc:
                def set_condition(self,*x):return self
                def run_poly_command(self,x):self.command=x;return self
                def disable_global_minimization(self):return self
                def calculate(self):return self
            c=Calc();obj.Hessian_calculation(c,1450,[.2,.1])
            self.assertIn('X(W)=0.2',c.command);self.assertIn('X(C)=0.1',c.command)
            self.assertNotIn('W(W)',c.command)
        finally:
            if old is None:sys.modules.pop('tc_python',None)
            else:sys.modules['tc_python']=old


class HostSource(unittest.TestCase):
    def test_actual_bridge_and_both_chemical_branches(self):
        source=(ROOT/'KKS_FD_CUDA_MPI/solverloop/computeElasticInhom.cu').read_text()
        chemistry=(ROOT/'KKS_FD_CUDA_MPI/solverloop/computeDrivingForce.cu').read_text()
        interpolation=(ROOT/'KKS_FD_CUDA_MPI/functions/functionH.cu').read_text()
        prefix='''#include <cmath>
#include <cstdio>
#define MAX_NUM_PHASES 2
#define MAX_NUM_COMP 2
struct D{long x,y,z;};D threadIdx{1,1,1},blockIdx{0,0,0},blockDim{1,1,1};
double calcPhaseEnergy(double**,long,double*,double*,double*,long,long,long){return 0;}
double calcDiffusionPotential(double**,long,long,double*,double*,long,long,long){return 0;}
double calcDoubleWellDerivative(double**,long,double*,double*,double*,long,long){return 0;}
void zeroEnergy(double,double*,double* p){*p=0;}
void (*free_energy_tdb_dev[2])(double,double*,double*)={zeroEnergy,zeroEnergy};
'''
        bodies='\n'.join([function(interpolation,'double calcInterp5thDiff('),
            function(source,'void inh_bridge_dfeldphi('),
            function(source,'void inh_eigsts_avg2('),
            function(chemistry,'void __computeDrivingForce__('),
            function(chemistry,'void __computeDrivingForce_02__(')])
        main=r'''
double h(double p){return p*p*p*(10-15*p+6*p*p);}
int main(){double x[27]={},y[27]={},f0[27]={},f1[27]={},cp0[27]={},cp1[27]={},u[27]={};double* phi[]={x,y};double* force[]={f0,f1};double* pc[]={cp0,cp1};double* mu[]={u};double z[16]={};long map[]={0,1};double raw[]={2.7};
for(int branch=0;branch<2;branch++)for(int precip=0;precip<2;precip++)for(double v:{.2,.5,.8}){
x[13]=precip==0?v:1-v;y[13]=1-x[13];f0[13]=f1[13]=0;
inh_bridge_dfeldphi(phi,force,raw,precip,2,9,3,3,3,3);
if(branch==0)__computeDrivingForce__(phi,nullptr,force,pc,z,z,z,1,z,z,z,1,2,2,3,3,3,3,9,3,1);
else __computeDrivingForce_02__(phi,nullptr,force,pc,mu,1,z,z,z,1,1000,map,2,2,3,3,3,3,9,3,1);
double fd=raw[0]*(h(v+1e-6)-h(v-1e-6))/(2e-6);
// MicroSim's binary constrained derivative gives opposite phase forces.
// updatePhi projects the pair; the half difference is the scalar derivative.
if(fabs(force[precip][13]-fd)>1e-8||fabs(force[1-precip][13]+fd)>1e-8)return 1;
}
threadIdx={0,0,0};double e1[1],e2[1],hp[]={.5},ch[]={300,100,80},ct[]={0,0,0};
inh_eigsts_avg2(e1,e2,hp,ch,ct,.005,2,1);
if(fabs(e1[0]-1.75)>1e-12||fabs(e2[0]-2.25)>1e-12)return 2;
return 0;}
'''
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);(tmp/'check.cpp').write_text(prefix+bodies+main)
            p=subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-include','initializer_list',str(tmp/'check.cpp'),'-o',str(tmp/'check')],text=True,capture_output=True)
            self.assertEqual(p.returncode,0,p.stderr)
            subprocess.run([str(tmp/'check')],check=True)

    def test_actual_cuda_table_initialization(self):
        source=(ROOT/'KKS_FD_CUDA_MPI/functions/functionF.cu').read_text()
        fn=function(source,'void function_F_04_init_propertymatrices(')
        prefix='''#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
typedef struct {long numThermoPhases,numComponents,numPhases;char **phases_tdb;long *thermo_phase_host;} domainInfo;
typedef struct {int unused;} controls;
typedef struct {double T;double ***F0_A_host,**F0_B_host,*F0_C_host;} simParameters;
double **MallocM(long a,long b){double **p=calloc(a,sizeof(*p));for(long i=0;i<a;i++)p[i]=calloc(b,sizeof(**p));return p;}
double ****Malloc4M(long a,long b,long c,long d){double ****p=calloc(a,sizeof(*p));for(long i=0;i<a;i++){p[i]=calloc(b,sizeof(**p));for(long j=0;j<b;j++)p[i][j]=MallocM(c,d);}return p;}
void Free4M(double ****p,long a,long b,long c){}
void FreeM(double **p,long a){}
'''
        main='''
int main(int argc,char **argv){int n=atoi(argv[1]),m=n-1;char *names[]={"SOLID","LIQUID"};long map[]={0,1};domainInfo d={2,n,2,names,map};controls c={0};simParameters p={0};p.T=1001.5;p.F0_A_host=calloc(2,sizeof(double**));for(int i=0;i<2;i++)p.F0_A_host[i]=MallocM(m,m);p.F0_B_host=MallocM(2,m);p.F0_C_host=calloc(2,sizeof(double));function_F_04_init_propertymatrices(&d,&c,&p);double gs=p.F0_C_host[0],gl=0;for(int i=0;i<m;i++){double cs=.04+.01*i,cl=.08+.02*i;double ms=2*p.F0_A_host[0][i][i]*cs+p.F0_B_host[0][i],ml=2*p.F0_A_host[1][i][i]*cl;gs+=p.F0_B_host[0][i]*cs;for(int j=0;j<m;j++){double sj=.04+.01*j,lj=.08+.02*j;if(i!=j){ms+=p.F0_A_host[0][i][j]*sj;ml+=p.F0_A_host[1][i][j]*lj;}if(i<=j){gs+=p.F0_A_host[0][i][j]*cs*sj;gl+=p.F0_A_host[1][i][j]*cl*lj;}}if(fabs(ms-ml)>1e-11)return 2;gs-=ms*cs;gl-=ml*cl;}return fabs(gs-gl)>1e-11?3:0;}
'''
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);(tmp/'probe.c').write_text(prefix+fn+main)
            gsl=subprocess.check_output(['pkg-config','--cflags','--libs','gsl'],text=True).split()
            cc=os.environ.get('CC','clang' if shutil.which('clang') else 'cc')
            subprocess.run([cc,'-g','-O1','-fsanitize=address','-fno-omit-frame-pointer','-Wno-deprecated-declarations',str(tmp/'probe.c'),*gsl,'-lm','-o',str(tmp/'probe')],check=True,capture_output=True)
            for n in (2,3,5):
                m=n-1;case=tmp/str(n);tdb=case/'tdbs_encrypted';tdb.mkdir(parents=True)
                comp=['T,'+','.join(['cs'+str(i) for i in range(m)]+['cl'+str(i) for i in range(m)])]
                hs=['T,'+','.join('h'+str(i) for i in range(m*(m+1)//2))];hl=hs.copy()
                for t in range(1000,1004):
                    comp.append(','.join(map(str,[t]+[.04+.01*i for i in range(m)]+[.08+.02*i for i in range(m)])))
                    hs.append(','.join(map(str,[t]+[6+i for i in range(m)]+[.7 for i in range(m) for j in range(i+1,m)])))
                    hl.append(','.join(map(str,[t]+[2+i for i in range(m)]+[.3 for i in range(m) for j in range(i+1,m)])))
                for name,lines in [('Composition_SOLID.csv',comp),('HSN_SOLID.csv',hs),('HSN_LIQUID.csv',hl)]:
                    (tdb/name).write_text('\n'.join(lines)+'\n')
                env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0'} # scaffolding allocations live until process exit
                p=subprocess.run([str(tmp/'probe'),str(n)],cwd=case,env=env,text=True,capture_output=True)
                self.assertEqual(p.returncode,0,p.stdout+p.stderr)

    def test_spline_endpoints_all_affected_readers(self):
        for name in ['Grand_Potential_MPI/functions/functionF_04.h','Grand_Potential_MPI/functions/functionF_06.h','KKS_OpenCl/functions/functionF_04.h']:
            text=(ROOT/name).read_text()
            for line in text.splitlines():
                if ('acc_ES[a][k]' in line or 'spline_ES[a][k]' in line) and 'malloc' in line:
                    self.assertIn('malloc(2*',line)

    def test_spectral_math(self):
        code=r'''#include "inhom_math.hpp"
#include <cstdio>
int main(){
if(inhom_max3(0,1,2)!=2)return 1;
if(fabs(inhom_wave_number(2,5,1)-4*acos(-1.)/5)>1e-12)return 2;
if(inhom_wave_number(2,4,1)!=0)return 3;
double k[]={.2,.4,.7},c[]={300,100,80},o[6];inhom_green_inverse(o,k,c);
double a[3][3]={{c[0]*k[0]*k[0]+c[2]*(k[1]*k[1]+k[2]*k[2]),(c[1]+c[2])*k[0]*k[1],(c[1]+c[2])*k[0]*k[2]},
{(c[1]+c[2])*k[0]*k[1],c[0]*k[1]*k[1]+c[2]*(k[0]*k[0]+k[2]*k[2]),(c[1]+c[2])*k[1]*k[2]},
{(c[1]+c[2])*k[0]*k[2],(c[1]+c[2])*k[1]*k[2],c[0]*k[2]*k[2]+c[2]*(k[0]*k[0]+k[1]*k[1])}};
double b[3][3]={{o[0],o[5],o[4]},{o[5],o[1],o[3]},{o[4],o[3],o[2]}};
for(int i=0;i<3;i++)for(int j=0;j<3;j++){double z=0;for(int l=0;l<3;l++)z+=a[i][l]*b[l][j];if(fabs(z-(i==j))>1e-12)return 4;}
for(int i=0;i<3;i++)c[i]*=1e-9;double q[6];inhom_green_inverse(q,k,c);for(int i=0;i<6;i++)if(fabs(q[i]*1e-9-o[i])>1e-12)return 5;
return 0;}
'''
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);(tmp/'check.cpp').write_text(code)
            subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-I'+str(ROOT/'KKS_FD_CUDA_MPI/solverloop'),str(tmp/'check.cpp'),'-o',str(tmp/'check')],check=True,capture_output=True)
            subprocess.run([str(tmp/'check')],check=True)


if __name__=='__main__':unittest.main(verbosity=2)
