#!/usr/bin/env python3
"""Prepare synthetic F1 inputs; all physical evolution runs in compiled C/CUDA."""
import argparse,json,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def replace(text,key,value):
 line=f'{key} = {value};';pattern=rf'(?m)^{re.escape(key)}\s*=.*?;'
 return re.sub(pattern,line,text) if re.search(pattern,text) else text+'\n'+line+'\n'
def h(p):return p**3*(10-15*p+6*p*p)
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--case',choices=['nucleus2d','cubic3d'],required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--grid',type=int);a.add_argument('--radius-nm',type=float,default=70);a.add_argument('--spacing-nm',type=float);a.add_argument('--box-um',type=float,default=1.28);a.add_argument('--epsilon-nm',type=float);a.add_argument('--supersaturation',type=float,default=.008);a.add_argument('--dt',type=float,default=1e-10);a.add_argument('--steps',type=int);a.add_argument('--save',type=int);args=a.parse_args()
 dim=2 if args.case=='nucleus2d' else 3;N=args.grid if args.grid is not None else (512 if dim==2 else 256);dx=(args.spacing_nm*1e-9 if args.spacing_nm is not None else (1.25e-9 if dim==2 else args.box_um*1e-6/N));eps=(args.epsilon_nm if args.epsilon_nm is not None else (20 if dim==2 else 40))*1e-9;steps=args.steps if args.steps is not None else (2000 if dim==2 else 20);save=args.save if args.save is not None else (100 if dim==2 else max(1,steps))
 ca=.926+1.2*args.supersaturation;cb=.817+args.supersaturation
 if not np.isfinite([dx,eps,args.dt,args.radius_nm,args.supersaturation]).all() or N<8 or dx<=0 or eps<=0 or args.dt<=0 or steps<0 or save<1 or not (0<ca<1 and 0<cb<1):a.error('Invalid mesh, time, interface, or composition parameters.')
 if args.output.exists():a.error('Output directory already exists; choose a fresh directory.')
 x=(np.arange(N)-(N-1)/2)*dx;scale=np.sqrt((3*.1*eps/(2*2.94))/(6*2.94*.1/eps));L=N*dx
 if dim==2:
  R=args.radius_nm*1e-9
  if R<=0 or R+12*scale>=L/2:a.error('Nucleus/interface does not fit inside this periodic cell.')
  radius=np.hypot(x[:,None],x[None,:]);p=(1/(1+np.exp(np.clip((radius-R)/scale,-700,700))))[:,:,None];radii=[R];centres=[[0,0,0]];cp=[4.5e11,1.5e11,1.5e11];cm=[3.2e11,.7e11,1.25e11]
 else:
  radii=(np.array([120,130,140,150,125,135,145,155])*1e-9).tolist();centres=(np.array([(i,j,k) for i in [-.25,.25] for j in [-.25,.25] for k in [-.25,.25]])*L).tolist()
  if max(radii)+12*scale>=L/4:a.error('Particles/interfaces do not fit the octants; increase box size.')
  p=np.zeros((N,N,N))
  for centre,R in zip(centres,radii):
   radius=np.sqrt((x[:,None,None]-centre[0])**2+(x[None,:,None]-centre[1])**2+(x[None,None,:]-centre[2])**2);q=1/(1+np.exp(np.clip((radius-R)/scale,-700,700)));p+=q*(1-p)
  cp=[4.5e11,1.5e11,1.2e11];cm=[3.2e11,.7e11,.85e11]
 c=h(p)*ca+(1-h(p))*cb;text=(ROOT/'KKS_FD_CUDA_MPI/examples/2d_f1/generalized.in').read_text()
 for key,value in dict(DIMENSION=dim,MESH_X=N,MESH_Y=N,MESH_Z=1 if dim==2 else N,DELTA_X=dx,DELTA_Y=dx,DELTA_Z=dx,DELTA_t=args.dt,NTIMESTEPS=steps,SAVET=save,TRACK_PROGRESS=100 if dim==2 else 1,RESTART=1,STARTTIME=0,epsilon=eps,Function_F=1,Function_anisotropy=0,ANTI_TRAPPING=0,WRITEFORMAT='BINARY',INHOM_TOLERANCE='1e-19',INHOM_MAX_ITERATIONS=500,INHOM_STRICT=1,INHOM_LOADING='STRESS_FREE',INHOM_DIAGNOSTICS=0).items():text=replace(text,key,str(value))
 for pattern,line in [(r'(?m)^A\s*=\s*\{1,.*?;','A = {1,1.2e4};'),(r'(?m)^DIFFUSIVITY\s*=\s*\{1,0,.*?;','DIFFUSIVITY = {1,0,2e-10};'),(r'(?m)^EIGEN_STRAIN\s*=\s*\{0,.*?;','EIGEN_STRAIN = {0,.001,.001,.001,0,0,0};')]:text=re.sub(pattern,line,text)
 for phase,constants in enumerate([cp,cm]):text=re.sub(r'(?m)^VOIGT_ISOTROPIC\s*=\s*\{'+str(phase)+r',.*?;','VOIGT_ISOTROPIC = {'+str(phase)+','+','.join(map(str,constants))+'};',text)
 if dim==3:text=re.sub(r'(?m)^INHOM_2D_MODE\s*=.*?;\s*','',text)
 out=args.output;out.mkdir(parents=True);(out/'input.in').write_text(text);(out/'fill.in').write_text('# Restart supplies the fields; no filling operation is used.\n');path=out/'DATA/Processor_0/state_0.vtk';path.parent.mkdir(parents=True)
 with path.open('wb') as f:
  f.write(f'# vtk DataFile Version 3.0\nSynthetic F1 initial condition\nBINARY\nDATASET STRUCTURED_POINTS\nDIMENSIONS {N} {N} {p.shape[2]}\nORIGIN 0 0 0\nSPACING {dx:.17e} {dx:.17e} {dx:.17e}\nPOINT_DATA {p.size}\n'.encode())
  for name,field in [('alpha',p),('liquid',1-p),('Composition_Al',c)]:f.write(f'SCALARS {name} double 1\nLOOKUP_TABLE default\n'.encode());field.astype('>f8').tofile(f);f.write(b'\n')
 (out/'case.json').write_text(json.dumps(dict(case=args.case,grid=list(p.shape),spacing_m=dx,epsilon_m=eps,dt_s=args.dt,steps=steps,radii_m=radii,centres_m=centres,supersaturation=args.supersaturation,initial_phase_compositions=[ca,cb],precipitate_cubic_Pa=cp,matrix_cubic_Pa=cm,mean_composition=float(c.mean()),array_order='physical_x,physical_y,physical_z; legacy MicroSim VTK scalar order'),indent=2)+'\n');print(out.resolve())
if __name__=='__main__':main()
