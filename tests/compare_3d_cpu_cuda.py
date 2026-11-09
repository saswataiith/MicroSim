#!/usr/bin/env python3
"""Independent CPU/FFTW-MPI check of the fixed cubic CUDA benchmark's fields."""
import argparse,json,shlex,subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cuda-csv',type=Path,required=True);p.add_argument('--grid',type=int,default=64);p.add_argument('--cpu-executable',type=Path,required=True);p.add_argument('--ranks',type=int,default=1);p.add_argument('--mpiexec',default='mpiexec');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
N=a.grid;n=N**3
if N<1 or a.ranks<1 or a.output.exists():p.error('Use positive grid/rank counts and a fresh output directory.')
d=np.loadtxt(a.cuda_csv,delimiter=',',skiprows=1)
if d.shape!=(n,8) or not np.isfinite(d).all():p.error('Expected a complete single-GPU volume diagnostic, not a slice.')
a.output.mkdir(parents=True);hfile=(a.output/'h.bin').resolve();d[:,0].tofile(hfile);prefix=(a.output/'cpu').resolve()
cmd=shlex.split(a.mpiexec)+['-n',str(a.ranks),str(a.cpu_executable.resolve()),str(N),str(hfile),str(prefix)]
q=subprocess.run(cmd,text=True,capture_output=True);(a.output/'cpu.log').write_text(q.stdout+q.stderr)
if q.returncode:raise RuntimeError('CPU solve failed; see cpu.log')
fields=np.empty((14,n));coverage=np.zeros(n,dtype=bool)
for rank in range(a.ranks):
 with Path(str(prefix)+f'-rank{rank}.bin').open('rb') as f:
  start,count=np.fromfile(f,dtype=np.uint64,count=2).astype(int);v=np.fromfile(f,dtype=float).reshape(14,count)
 offset=start*N*N;fields[:,offset:offset+count]=v;coverage[offset:offset+count]=True
if not coverage.all():raise RuntimeError('CPU slabs do not cover the volume')
e=d[:,1:7].T;hp=d[:,0];cp=np.array([4.5e11,1.5e11,1.2e11]);cm=np.array([3.2e11,.7e11,.85e11]);C=cm[:,None]+(cp-cm)[:,None]*hp;ee=e.copy();ee[:3]-=.001*hp;s=np.empty_like(ee)
for i in range(3):s[i]=C[0]*ee[i]+C[1]*(ee[(i+1)%3]+ee[(i+2)%3])
s[3:]=2*C[2]*ee[3:];W=.5*np.sum(ee*s*np.array([1,1,1,2,2,2])[:,None],axis=0)
r=dict(grid=N,ranks=a.ranks,strain_max_error=float(np.max(abs(e-fields[:6]))),stress_max_error_Pa=float(np.max(abs(s-fields[6:12]))),energy_max_error_J_per_m3=float(np.max(abs(W-fields[12]))),elastic_force_max_error_Pa=float(np.max(abs(d[:,7]-fields[13]))))
(a.output/'comparison.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if r['strain_max_error']>=1e-10 or r['stress_max_error_Pa']>=10 or r['elastic_force_max_error_Pa']>=.1:raise SystemExit('Independent reference tolerance exceeded')
