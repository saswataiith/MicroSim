#!/usr/bin/env python3
"""Restart a prepared F1 case, preserving physical time if the step changes."""
import argparse,json,re,shutil,math
from pathlib import Path
from prepare_elasticity_case import replace
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--step',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--dt',type=float);p.add_argument('--until',type=float,required=True,help='Absolute final time in seconds');p.add_argument('--save',type=int,default=500);a=p.parse_args()
text=(a.source/'input.in').read_text()
def value(key):
 m=re.search(r'(?m)^'+re.escape(key)+r'\s*=\s*([^;]+);',text)
 if not m:p.error('Missing '+key)
 return m.group(1).strip()
old_dt=float(value('DELTA_t'));dt=a.dt if a.dt is not None else old_dt;time=a.step*old_dt
if not all(map(math.isfinite,[old_dt,dt,a.until])) or old_dt<=0 or dt<=0 or a.step<0 or a.save<1 or a.until<=time:p.error('Invalid timestep, restart step, save interval or final time.')
start=round(time/dt);end=round(a.until/dt)
if abs(start*dt-time)>1e-12*max(dt,time) or abs(end*dt-a.until)>1e-12*max(dt,a.until):p.error('Requested times must be integer multiples of the new timestep.')
if int(value('Function_F'))!=1:p.error('This helper is for the published F1 fixtures.')
source=a.source/f'DATA/Processor_0/state_{a.step}.vtk'
if not source.is_file():p.error('Checkpoint does not exist: '+str(source))
if a.output.exists():p.error('Output directory exists; choose a fresh directory.')
a.output.mkdir(parents=True);dest=a.output/'DATA/Processor_0';dest.mkdir(parents=True);shutil.copy2(source,dest/f'state_{start}.vtk')
for key,val in dict(DELTA_t=dt,STARTTIME=start,NTIMESTEPS=end-start,SAVET=a.save,TRACK_PROGRESS=a.save,RESTART=1).items():text=replace(text,key,str(val))
(a.output/'input.in').write_text(text);shutil.copy2(a.source/'fill.in',a.output/'fill.in');(a.output/'restart.json').write_text(json.dumps(dict(source=str(a.source.resolve()),source_step=a.step,source_dt_s=old_dt,restart_time_s=time,dt_s=dt,start_count=start,additional_steps=end-start,final_time_s=end*dt),indent=2)+'\n');print(a.output.resolve())
