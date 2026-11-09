#!/usr/bin/env python3
"""Check common chemical potential and restart time/data preservation, no GPU."""
import hashlib,json,re,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def command(script,*args,ok=True):
 p=subprocess.run([sys.executable,str(ROOT/'tests'/script),*map(str,args)],capture_output=True,text=True)
 if ok and p.returncode:raise AssertionError(p.stderr)
 if not ok and not p.returncode:raise AssertionError('Expected rejection')
 return p
def fields(path):
 arrays={}
 with path.open('rb') as f:
  while line:=f.readline():
   if line.startswith(b'POINT_DATA'):n=int(line.split()[1])
   if line.startswith(b'SCALARS'):
    key=line.split()[1].decode();assert f.readline().strip()==b'LOOKUP_TABLE default';arrays[key]=np.fromfile(f,dtype='>f8',count=n).astype(float);assert f.read(1)==b'\n'
 return arrays
class Cases(unittest.TestCase):
 def test_common_potential_and_safe_output(self):
  with tempfile.TemporaryDirectory() as tmp:
   for case in ['nucleus2d','cubic3d']:
    out=Path(tmp)/case;extra=['--spacing-nm','40'] if case=='nucleus2d' else []
    command('prepare_elasticity_case.py','--case',case,'--grid',16,'--steps',0,'--output',out,*extra)
    d=fields(out/'DATA/Processor_0/state_0.vtk');p=d['alpha'];hp=p**3*(10-15*p+6*p*p);c=d['Composition_Al'];mu=(c+hp*1088/20000)/(hp/20000+(1-hp)/24000)
    self.assertLess(np.max(abs(mu-19800)),2e-11);self.assertTrue(np.array_equal(p+d['liquid'],np.ones_like(p)))
    before=(out/'input.in').read_bytes();command('prepare_elasticity_case.py','--case',case,'--output',out,ok=False);self.assertEqual(before,(out/'input.in').read_bytes())
 def test_restart_remaps_time_without_changing_fields(self):
  with tempfile.TemporaryDirectory() as tmp:
   source=Path(tmp)/'source';command('prepare_elasticity_case.py','--case','nucleus2d','--grid',16,'--spacing-nm',40,'--output',source)
   data=source/'DATA/Processor_0';(data/'state_2000.vtk').write_bytes((data/'state_0.vtk').read_bytes());out=Path(tmp)/'restart'
   command('restart_elasticity_case.py','--source',source,'--step',2000,'--output',out,'--dt',2e-10,'--until',2e-6)
   m=json.loads((out/'restart.json').read_text());self.assertEqual(m['start_count'],1000);self.assertEqual(m['additional_steps'],9000)
   self.assertEqual(hashlib.sha256((data/'state_2000.vtk').read_bytes()).digest(),hashlib.sha256((out/'DATA/Processor_0/state_1000.vtk').read_bytes()).digest())
   command('restart_elasticity_case.py','--source',source,'--step',2000,'--output',Path(tmp)/'bad','--dt',3e-10,'--until',2e-6,ok=False)
if __name__=='__main__':unittest.main()
