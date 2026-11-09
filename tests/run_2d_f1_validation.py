#!/usr/bin/env python3
"""Execute isolated CUDA F1 cases and verify mechanics AND coupled updates.

No CUDA Green tensor or CUDA phase/transport routines are used by the oracles.
The mechanical oracle independently minimizes the compatible-strain energy.
The F1 oracle analytically solves the two-parabola KKS equations and applies
periodic finite-volume transport and the constrained phase-energy derivative.
"""
import argparse,json,os,re,shlex,subprocess,time
from pathlib import Path
import numpy as np
from check_inhom_2d_fields import check
ROOT=Path(__file__).resolve().parents[1]

def write_state(path,p,c,dx,dy):
    nx,ny=p.shape
    text=f'# vtk DataFile Version 3.0\nIndependent F1 initial state\nASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS {ny} {nx} 1\nORIGIN 0 0 0\nSPACING {dy:.17e} {dx:.17e} {dx:.17e}\nPOINT_DATA {nx*ny}\n'
    for name,a in [('alpha',p),('liquid',1-p),('Composition_Al',c)]:
        text+=f'SCALARS {name} double 1\nLOOKUP_TABLE default\n'+''.join(f'{v:.17e}\n' for v in a.ravel())
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)

def read_state(path,shape):
    words=path.read_text().split();arrays={};n=np.prod(shape)
    for i,word in enumerate(words):
        if word=='SCALARS':arrays[words[i+1]]=np.array(words[i+6:i+6+n],float).reshape(shape)
    return arrays

def set_value(text,key,value):
    pattern=rf'(?m)^{re.escape(key)}\s*=.*?;'
    line=f'{key} = {value};'
    return re.sub(pattern,line,text) if re.search(pattern,text) else text+'\n'+line+'\n'

def h(p):return p**3*(10-15*p+6*p*p)

def f1_step(p,c,force,dx,dy,dt,antitrapping=False):
    # F_p=A_p*c_p^2+B_p*c_p+C_p, fixed by the supplied common tangent.
    A0,A1=1e4,1.2e4;eq0,eq1=.926,.817;V=1e-5
    B0=2*(A1*eq1-A0*eq0);C0=A0*eq0**2-A1*eq1**2
    hp=h(p);mu=(c+hp*B0/(2*A0))/(hp/(2*A0)+(1-hp)/(2*A1))
    c0=(mu-B0)/(2*A0);c1=mu/(2*A1)
    assert np.max(abs(hp*c0+(1-hp)*c1-c))<5e-15
    assert np.max(abs(2*A0*c0+B0-2*A1*c1))<1e-10
    psi_difference=C0-A0*c0*c0+A1*c1*c1
    gamma,epsilon,alpha=.1,8e-8,2.94
    kappa=3*gamma*epsilon/(2*alpha);theta=6*alpha*gamma/epsilon
    M0,M1=2e-10/(2*A0),1e-9/(2*A1)
    tau=3*.783333*kappa*(eq1-eq0)**2/(M1*theta*V)
    # Rectangular-grid Laplacian derived from zero sum and quadratic moments.
    diagonal=1/(2*(dx*dx+dy*dy));wx=1/(dx*dx)-2*diagonal;wy=1/(dy*dy)-2*diagonal
    lap=wx*(np.roll(p,1,0)+np.roll(p,-1,0)-2*p)+wy*(np.roll(p,1,1)+np.roll(p,-1,1)-2*p)
    lap+=diagonal*(sum(np.roll(np.roll(p,i,0),j,1) for i in [-1,1] for j in [-1,1])-4*p)
    # Scalar energy: h*f0/V+(1-h)*f1/V + W_el + theta*p^2*(1-p)^2
    # + kappa*|grad p|^2. MicroSim's binary cross derivatives are constrained:
    # chemical/elastic phase forces have opposite signs, then updatePhi projects.
    tangent_force=30*p*p*(1-p)**2*(psi_difference/V+force)+2*theta*p*(1-p)*(1-2*p)-2*kappa*lap
    pnew=p-dt*tangent_force/tau
    mobility=M0*hp+M1*(1-hp)
    divergence=np.zeros_like(c)
    for axis,spacing in [(0,dx),(1,dy)]:
        # Each face has one flux; neighboring cells receive opposite contributions.
        flux=.5*(mobility+np.roll(mobility,-1,axis))*(np.roll(mu,-1,axis)-mu)/spacing
        divergence+=(flux-np.roll(flux,1,axis))/spacing
    if antitrapping:
        amplitude=np.sqrt(2*kappa/theta)*(c1-c0)*(pnew-p)/dt
        transverse=[(np.roll(p,-1,axis)-np.roll(p,1,axis))/(2*spacing) for axis,spacing in [(0,dx),(1,dy)]]
        for axis,spacing in [(0,dx),(1,dy)]:
            normal=(np.roll(p,-1,axis)-p)/spacing
            tangent=.5*(transverse[1-axis]+np.roll(transverse[1-axis],-1,axis))
            norm=np.hypot(normal,tangent)
            direction=np.divide(normal,norm,out=np.zeros_like(normal),where=norm*min(dx,dy)>64*np.finfo(float).eps)
            flux=.5*(amplitude+np.roll(amplitude,-1,axis))*direction*(1-2e-10/1e-9)
            divergence+=(flux-np.roll(flux,1,axis))/spacing
    return pnew,c+dt*divergence

def scalar_energy(p,c,dx,dy,mechanical_potential):
    A0,A1=1e4,1.2e4;eq0,eq1=.926,.817;B0=2*(A1*eq1-A0*eq0);C0=A0*eq0**2-A1*eq1**2
    hp=h(p);mu=(c+hp*B0/(2*A0))/(hp/(2*A0)+(1-hp)/(2*A1))
    c0=(mu-B0)/(2*A0);c1=mu/(2*A1)
    chemical=np.mean(hp*(A0*c0*c0+B0*c0+C0)+(1-hp)*A1*c1*c1)/1e-5
    kappa=3*.1*8e-8/(2*2.94);theta=6*2.94*.1/8e-8
    wd=1/(2*(dx*dx+dy*dy));wx=1/(dx*dx)-2*wd;wy=1/(dy*dy)-2*wd
    gradient=wx*(np.roll(p,-1,0)-p)**2+wy*(np.roll(p,-1,1)-p)**2
    gradient+=wd*sum((np.roll(np.roll(p,-1,0),j,1)-p)**2 for j in [-1,1])
    return float(chemical+np.mean(theta*p*p*(1-p)**2+kappa*gradient)+mechanical_potential)

def run_case(base,executable,launcher,name,mode,loading,nx=9,ny=7,uniform=False,homogeneous=False,eigen_z=None,steps=4,maxiter=500,expect_failure=False,dt=1e-11,antitrapping=False):
    folder=base/name;folder.mkdir(parents=True,exist_ok=False)
    dx,dy=1e-8,1.3e-8
    x,y=np.indices((nx,ny));p=.35+.15*np.cos(2*np.pi*x/nx)*np.cos(2*np.pi*y/ny)
    c=.84+.004*np.sin(2*np.pi*x/nx)*np.cos(2*np.pi*y/ny)
    if uniform:p[:]=.35;c[:]=.84
    initial_mass=float(c.mean());cp=(4.5,1.5,1.2);cm=(3.2,.7,.85)
    if homogeneous:cp=cm
    eig=(.005,.01,(.003 if mode==1 else 0.) if eigen_z is None else eigen_z)
    sigma=(1e8,-2e8,0,0,0,5e7) if loading=='APPLIED_STRESS' else (0,)*6
    E=(.002,-.001,.004 if mode==1 else 0.,0,0,.003) if loading=='CONSTRAINED' else (0,)*6
    text=(ROOT/'KKS_FD_CUDA_MPI/Validate_Input_2d_generalized.in').read_text()
    changes={'MESH_X':nx,'MESH_Y':ny,'RESTART':1,'NTIMESTEPS':steps,'INHOM_2D_MODE':mode,'INHOM_LOADING':loading,'INHOM_MAX_ITERATIONS':maxiter,
        'DELTA_t':dt,'Function_anisotropy':0,'ANTI_TRAPPING':int(antitrapping),'APPLIED_STRESS':'{'+','.join(map(str,sigma))+'}',
        'HOMOGENEOUS_STRAIN':'{'+','.join(map(str,E))+'}'}
    for key,value in changes.items():text=set_value(text,key,str(value))
    text=re.sub(r'(?m)^A = \{1,.*?;', 'A = {1,1.2e4};',text)
    text=re.sub(r'(?m)^DIFFUSIVITY = \{1,0,.*?;','DIFFUSIVITY = {1,0,2e-10};',text)
    text=re.sub(r'(?m)^EIGEN_STRAIN = \{0,.*?;','EIGEN_STRAIN = {0,'+','.join(map(str,(*eig,0,0,0)))+'};',text)
    for phase,constants in [(0,cp),(1,cm)]:
        text=re.sub(r'(?m)^VOIGT_ISOTROPIC = \{'+str(phase)+r',.*?;','VOIGT_ISOTROPIC = {'+str(phase)+','+','.join(str(v*1e11) for v in constants)+'};',text)
    (folder/'input.in').write_text(text);(folder/'fill.in').write_text('FILLCYLINDER = {0,4,3,0,1,2};\n')
    write_state(folder/'DATA/Processor_0/state_0.vtk',p,c,dx,dy)
    start=time.monotonic();command=launcher+[str(executable),'input.in','fill.in','state']
    completed=subprocess.run(command,cwd=folder,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
    (folder/'run.log').write_text(completed.stdout)
    if expect_failure:
        assert completed.returncode!=0,'Unconverged solve was accepted'
        assert 'No force returned.' in completed.stdout,'Failure was not the intended mechanical iteration limit'
        assert not (folder/'DATA/Processor_0/state_1.vtk').exists(),'Advanced with unconverged mechanics'
        return dict(name=name,status='PASS',expected_failure=True,seconds=time.monotonic()-start)
    assert completed.returncode==0,completed.stdout[-4000:]
    parameters=np.genfromtxt(folder/'f1_parameters.csv',delimiter=',',names=True)
    A0,A1=1e4,1.2e4;eq0,eq1=.926,.817;kappa=3*.1*8e-8/(2*2.94);theta=6*2.94*.1/8e-8
    M1=1e-9/(2*A1);tau=3*.783333*kappa*(eq1-eq0)**2/(M1*theta*1e-5)
    expected=[A0,A1,2*(A1*eq1-A0*eq0),0,A0*eq0**2-A1*eq1**2,0,2e-10/(2*A0),M1,kappa,theta,1/tau,1e-5,int(antitrapping)]
    actual_parameters=[float(parameters[k]) for k in parameters.dtype.names]
    assert np.allclose(actual_parameters,expected,rtol=2e-13,atol=1e-15),'F1 initialization differs from input thermodynamics'
    history=[];previous_energy=None
    for step in range(steps):
        csv=folder/f'elastic_rank0_step{step}.csv'
        data=np.loadtxt(csv,delimiter=',',skiprows=1)
        assert np.max(abs(data[:,0].reshape(nx,ny)-h(p)))<2e-11,'Mechanical interpolation differs from current phase field'
        result=check(csv,nx,ny,mode,dy=1.3,stress=(sigma[0]/1e11,sigma[1]/1e11,sigma[5]/1e11),
            constrained=loading=='CONSTRAINED',prescribed=E,cp_values=cp,cm_values=cm,eigen_values=eig,with_fields=True)
        if step==0 and uniform and loading=='STRESS_FREE':
            expected=np.array([eig[0],eig[1],eig[2],0,0,0])*h(p)[0,0]
            assert np.max(abs(np.array(result['mean_strain'])-expected))<1e-11
            assert np.max(abs(np.array(result['mean_stress_pa'])))<.01
        if step==0 and mode==1 and eigen_z==0. and not uniform and not homogeneous:
            assert abs(result['mean_strain'][2])>1e-8,'Generalized Ezz was clamped instead of evaluated'
        total_energy=scalar_energy(p,c,dx,dy,result['reference_potential_density'])
        if previous_energy is not None and not antitrapping:assert total_energy<=previous_energy+1e-5,(name,step,'energy increased',total_energy-previous_energy)
        previous_energy=total_energy
        p,c=f1_step(p,c,result.pop('reference_force'),dx,dy,dt,antitrapping);result.pop('reference_strain')
        actual=read_state(folder/f'DATA/Processor_0/state_{step+1}.vtk',(nx,ny))
        pe=float(np.max(abs(actual['alpha']-p)));ce=float(np.max(abs(actual['Composition_Al']-c)))
        mass=abs(float(actual['Composition_Al'].mean())-initial_mass)
        assert pe<2e-10,(name,step,'phase update',pe)
        assert ce<2e-12,(name,step,'composition update',ce)
        assert mass<2e-13,(name,step,'mass',mass)
        assert np.max(abs(actual['alpha']+actual['liquid']-1))<2e-14
        assert 0<=actual['alpha'].min()<=actual['alpha'].max()<=1
        assert np.isfinite(actual['Composition_Al']).all()
        history.append(dict(step=step,total_potential_density=total_energy,phase_max_abs_error=pe,composition_max_abs_error=ce,mass_mean_drift=mass,mechanics=result))
    return dict(name=name,status='PASS',model=mode,loading=loading,dt=dt,antitrapping=antitrapping,shape=[nx,ny],steps=history,seconds=time.monotonic()-start)

def main():
    a=argparse.ArgumentParser();a.add_argument('--executable',type=Path,default=ROOT/'KKS_FD_CUDA_MPI/microsim_kks_fd_cuda_mpi');a.add_argument('--launcher',default='mpiexec -n 1');a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    sources=[p for p in (ROOT/'KKS_FD_CUDA_MPI').rglob('*') if p.suffix in ('.c','.cu','.h','.cuh','.hpp')]
    assert args.executable.stat().st_mtime >= max(p.stat().st_mtime for p in sources),'Rebuild the CUDA executable after source changes before validating.'
    base=args.output.resolve();base.mkdir(parents=True,exist_ok=True);records=[]
    try:
        for mode,label in [(1,'generalized'),(2,'intrinsic')]:
            for loading in ['STRESS_FREE','CONSTRAINED','APPLIED_STRESS']:
                name=label+'_'+loading.lower();records.append(run_case(base,args.executable.resolve(),shlex.split(args.launcher),name,mode,loading));print('PASS',name,flush=True)
        extras=[('uniform_generalized',1,dict(uniform=True)),('uniform_intrinsic',2,dict(uniform=True)),('even_generalized',1,dict(nx=10,ny=8)),('even_intrinsic',2,dict(nx=10,ny=8)),('homogeneous_generalized',1,dict(homogeneous=True)),('zero_zz_generalized',1,dict(eigen_z=0.)),('antitrapping_generalized',1,dict(antitrapping=True)),('antitrapping_intrinsic',2,dict(antitrapping=True)),('uniform_antitrapping',1,dict(antitrapping=True,uniform=True)),('time_coarse',1,dict(dt=1e-10,steps=4)),('time_medium',1,dict(dt=5e-11,steps=8)),('time_fine',1,dict(dt=2.5e-11,steps=16))]
        for name,mode,kw in extras:
            records.append(run_case(base,args.executable.resolve(),shlex.split(args.launcher),name,mode,'STRESS_FREE',**kw));print('PASS',name,flush=True)
        temporal=[]
        for name,step in [('time_coarse',4),('time_medium',8),('time_fine',16)]:
            state=read_state(base/name/f'DATA/Processor_0/state_{step}.vtk',(9,7))
            temporal.append(np.stack([state['alpha'],state['Composition_Al']]))
        errors=[float(np.linalg.norm(temporal[i]-temporal[i+1])) for i in [0,1]]
        ratio=errors[0]/errors[1]
        assert 1.7<ratio<2.3,('Expected first-order temporal convergence',ratio)
        records.append(dict(name='temporal_convergence',status='PASS',error_coarse_medium=errors[0],error_medium_fine=errors[1],ratio=ratio))
        print('PASS temporal_convergence',ratio,flush=True)
        records.append(run_case(base,args.executable.resolve(),shlex.split(args.launcher),'reject_unconverged',1,'STRESS_FREE',maxiter=1,expect_failure=True));print('PASS reject_unconverged',flush=True)
    finally:
        (base/'validation.json').write_text(json.dumps(dict(checks=records),indent=2)+'\n')
    print('ALL 2D FUNCTION F1 CHECKS PASSED',flush=True)
if __name__=='__main__':main()
