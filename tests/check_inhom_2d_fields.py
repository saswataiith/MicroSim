#!/usr/bin/env python3
"""Independent dense variational solve against opt-in CUDA diagnostic fields.

This oracle constructs its stiffness matrix by differentiating compatible strains
and contracting the constitutive tensor. It does not use the CUDA Green tensor,
fixed-point iteration, homogeneous-compliance helper, or force implementation.
"""
import argparse,json
from pathlib import Path
import numpy as np


def check(path,nx,ny,mode,dy=1.3,stress=(0,0,0),homogeneous=False,constrained=False,prescribed=(0,0,0,0,0,0),cp_values=(4.5,1.5,1.2),cm_values=(3.,1.,.8),eigen_values=None,with_fields=False):
    data=np.loadtxt(path,delimiter=',',skiprows=1)
    assert data.shape==(nx*ny,8) and np.isfinite(data).all()
    shape=(nx,ny); n=nx*ny; rootn=np.sqrt(n)
    nm=0 if constrained else (4 if mode==1 else 3)
    mean_slots=[0,1,5,2] if mode==1 else [0,1,5]
    target=np.array([*stress,0.] if mode==1 else stress)
    h=data[:,0].reshape(shape)
    eig=np.array(eigen_values if eigen_values is not None else [.005,.01,.01 if mode==1 else 0])
    # Stiffness scaled by 1e11 Pa; dimensionless displacement measured in dx.
    cp=np.array(cp_values);cm=np.array(cm_values)
    if homogeneous:cp=cm.copy()
    c=cm[:,None,None]+(cp-cm)[:,None,None]*h
    kx=2*np.pi*np.fft.fftfreq(nx)[:,None]
    ky=2*np.pi*np.fft.fftfreq(ny,d=dy)[None,:]
    if nx%2==0:kx[nx//2]=0
    if ny%2==0:ky[:,ny//2]=0
    def derivative(a,k):return np.fft.ifft2(1j*k*np.fft.fft2(a)).real
    def strain(q):
        u=q[:2*n].reshape(2,nx,ny);mean=np.array([prescribed[0],prescribed[1],2*prescribed[5],prescribed[2]]) if constrained else q[2*n:]/rootn
        e=np.zeros((6,nx,ny))
        e[0]=derivative(u[0],kx)+mean[0]
        e[1]=derivative(u[1],ky)+mean[1]
        e[5]=.5*(derivative(u[0],ky)+derivative(u[1],kx)+mean[2])
        if mode==1:e[2]=mean[3]
        return e
    def constitutive(e):
        s=np.zeros_like(e)
        for i in range(3):s[i]=c[0]*e[i]+c[1]*(e[(i+1)%3]+e[(i+2)%3])
        s[3:]=2*c[2]*e[3:]
        return s
    def gradient(s):
        g=np.empty(2*n+nm)
        g[:n]=(-derivative(s[0],kx)-derivative(s[5],ky)).ravel()
        g[n:2*n]=(-derivative(s[5],kx)-derivative(s[1],ky)).ravel()
        if not constrained:g[2*n:]=rootn*np.mean(s[mean_slots],axis=(1,2))
        return g
    eigen=np.zeros((6,nx,ny));eigen[:3]=eig[:,None,None]*h
    fixed=strain(np.zeros(2*n+nm))
    rhs=gradient(constitutive(eigen-fixed))
    if not constrained:rhs[2*n:]+=rootn*target
    eye=np.eye(2*n+nm)
    mat=np.column_stack([gradient(constitutive(strain(q)-fixed)) for q in eye])
    assert np.max(np.abs(mat-mat.T))<1e-12
    q=np.linalg.lstsq(mat,rhs,rcond=1e-12)[0]
    ref=strain(q)
    gpu=data[:,1:7].T.reshape(6,nx,ny)
    strain_error=float(np.max(np.abs(gpu-ref)))
    s=constitutive(gpu-eigen)
    gres=gradient(s)
    if not constrained:gres[2*n:]-=rootn*target
    equilibrium=float(np.linalg.norm(gres)/max(np.linalg.norm(rhs),1e-12))
    # Independent tensor energy derivative via central finite differences at fixed total strain.
    def energy(hh,total=gpu):
        cc=cm[:,None,None]+(cp-cm)[:,None,None]*hh
        e=total.copy();e[:3]-=eig[:,None,None]*hh
        return .5*cc[0]*np.sum(e[:3]**2,axis=0)+cc[1]*(e[0]*e[1]+e[0]*e[2]+e[1]*e[2])+2*cc[2]*np.sum(e[3:]**2,axis=0)
    def derivative(total):
        # W(h) is cubic at fixed total strain; Richardson removes the O(delta^2)
        # central-difference term, essential for a truly stress-free zero-force case.
        delta=1e-5
        full=(energy(h+delta,total)-energy(h-delta,total))/(2*delta)
        half=(energy(h+delta/2,total)-energy(h-delta/2,total))/delta
        return (4*half-full)/3*1e11
    df=derivative(gpu)
    force_error=float(np.max(np.abs(data[:,7].reshape(shape)-df))/max(np.max(np.abs(df)),1))
    assert strain_error<2e-9,(path,'strain',strain_error)
    assert equilibrium<2e-7,(path,'equilibrium',equilibrium)
    assert force_error<2e-7,(path,'force',force_error)
    assert np.max(np.abs(gpu[[3,4]]))<1e-15
    if mode==2:assert np.max(np.abs(gpu[2]))<1e-15
    else:
        assert np.ptp(gpu[2])<1e-15
        if not constrained:assert abs(np.mean(s[2]))<2e-8
    result=dict(file=str(path),strain_max_abs_error=strain_error,equilibrium_relative_residual=equilibrium,force_relative_error=force_error,
                mean_strain=np.mean(gpu,axis=(1,2)).tolist(),mean_stress_pa=(np.mean(s,axis=(1,2))*1e11).tolist())
    if with_fields:
        result['reference_strain']=ref
        result['reference_force']=derivative(ref)
        mean=np.mean(ref,axis=(1,2))
        work=(stress[0]*mean[0]+stress[1]*mean[1]+2*stress[2]*mean[5])*1e11 if not constrained else 0.
        result['reference_potential_density']=float(np.mean(energy(h,ref))*1e11-work)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('files',nargs='+',type=Path);p.add_argument('--nx',type=int,default=9);p.add_argument('--ny',type=int,default=7);p.add_argument('--mode',type=int,choices=[1,2],required=True);p.add_argument('--stress',type=float,nargs=3,default=[0,0,0]);p.add_argument('--dy',type=float,default=1.3);p.add_argument('--homogeneous',action='store_true');p.add_argument('--constrained',action='store_true');p.add_argument('--prescribed',type=float,nargs=6,default=[0]*6);a=p.parse_args()
    print(json.dumps([check(f,a.nx,a.ny,a.mode,a.dy,a.stress,a.homogeneous,a.constrained,a.prescribed) for f in a.files],indent=2))
