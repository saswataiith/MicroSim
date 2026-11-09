from pathlib import Path
import json,sys,numpy as np
records=[]
for p in sorted(Path(sys.argv[1] if len(sys.argv)>1 else 'build/c_api_fields').glob('*.csv')):
 mode=int(p.name.split('model')[1][0]);loading=int(p.name.split('loading')[1][0]);d=np.loadtxt(p,delimiter=',',skiprows=1);h=d[:,2].reshape(9,7)
 # C API supports rectangular grids; make independent dense variational operator.
 nx,ny=h.shape;N=nx*ny;root=np.sqrt(N);slots=[0,1,5,2] if mode==1 else [0,1,5];nm=len(slots)
 kx=2*np.pi*np.fft.fftfreq(nx)[:,None];ky=2*np.pi*np.fft.fftfreq(ny,1.3)[None,:]
 cp=np.array([4.5,1.5,1.2]);cm=np.array([3.,1.,.8]);c=cm[:,None,None]+(cp-cm)[:,None,None]*h
 eig=np.array([.005,.01,.003 if mode==1 else 0,0,0,.002]);e0=eig[:,None,None]*h
 def deriv(a,k):return np.fft.ifft2(1j*k*np.fft.fft2(a)).real
 def B(q):
  u=q[:2*N].reshape(2,nx,ny);e=np.zeros((6,nx,ny));e[0]=deriv(u[0],kx);e[1]=deriv(u[1],ky);e[5]=.5*(deriv(u[0],ky)+deriv(u[1],kx))
  if loading!=2:
   means=q[2*N:]/root;e[0]+=means[0];e[1]+=means[1];e[5]+=.5*means[2]
   if mode==1:e[2]=means[3]
  return e
 def law(e):
  s=np.zeros_like(e)
  for j in range(3):s[j]=c[0]*e[j]+c[1]*(e[(j+1)%3]+e[(j+2)%3])
  s[3:]=2*c[2]*e[3:];return s
 def Bt(s):
  a=[(-deriv(s[0],kx)-deriv(s[5],ky)).ravel(),(-deriv(s[5],kx)-deriv(s[1],ky)).ravel()]
  if loading!=2:a.append(root*np.mean(s[slots],axis=(1,2)))
  return np.concatenate(a)
 size=2*N+(nm if loading!=2 else 0);rhs=Bt(law(e0));target=np.zeros(6)
 if loading==1:target[[0,1,5]]=[.01,-.02,.005]
 if loading!=2:rhs[2*N:]+=root*target[slots]
 mat=np.column_stack([Bt(law(B(q))) for q in np.eye(size)]);assert np.max(abs(mat-mat.T))<1e-12
 q=np.linalg.lstsq(mat,rhs,rcond=1e-12)[0];ref=B(q);actual=d[:,3:9].T.reshape(6,nx,ny);err=np.max(abs(ref-actual));assert err<1e-10,(p,err)
 def W(hh):
  cc=cm[:,None,None]+(cp-cm)[:,None,None]*hh;e=actual-eig[:,None,None]*hh
  return .5*cc[0]*np.sum(e[:3]**2,axis=0)+cc[1]*(e[0]*e[1]+e[0]*e[2]+e[1]*e[2])+2*cc[2]*np.sum(e[3:]**2,axis=0)
 derivative=(W(h+1e-5)-W(h-1e-5))/2e-5;force=d[:,-1].reshape(nx,ny);ferr=np.max(abs(derivative-force));assert ferr<1e-10,(p,ferr)
 assert np.max(abs(W(h)-d[:,-2].reshape(nx,ny)))<1e-13
 records.append(dict(model=mode,loading=loading,max_strain_error=float(err),max_force_error=float(ferr)))
assert len(records)==6,'Expected both models and all three loading choices.'
print(json.dumps(records,indent=2))
