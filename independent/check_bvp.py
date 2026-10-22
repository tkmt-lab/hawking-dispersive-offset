#!/usr/bin/env python3
"""Cross-check original fourth-order equation with adaptive cubic BVP collocation.

Profile and dispersion root routines are shared; state, ODE, discretization and
boundary solve are different from the primary multiple-shooting implementation.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_bvp
from scatter import profile, modes


def solve_original(name, lam, omega, extent=7, tol=1e-9):
    start=time.monotonic()
    def matrix(x):
        v,dv=profile(name,x)
        a=np.zeros((4,4,len(x)),complex)
        a[0,1]=lam; a[1,2]=lam; a[2,3]=lam
        a[3,0]=(omega**2+1j*omega*dv)/lam
        a[3,1]=2j*omega*v-2*v*dv
        a[3,2]=lam*(1-v*v)
        return a
    def fun(x,y):
        return np.einsum('ijn,jcn->icn',matrix(x),y.reshape(4,3,-1)).reshape(12,-1)
    def jac(x,y):
        return np.einsum('ijn,cd->icjdn',matrix(x),np.eye(3)).reshape(12,12,-1)
    ml,mr=[modes(float(profile(name,x)[0]),lam,omega) for x in [-extent,extent]]
    def eigenbasis(m):
        q=m['k']/lam
        return np.stack([q*0+1,1j*q,-q*q,-1j*q**3])/m['norm']
    def idx(m,vg,om):
        out=np.flatnonzero(m['real'] & (m['vg'].real*vg>0) & (m['omega_comoving'].real*om>0))
        assert len(out)==1
        return int(out[0])
    il=[idx(ml,1,1),idx(ml,1,-1)]
    ol=[idx(ml,-1,1),idx(ml,-1,-1)]
    ir=idx(mr,-1,1); ou=idx(mr,1,1)
    grow=int(np.flatnonzero(mr['k'].imag < -1e-9)[0])
    bl,br=np.linalg.inv(eigenbasis(ml)),np.linalg.inv(eigenbasis(mr))
    left_rhs=np.array([[0,1,0],[0,0,1]],complex)
    right_rhs=np.array([[1,0,0],[0,0,0]],complex)
    def bc(ya,yb):
        return np.concatenate([(bl[il]@ya.reshape(4,3)-left_rhs).ravel(),
                               (br[[ir,grow]]@yb.reshape(4,3)-right_rhs).ravel()])
    def bc_jac(ya,yb):
        dl=np.zeros((12,12),complex); dr=dl.copy()
        dl[:6]=np.kron(bl[il],np.eye(3))
        dr[6:]=np.kron(br[[ir,grow]],np.eye(3))
        return dl,dr
    x=np.linspace(-extent,extent,int(np.ceil(4*extent*lam))+1)
    # Explicit real/imaginary split: the first native-complex pilot did not
    # converge (recorded separately); no output from that pilot is accepted.
    unpack=lambda z: z[:12]+1j*z[12:]
    pack=lambda z: np.concatenate([z.real,z.imag],axis=0)
    def realjac(a):
        return np.concatenate([np.concatenate([a.real,-a.imag],axis=1),
                               np.concatenate([a.imag,a.real],axis=1)],axis=0)
    def rfun(x,y): return pack(fun(x,unpack(y)))
    def rjac(x,y): return realjac(jac(x,unpack(y)))
    def rbc(ya,yb): return pack(bc(unpack(ya),unpack(yb)))
    def rbcjac(ya,yb): return tuple(realjac(a) for a in bc_jac(unpack(ya),unpack(yb)))
    sol=solve_bvp(rfun,rbc,x,np.zeros((24,len(x))),fun_jac=rjac,bc_jac=rbcjac,
                  tol=tol,bc_tol=1e-10,max_nodes=30000)
    yy=unpack(sol.y)
    cl=bl@yy[:,0].reshape(4,3); cr=br@yy[:,-1].reshape(4,3)
    s=np.stack([cr[ou],cl[ol[0]],cl[ol[1]]])
    alpha2,beta2,g=abs(s[0,1])**2,abs(s[0,2])**2,abs(s[0,0])**2
    t=omega/np.log(alpha2/beta2)
    eta=np.diag([1,1,-1])
    return {'profile':name,'Lambda':lam,'omega':omega,'extent':extent,'tolerance':tol,
            'success':bool(sol.success),'message':sol.message,'nodes':len(sol.x),
            'rms_residual_max':float(sol.rms_residuals.max()),'bc_residual_max':float(abs(bc(yy[:,0],yy[:,-1])).max()),
            'alpha2':float(alpha2),'beta2':float(beta2),'G':float(g),'T_ratio':float(t),
            'T_over_TH_minus1':float(2*np.pi*t-1),
            'pseudo_unitarity_max':float(abs(s.conj().T@eta@s-eta).max()),
            'S_complex_pairs':np.stack([s.real,s.imag],axis=-1).tolist(),'seconds':time.monotonic()-start}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--profiles',nargs='+',default=['T05'])
    p.add_argument('--lambdas',nargs='+',type=float,default=[10.])
    p.add_argument('--omega',type=float,default=.01)
    p.add_argument('--tol',type=float,default=1e-9)
    p.add_argument('--output',required=True)
    args=p.parse_args(); rows=[]
    for name in args.profiles:
        for lam in args.lambdas:
            row=solve_original(name,lam,args.omega,tol=args.tol); rows.append(row)
            Path(args.output).write_text(json.dumps(rows,indent=2)+'\n')
            print(name,lam,row['success'],'delta',row['T_over_TH_minus1'],'eta',row['pseudo_unitarity_max'],'nodes',row['nodes'],'sec',row['seconds'],flush=True)
            if not row['success']:
                raise SystemExit(1)
