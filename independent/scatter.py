#!/usr/bin/env python3
"""Independent scattering solver; definitions only, no imports from the main implementation."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import leggauss
from scipy.special import erf
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def profile(name, x):
    x = np.asarray(x)
    if name == "T05":
        z, dz, a = 2*x, 2+0*x, .5
    elif name == "B4":
        z, dz, a = x/.6-.7*x*x+.5*x**3, 1/.6-1.4*x+1.5*x*x, .6
    elif name == "N1":
        z, dz, a = 2*x+.4*x*x+.3*x**3, 2+.8*x+.9*x*x, .5
    elif name == "N2":
        a = .7*np.sqrt(np.pi)/2*np.exp((.3/.7)**2)
        return (-1+a*(erf((x+.3)/.7)-erf(.3/.7)),
                a*2/np.sqrt(np.pi)/.7*np.exp(-((x+.3)/.7)**2))
    else:
        raise ValueError(name)
    t = np.tanh(z)
    return -1+a*t, a*dz*(1-t*t)


def matrix(name, x, lam, omega):
    v, _ = profile(name, x)
    a = np.zeros(np.shape(v)+(4,4), dtype=complex)
    a[...,0,1] = lam
    a[...,1,2] = lam
    a[...,2,0] = 1j*omega*v
    a[...,2,1] = lam*(1-v*v)
    a[...,2,3] = 1j*omega
    a[...,3,0] = -1j*omega
    a[...,3,1] = lam*v
    return a


def gauss_table(stages):
    c, b = leggauss(stages)
    c, b = (c+1)/2, b/2
    a = np.empty((stages, stages))
    for j in range(stages):
        p = Polynomial.fromroots(np.delete(c, j))
        p = p/p(c[j])
        pi = p.integ()
        a[:,j] = pi(c)-pi(0)
    return c, a, b


def transfers(name, grid, lam, omega, method, stages, rtol):
    n = len(grid)-1
    h = np.diff(grid)
    out = np.empty((n,4,4), complex)
    if method == "gauss":
        c, a, b = gauss_table(stages)
        for start in range(0,n,128):
            end = min(start+128,n)
            hh = h[start:end]
            aa = matrix(name, grid[start:end,None]+hh[:,None]*c,lam,omega)
            # Stage values X_i = I + h sum_j a_ij A_j X_j.
            blocks = -hh[:,None,None,None,None]*a[None,:,None,:,None]*aa[:,None,:,:,:].transpose(0,1,3,2,4)
            lhs = blocks.reshape(-1,4*stages,4*stages)+np.eye(4*stages)
            rhs = np.broadcast_to(np.tile(np.eye(4),(stages,1)), (end-start,4*stages,4))
            xx = np.linalg.solve(lhs,rhs).reshape(-1,stages,4,4)
            out[start:end] = np.eye(4)+hh[:,None,None]*np.einsum("j,njab,njbc->nac",b,aa,xx)
    elif method == "dop853":
        # Separate explicit RK code, vectorized over short independent intervals.
        for start in range(0,n,128):
            end = min(start+128,n)
            hh = h[start:end]
            def rhs(t, y):
                aa = matrix(name,grid[start:end]+hh*t,lam,omega)
                return (hh[:,None,None]*(aa@y.reshape(-1,4,4))).ravel()
            initial = np.broadcast_to(np.eye(4,dtype=complex),(end-start,4,4)).copy()
            sol = solve_ivp(rhs,(0,1),initial.ravel(),method="DOP853",rtol=rtol,atol=rtol*.01)
            if not sol.success:
                raise RuntimeError(sol.message)
            out[start:end] = sol.y[:,-1].reshape(-1,4,4)
    else:
        raise ValueError(method)
    return out


def modes(v, lam, omega):
    q = np.roots([1,0,1-v*v,2*v*omega/lam,-(omega/lam)**2]).astype(complex)
    for _ in range(3):
        q -= (q**4+(1-v*v)*q*q+2*v*omega/lam*q-(omega/lam)**2)/(4*q**3+2*(1-v*v)*q+2*v*omega/lam)
    k = q*lam
    real = abs(k.imag)<1e-9
    k[real] = k[real].real
    om = omega-v*k
    vg = v+(k+2*k**3/lam**2)/om
    norm = np.ones(4)
    norm[real] = np.sqrt(abs((om*vg)[real]))
    basis = np.stack([np.ones(4),1j*k/lam,-(k/lam)**2,v-omega/k])/norm
    return {"k":k,"omega_comoving":om,"vg":vg,"real":real,"basis":basis,"norm":norm}


def current(y, v, lam, omega):
    # Physical derivatives and given sesquilinear J, without mode flux formula.
    f, d1, d2, p = y[0],lam*y[1],lam*lam*y[2],y[3]
    d3 = lam*lam*(1j*omega*(p+v*f)+(1-v*v)*d1)
    df = -1j*omega*f+v*d1
    outer = lambda a,b: np.outer(a.conj(), b)
    j = v*(outer(f,df)-outer(df,f))-(outer(f,d1)-outer(d1,f))
    j += (outer(f,d3)-outer(d1,d2)+outer(d2,d1)-outer(d3,f))/lam**2
    return .5j*j


def solve(name, lam, omega, extent=7., step=1., stages=6, method="gauss", rtol=2e-13):
    start = time.monotonic()
    # step caps h*Lambda. Profile is sampled continuously inside each interval.
    n = int(np.ceil(2*extent*lam/step))
    grid = np.linspace(-extent,extent,n+1)
    tt = transfers(name,grid,lam,omega,method,stages,rtol)
    vl, vr = float(profile(name,-extent)[0]), float(profile(name,extent)[0])
    ml, mr = modes(vl,lam,omega), modes(vr,lam,omega)
    def select(m, sign_vg, sign_om):
        indices = np.flatnonzero(m["real"] & (m["vg"].real*sign_vg>0) & (m["omega_comoving"].real*sign_om>0))
        if len(indices)!=1:
            raise ValueError((name,lam,omega,"channel count",indices))
        return int(indices[0])
    ilp, iln = select(ml,1,1), select(ml,1,-1)
    olp, oln = select(ml,-1,1), select(ml,-1,-1)
    ir, ou = select(mr,-1,1), select(mr,1,1)
    growing = np.flatnonzero(mr["k"].imag < -1e-9)
    if sum(ml["real"])!=4 or sum(mr["real"])!=2 or len(growing)!=1:
        raise ValueError("unexpected dispersion regime")
    growing = int(growing[0])
    invl, invr = np.linalg.inv(ml["basis"]), np.linalg.inv(mr["basis"])
    rows = (4*np.arange(n)[:,None,None]+np.arange(4)[None,:,None]+np.zeros((1,1,4),int)).ravel()
    cols = (4*np.arange(n)[:,None,None]+np.zeros((1,4,1),int)+np.arange(4)[None,None,:]).ravel()
    rr = np.concatenate([rows, np.arange(4*n),np.repeat(np.arange(4*n,4*n+4),4)])
    cc = np.concatenate([cols, np.arange(4,4*(n+1)),np.tile(np.arange(4),2),np.tile(4*n+np.arange(4),2)])
    dd = np.concatenate([-tt.ravel(),np.ones(4*n),invl[[ilp,iln]].ravel(),invr[[ir,growing]].ravel()])
    a = coo_matrix((dd,(rr,cc)),shape=(4*(n+1),4*(n+1))).tocsc()
    rhs = np.zeros((4*(n+1),3),complex)
    rhs[4*n,1],rhs[4*n+1,2],rhs[4*n+2,0] = 1,1,1
    lu = splu(a)
    yy = lu.solve(rhs)
    # One residual correction improves sparse solve accuracy with no changed model.
    yy += lu.solve(rhs-a@yy)
    y = yy.reshape(n+1,4,3)
    cl, cr = invl@y[0], invr@y[-1]
    s = np.stack([cr[ou],cl[olp],cl[oln]])
    eta = np.diag([1.,1.,-1.])
    defect = s.conj().T@eta@s-eta
    alpha2,beta2,g = abs(s[0,1])**2,abs(s[0,2])**2,abs(s[0,0])**2
    temperature = omega/np.log(alpha2/beta2)
    samples = np.linspace(0,n,7,dtype=int)
    fluxes = [current(y[j],float(profile(name,grid[j])[0]),lam,omega) for j in samples]
    mode_error = 0.
    for m,v in [(ml,vl),(mr,vr)]:
        inds = np.flatnonzero(m["real"])
        flux = current(m["basis"][:,inds],v,lam,omega)
        expected = np.diag(np.sign((m["omega_comoving"]*m["vg"])[inds].real))
        mode_error = max(mode_error,float(np.max(abs(flux-expected))))
    def encode(z):
        return np.stack([z.real,z.imag],axis=-1).tolist()
    return {"profile":name,"Lambda":lam,"omega":omega,"method":method,"stages":stages,
            "step":step,"extent":extent,"rtol":rtol,"intervals":n,
            "alpha2":float(alpha2),"beta2":float(beta2),"G":float(g),
            "T_ratio":float(temperature),"T_over_TH_minus1":float(temperature*2*np.pi-1),
            "pseudo_unitarity_max":float(np.max(abs(defect))),
            "u_row_identity_error":float(abs(alpha2-beta2-(1-g))),
            "flux_drift_max":float(max(np.max(abs(f-fluxes[0])) for f in fluxes)),
            "plane_wave_flux_error":mode_error,"linear_residual_max":float(np.max(abs(a@yy-rhs))),
            "boundary_forbidden_amplitude":float(abs(cr[growing]).max()),
            "S_complex_pairs":encode(s),"left_k_pairs":encode(ml["k"]),"right_k_pairs":encode(mr["k"]),
            "left_v":vl,"right_v":vr,"seconds":time.monotonic()-start}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profiles",nargs="+",default=["T05"])
    p.add_argument("--lambdas",nargs="+",type=float,default=[10.])
    p.add_argument("--omegas",nargs="+",type=float,default=[.02])
    p.add_argument("--extent",type=float,default=7.)
    p.add_argument("--step",type=float,default=1.)
    p.add_argument("--stages",type=int,default=6)
    p.add_argument("--method",choices=["gauss","dop853"],default="gauss")
    p.add_argument("--rtol",type=float,default=2e-13)
    p.add_argument("--output",required=True)
    args = p.parse_args()
    rows=[]
    for name in args.profiles:
        for lam in args.lambdas:
            for w in args.omegas:
                row=solve(name,lam,w,args.extent,args.step,args.stages,args.method,args.rtol)
                rows.append(row)
                Path(args.output).write_text(json.dumps(rows,indent=2)+"\n")
                print(name,lam,w,"delta=%.12g"%row["T_over_TH_minus1"],"eta=%.3g"%row["pseudo_unitarity_max"],"sec=%.2f"%row["seconds"],flush=True)


if __name__ == "__main__":
    main()
