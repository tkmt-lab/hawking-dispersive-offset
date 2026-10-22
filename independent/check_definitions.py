#!/usr/bin/env python3
"""Direct algebra/flux checks, separate from fitting measured temperatures."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
import scatter


def main():
    worst={'state_equivalence_relative':0.,'current_conservation_relative':0.,
           'profile_derivative':0.,'constant_propagator':0.}
    tested=0
    for name in ['T05','B4','N1','N2']:
        v0,d0=scatter.profile(name,0.)
        assert abs(v0+1)<1e-14 and abs(d0-1)<1e-14
        for x in [-2.,-.3,0.,.2,1.5]:
            v,dv=scatter.profile(name,x)
            cs=scatter.profile(name,x+1e-20j)[0].imag/1e-20
            worst['profile_derivative']=max(worst['profile_derivative'],abs(cs-dv))
            for lam in [10.,160.]:
                for w in [.0025,.02]:
                    # Direct, unscaled original fourth-order system.
                    a=np.zeros((4,4),complex)
                    a[0,1]=a[1,2]=a[2,3]=1
                    a[3,:3]=lam**2*np.array([w*w+1j*w*dv,2j*w*v-2*v*dv,1-v*v])
                    t=np.diag([1,1/lam,1/lam**2,0]).astype(complex)
                    t[3]=[-v,(v*v-1)/(1j*w),0,1/(1j*w*lam**2)]
                    dt=np.zeros((4,4),complex)
                    dt[3,:2]=[-dv,2*v*dv/(1j*w)]
                    lhs=t@a+dt; rhs=scatter.matrix(name,x,lam,w)@t
                    err=np.max(abs(lhs-rhs))/max(1,np.max(abs(lhs)),np.max(abs(rhs)))
                    worst['state_equivalence_relative']=max(worst['state_equivalence_relative'],err)
                    # Matrix of J in (f,f',f'',f'''); its derivative must vanish.
                    k=np.zeros((4,4),complex); dk=k.copy()
                    k[0,0]=-2j*w*v; k[0,1]=v*v-1; k[1,0]=1-v*v
                    k[0,3]=k[2,1]=1/lam**2
                    k[1,2]=k[3,0]=-1/lam**2
                    dk[0,0]=-2j*w*dv; dk[0,1]=2*v*dv; dk[1,0]=-2*v*dv
                    terms=[a.conj().T@k,k@a,dk]
                    err=np.max(abs(sum(terms)))/max(1,*[np.max(abs(q)) for q in terms])
                    worst['current_conservation_relative']=max(worst['current_conservation_relative'],err)
                    tested+=1
    original=scatter.profile
    try:
        for v in [-2.,-1.5,-.5]:
            scatter.profile=lambda name,x: (v+np.asarray(x)*0,np.asarray(x)*0)
            grid=np.array([0.,.01,.02])
            tt=scatter.transfers('constant',grid,40.,.01,'gauss',6,2e-13)
            exact=expm(.01*scatter.matrix('constant',0.,40.,.01))
            worst['constant_propagator']=max(worst['constant_propagator'],float(np.max(abs(tt-exact))))
    finally:
        scatter.profile=original
    for key,value in worst.items():
        assert value<1e-11,(key,value)
    report={'status':'PASS','algebra_cases':tested,'errors':{k:float(v) for k,v in worst.items()},
            'scope':'same-agent alternative algebra; no independent-agent review'}
    Path(__file__).with_name('definition-checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__': main()
