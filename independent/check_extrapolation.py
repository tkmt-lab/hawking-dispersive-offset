#!/usr/bin/env python3
"""60-digit Decimal normal equations vs NumPy SVD-based central fits."""
from decimal import Decimal as D, localcontext
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def intercept(x,y,degree=2):
    n=degree+1
    a=[[sum(v**(i+j) for v in x) for j in range(n)] for i in range(n)]
    b=[sum(v**i*w for v,w in zip(x,y)) for i in range(n)]
    for j in range(n):
        pivot=max(range(j,n),key=lambda i:abs(a[i][j]))
        a[j],a[pivot]=a[pivot],a[j]; b[j],b[pivot]=b[pivot],b[j]
        p=a[j][j]; a[j]=[z/p for z in a[j]]; b[j]/=p
        for i in range(n):
            if i==j: continue
            p=a[i][j]; a[i]=[z-p*w for z,w in zip(a[i],a[j])]; b[i]-=p*b[j]
    return b[0]


def main():
    data=json.loads((ROOT/'refined.json').read_text(),parse_float=D)
    summary=json.loads((ROOT/'analysis.json').read_text(),parse_float=D)
    out={}
    with localcontext() as ctx:
        ctx.prec=60
        for name in ['T05','B4','N1','N2']:
            lamvals=[D(20),D(40),D(80),D(160)]
            yy=[]
            for lam in lamvals:
                rr=[r for r in data if r['profile']==name and r['Lambda']==lam]
                delta=intercept([r['omega']**2 for r in rr],[r['T_over_TH_minus1'] for r in rr])
                yy.append(lam**2*delta)
            b0=intercept([1/lam**2 for lam in lamvals],yy)
            difference=b0-summary['profiles'][name]['b0_num']
            assert abs(difference)<D('1e-11')
            out[name]={'b0_decimal':str(b0),'minus_numpy':str(difference)}
    report={'status':'PASS','precision_decimal_digits':60,'profiles':out}
    (ROOT/'extrapolation-check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
