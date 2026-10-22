import math
N=8
def mul(a,b): return [sum(a[i]*b[k-i] for i in range(k+1)) for k in range(N)]
def tanh_series(a):
    # tanh(a) for series a with a[0]=0 via t' relation: integrate t = sum; use odd Taylor of tanh
    c={1:1,3:-1/3,5:2/15,7:-17/315}
    out=[0.0]*N; p=[1.0]+[0.0]*(N-1)
    for n in range(1,N):
        p=mul(p,a)
        if n in c: out=[o+c[n]*q for o,q in zip(out,p)]
    return out
def b0_from(vs):  # vs = Taylor coeffs of v+1
    d=[vs[k]*math.factorial(k) for k in range(N)]
    k=d[1]; a,b,dd=d[2],d[3],d[4]
    return k,(dd/k**3-4*a*b/k**4+3*a**3/k**5)/12
# N1
a=[0,2,0.4,0.3,0,0,0,0]
t=tanh_series(a); vs=[0.5*x for x in t]
print("N1 kappa,b0",b0_from(vs), "v-,v+",-1.5,-0.5)
# N2: A[erf((x+s)/w)-erf(s/w)]
s,w=0.3,0.7; u=s/w; A=w*math.sqrt(math.pi)/2*math.exp(u*u)
v1=1; v2=-2*u/w; v3=(4*u*u-2)/w**2; v4=-(8*u**3-12*u)/w**3
b0=(v4-4*v2*v3+3*v2**3)/12
vp=-1+A*(1-math.erf(u)); vm=-1+A*(-1-math.erf(u))
print("N2 A",A,"b0",b0,"v-,v+",vm,vp)
# T0.5 and B4 checks
t=tanh_series([0,1/0.6*0+2,0,0,0,0,0,0]); print("T0.5", b0_from([0.5*x for x in t]))
t=tanh_series([0,1/0.6,-0.7,0.5,0,0,0,0]); print("B4", b0_from([0.6*x for x in t]))
