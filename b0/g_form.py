"""Rewrite b0, b1 (and c) in terms of g(u) = v'(x) expressed as a function of u = 1 + v  (u' = g(u), g(0) = kappa = 1).
g(u) = 1 + g1 u + g2 u^2 + ... ; v^(n)(0) = [(g d/du)^(n-1) g](0).  Exact rational arithmetic."""
import json, os
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__))
NG = 6
def cadd(p, q, s=1):
    r = dict(p)
    for k, c in q.items():
        r[k] = r.get(k, 0) + s * c
        if r[k] == 0: del r[k]
    return r
def cmul(p, q):
    r = {}
    for k1, c1 in p.items():
        for k2, c2 in q.items():
            k = tuple(a + b for a, b in zip(k1, k2)); r[k] = r.get(k, 0) + c1 * c2
    return {k: c for k, c in r.items() if c}
ONE = {(0,) * NG: Fr(1)}
def gc(n):  # coefficient g_n as a Q[g]-poly
    return ONE if n == 0 else {tuple(1 if i == n - 1 else 0 for i in range(NG)): Fr(1)}
UMAX = 8
g = {n: gc(n) for n in range(NG + 1)}  # u-poly: power -> coeff
def umul(p, q):
    r = {}
    for a, ca in p.items():
        for b, cb in q.items():
            if a + b <= UMAX: r[a + b] = cadd(r.get(a + b, {}), cmul(ca, cb))
    return {k: v for k, v in r.items() if v}
def uder(p):
    return {n - 1: {k: c * n for k, c in cc.items()} for n, cc in p.items() if n > 0}
D = {1: g}
for n in range(2, 8):
    D[n] = umul(g, uder(D[n - 1]))
vder = {n: D[n].get(0, {}) for n in range(1, 8)}  # v^(n)(0) as Q[g]-polys
assert vder[1] == ONE
loc = json.load(open(os.path.join(HERE, 'local_expansion_out_order6.json')))
def subst(name):
    tot = {}
    for k, v in loc[name].items():
        ex = tuple(int(t) for t in k.strip('()').split(','))
        term = {(0,) * NG: Fr(v)}
        for i, e in enumerate(ex):
            for _ in range(e): term = cmul(term, vder[i + 2])
        tot = cadd(tot, term)
    return tot
def fmt(p):
    s = []
    for k, c in sorted(p.items()):
        mon = '*'.join(f'g{i+1}' + (f'^{e}' if e > 1 else '') for i, e in enumerate(k) if e) or '1'
        s.append(f'{c}*{mon}')
    return ' + '.join(s) if s else '0'
res = {}
for name in ('b0_vmonomials', 'b1_vmonomials', 'delta_Lam^-2_vmonomials', 'delta_Lam^-4_vmonomials', 'F_Lam^-2_vmonomials'):
    res[name.replace('vmonomials', 'gform')] = fmt(subst(name))
    print(name, '->', res[name.replace('vmonomials', 'gform')])
json.dump(res, open(os.path.join(HERE, 'g_form_out.json'), 'w'), indent=1)
