"""b0: check each step of the derivation numerically.

For each profile and Lambda:
  (1) zero-mode quantities from the w = 0 ODE (zero_mode.py): F, delta, T(0) by the exact formula;
  (2) the full 3x3 scattering (src/hawking_bvp.py) at small w, extrapolated to w -> 0:
        T_row(0), C_beta = lim w |beta|^2, G0 = lim |A|^2;
  (3) the local expansion (local_expansion.py):
        T(0)/T_H = 1 + b0/Lam^2 + b1/Lam^4,  delta = -(1 + v'')/(4 Lam^2) + d1/Lam^4,
        F/T_H = 1 + (b0 - (1+v'')/4)/Lam^2 + ...
Usage: check_steps.py OUTNAME key1 key2 ...   (keys from PROFILES below or from blind_profiles.py)"""
import json, os, sys
from fractions import Fraction
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
sys.path.insert(0, HERE)
import hawking_bvp as hb
from zero_mode import zero_mode

PROFILES = {  # profiles of the curvature study (not prediction-first)
    'tanh05': lambda: hb.Tanh(0.5),
    'sh_q2_tp025': lambda: hb.SumTanh([(1.0, 2.0, 0.25)]),
    'sum_asym1': lambda: hb.SumTanh([(0.5, 2.0, 0.25), (0.5, 1.0, -0.5)]),
    'sum_mix4': lambda: hb.SumTanh([(0.6, 1.5, -0.3), (0.4, 3.0, 0.1)]),
    'blind_b': lambda: hb.SumTanh([(0.6, 1.2, 0.4), (0.4, 3.5, -0.15)]),
}
try:
    from blind_profiles import BLIND
    PROFILES.update(BLIND)
except ImportError:
    pass


def taylor(prof, nmax=7, r=None, n=64):
    """v^(k)(0)/k! for k <= nmax via a Cauchy integral on |z| = r (profiles are analytic, numpy-complex safe)."""
    if r is None:
        r = getattr(prof, 'cauchy_r', 0.15)
    th = 2 * np.pi * np.arange(n) / n
    z = r * np.exp(1j * th)
    vals = np.array([complex(prof.v(zz)) for zz in z])
    a = np.fft.fft(vals) / n
    return np.array([(a[k] / r ** k).real for k in range(nmax + 1)])


def load_poly(name):
    d = json.load(open(os.path.join(HERE, 'local_expansion_out_order6.json')))[name]
    return [(tuple(int(t) for t in k.strip('()').split(',')), float(Fraction(v))) for k, v in d.items()]


def eval_poly(poly, vd):  # vd[n] = v^(n)(0), n = 2..7
    return sum(c * np.prod([vd[i + 2] ** e for i, e in enumerate(k)]) for k, c in poly)


def bvp_limit(prof, Lam, ws=(0.02, 0.01, 0.005, 0.0025)):
    out = []
    for w in ws:
        res = hb.solve(prof, w, Lam)
        t = hb.temps(res, w)
        out.append((w, t['T_row'], w * t['beta2_row'] * 2 * np.pi, t['A2_row'], res['unit_err'], res['flux_err']))
    W = np.array([o[0] for o in out])
    # T_row(w) = T0 + a w^2 + b w^4 + d w^6 ; w|beta|^2 and |A|^2: cubic polynomial in w (odd powers appear via -w/2)
    A = np.vstack([np.ones(4), W ** 2, W ** 4, W ** 6]).T
    T0 = np.linalg.solve(A, [o[1] for o in out])[0]
    B = np.vstack([np.ones(4), W, W ** 2, W ** 3]).T
    Cb = np.linalg.solve(B, [o[2] for o in out])[0]
    G0 = np.linalg.solve(B, [o[3] for o in out])[0]
    return dict(T0_over_TH=float(T0), Cbeta_over_TH=float(Cb), G0=float(G0), ws=list(ws),
                raw=[list(map(float, o)) for o in out], max_unit_err=float(max(o[4] for o in out)),
                max_flux_err=float(max(o[5] for o in out)))


def main():
    outname = sys.argv[1]
    keys = sys.argv[2:]
    Lams = [float(x) for x in os.environ.get('B0_LAMS', '10,20,40,80').split(',')]
    do_bvp = os.environ.get('B0_BVP', '1') == '1'
    bvp_max = float(os.environ.get('B0_BVP_MAXLAM', '80'))
    b0p, b1p, d0p, d1p = (load_poly(k) for k in ('b0_vmonomials', 'b1_vmonomials', 'delta_Lam^-2_vmonomials', 'delta_Lam^-4_vmonomials'))
    rec = {}
    for key in keys:
        prof = PROFILES[key]()
        a = taylor(prof)
        from math import factorial
        vd = {n: a[n] * factorial(n) for n in range(8)}
        b0 = eval_poly(b0p, vd); b1 = eval_poly(b1p, vd); d0 = eval_poly(d0p, vd); d1 = eval_poly(d1p, vd)
        print(f"== {key}: v(0)={vd[0]:+.3e} v'={vd[1]:.12f} v''={vd[2]:+.6f} v'''={vd[3]:+.6f} v''''={vd[4]:+.6f}  b0={b0:+.6f} b1={b1:+.6f}", flush=True)
        R = dict(vd={str(n): vd[n] for n in vd}, b0_local=b0, b1_local=b1, delta_c2=d0, delta_c4=d1, per_Lam=[])
        for Lam in Lams:
            z = zero_mode(prof, Lam)
            pred_T = 1 + b0 / Lam ** 2 + b1 / Lam ** 4
            pred_d = d0 / Lam ** 2 + d1 / Lam ** 4
            row = dict(Lam=Lam, zero=z, pred_T0_over_TH=pred_T, pred_delta=pred_d)
            line = (f"  Lam={Lam:5.0f}  zero-mode: T0/TH-1={z['T0_over_TH'] - 1:+.10e}  Lam^2(.)={Lam**2 * (z['T0_over_TH'] - 1):+.7f}"
                    f"  local: {pred_T - 1:+.10e} | delta={z['delta']:+.6e} local {pred_d:+.6e}")
            if do_bvp and Lam <= bvp_max:
                bl = bvp_limit(prof, Lam)
                row['bvp'] = bl
                line += (f"\n            BVP(w->0): T0/TH-1={bl['T0_over_TH'] - 1:+.10e}  diff vs zero-mode {bl['T0_over_TH'] - z['T0_over_TH']:+.1e}"
                         f" | C_beta/TH-1={bl['Cbeta_over_TH'] - 1:+.6e} pred {z['Cbeta_pred_over_TH'] - 1:+.6e}"
                         f" | G0={bl['G0']:.6e} pred {z['G0_pred']:.6e}")
            print(line, flush=True)
            R['per_Lam'].append(row)
        rec[key] = R
    json.dump(rec, open(os.path.join(HERE, f'{outname}.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
