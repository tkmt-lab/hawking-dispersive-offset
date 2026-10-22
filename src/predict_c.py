"""Candidate closed form for c found after the shifted-tanh scan (2026-10-21):

    c = 2 kap * x4,   x(u) = u/kap + x2 u^2 + x3 u^3 + x4 u^4 + ...  the inverse series of u(x) = v(x) + 1,

i.e.  c = -(5/4) v''^3/kap^6 + (5/6) v'' v'''/kap^5 - (1/12) v''''/kap^4   (derivatives at the horizon).

Origin (heuristic, see report): in the p-representation the conserved-frequency
mode has phase W = -int X_w(p) dp with  w - v(X) p = F(p),  u = v(X)+1 = w/p - p^2/(2 Lam^2) + O(p^4).
The coefficient of ln p in W is the 1/p coefficient of X_w(p) = x(u(p)):
    Res = w/kap - 2 x4 w^3/Lam^2 + ...,   T = w/Res/(2 pi)  =>  T/T_H = 1 + 2 kap x4 w^2/Lam^2 + ...
This file writes predictions BEFORE the corresponding scans are run (pre-registration for the
profiles not used to find the formula)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))


def tanh_derivs(T):
    return dict(d1=1 - T * T, d2=-2 * T * (1 - T * T), d3=-2 * (1 - T * T) * (1 - 3 * T * T),
                d4=8 * T * (1 - T * T) * (2 - 3 * T * T))


def derivs_sumtanh(comps):
    """v^(n)(0), n=1..4, for SumTanh components (kap_i, q_i, t_i)."""
    out = np.zeros(5)
    for k, q, t in comps:
        d = tanh_derivs(-t)
        for n in range(1, 5):
            out[n] += k * q ** (n - 1) * d[f'd{n}'] / (1 - t * t)
    return out


def c_formula(v1, v2, v3, v4):
    return -1.25 * v2 ** 3 / v1 ** 6 + (5.0 / 6.0) * v2 * v3 / v1 ** 5 - v4 / (12.0 * v1 ** 4)


def c_dpls(v1, v2):
    return 0.25 - 0.375 * v2 / v1 ** 2


PROFILES = {
    # used to FIND the formula (not a test):
    'sh_q2_tp0125': [(1.0, 2.0, 0.125)], 'sh_q2_tm0125': [(1.0, 2.0, -0.125)],
    'sh_q2_tp025': [(1.0, 2.0, 0.25)], 'sh_q2_tm025': [(1.0, 2.0, -0.25)],
    'sh_q4_tp0125': [(1.0, 4.0, 0.125)], 'sh_q4_tm0125': [(1.0, 4.0, -0.125)],
    # pre-registered tests (asymmetric; some with v''(0) = 0):
    'sum_asym1': [(0.5, 2.0, 0.25), (0.5, 1.0, -0.5)],
    'sum_asym2': [(0.5, 2.0, -0.25), (0.5, 1.0, 0.5)],
    'sum_mix3': [(0.3, 3.0, 0.2), (0.7, 1.0, -0.1)],
    'sum_mix4': [(0.6, 1.5, -0.3), (0.4, 3.0, 0.1)],
    'sum_mix5': [(0.5, 3.0, 0.3), (0.5, 1.2, 0.0)],
}
if __name__ == '__main__':
    pred = {}
    for key, comps in PROFILES.items():
        d = derivs_sumtanh(comps)
        pred[key] = dict(comps=comps, v1=d[1], v2=d[2], v3=d[3], v4=d[4],
                         c_formula=c_formula(*d[1:]), c_DPLS=c_dpls(d[1], d[2]))
    # symmetric profiles (v' even => v'' = v'''' = 0): formula predicts c = 0
    for key in ('tanh05', 'tanh10', 'flat_m1', 'flat_m2', 'flat_m3', 'flat_m4'):
        pred[key] = dict(v2=0.0, v4=0.0, c_formula=0.0, c_DPLS=0.25, note='v\' even')
    json.dump(pred, open(os.path.join(OUT, 'c_predictions_preregistered.json'), 'w'), indent=1)
    for k, p in pred.items():
        print(f"{k:14s} v''={p['v2']:+.5f}  c_formula={p['c_formula']:+.6f}  c_DPLS={p['c_DPLS']:+.4f}")


# ---- stage 2 (written 2026-10-21 after the stage-1 scans): blind test of c and of the
# low-frequency offset guess  b0 = (v'''' - 4 v'' v''' + 3 v''^3)/12  (kap = 1), i.e.
#   T(w -> 0)/T_H - 1 = b0 kap^2/Lam^2 + ...   (guess formed from sh_* (b0 = 0) and sum_asym1;
#   sum_mix3/4/5 were already computed when the guess was formed, so they are not blind).
BLIND2 = {
    'blind_a': [(0.4, 2.5, 0.3), (0.35, 1.0, -0.2), (0.25, 4.0, 0.05)],
    'blind_b': [(0.6, 1.2, 0.4), (0.4, 3.5, -0.15)],
}


def b0_guess(v1, v2, v3, v4):
    return (v4 / v1 ** 4 - 4 * v2 * v3 / v1 ** 5 + 3 * v2 ** 3 / v1 ** 6) / 12.0


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'stage2':
    pred2 = {}
    for key, comps in BLIND2.items():
        d = derivs_sumtanh(comps)
        pred2[key] = dict(comps=comps, v1=d[1], v2=d[2], v3=d[3], v4=d[4], c_formula=c_formula(*d[1:]),
                          b0_guess=b0_guess(*d[1:]), c_DPLS=c_dpls(d[1], d[2]))
        print(key, pred2[key])
    json.dump(pred2, open(os.path.join(OUT, 'c_predictions_blind_stage2.json'), 'w'), indent=1)
