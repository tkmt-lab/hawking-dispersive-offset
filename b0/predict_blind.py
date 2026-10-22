"""Write the blind predictions (local formulas only; no zero-mode / scattering run) to blind_predictions.json."""
import json, os, sys
from math import factorial
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from blind_profiles import BLIND
from check_steps import taylor, load_poly, eval_poly

polys = {k: load_poly(k) for k in ('b0_vmonomials', 'b1_vmonomials', 'delta_Lam^-2_vmonomials', 'delta_Lam^-4_vmonomials')}
out = {}
for key, mk in BLIND.items():
    prof = mk()
    L = prof.default_L()
    xs = np.linspace(-L, L, 20001)
    v = prof.v(xs)
    assert np.all(prof.dv(xs) >= 0) and np.all(np.diff(v) >= -1e-15), 'not monotone'
    assert np.all(v[xs < 0] < -1) and np.all(v[xs > 0] > -1) and v[-1] < 0
    a1 = taylor(prof, r=prof.cauchy_r); a2 = taylor(prof, r=1.5 * prof.cauchy_r)
    vd = {n: a1[n] * factorial(n) for n in range(8)}
    tay_diff = float(np.max(np.abs(a1 - a2)[:8] * np.array([factorial(n) for n in range(8)])))
    b0, b1, d2, d4 = (eval_poly(polys[k], vd) for k in polys)
    b0_closed = (vd[4] - 4 * vd[2] * vd[3] + 3 * vd[2] ** 3) / 12
    P = dict(name=prof.name, L=L, v_minus=float(prof.v(-L)), v_plus=float(prof.v(L)), vderivs=[vd[n] for n in range(8)],
             taylor_two_radii_maxdiff=tay_diff, b0=b0, b0_closed_form=b0_closed, b1=b1, delta_Lam2=d2, delta_Lam4=d4,
             G0_Lam4_leading=(1 + vd[2]) ** 2 / 64,
             pred_T0_over_TH_minus1={str(L_): b0 / L_ ** 2 + b1 / L_ ** 4 for L_ in (10, 20, 40, 80, 160, 320)})
    out[key] = P
    print(key, json.dumps({k: P[k] for k in ('name', 'v_minus', 'v_plus', 'taylor_two_radii_maxdiff', 'b0', 'b0_closed_form', 'b1', 'delta_Lam2', 'G0_Lam4_leading')}))
    print('   v^(n)(0):', [round(x, 6) for x in P['vderivs']])
out['_criteria'] = ('Success if (1) b0 extrapolated from the full scattering T_row(w->0) at Lam=20,40,80 '
                    '(fit b0 + e1/Lam^2 + e2/Lam^4) agrees with the predicted b0 within 1e-3*max(1,|b0|); '
                    '(2) the zero-mode formula T(0) agrees with the scattering limit within 1e-9 at every Lam; '
                    '(3) as a secondary test, b1 from the zero-mode values at Lam=20..160 agrees within 10%.')
json.dump(out, open(os.path.join(HERE, 'blind_predictions.json'), 'w'), indent=1)
