"""Secondary check of the Lam^-4 coefficient b1 with tighter ODE tolerances (zero-mode formula only).
Fit Lam^4 (T0/T_H - 1 - b0/Lam^2) = b1 + e2/Lam^2 + e3/Lam^4 on several Lam windows."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from check_steps import PROFILES
from zero_mode import zero_mode
known = json.load(open(os.path.join(HERE, 'check_known.json')))
known.update(json.load(open(os.path.join(HERE, 'check_blind.json'))))
out = {}
Lams = [10.0, 14.0, 20.0, 28.0, 40.0, 56.0, 80.0]
for key in sys.argv[1:]:
    prof = PROFILES[key]()
    b0, b1 = known[key]['b0_local'], known[key]['b1_local']
    t = {}
    for L in Lams:
        z1 = zero_mode(prof, L, rtol=1e-13, atol=1e-16)
        t[L] = z1['T0_over_TH'] - 1
    y = np.array([L ** 4 * (t[L] - b0 / L ** 2) for L in Lams])
    fits = {}
    for lo in (2, 3):
        Ls = np.array(Lams[lo:])
        A = np.vstack([np.ones(len(Ls)), Ls ** -2, Ls ** -4]).T
        fits[f'Lam>={Ls[0]:.0f}'] = float(np.linalg.lstsq(A, y[lo:], rcond=None)[0][0])
    out[key] = dict(b1_pred=b1, fits=fits, y=dict(zip(map(str, Lams), y.tolist())))
    print(f"{key:12s} b1 pred {b1:+.5f}  fits {fits}  Lam^4(T-1-b0/Lam^2): " + ' '.join(f'{v:+.4f}' for v in y), flush=True)
json.dump(out, open(os.path.join(HERE, 'b1_check.json'), 'w'), indent=1)
