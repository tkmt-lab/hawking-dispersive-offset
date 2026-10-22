"""Judge the pre-registered criteria (blind_predictions.json) and the same fits for the known profiles.
b0_ext: fit Lam^2 (T0/T_H - 1) = b0 + e1/Lam^2 + e2/Lam^4 on Lam = 20, 40, 80 (scattering limit, w -> 0).
b1_ext: fit Lam^4 (T0/T_H - 1 - b0_pred/Lam^2) = b1 + e/Lam^2 + e'/Lam^4 on Lam = 20..160 (zero-mode formula)."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
pred = json.load(open(os.path.join(HERE, 'blind_predictions.json')))
cfit = json.load(open(os.path.join(HERE, '..', 'results', 'c_final.json')))
lines, summ = [], {}
for fn, blind in (('check_known.json', False), ('check_blind.json', True)):
    rec = json.load(open(os.path.join(HERE, fn)))
    for key, R in rec.items():
        b0p, b1p = R['b0_local'], R['b1_local']
        per = {row['Lam']: row for row in R['per_Lam']}
        Ls = np.array([20.0, 40.0, 80.0])
        y = np.array([L * L * (per[L]['bvp']['T0_over_TH'] - 1) for L in Ls])
        A = np.vstack([np.ones(3), Ls ** -2, Ls ** -4]).T
        b0e = float(np.linalg.solve(A, y)[0])
        Lz = np.array([L for L in (20.0, 40.0, 80.0, 160.0) if L in per])
        yz = np.array([L ** 4 * (per[L]['zero']['T0_over_TH'] - 1 - b0p / L ** 2) for L in Lz])
        Az = np.vstack([np.ones(len(Lz)), Lz ** -2, Lz ** -4]).T
        b1e = float(np.linalg.lstsq(Az, yz, rcond=None)[0][0])
        maxdiff = max(abs(row['bvp']['T0_over_TH'] - row['zero']['T0_over_TH']) for row in R['per_Lam'] if 'bvp' in row)
        g0 = max(abs(row['bvp']['G0'] / row['zero']['G0_pred'] - 1) for row in R['per_Lam'] if 'bvp' in row)
        dl = [(row['Lam'], row['Lam'] ** 2 * row['zero']['delta'], R['delta_c2']) for row in R['per_Lam']]
        ok1 = abs(b0e - b0p) < 1e-3 * max(1, abs(b0p)); ok2 = maxdiff < 1e-9; ok3 = abs(b1e - b1p) < 0.1 * max(abs(b1p), 1e-3)
        S = dict(blind=blind, b0_pred=b0p, b0_ext_scattering=b0e, b1_pred=b1p, b1_ext_zeromode=b1e,
                 max_T0_diff_scattering_vs_zeromode=maxdiff, max_rel_G0_diff=g0,
                 delta_Lam2_zero_vs_local=dl, crit1=ok1, crit2=ok2, crit3=ok3,
                 b0_inf_frequency_fit=cfit.get(key, {}).get('b0_inf'))
        summ[key] = S
        lines.append(f"{key:12s} {'BLIND' if blind else 'known'}  b0 pred {b0p:+.6f}  ext(scatt) {b0e:+.6f}  diff {b0e - b0p:+.1e}"
                     f" | b1 pred {b1p:+.4f} ext(zero) {b1e:+.4f}"
                     f" | max|T0 scatt-zero| {maxdiff:.1e} | max rel G0 err {g0:.1e}"
                     f" | frequency-fit b0_inf {S['b0_inf_frequency_fit']}  | crit {ok1},{ok2},{ok3}")
open(os.path.join(HERE, 'b0_summary.txt'), 'w').write('\n'.join(lines) + '\n')
json.dump(summ, open(os.path.join(HERE, 'b0_summary.json'), 'w'), indent=1)
print('\n'.join(lines))
